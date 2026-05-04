from typing import Dict, List, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class ModelingPlan(BaseModel):
    """Planner 结构化输出 - 水文模型构建蓝图 (v1.3.1 模块化版本)"""
    
    reasoning: str = Field(
        description="根据流域属性分析为什么选择以下产流和汇流模块组合"
    )
    
    # 产流模块选择
    runoff_module_id: Literal[
        "scs_runoff", "xaj_runoff", "tank_runoff", "hbv_runoff", "simple_runoff"
    ] = Field(
        description="选择最适合该流域的产流模块: scs_runoff=SCS-CN产流, xaj_runoff=新安江蓄满产流, tank_runoff=Tank产流, hbv_runoff=HBV土壤水分, simple_runoff=简单产流"
    )
    
    # 汇流模块选择
    routing_module_id: Literal[
        "linear_routing", "nonlinear_routing", "nash_routing", "direct_routing"
    ] = Field(
        description="选择最适合的汇流模块: linear_routing=线性水库, nonlinear_routing=非线性水库, nash_routing=Nash单位线, direct_routing=直接汇流"
    )
    
    response_type: Literal["快速响应", "中等响应", "慢速响应"] = Field(
        description="水文响应速度"
    )
    
    param_suggestions: Dict[str, List[float]] = Field(
        description="建议参数及上下界列表，必须严格遵循 {'参数名_后缀': [下界, 上界]} 格式。产流参数用 _runoff 后缀，汇流参数用 _routing 后缀，如 {'k_routing': [0.05, 0.8], 'CN_runoff': [30, 95]}"
    )
    
    description: str = Field(description="完整的建模方案描述")


PLANNER_SYSTEM_PROMPT = """你是一位顶尖的水文学家。你的任务是根据流域特征，设计严谨的集总式水文模型模块化组合方案。

【🛑 强制水文选型铁律】 - 违反此规律将导致模拟彻底失败：
1. 湿润地区 / 南方流域 / 高渗透性：严禁使用超渗产流！必须选择【蓄满产流】模块（如 xaj_runoff, hbv_runoff）。
2. 干旱或半干旱地区 / 北方小流域 / 强降雨引发短时洪水：必须选择【超渗产流】模块（如 scs_runoff）。
3. 需要体现"地表快速+地下慢速"多重响应的复杂流域：优先选择【Tank 模型】模块（tank_runoff）。

【🛑 强制参数命名铁律】 - 为了防止下游全局优化算法（SCE-UA）发生参数命名空间冲突，你输出的 param_suggestions 必须严格遵守以下前缀命名规范：
1. 产流模块的参数，必须以模块名或特定前缀开头，例如："CN_runoff", "Smax_soil", "k1_tank"。
2. 汇流模块的参数，必须以 routing 结尾或加前缀，例如："k_routing", "S0_routing"。
3. 参数值必须是包含上下界的列表格式，例如：[0.05, 0.8]。绝对不允许输出单个数值的点估计！

请在 reasoning 字段中首先默念上述选型铁律，分析流域的干湿特征，然后再严格按 Pydantic 约束输出方案。

【可选产流模块】: scs_runoff, xaj_runoff, tank_runoff, hbv_runoff, simple_runoff
【可选汇流模块】: linear_routing, nonlinear_routing, nash_routing, direct_routing"""


class Planner:

    def __init__(self, openai_api_key, model_name="gpt-4o"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            openai_api_key=openai_api_key,
        )
        self._structured_llm = None

    def _get_structured_llm(self):
        if self._structured_llm is None:
            try:
                self._structured_llm = self.llm.with_structured_output(ModelingPlan)
            except Exception:
                self._structured_llm = None
        return self._structured_llm

    def plan(self, attributes, context, user_request="", use_structured=False):
        extra = ""
        if user_request:
            extra = f"\n用户额外需求（务必遵守）:\n{user_request}\n"
        
        user_message = (
            f"流域特征: {attributes}\n"
            f"参考知识:\n{context}\n"
            f"{extra}"
            f"请根据以上信息，提出产流+汇流模拟方案，并推荐模型参数。"
        )

        if use_structured:
            try:
                structured_llm = self._get_structured_llm()
                if structured_llm:
                    messages = [
                        ("system", "你是一位资深水文建模专家，请严格按照要求的结构输出水文建模蓝图。"),
                        ("human", user_message)
                    ]
                    result = structured_llm.invoke(messages)
                    return result
            except Exception as e:
                print(f"Planner 结构化输出失败，退回普通模式: {e}")

        prompt = ChatPromptTemplate.from_messages([
            ("system", PLANNER_SYSTEM_PROMPT),
            ("user", "{user_message}"),
        ])
        chain = prompt | self.llm
        response = chain.invoke({"user_message": user_message})
        return response.content