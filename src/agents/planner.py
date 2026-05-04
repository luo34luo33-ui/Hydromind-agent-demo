from typing import Dict, List, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class ModelingPlan(BaseModel):
    """Planner 结构化输出 - 水文模型构建蓝图"""
    
    reasoning: str = Field(
        description="根据流域属性分析为什么选择以下产汇流机制"
    )
    
    runoff_type: Literal["SCS-CN", "新安江", "HBV", "Tank Model"] = Field(
        description="选择最适合该流域的产流模型"
    )
    
    flow_routing: Literal["线性水库", "非线性水库", "Nash瞬时单位线", "马斯京根"] = Field(
        description="选择最适合的汇流或演进机制"
    )
    
    response_type: Literal["快速响应", "中等响应", "慢速响应"] = Field(
        description="水文响应速度"
    )
    
    param_suggestions: Dict[str, List[float]] = Field(
        description="建议参数及上下界列表，必须严格遵循 {'参数名': [下界, 上界]} 格式，如 {'k': [0.05, 0.8], 'S0': [0, 200]}"
    )
    
    description: str = Field(description="完整的建模方案描述")
    
    template_keywords: List[str] = Field(
        description="用于检索代码模板的关键词"
    )


PLANNER_SYSTEM_PROMPT = """你是一位资深水文学家，擅长根据流域特征设计概念性水文模型模拟方案。

你的任务：
1. 分析给定的流域属性（面积、坡度、渗透性、气候类型等）
2. 判定该流域的水文响应特征（快速/中等/慢速响应）
3. 提出一个产流+汇流模拟方案，推荐合适的模型参数

原则：
- 大面积 + 低坡度 + 干旱 → 慢速响应，适合低出流系数 + 较大储水容量
- 小面积 + 高坡度 + 湿润 → 快速响应，适合高出流系数 + 较小储水容量
- 中等面积 + 中等坡度 → 中等响应
- 气候湿润 → 蒸散发量较大
- 气候干旱 → 蒸散发量较小但降雨稀少

输出格式：
先给出 1-2 句总体判断，然后列出具体的产流方案和汇流方案，最后给出建议参数值。"""


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