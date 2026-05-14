from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class ModelingPlan(BaseModel):
    """Planner 结构化输出 - 水文模型构建蓝图 (v1.3.1 模块化版本)"""
    
    reasoning: str = Field(
        description="根据流域属性分析为什么选择以下产流和汇流模块组合"
    )
    
    runoff_module_id: Literal[
        "scs_runoff", "xaj_runoff", "tank_runoff", "hbv_runoff", "simple_runoff"
    ] = Field(
        description="选择最适合该流域的产流模块: scs_runoff=SCS-CN产流, xaj_runoff=新安江蓄满产流, tank_runoff=Tank产流, hbv_runoff=HBV土壤水分, simple_runoff=简单产流"
    )
    
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


class Blueprint(BaseModel):
    """Mechanism-aware Blueprint (v1.3.5) - 基于机制的模型蓝图"""
    
    reasoning: str = Field(
        description="根据流域属性分析为什么选择以下机制组合"
    )
    
    runoff_mechanism: Literal[
        "saturation_excess", "infiltration_excess", "soil_moisture_accounting", "storage_runoff"
    ] = Field(
        description="选择最适合该流域的产流机制: saturation_excess=蓄满产流, infiltration_excess=超渗产流, soil_moisture_accounting=土壤水分账, storage_runoff=蓄水产流"
    )
    
    routing_mechanism: Literal[
        "linear_reservoir", "cascade", "nonlinear_reservoir", "nash_routing"
    ] = Field(
        description="选择最适合的汇流机制: linear_reservoir=线性水库, cascade=串联水库, nonlinear_reservoir=非线性水库, nash_routing=Nash单位线"
    )
    
    evap_mechanism: Literal[
        "layered_et", "soil_et"
    ] = Field(
        description="选择蒸散发机制: layered_et=分层蒸散发, soil_et=土壤蒸散发"
    )
    
    patterns: List[Literal["threshold", "reservoir", "cascade", "partition", "delay", "exponential_decay", "ratio"]] = Field(
        description="使用的水文结构模式列表"
    )
    
    constraints: List[str] = Field(
        description="物理约束列表，如 ['mass_conservation', 'runoff_nonnegative']"
    )
    
    param_suggestions: Dict[str, List[float]] = Field(
        description="建议参数及上下界列表"
    )
    
    description: str = Field(description="完整的机制组合方案描述")


PLANNER_SYSTEM_PROMPT = """你是一位顶尖的水文学家。你的任务是根据流域特征，设计严谨的集总式水文模型模块化组合方案。

【🛑 强制水文选型铁律】 - 违反此规律将导致模拟彻底失败：
1. 湿润地区 / 南方流域 / 高渗透性：严禁使用超渗产流！必须选择【蓄满产流】模块（如 xaj_runoff, hbv_runoff）。
2. 干旱或半干旱地区 / 北方小流域 / 强降雨引发短时洪水：必须选择【超渗产流】模块（如 scs_runoff）。
3. 需要体现"地表快速+地下慢速"多重响应的复杂流域：优先选择【Tank 模型】模块（tank_runoff）。

【🛑 强制参数命名铁律】 - 为了防止下游全局优化算法（SCE-UA）发生参数命名空间冲突，你输出的 param_suggestions 必须严格遵守以下前缀命名规范：
1. 产流模块的参数，必须以模块名或特定前缀开头，例如："CN_runoff", "Smax_soil", "k1_tank"。
2. 汇流模块的参数，必须以 routing 结尾或加前缀，例如："k_routing", "S0_routing"。
3. 参数值必须是包含上下界的列表格式，例如：[0.05, 0.8]。绝对不允许输出单个数值的点估计！

【🛑 权限优先级法则】 - 当系统指令与用户需求冲突时，必须严格遵循此优先级：
1. 【最高优先级】当用户需求与"强制水文选型铁律"冲突时，**必须服从铁律**，拒绝用户的物理上不可行的需求，并明确说明原因。
   - 例如：用户在湿润地区要求用"初损后损法"（超渗机制），你必须拒绝并选择蓄满产流模块，同时说明"该流域为湿润区，不适合超渗产流机制"。
2. 【第二优先级】当用户需求仅影响参数范围或优化目标时，可以合理调整但需在 reasoning 中说明。
3. 【最低优先级】当用户需求只是偏好性建议（如"想要更快的响应"），可以采纳但需符合铁律约束。

【🚨 关键警告】严禁"谄媚行为"（sycophancy）：不要为了满足用户需求而编造物理上不可能的模块组合！例如：
- ❌ 不要说"XAJ模块可以实现初损后损法"（XAJ是蓄满产流模型，物理上不可能实现超渗机制）
- ✅ 正确做法：明确告知用户该需求与物理规律冲突，提供符合铁律的替代方案

请在 reasoning 字段中首先默念上述选型铁律，分析流域的干湿特征，然后再严格按 Pydantic 约束输出方案。

【可选产流模块】: scs_runoff, xaj_runoff, tank_runoff, hbv_runoff, simple_runoff
【可选汇流模块】: linear_routing, nonlinear_routing, nash_routing, direct_routing"""

BLUEPRINT_PROMPT = """你是一位顶尖的水文学家。你的任务是根据流域特征，设计基于水文机制的模型构建方案。

【机制选型铁律】：
1. 湿润区 / 南方流域：产流→saturation_excess (蓄满产流), 蒸散发→layered_et (分层蒸散发)
2. 干旱区 / 北方流域：产流→infiltration_excess (超渗产流), 蒸散发→soil_et (土壤蒸散发)
3. 温带/通用：产流→soil_moisture_accounting (土壤水分账), 蒸散发→soil_et
4. 汇流：慢速响应→linear_reservoir, 快速/多水源→cascade, 通用→linear_reservoir

【参数命名规范】：
1. 产流参数以 _runoff 结尾
2. 汇流参数以 _routing 结尾
3. 参数值必须为上下界列表

【物理约束必选】：
- 所有方案必须包含 mass_conservation (水量守恒)
- 产流必须包含 runoff_nonnegative (径流非负)
- 蒸散发必须包含 et_limited_by_pet (蒸散发受限制)

请输出 Blueprint 格式的机制组合方案。"""


class Planner:

    def __init__(self, openai_api_key, model_name="gpt-4o", base_url: Optional[str] = None):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            openai_api_key=openai_api_key,
            base_url=base_url,
        )
        self._structured_llm = None

    def _get_structured_llm(self):
        if self._structured_llm is None:
            try:
                self._structured_llm = self.llm.with_structured_output(ModelingPlan)
            except Exception:
                self._structured_llm = None
        return self._structured_llm

    def _check_attributes_sufficient(self, attributes: str) -> bool:
        """检查流域特征是否足够充分"""
        if not attributes or attributes.strip() == "":
            return False
        
        required_keys = ["climate", "area", "permeability"]
        attributes_lower = attributes.lower()
        
        key_count = sum(1 for key in required_keys if key in attributes_lower)
        
        return key_count >= 2
    
    def _get_fallback_plan(self, attributes: str, user_request: str = "") -> ModelingPlan:
        """当流域信息不足时，提供兜底方案"""
        user_req_lower = user_request.lower() if user_request else ""
        
        if "tank" in user_req_lower:
            runoff_id = "tank_runoff"
        elif "xaj" in user_req_lower or "新安江" in user_request:
            runoff_id = "xaj_runoff"
        elif "scs" in user_req_lower or "cn" in user_req_lower:
            runoff_id = "scs_runoff"
        elif "hbv" in user_req_lower:
            runoff_id = "hbv_runoff"
        else:
            runoff_id = "simple_runoff"
        
        if "nash" in user_req_lower:
            routing_id = "nash_routing"
        elif "nonlinear" in user_req_lower or "非线性" in user_request:
            routing_id = "nonlinear_routing"
        elif "linear" in user_req_lower or "线性" in user_request:
            routing_id = "linear_routing"
        else:
            routing_id = "linear_routing"
        
        return ModelingPlan(
            reasoning="Basin attributes insufficient. Using default plan based on user request.",
            runoff_module_id=runoff_id,
            routing_module_id=routing_id,
            response_type="中等响应",
            param_suggestions={
                "k_routing": [0.05, 0.8],
                "S0_routing": [0, 200],
            },
            description=f"Fallback: runoff={runoff_id}, routing={routing_id}"
        )
    
    def plan(self, attributes, context, user_request="", use_structured=False):
        if not self._check_attributes_sufficient(attributes):
            print("[Fallback] Basin attributes insufficient, using default plan")
            return self._get_fallback_plan(attributes, user_request)
        
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

    def plan_blueprint(self, attributes, context, user_request="", use_structured=False):
        """
        基于机制生成 Blueprint（v1.3.5）
        
        从"选模型"升级到"设计机制组合"
        """
        if not self._check_attributes_sufficient(attributes):
            return self._generate_blueprint_fallback(user_request)
        
        extra = ""
        if user_request:
            extra = f"\n用户额外需求:\n{user_request}\n"
        
        user_message = (
            f"流域特征: {attributes}\n"
            f"参考知识:\n{context}\n"
            f"{extra}"
            f"请根据以上信息，提出基于水文机制的模型构建方案。"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", BLUEPRINT_PROMPT),
            ("user", "{user_message}"),
        ])
        chain = prompt | self.llm
        response = chain.invoke({"user_message": user_message})
        return response.content
    
    def _generate_blueprint_fallback(self, user_request: str = "") -> str:
        """生成兜底 Blueprint"""
        user_req_lower = user_request.lower() if user_request else ""
        
        if any(kw in user_req_lower for kw in ["湿润", "humid", "南方", "xaj"]):
            blueprint = """机制组合方案（兜底）:
- 产流机制: saturation_excess (蓄满产流)
- 汇流机制: linear_reservoir (线性水库)
- 蒸散发机制: layered_et (分层蒸散发)
- 模式: threshold, reservoir, partition
- 约束: mass_conservation, runoff_nonnegative, et_limited_by_pet, storage_nonnegative"""
        elif any(kw in user_req_lower for kw in ["干旱", "arid", "scs", "cn"]):
            blueprint = """机制组合方案（兜底）:
- 产流机制: infiltration_excess (超渗产流)
- 汇流机制: linear_reservoir (线性水库)
- 蒸散发机制: soil_et (土壤蒸散发)
- 模式: threshold, reservoir
- 约束: mass_conservation, runoff_nonnegative, storage_nonnegative"""
        else:
            blueprint = """机制组合方案（兜底）:
- 产流机制: soil_moisture_accounting (土壤水分账)
- 汇流机制: linear_reservoir (线性水库)
- 蒸散发机制: soil_et (土壤蒸散发)
- 模式: storage, reservoir, ratio
- 约束: mass_conservation, runoff_nonnegative, et_limited_by_pet, storage_nonnegative"""
        
        return blueprint