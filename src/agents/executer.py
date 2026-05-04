from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class HydroModelCode(BaseModel):
    """结构化输出的模型代码"""
    reasoning: str = Field(description="代码设计与参数使用的思考过程")
    simulate_function: str = Field(description="纯净的 simulate_runoff 函数代码，不含 import")
    parameters_config: Dict[str, List[float]] = Field(
        description="代码中实际使用的参数及其寻优上下界，格式为 {'参数名': [下界, 上界]}，如 {'k': [0.05, 0.8], 'S0': [0, 200]}"
    )


EXECUTER_SYSTEM_PROMPT = """你是一位科学计算 Python 程序员，专门实现水文模型模拟函数。

你需要根据给定的建模方案，生成一个完整的 Python 函数：

```python
def simulate_runoff(precip, pet, params):
    # precip: numpy 数组，日降雨量 (float)
    # pet: numpy 数组，日潜在蒸散发 (float)
    # params: 字典，模型参数
    # 返回: numpy 数组，日径流量序列，长度与输入相同
```

严格要求：
1. 仅返回函数代码本身，不要任何 Markdown 标记（不要用 ```python 包裹）
2. 不要写 import 语句（假设 numpy as np 已可用）
3. 使用 np.zeros、np.roll 等 NumPy 函数进行向量化
4. 确保返回的数组长度与 precip 一致
5. 必须包含 params 的默认参数读取，如 params.get("k", 0.3)
6. 必须包含完整的水量平衡：降雨 + 输入 - 蒸散发 - 出流 = 储水变化
7. **所有代码注释必须使用中文**，包括函数说明、变量说明、计算步骤说明等
8. **径流量不能全为零**，必须包含基流成分（即使无降雨也应该有基流出流），防止模拟结果全部为零
9. **如需加速计算，可使用 @jit(nopython=True) 装饰器"""


class Executer:

    def __init__(self, openai_api_key, model_name="gpt-4o"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,
            openai_api_key=openai_api_key,
        )
        self._structured_llm = None
        
    def _get_structured_llm(self):
        """获取结构化输出的 LLM"""
        if self._structured_llm is None:
            try:
                self._structured_llm = self.llm.with_structured_output(HydroModelCode)
            except Exception:
                self._structured_llm = None
        return self._structured_llm
    
    def generate_code(self, plan, code_template="", param_constraints=None, use_structured=False):
        plan_str = plan if isinstance(plan, str) else str(plan)
        
        template_context = ""
        if code_template:
            param_list_str = ", ".join([f"{k}: {v[0]}-{v[1]}" for k, v in param_constraints.items()]) if param_constraints else "无"
            template_context = f"""
参考基准代码模板：
{code_template}

【开发指令】：
1. 核心物理机制约束：必须绝对保留上述模板中的产流/汇流核心逻辑和水量平衡（Mass Balance）机制，切勿随意修改物理运算步骤。
2. 接口适配约束：请在此模板基础上，根据 Planner 提供的参数建议（{param_list_str}），仅修改参数读取方式（如 params.get(...)）和输入输出接口，以满足主函数的调用需求。
"""
        
        param_context = ""
        if param_constraints:
            param_list = ", ".join([f"{k}: {v[0]}-{v[1]}" for k, v in param_constraints.items()])
            param_context = f"""
约束参数（必须使用这些参数名，参数范围仅作参考）：
- 参数列表: {param_list}
- 函数内必须使用 params.get("参数名") 读取这些参数
"""
        
        user_message = (
            f"建模方案:\n{plan_str}\n\n"
            f"{param_context}"
            f"{template_context}"
            f"请根据以上方案生成 simulate_runoff(precip, pet, params) 函数。"
        )
        
        if use_structured:
            structured_llm = self._get_structured_llm()
            if structured_llm:
                try:
                    result = structured_llm.invoke(user_message)
                    if hasattr(result, 'simulate_function'):
                        return {
                            "code": result.simulate_function,
                            "parameters_config": result.parameters_config,
                            "reasoning": result.reasoning
                        }
                except Exception as e:
                    print(f"Executer 结构化输出失败: {e}")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", EXECUTER_SYSTEM_PROMPT),
            ("user", "{user_message}"),
        ])
        chain = prompt | self.llm
        response = chain.invoke({"user_message": user_message})
        
        content = response.content
        if isinstance(content, list):
            content = content[0].get("text", "") if content else ""
        code = str(content).strip()
        code = code.replace("```python", "").replace("```", "").strip()
        
        return {"code": code, "parameters_config": {}, "reasoning": ""}

    def retry_with_error(self, previous_code, error_message, use_structured=False):
        """传入上一次生成的代码和错误信息，让 LLM 修正"""
        user_message = (
            f"上一版代码:\n```\n{previous_code}\n```\n"
            f"执行时报错:\n{error_message}\n\n"
            f"请修正错误后重新生成 simulate_runoff 函数。"
        )
        
        if use_structured and isinstance(previous_code, dict):
            previous_code_str = previous_code.get("code", "")
        else:
            previous_code_str = previous_code if isinstance(previous_code, str) else str(previous_code)
        
        user_message = (
            f"上一版代码:\n```\n{previous_code_str}\n```\n"
            f"执行时报错:\n{error_message}\n\n"
            f"请修正错误后重新生成 simulate_runoff 函数。\n\n"
            f"【重要约束】：修正时必须保留原始代码中的物理机制和水量平衡逻辑，仅修复语法错误或逻辑错误，不要随意修改物理公式。"
        )
        
        if use_structured:
            structured_llm = self._get_structured_llm()
            if structured_llm:
                try:
                    result = structured_llm.invoke(user_message)
                    if hasattr(result, 'simulate_function'):
                        return {
                            "code": result.simulate_function,
                            "parameters_config": result.parameters_config,
                            "reasoning": result.reasoning
                        }
                except Exception:
                    pass
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", EXECUTER_SYSTEM_PROMPT),
            ("user", "{user_message}"),
        ])
        chain = prompt | self.llm
        response = chain.invoke({"user_message": user_message})
        
        content = response.content
        if isinstance(content, list):
            content = content[0].get("text", "") if content else ""
        code = str(content).strip()
        code = code.replace("```python", "").replace("```", "").strip()
        
        return {"code": code, "parameters_config": {}, "reasoning": ""}