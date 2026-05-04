from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import numpy as np
from agents.executer import Executer
from agents.validator import CodeValidator


class AgentState(TypedDict):
    plan: str
    code_template: str
    code: str | None
    error: str | None
    attempt: int
    validated: bool
    inputs: dict


def create_agent_graph(openai_api_key: str, model_name: str = "gpt-4o"):
    executer = Executer(openai_api_key, model_name)
    validator = CodeValidator()
    
    def generate_code_node(state: AgentState) -> AgentState:
        attempt = state["attempt"]
        
        if attempt == 0:
            code = executer.generate_code(
                state["plan"], 
                code_template=state.get("code_template", "")
            )
        else:
            error_msg = state.get("error", "")
            code = executer.retry_with_error(
                state.get("code", ""), 
                error_msg
            )
        
        return {
            **state,
            "code": code,
            "attempt": attempt + 1
        }
    
    def validate_code_node(state: AgentState) -> AgentState:
        code = state.get("code", "")
        inputs = state.get("inputs", {})
        
        if not code:
            return {
                **state,
                "validated": False,
                "error": "代码为空"
            }
        
        syntax_ok, syntax_err = validator.validate_syntax(code)
        if not syntax_ok:
            return {
                **state,
                "validated": False,
                "error": f"语法错误: {syntax_err}"
            }
        
        exec_ok, _, exec_err = validator.execute_safe(code, inputs)
        if not exec_ok:
            return {
                **state,
                "validated": False,
                "error": exec_err or "执行失败"
            }
        
        return {
            **state,
            "validated": True,
            "error": None
        }
    
    def should_continue(state: AgentState) -> str:
        if state.get("validated", False):
            return "end"
        
        max_attempts = 3
        if state.get("attempt", 0) >= max_attempts:
            return "end"
        
        return "retry"
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("generate", generate_code_node)
    workflow.add_node("validate", validate_code_node)
    
    workflow.set_entry_point("generate")
    
    workflow.add_edge("generate", "validate")
    
    workflow.add_conditional_edges(
        "validate",
        should_continue,
        {
            "retry": "generate",
            "end": END
        }
    )
    
    return workflow.compile()


def run_agent_loop(
    plan: str,
    code_template: str,
    inputs: dict,
    openai_api_key: str,
    model_name: str = "gpt-4o",
    max_attempts: int = 3
):
    """
    运行 Agent 循环：生成代码 → 验证 → 失败则重试
    
    参数:
        plan: Planner 生成的建模方案
        code_template: 代码模板
        inputs: 测试输入 {"precip": np.array, "pet": np.array, "params": dict}
        openai_api_key: OpenAI API 密钥
        model_name: 模型名称
        max_attempts: 最大尝试次数
    
    返回:
        {"code": str, "validated": bool, "error": str|None, "attempts": int}
    """
    graph = create_agent_graph(openai_api_key, model_name)
    
    initial_state: AgentState = {
        "plan": plan,
        "code_template": code_template,
        "code": None,
        "error": None,
        "attempt": 0,
        "validated": False,
        "inputs": inputs
    }
    
    final_state = graph.invoke(initial_state)
    
    return {
        "code": final_state.get("code"),
        "validated": final_state.get("validated", False),
        "error": final_state.get("error"),
        "attempts": final_state.get("attempt", 0)
    }