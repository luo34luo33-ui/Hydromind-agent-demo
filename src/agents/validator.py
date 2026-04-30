import ast
import traceback
import numpy as np
import pandas as pd


FALLBACK_CODE = '''
def simulate_runoff(precip, pet, params):
    k = params.get("k", 0.3)
    n = len(precip)
    q = np.zeros(n)
    S = params.get("S0", 50.0)
    for t in range(n):
        S += precip[t] - 0.5 * pet[t]
        if S < 0:
            S = 0
        q[t] = k * S
        S -= q[t]
    return q
'''


class CodeValidator:

    @staticmethod
    def validate_syntax(code_str):
        """用 AST 检查 Python 语法是否正确"""
        try:
            ast.parse(code_str)
            return True, None
        except SyntaxError as e:
            return False, str(e)

    @staticmethod
    def execute_safe(code_str, inputs):
        """
        在受限沙盒命名空间中执行生成的模型函数。
        inputs 应包含 'precip', 'pet' 两个 numpy 数组，
        可选 'params' 字典。
        """
        sandbox = {
            "np": np,
            "pd": pd,
        }

        exec_globals = sandbox.copy()
        exec_locals = {}

        try:
            exec(code_str, exec_globals, exec_locals)
        except Exception as e:
            return False, None, traceback.format_exc()

        if "simulate_runoff" not in exec_locals:
            exec_globals.pop("np", None)
            exec_globals.pop("pd", None)
            if "simulate_runoff" not in exec_globals:
                return False, None, "生成的代码中没有定义 simulate_runoff 函数"

        func = exec_locals.get("simulate_runoff", exec_globals.get("simulate_runoff"))

        precip = inputs.get("precip")
        pet = inputs.get("pet")
        params = inputs.get("params", {})

        try:
            result = func(precip, pet, params)
        except Exception as e:
            return False, None, traceback.format_exc()

        if result is None:
            return False, None, "函数返回了 None"
        if not isinstance(result, np.ndarray):
            return False, None, f"函数返回值类型错误: {type(result)}，应为 numpy.ndarray"
        if len(result) != len(precip):
            return False, None, f"返回值长度不匹配: 期望 {len(precip)}，实际 {len(result)}"

        return True, result, None


def execute_with_fallback(code_str, inputs, fallback_code=FALLBACK_CODE):
    """
    先尝试执行生成的代码 → 失败则执行兜底模型。
    返回 (success, q_sim, error_message, used_fallback)。
    """
    success, result, error = CodeValidator.execute_safe(code_str, inputs)
    if success:
        return True, result, None, False

    success, result, fb_error = CodeValidator.execute_safe(fallback_code, inputs)
    if success:
        return True, result, f"生成的代码执行失败，已切换至兜底模型。原始错误:\n{error}", True

    return False, None, f"生成的代码和兜底模型均失败。\n原始错误:\n{error}\n兜底错误:\n{fb_error}", True
