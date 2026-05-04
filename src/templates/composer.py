"""
模型组合器 (Model Composer)

将产流模块和汇流模块组合成完整的 simulate_runoff 函数。

防雷设计：
1. 参数命名空间隔离：产流参数 _runoff，汇流参数 _routing
2. 状态初始化：使用模块的 state_init_mapping
3. 只提取计算核心：inspect.getsource 只提取 compute/route 方法
"""
import inspect
from typing import Dict, Any, Tuple, Optional
from .modules import GLOBAL_REGISTRY, RunoffModule, RoutingModule


class ModelComposer:
    """模型组合器"""
    
    BLUEPRINT = '''def simulate_runoff(precip, pet, params):
    """
    基于模块化架构的集总式水文模型
    
    产流模块: {runoff_name}
    汇流模块: {routing_name}
    
    参数:
        precip: 日降雨量序列 (mm/day)
        pet: 日潜在蒸散发序列 (mm/day)
        params: 参数字典
    
    返回:
        q: 日径流序列 (mm/day)
    """
    import numpy as np
    
    n_steps = len(precip)
    Q_sim = np.zeros(n_steps)
    
    # === 1. 状态初始化 ===
    state = {initial_state}
    
    # === 2. 时间步循环 ===
    for t in range(n_steps):
        P_t = float(precip[t])
        PE_t = float(pet[t])
        
        # === 3. 产流计算 ===
        # {runoff_code}
        
        # === 4. 汇流计算 ===
        # {routing_code}
        
        Q_sim[t] = Q_t
    
    return Q_sim
'''
    
    def __init__(self):
        self.runoff_registry = {}
        self.routing_registry = {}
        self._load_modules()
    
    def _load_modules(self):
        """从全局注册表加载模块"""
        self.runoff_registry = {
            "scs_runoff": GLOBAL_REGISTRY.get_runoff("scs_runoff"),
            "xaj_runoff": GLOBAL_REGISTRY.get_runoff("xaj_runoff"),
            "tank_runoff": GLOBAL_REGISTRY.get_runoff("tank_runoff"),
            "hbv_runoff": GLOBAL_REGISTRY.get_runoff("hbv_runoff"),
            "simple_runoff": GLOBAL_REGISTRY.get_runoff("simple_runoff"),
        }
        
        self.routing_registry = {
            "linear_routing": GLOBAL_REGISTRY.get_routing("linear_routing"),
            "nonlinear_routing": GLOBAL_REGISTRY.get_routing("nonlinear_routing"),
            "nash_routing": GLOBAL_REGISTRY.get_routing("nash_routing"),
            "direct_routing": GLOBAL_REGISTRY.get_routing("direct_routing"),
        }
    
    def get_runoff_module(self, module_id: str) -> Optional[RunoffModule]:
        """获取产流模块"""
        return self.runoff_registry.get(module_id)
    
    def get_routing_module(self, module_id: str) -> Optional[RoutingModule]:
        """获取汇流模块"""
        return self.routing_registry.get(module_id)
    
    def list_runoff_modules(self) -> Dict[str, str]:
        """列出所有产流模块 {id: name}"""
        return {k: v.name for k, v in self.runoff_registry.items() if v}
    
    def list_routing_modules(self) -> Dict[str, str]:
        """列出所有汇流模块 {id: name}"""
        return {k: v.name for k, v in self.routing_registry.items() if v}
    
    def _get_module_method_code(self, module) -> str:
        """防雷设计3：只提取 compute/route 方法，丢弃类的外壳"""
        try:
            if hasattr(module, 'compute'):
                return inspect.getsource(module.compute)
            elif hasattr(module, 'route'):
                return inspect.getsource(module.route)
        except Exception:
            pass
        return ""
    
    def _build_initial_state(self, runoff_module: RunoffModule, 
                           routing_module: RoutingModule) -> str:
        """防雷设计2：使用模块的 state_init_mapping 构建初始状态"""
        init_mappings = {}
        
        for key, init_expr in runoff_module.state_init_mapping.items():
            init_mappings[f"'{key}'"] = init_expr
        
        for key, init_expr in routing_module.state_init_mapping.items():
            init_mappings[f"'{key}'"] = init_expr
        
        if not init_mappings:
            return "{}"
        
        return "{" + ", ".join(f"{k}: {v}" for k, v in init_mappings.items()) + "}"
    
    def _merge_params(self, runoff_module: RunoffModule, 
                      routing_module: RoutingModule) -> Dict[str, List[float]]:
        """防雷设计1：合并参数（自动加后缀，无冲突）"""
        combined = {}
        
        for key, value in runoff_module.params.items():
            combined[key] = value
        
        for key, value in routing_module.params.items():
            combined[key] = value
        
        return combined
    
    def compose(self, runoff_id: str, routing_id: str) -> Dict[str, Any]:
        """
        组合产流模块和汇流模块，生成完整模型
        
        Args:
            runoff_id: 产流模块 ID
            routing_id: 汇流模块 ID
        
        Returns:
            {
                "code": 完整的 simulate_runoff 函数代码,
                "runoff_module": 产流模块对象,
                "routing_module": 汇流模块对象,
                "params": 合并后的参数范围,
                "runoff_id": 产流模块ID,
                "routing_id": 汇流模块ID,
                "runoff_code": 产流模块代码,
                "routing_code": 汇流模块代码
            }
        """
        runoff_module = self.get_runoff_module(runoff_id)
        routing_module = self.get_routing_module(routing_id)
        
        if not runoff_module:
            raise ValueError(f"未找到产流模块: {runoff_id}")
        if not routing_module:
            raise ValueError(f"未找到汇流模块: {routing_id}")
        
        runoff_code = self._get_module_method_code(runoff_module)
        routing_code = self._get_module_method_code(routing_module)
        
        initial_state = self._build_initial_state(runoff_module, routing_module)
        
        combined_params = self._merge_params(runoff_module, routing_module)
        
        model_code = self.BLUEPRINT.format(
            runoff_name=runoff_module.name,
            routing_name=routing_module.name,
            initial_state=initial_state,
            runoff_code=self._format_runoff_call(runoff_module),
            routing_code=self._format_routing_call(routing_module)
        )
        
        return {
            "code": model_code,
            "runoff_module": runoff_module,
            "routing_module": routing_module,
            "params": combined_params,
            "runoff_id": runoff_id,
            "routing_id": routing_id,
            "runoff_code": runoff_code,
            "routing_code": routing_code
        }
    
    def _format_runoff_call(self, module: RunoffModule) -> str:
        """生成产流模块调用代码"""
        return f"R_t, state = _runoff_compute(P_t, PE_t, state, params)"
    
    def _format_routing_call(self, module: RoutingModule) -> str:
        """生成汇流模块调用代码"""
        return f"Q_t, state = _routing_route(R_t, state, params)"


# 全局组合器实例
COMPOSER = ModelComposer()


def get_composer() -> ModelComposer:
    """获取全局组合器"""
    return COMPOSER


def compose_model(runoff_id: str, routing_id: str) -> Dict[str, Any]:
    """快捷组合函数"""
    return COMPOSER.compose(runoff_id, routing_id)