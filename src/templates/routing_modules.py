"""
汇流模块实现

包含以下模块：
1. 线性水库汇流 (linear_routing)
2. 非线性水库汇流 (nonlinear_routing)
3. Nash 单位线汇流 (nash_routing)
4. 直接汇流 (direct_routing)

注意：参数名必须以 _routing 结尾，避免与产流模块参数冲突
"""
from typing import Dict, List, Tuple
from .modules import RoutingModule


class LinearRoutingModule(RoutingModule):
    """
    线性水库汇流
    
    最简单的汇流模型：Q = k * S
    储水衰减呈指数形式。
    """
    
    @property
    def id(self) -> str:
        return "linear_routing"
    
    @property
    def name(self) -> str:
        return "线性水库"
    
    @property
    def keywords(self) -> List[str]:
        return ["线性水库", "k*S", "慢速响应", "线性", "k"]
    
    @property
    def state_keys(self) -> List[str]:
        return ["routing_storage"]
    
    @property
    def state_init_mapping(self) -> Dict[str, str]:
        return {
            "routing_storage": "params.get('S0_routing', 50.0)"
        }
    
    @property
    def params(self) -> Dict[str, List[float]]:
        return {
            "k_routing": [0.05, 0.8],
            "S0_routing": [0, 200]
        }
    
    @property
    def description(self) -> str:
        return "线性水库汇流，适用于慢速响应流域"
    
    def route(self, R_t: float, state: Dict[str, float],
              params: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        k = params.get("k_routing", 0.3)
        S = state.get("routing_storage", params.get("S0_routing", 50.0))
        
        S = S + R_t
        Q = k * S
        S = S - Q
        
        state["routing_storage"] = S
        
        return Q, state


class NonlinearRoutingModule(RoutingModule):
    """
    非线性水库汇流
    
    Q = k * S^beta，非线性响应。
    """
    
    @property
    def id(self) -> str:
        return "nonlinear_routing"
    
    @property
    def name(self) -> str:
        return "非线性水库"
    
    @property
    def keywords(self) -> List[str]:
        return ["非线性", "非线性水库", "beta", "S^beta", "非线性响应"]
    
    @property
    def state_keys(self) -> List[str]:
        return ["routing_storage"]
    
    @property
    def state_init_mapping(self) -> Dict[str, str]:
        return {
            "routing_storage": "params.get('S0_routing', 50.0)"
        }
    
    @property
    def params(self) -> Dict[str, List[float]]:
        return {
            "k_routing": [0.01, 0.5],
            "beta_routing": [1.0, 3.0],
            "S0_routing": [0, 200]
        }
    
    @property
    def description(self) -> str:
        return "非线性水库汇流，适用于非线性响应流域"
    
    def route(self, R_t: float, state: Dict[str, float],
              params: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        k = params.get("k_routing", 0.1)
        beta = params.get("beta_routing", 1.5)
        S = state.get("routing_storage", params.get("S0_routing", 50.0))
        
        S = S + R_t
        Q = k * (S ** beta)
        S = S - Q
        
        if S < 0:
            S = 0
            Q = max(R_t, 0)
        
        state["routing_storage"] = S
        
        return Q, state


class NashRoutingModule(RoutingModule):
    """
    Nash 瞬时单位线
    
    n 个线性水库串联，模拟流域汇流的延迟效应。
    """
    
    def __init__(self, n: int = 3):
        self._n = n
    
    @property
    def id(self) -> str:
        return "nash_routing"
    
    @property
    def name(self) -> str:
        return f"Nash 单位线 (n={self._n})"
    
    @property
    def keywords(self) -> List[str]:
        return ["Nash", "单位线", "串联", "瞬时单位线", "n个水库"]
    
    @property
    def state_keys(self) -> List[str]:
        return [f"nash_{i}" for i in range(1, self._n + 1)]
    
    @property
    def state_init_mapping(self) -> Dict[str, str]:
        mapping = {}
        for i in range(1, self._n + 1):
            mapping[f"nash_{i}"] = "0.0"
        return mapping
    
    @property
    def params(self) -> Dict[str, List[float]]:
        return {
            "k_routing": [0.1, 0.8],
            "n_routing": [2, 5]
        }
    
    @property
    def description(self) -> str:
        return "Nash 瞬时单位线，适用于模拟流域汇流延迟效应"
    
    def route(self, R_t: float, state: Dict[str, float],
              params: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        k = params.get("k_routing", 0.3)
        
        S1 = state.get("nash_1", 0.0)
        
        S1 = S1 + R_t
        Q1 = k * S1
        S1 = S1 - Q1
        state["nash_1"] = S1
        
        for i in range(2, self._n + 1):
            Si = state.get(f"nash_{i}", 0.0)
            Si = Si + Q1
            Qi = k * Si
            Si = Si - Qi
            state[f"nash_{i}"] = Si
            Q1 = Qi
        
        return Q1, state
    
    @property
    def state_keys(self) -> List[str]:
        n = int(self.params.get("n_routing", [3, 3])[1])
        return [f"nash_{i}" for i in range(1, n + 1)]


class DirectRoutingModule(RoutingModule):
    """
    直接汇流
    
    简单的延迟汇流，产流量直接传递到出口。
    """
    
    @property
    def id(self) -> str:
        return "direct_routing"
    
    @property
    def name(self) -> str:
        return "直接汇流"
    
    @property
    def keywords(self) -> List[str]:
        return ["直接", "无延迟", "即时", "直接汇流"]
    
    @property
    def state_keys(self) -> List[str]:
        return []
    
    @property
    def state_init_mapping(self) -> Dict[str, str]:
        return {}
    
    @property
    def params(self) -> Dict[str, List[float]]:
        return {}
    
    @property
    def description(self) -> str:
        return "直接汇流，无延迟，适用于快速响应"
    
    def route(self, R_t: float, state: Dict[str, float],
              params: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        return R_t, state


# 模块注册
from .modules import GLOBAL_REGISTRY

GLOBAL_REGISTRY.register_routing(LinearRoutingModule())
GLOBAL_REGISTRY.register_routing(NonlinearRoutingModule())
GLOBAL_REGISTRY.register_routing(NashRoutingModule(n=3))
GLOBAL_REGISTRY.register_routing(DirectRoutingModule())