"""
模块化水文模型接口定义

包含产流模块和汇流模块的抽象基类，定义统一的接口规范。
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any


class RunoffModule(ABC):
    """
    产流模块基类
    
    负责计算流域对降水的水文响应，输出有效产流量。
    
    接口规范：
    - compute(P_t, PE_t, state, params) -> (R_t, updated_state)
    - state_init_mapping: 状态初始化映射
    - 参数名必须以 _runoff 后缀结尾（防冲突）
    """
    
    @property
    @abstractmethod
    def id(self) -> str:
        """模块唯一标识符"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """模块名称"""
        pass
    
    @property
    @abstractmethod
    def keywords(self) -> List[str]:
        """检索关键词"""
        pass
    
    @property
    @abstractmethod
    def state_keys(self) -> List[str]:
        """状态字典需要的键"""
        pass
    
    @property
    @abstractmethod
    def state_init_mapping(self) -> Dict[str, str]:
        """
        状态初始化映射
        
        返回格式: {"state_key": "params.get('param_name', default)"}
        例如: {"soil_storage": "params.get('S0_runoff', 0.0)"}
        """
        pass
    
    @property
    @abstractmethod
    def params(self) -> Dict[str, List[float]]:
        """
        参数范围（必须以 _runoff 结尾）
        
        格式: {"参数名_runoff": [下界, 上界]}
        """
        pass
    
    @abstractmethod
    def compute(self, P_t: float, PE_t: float, state: Dict[str, float],
                params: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        计算单步产流
        
        Args:
            P_t: 当前时刻降水量 (mm)
            PE_t: 当前时刻潜在蒸散发 (mm)
            state: 状态字典（原地更新）
            params: 参数字典
        
        Returns:
            (R_t, updated_state): 产流量(mm), 更新后的状态
        """
        pass
    
    @property
    def description(self) -> str:
        """模块描述"""
        return ""


class RoutingModule(ABC):
    """
    汇流模块基类
    
    负责将产流量从流域汇流至出口，输出河道流量。
    
    接口规范：
    - route(R_t, state, params) -> (Q_t, updated_state)
    - state_init_mapping: 状态初始化映射
    - 参数名必须以 _routing 后缀结尾（防冲突）
    """
    
    @property
    @abstractmethod
    def id(self) -> str:
        """模块唯一标识符"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """模块名称"""
        pass
    
    @property
    @abstractmethod
    def keywords(self) -> List[str]:
        """检索关键词"""
        pass
    
    @property
    @abstractmethod
    def state_keys(self) -> List[str]:
        """状态字典需要的键"""
        pass
    
    @property
    @abstractmethod
    def state_init_mapping(self) -> Dict[str, str]:
        """
        状态初始化映射
        
        返回格式: {"state_key": "params.get('param_name', default)"}
        例如: {"routing_storage": "params.get('S0_routing', 50.0)"}
        """
        pass
    
    @property
    @abstractmethod
    def params(self) -> Dict[str, List[float]]:
        """
        参数范围（必须以 _routing 结尾）
        
        格式: {"参数名_routing": [下界, 上界]}
        """
        pass
    
    @abstractmethod
    def route(self, R_t: float, state: Dict[str, float],
              params: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        计算单步汇流
        
        Args:
            R_t: 当前时刻输入水量 (mm)，来自产流模块
            state: 状态字典（原地更新）
            params: 参数字典
        
        Returns:
            (Q_t, updated_state): 出口流量(mm), 更新后的状态
        """
        pass
    
    @property
    def description(self) -> str:
        """模块描述"""
        return ""


class ModuleRegistry:
    """模块注册表 - 存储所有可用模块"""
    
    def __init__(self):
        self._runoff_modules: Dict[str, RunoffModule] = {}
        self._routing_modules: Dict[str, RoutingModule] = {}
    
    def register_runoff(self, module: RunoffModule):
        """注册产流模块"""
        self._runoff_modules[module.id] = module
    
    def register_routing(self, module: RoutingModule):
        """注册汇流模块"""
        self._routing_modules[module.id] = module
    
    def get_runoff(self, module_id: str) -> RunoffModule:
        """获取产流模块"""
        return self._runoff_modules.get(module_id)
    
    def get_routing(self, module_id: str) -> RoutingModule:
        """获取汇流模块"""
        return self._routing_modules.get(module_id)
    
    def list_runoff_modules(self) -> List[str]:
        """列出所有产流模块ID"""
        return list(self._runoff_modules.keys())
    
    def list_routing_modules(self) -> List[str]:
        """列出所有汇流模块ID"""
        return list(self._routing_modules.keys())


# 全局模块注册表
GLOBAL_REGISTRY = ModuleRegistry()