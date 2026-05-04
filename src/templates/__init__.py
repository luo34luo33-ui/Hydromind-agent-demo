"""
Hydromind 模板系统

包含模块化架构：
- modules.py: 接口定义
- runoff_modules.py: 产流模块 (5个)
- routing_modules.py: 汇流模块 (4个)
- composer.py: 模型组合器
"""
from .modules import RunoffModule, RoutingModule, ModuleRegistry, GLOBAL_REGISTRY
from .runoff_modules import (
    SCSRunoffModule, XAJRunoffModule, TankRunoffModule, 
    HBVRunoffModule, SimpleRunoffModule
)
from .routing_modules import (
    LinearRoutingModule, NonlinearRoutingModule, 
    NashRoutingModule, DirectRoutingModule
)
from .composer import ModelComposer, COMPOSER, get_composer, compose_model

__all__ = [
    "RunoffModule",
    "RoutingModule",
    "ModuleRegistry",
    "GLOBAL_REGISTRY",
    "SCSRunoffModule",
    "XAJRunoffModule",
    "TankRunoffModule",
    "HBVRunoffModule",
    "SimpleRunoffModule",
    "LinearRoutingModule",
    "NonlinearRoutingModule",
    "NashRoutingModule",
    "DirectRoutingModule",
    "ModelComposer",
    "COMPOSER",
    "get_composer",
    "compose_model",
]