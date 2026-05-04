"""
产流模块实现

包含以下模块：
1. SCS-CN 产流模块 (scs_runoff)
2. 新安江蓄满产流模块 (xaj_runoff)
3. Tank 产流模块 (tank_runoff)
4. HBV 土壤水分模块 (hbv_runoff)
5. 简单产流模块 (simple_runoff)

注意：参数名必须以 _runoff 结尾，避免与汇流模块参数冲突
"""
from typing import Dict, List, Tuple
from .modules import RunoffModule


class SCSRunoffModule(RunoffModule):
    """
    SCS-CN 产流模型
    
    基于超额蓄水量的产流机制。
    - Ia = 0.2 * Smax (初始截留)
    - Q = (P - Ia)^2 / (P - Ia + Smax) (当 P > Ia)
    """
    
    @property
    def id(self) -> str:
        return "scs_runoff"
    
    @property
    def name(self) -> str:
        return "SCS-CN 产流模型"
    
    @property
    def keywords(self) -> List[str]:
        return ["SCS", "CN", "曲线数", "超额蓄水", "超渗产流", "SCS-CN"]
    
    @property
    def state_keys(self) -> List[str]:
        return ["soil_storage"]
    
    @property
    def state_init_mapping(self) -> Dict[str, str]:
        return {
            "soil_storage": "params.get('S0_runoff', 0.0)"
        }
    
    @property
    def params(self) -> Dict[str, List[float]]:
        return {
            "CN_runoff": [30, 95],
            "Smax_runoff": [0, 500],
            "Ia_ratio_runoff": [0.1, 0.3]
        }
    
    @property
    def description(self) -> str:
        return "SCS-CN 产流模型，适用于干旱小流域、降雨径流快速推求"
    
    def compute(self, P_t: float, PE_t: float, state: Dict[str, float],
                params: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        CN = state.get("CN_runoff", params.get("CN_runoff", 75))
        Smax = 25400.0 / CN - 254.0 if CN > 0 else 100.0
        Ia_ratio = params.get("Ia_ratio_runoff", 0.2)
        Ia = Ia_ratio * Smax
        
        soil = state.get("soil_storage", 0.0)
        
        soil = max(soil - PE_t, 0.0)
        
        if P_t <= Ia:
            R_t = 0.0
        else:
            P_eff = P_t - Ia
            R_t = (P_eff ** 2) / (P_eff + Smax)
        
        soil = soil + P_t - R_t - PE_t
        state["soil_storage"] = max(soil, 0.0)
        
        return R_t, state


class XAJRunoffModule(RunoffModule):
    """
    新安江蓄满产流模型
    
    基于蓄满产流机制，使用 B 曲线描述张力水容量分布。
    """
    
    @property
    def id(self) -> str:
        return "xaj_runoff"
    
    @property
    def name(self) -> str:
        return "新安江蓄满产流模型"
    
    @property
    def keywords(self) -> List[str]:
        return ["新安江", "蓄满产流", "张力水", "湿润区", "南方流域", "B曲线"]
    
    @property
    def state_keys(self) -> List[str]:
        return ["W", "S"]
    
    @property
    def state_init_mapping(self) -> Dict[str, str]:
        return {
            "W": "params.get('WM_runoff', 120.0) * 0.5",
            "S": "params.get('SM_runoff', 30.0) * 0.5"
        }
    
    @property
    def params(self) -> Dict[str, List[float]]:
        return {
            "WM_runoff": [50, 200],
            "UM_runoff": [20, 80],
            "LM_runoff": [30, 120],
            "B_runoff": [0.1, 1.0],
            "K_runoff": [0.5, 1.0],
            "C_runoff": [0.1, 0.3],
            "SM_runoff": [10, 50]
        }
    
    @property
    def description(self) -> str:
        return "新安江蓄满产流模型，适用于湿润区、南方流域"
    
    def compute(self, P_t: float, PE_t: float, state: Dict[str, float],
                params: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        WM = params.get("WM_runoff", 120.0)
        UM = params.get("UM_runoff", 40.0)
        LM = params.get("LM_runoff", 80.0)
        B = params.get("B_runoff", 0.4)
        K = params.get("K_runoff", 0.9)
        C = params.get("C_runoff", 0.15)
        SM = params.get("SM_runoff", 30.0)
        
        W = state.get("W", WM * 0.5)
        S = state.get("S", SM * 0.5)
        
        PE = K * PE_t
        
        if W >= UM:
            EU = PE
            EL = 0
            EC = 0
        elif W >= UM - LM:
            EU = W - (UM - LM)
            EL = PE - EU
            EC = 0
        else:
            EU = 0
            EL = max(W - (UM - LM), 0)
            EC = PE - EU - EL
            EC = min(EC, C * PE)
        
        W = max(W - EU - EL - EC, 0)
        
        P_total = P_t + W
        WMM = WM * (1 + B)
        
        if P_total < WMM:
            WMM_W = WMM * (1 - (P_total / WMM) ** (1 / (1 + B)))
            if P_total <= WMM_W:
                W = P_total
                Q = 0
            else:
                FR = 1 - (WMM_W / WMM) ** (1 + B)
                W = WMM_W
                Q = FR * P_t
        else:
            Q = P_t + W - WM
            W = WM
        
        S = S + Q
        if S < 0:
            S = 0
        
        EX = min(S / SM, 1.0)
        
        state["W"] = W
        state["S"] = S
        
        return Q, state


class TankRunoffModule(RunoffModule):
    """
    Tank 产流模型 - 多水箱串联结构
    
    上层水箱接收降雨，中层和下层水箱接收侧向补给。
    """
    
    @property
    def id(self) -> str:
        return "tank_runoff"
    
    @property
    def name(self) -> str:
        return "Tank 产流模型"
    
    @property
    def keywords(self) -> List[str]:
        return ["Tank", "水箱", "多水箱", "串联", "k1", "k2", "k3"]
    
    @property
    def state_keys(self) -> List[str]:
        return ["S1", "S2", "S3"]
    
    @property
    def state_init_mapping(self) -> Dict[str, str]:
        return {
            "S1": "params.get('S1_runoff', 50.0)",
            "S2": "params.get('S2_runoff', 80.0)",
            "S3": "params.get('S3_runoff', 100.0)"
        }
    
    @property
    def params(self) -> Dict[str, List[float]]:
        return {
            "k1_runoff": [0.1, 0.8],
            "k2_runoff": [0.01, 0.3],
            "k3_runoff": [0.001, 0.1],
            "S1_runoff": [0, 100],
            "S2_runoff": [0, 150],
            "S3_runoff": [0, 200]
        }
    
    @property
    def description(self) -> str:
        return "Tank 产流模型，适用于需要模拟快速+慢速双重响应的复杂流域"
    
    def compute(self, P_t: float, PE_t: float, state: Dict[str, float],
                params: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        k1 = params.get("k1_runoff", 0.3)
        k2 = params.get("k2_runoff", 0.1)
        k3 = params.get("k3_runoff", 0.01)
        
        S1 = state.get("S1", params.get("S1_runoff", 50.0))
        S2 = state.get("S2", params.get("S2_runoff", 80.0))
        S3 = state.get("S3", params.get("S3_runoff", 100.0))
        
        S1 = S1 + P_t - 0.6 * PE_t
        if S1 < 0:
            S1 = 0
        
        q1 = k1 * max(S1 - 20, 0)
        S1 = S1 - q1
        
        to_middle = 0.15 * S1
        S1 = S1 - to_middle
        S2 = S2 + to_middle
        
        S2 = S2 - 0.4 * PE_t
        if S2 < 0:
            S2 = 0
        
        q2 = k2 * S2
        S2 = S2 - q2
        
        to_lower = 0.10 * S2
        S2 = S2 - to_lower
        S3 = S3 + to_lower
        
        S3 = S3 - 0.2 * PE_t
        if S3 < 0:
            S3 = 0
        
        q3 = k3 * S3
        S3 = S3 - q3
        
        R_t = q1 + q2 + q3
        
        state["S1"] = S1
        state["S2"] = S2
        state["S3"] = S3
        
        return R_t, state


class HBVRunoffModule(RunoffModule):
    """
    HBV 土壤水分模型
    
    基于土壤含水量的非线性产流机制。
    """
    
    @property
    def id(self) -> str:
        return "hbv_runoff"
    
    @property
    def name(self) -> str:
        return "HBV 土壤水分模型"
    
    @property
    def keywords(self) -> List[str]:
        return ["HBV", "土壤含水量", "土壤水分", "温带", "FC", "BETA"]
    
    @property
    def state_keys(self) -> List[str]:
        return ["soil"]
    
    @property
    def state_init_mapping(self) -> Dict[str, str]:
        return {
            "soil": "params.get('FC_runoff', 300.0) * 0.5"
        }
    
    @property
    def params(self) -> Dict[str, List[float]]:
        return {
            "FC_runoff": [100, 500],
            "LP_runoff": [0.3, 0.9],
            "BETA_runoff": [1.0, 6.0]
        }
    
    @property
    def description(self) -> str:
        return "HBV 土壤水分模型，适用于温带、寒带流域"
    
    def compute(self, P_t: float, PE_t: float, state: Dict[str, float],
                params: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        FC = params.get("FC_runoff", 300.0)
        LP = params.get("LP_runoff", 0.5)
        BETA = params.get("BETA_runoff", 2.0)
        
        soil = state.get("soil", FC * 0.5)
        
        if soil < FC:
            soil_change = P_t * (soil / FC) ** BETA
        else:
            soil_change = P_t
        
        soil_change = max(soil_change, -soil)
        soil = soil + soil_change
        
        if soil > LP * FC:
            E = PE_t
        else:
            E = PE_t * (soil / (LP * FC))
        
        soil = max(soil - E, 0)
        
        P_eff = soil_change - max(soil_change - (soil - LP * FC), 0)
        
        state["soil"] = soil
        
        return P_eff, state


class SimpleRunoffModule(RunoffModule):
    """
    简单产流模型
    
    最基本的产流机制：P - ET 直接产生径流。
    适用于概念性模拟或快速原型开发。
    """
    
    @property
    def id(self) -> str:
        return "simple_runoff"
    
    @property
    def name(self) -> str:
        return "简单产流模型"
    
    @property
    def keywords(self) -> List[str]:
        return ["简单", "直接", "基本", "P-ET"]
    
    @property
    def state_keys(self) -> List[str]:
        return ["storage"]
    
    @property
    def state_init_mapping(self) -> Dict[str, str]:
        return {
            "storage": "params.get('S0_runoff', 50.0)"
        }
    
    @property
    def params(self) -> Dict[str, List[float]]:
        return {
            "k_runoff": [0.05, 0.8],
            "S0_runoff": [0, 200]
        }
    
    @property
    def description(self) -> str:
        return "简单产流模型，适用于概念性模拟或快速原型开发"
    
    def compute(self, P_t: float, PE_t: float, state: Dict[str, float],
                params: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        k = params.get("k_runoff", 0.3)
        S = state.get("storage", params.get("S0_runoff", 50.0))
        
        S = S + P_t - PE_t
        if S < 0:
            S = 0
        
        R_t = k * S
        S = S - R_t
        
        state["storage"] = S
        
        return R_t, state


# 模块注册
from .modules import GLOBAL_REGISTRY

GLOBAL_REGISTRY.register_runoff(SCSRunoffModule())
GLOBAL_REGISTRY.register_runoff(XAJRunoffModule())
GLOBAL_REGISTRY.register_runoff(TankRunoffModule())
GLOBAL_REGISTRY.register_runoff(HBVRunoffModule())
GLOBAL_REGISTRY.register_runoff(SimpleRunoffModule())