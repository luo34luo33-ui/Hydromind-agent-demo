import inspect
from typing import Dict, List, Any, Callable


class ModelTemplate:
    """模型模板类"""
    
    def __init__(
        self,
        id: str,
        name: str,
        keywords: List[str],
        func: Callable,
        params: Dict[str, List[float]],
        description: str = ""
    ):
        self.id = id
        self.name = name
        self.keywords = keywords
        self.func = func
        self.params = params
        self.description = description
    
    @property
    def code(self) -> str:
        """动态获取函数源代码"""
        try:
            return inspect.getsource(self.func)
        except Exception:
            return ""


def _linear_reservoir_simulate(precip, pet, params):
    """
    线性水库模型 - 单水箱水量平衡
    
    水量平衡: dS/dt = P - ET - Q
    汇流公式: Q = k * S
    
    参数:
        precip: 日降雨量序列 (mm/day)
        pet: 日潜在蒸散发序列 (mm/day)
        params: 参数字典 {"k": 出流系数, "S0": 初始储水量}
    
    返回:
        q: 日径流序列 (mm/day)
    """
    k = params.get("k", 0.3)
    S0 = params.get("S0", 50.0)
    n = len(precip)
    
    q = np.zeros(n)
    S = S0
    
    for t in range(n):
        S = S + precip[t] - 0.5 * pet[t]
        if S < 0:
            S = 0
        q[t] = k * S
        S = S - q[t]
    
    return q


def _tank_model_simulate(precip, pet, params):
    """
    三水箱模型 - 串联结构
    
    上层水箱: 直接接收降雨，蒸散发消耗，产生快速径流
    中层水箱: 接收上层侧向流入，产生中等径流
    下层水箱: 接收中层侧向流入，产生慢速基流
    
    参数:
        precip: 日降雨量序列 (mm/day)
        pet: 日潜在蒸散发序列 (mm/day)
        params: 参数字典 {"k1": 上层系数, "k2": 中层系数, "k3": 下层系数, "S1": 上层初始, "S2": 中层初始, "S3": 下层初始}
    
    返回:
        q: 日径流序列 (mm/day)，三箱之和
    """
    k1 = params.get("k1", 0.30)
    k2 = params.get("k2", 0.10)
    k3 = params.get("k3", 0.01)
    S1 = params.get("S1", 50.0)
    S2 = params.get("S2", 80.0)
    S3 = params.get("S3", 100.0)
    
    n = len(precip)
    q = np.zeros(n)
    
    for t in range(n):
        S1 = S1 + precip[t] - 0.6 * pet[t]
        if S1 < 0:
            S1 = 0
        
        q1 = k1 * max(S1 - 20, 0)
        S1 = S1 - q1
        
        to_middle = 0.15 * S1
        S1 = S1 - to_middle
        S2 = S2 + to_middle
        
        S2 = S2 - 0.4 * pet[t]
        if S2 < 0:
            S2 = 0
        
        q2 = k2 * S2
        S2 = S2 - q2
        
        to_lower = 0.10 * S2
        S2 = S2 - to_lower
        S3 = S3 + to_lower
        
        S3 = S3 - 0.2 * pet[t]
        if S3 < 0:
            S3 = 0
        
        q3 = k3 * S3
        S3 = S3 - q3
        
        q[t] = q1 + q2 + q3
    
    return q


def _scs_cn_simulate(precip, pet, params):
    """
    SCS-CN 模型 - 基于超额蓄水量的产流模型
    
    核心公式:
    - Ia = 0.2 * Smax (初始截留)
    - Q = (P - Ia)^2 / (P - Ia + Smax) (当 P > Ia)
    - Q = 0 (当 P <= Ia)
    
    参数:
        precip: 日降雨量序列 (mm/day)
        pet: 日潜在蒸散发序列 (mm/day)
        params: 参数字典 {"CN": 曲线数, "Smax": 最大截留量, "Ia": 初始截留}
    
    返回:
        q: 日径流序列 (mm/day)
    """
    CN = params.get("CN", 75.0)
    
    Smax = 25400.0 / CN - 254.0
    if Smax < 0:
        Smax = 0
    
    Ia = params.get("Ia", 0.2 * Smax)
    
    n = len(precip)
    q = np.zeros(n)
    
    for t in range(n):
        P = precip[t]
        
        if P <= Ia:
            q[t] = 0.0
        else:
            P_eff = P - Ia
            q[t] = (P_eff ** 2) / (P_eff + Smax)
    
    return q


def _xaj_simulate(precip, pet, params):
    """
    新安江模型 - 蓄满产流模型
    
    核心思想:
    - 流域内各点张力水容量不同，用 B 曲线描述分布
    - 当某点张力水蓄满后，降雨全部转化为径流
    - 蓄满后才发生自由水蓄水产流
    
    参数:
        precip: 日降雨量序列 (mm/day)
        pet: 日潜在蒸散发序列 (mm/day)
        params: 参数字典
    
    返回:
        q: 日径流序列 (mm/day)
    """
    WM = params.get("WM", 120.0)
    UM = params.get("UM", 40.0)
    LM = params.get("LM", 80.0)
    B = params.get("B", 0.4)
    K = params.get("K", 0.9)
    C = params.get("C", 0.15)
    SM = params.get("SM", 30.0)
    
    WM = UM + LM
    WUM = UM
    WLM = LM
    
    n = len(precip)
    q = np.zeros(n)
    
    W = WM * 0.5
    S = SM * 0.5
    
    for t in range(n):
        P = precip[t]
        PE = K * pet[t]
        
        if W >= WUM:
            EU = PE
            EL = 0
            EC = 0
        elif W >= WUM - WLM:
            EU = W - (WUM - WLM)
            EL = PE - EU
            EC = 0
        else:
            EU = 0
            EL = max(W - (WUM - WLM), 0)
            EC = PE - EU - EL
            EC = min(EC, C * PE)
        
        W = W - EU - EL - EC
        if W < 0:
            W = 0
        
        P_total = P + W
        WMM = WM * (1 + B)
        
        if P_total < WMM:
            WMM_W = WMM * (1 - (P_total / WMM) ** (1 / (1 + B)))
            WMM_P = WMM * (1 - ((P_total - P) / WMM) ** (1 / (1 + B)))
            
            if P_total <= WMM_W:
                W = P_total
                Q = 0
            else:
                FR = 1 - (WMM_W / WMM) ** (1 + B)
                W = WMM_W
                Q = FR * P
        else:
            Q = P + W - WM
            W = WM
        
        S = S + Q
        if S < 0:
            S = 0
        
        EX = S / SM
        if EX > 1:
            EX = 1
        
        q[t] = EX * S * 0.1
        S = S * (1 - 0.1)
    
    return q


def _hbv_simulate(precip, pet, params):
    """
    HBV 模型 - 概念性水文模型
    
    包含四个模块:
    1. 土壤含水量模块 (Soil Moisture)
    2. 响应函数模块 (Response Functions)
    3. 汇流模块 (Routing)
    4. 蒸散发模块
    
    参数:
        precip: 日降雨量序列 (mm/day)
        pet: 日潜在蒸散发序列 (mm/day)
        params: 参数字典
    
    返回:
        q: 日径流序列 (mm/day)
    """
    FC = params.get("FC", 300.0)
    LP = params.get("LP", 0.5)
    BETA = params.get("BETA", 2.0)
    K0 = params.get("K0", 0.2)
    K1 = params.get("K1", 0.05)
    UZL = params.get("UZL", 50.0)
    
    n = len(precip)
    q = np.zeros(n)
    
    soil = FC * 0.5
    U1 = 0.0
    U2 = 0.0
    L = 0.0
    
    for t in range(n):
        P = precip[t]
        PE = pet[t]
        
        soil_change = (P * (soil / FC) ** BETA) if soil < FC else P
        soil_change = max(soil_change, -soil)
        soil = soil + soil_change
        
        if soil > LP * FC:
            E = PE
        else:
            E = PE * (soil / (LP * FC))
        soil = max(soil - E, 0)
        
        P_eff = soil_change - max(soil_change - (soil - LP * FC), 0)
        
        U1 = U1 + P_eff * 0.5
        
        Q0 = K0 * max(U1 - UZL, 0)
        U1 = max(U1 - Q0, 0)
        
        U2 = U2 + Q0
        Q1 = K1 * U2
        U2 = max(U2 - Q1, 0)
        
        q[t] = Q0 + Q1
    
    return q


TEMPLATE_REGISTRY: Dict[str, ModelTemplate] = {
    "linear_reservoir": ModelTemplate(
        id="linear_reservoir",
        name="线性水库",
        keywords=["线性水库", "线性水库模型", "慢速响应", "单水箱", "k * S", "kS"],
        func=_linear_reservoir_simulate,
        params={"k": [0.05, 0.8], "S0": [0, 200]},
        description="适用于慢速响应流域、低坡度、干旱气候"
    ),
    "tank_model": ModelTemplate(
        id="tank_model",
        name="Tank Model",
        keywords=["水箱", "Tank", "三水箱", "多水箱", "串联", "k1", "k2", "k3"],
        func=_tank_model_simulate,
        params={"k1": [0.1, 0.8], "k2": [0.01, 0.3], "k3": [0.001, 0.1], "S1": [0, 100], "S2": [0, 150], "S3": [0, 200]},
        description="适用于需要模拟快速+慢速双重响应的复杂流域"
    ),
    "scs_cn": ModelTemplate(
        id="scs_cn",
        name="SCS-CN",
        keywords=["SCS", "CN", "曲线数", "超额蓄水", "SCS-CN"],
        func=_scs_cn_simulate,
        params={"CN": [30, 95], "Smax": [0, 500], "Ia": [0, 20]},
        description="适用于干旱小流域、降雨径流快速推求"
    ),
    "xaj": ModelTemplate(
        id="xaj",
        name="新安江",
        keywords=["新安江", "蓄满产流", "流域蒸发", "张力水", "湿润区", "南方流域"],
        func=_xaj_simulate,
        params={"WM": [50, 200], "UM": [20, 80], "LM": [30, 120], "B": [0.1, 1.0], "K": [0.5, 1.0], "C": [0.1, 0.3], "SM": [10, 50]},
        description="适用于湿润区、湿润季节的南方流域"
    ),
    "hbv": ModelTemplate(
        id="hbv",
        name="HBV",
        keywords=["HBV", "土壤含水量", "上层下层", "rate routine", "温带", "寒带"],
        func=_hbv_simulate,
        params={"FC": [100, 500], "LP": [0.3, 0.9], "BETA": [1.0, 6.0], "K0": [0.1, 0.5], "K1": [0.01, 0.1], "UZL": [0, 100]},
        description="通用型水文模型，适用于温带、寒带流域"
    ),
}


def get_template(template_id: str) -> ModelTemplate:
    """根据 ID 获取模板"""
    return TEMPLATE_REGISTRY.get(template_id)


def get_all_templates() -> Dict[str, ModelTemplate]:
    """获取所有模板"""
    return TEMPLATE_REGISTRY


def get_template_by_keyword(keyword: str) -> ModelTemplate:
    """根据关键词获取最佳匹配模板"""
    keyword_lower = keyword.lower()
    best_match = None
    best_score = 0
    
    for template in TEMPLATE_REGISTRY.values():
        score = sum(1 for kw in template.keywords if kw.lower() in keyword_lower)
        if score > best_score:
            best_score = score
            best_match = template
    
    return best_match or TEMPLATE_REGISTRY["linear_reservoir"]