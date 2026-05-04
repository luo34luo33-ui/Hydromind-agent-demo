# 水文模型代码模板库

以下模板可直接嵌入 Agent 上下文，供 Executer 参考修改。每个模板都经过向量化验证，确保能正确运行。

---

## 线性水库模型 (Linear Reservoir)

### 适用场景
- 慢速响应流域
- 低坡度、土壤渗透性低
- 干旱或半干旱气候
- 中大型流域面积

### 参数范围
| 参数 | 下界 | 上界 | 单位 | 说明 |
|------|------|------|------|------|
| k    | 0.05 | 0.80 | 无量纲 | 出流系数，越大出流越快 |
| S0   | 0    | 200  | mm    | 初始储水量 |

### 代码模板
```python
def simulate_runoff(precip, pet, params):
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
        # 降雨进入储水，蒸散发消耗储水
        S = S + precip[t] - 0.5 * pet[t]
        
        # 储水不能为负
        if S < 0:
            S = 0
        
        # 线性出流
        q[t] = k * S
        
        # 储水变化
        S = S - q[t]
    
    return q
```

### 物理特性
- 无降雨时仍有基流（因为 S > 0）
- 储水衰减呈指数形式
- 适用于单一蓄水层的简单流域

---

## Tank Model - 三水箱 (Three-Tank Model)

### 适用场景
- 需要模拟快速+慢速双重响应
- 流域内有不同蓄水层（上层土壤、下层土壤、地下水）
- 湿润区或湿润季节

### 参数范围
| 参数 | 下界 | 上界 | 单位 | 说明 |
|------|------|------|------|------|
| k1   | 0.10 | 0.80 | 无量纲 | 上层水箱出流系数（快速响应） |
| k2   | 0.01 | 0.30 | 无量纲 | 中层水箱出流系数（中等响应） |
| k3   | 0.001| 0.10 | 无量纲 | 下层水箱出流系数（慢速基流） |
| S1   | 0    | 100  | mm    | 上层初始储水量 |
| S2   | 0    | 150  | mm    | 中层初始储水量 |
| S3   | 0    | 200  | mm    | 下层初始储水量 |

### 代码模板
```python
def simulate_runoff(precip, pet, params):
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
        # 上层水箱水量平衡
        S1 = S1 + precip[t] - 0.6 * pet[t]
        if S1 < 0:
            S1 = 0
        
        # 上层出流（快速径流）
        q1 = k1 * max(S1 - 20, 0)
        S1 = S1 - q1
        
        # 侧向补给中层
        to_middle = 0.15 * S1
        S1 = S1 - to_middle
        S2 = S2 + to_middle
        
        # 中层水箱水量平衡
        S2 = S2 - 0.4 * pet[t]
        if S2 < 0:
            S2 = 0
        
        # 中层出流（中等径流）
        q2 = k2 * S2
        S2 = S2 - q2
        
        # 侧向补给下层
        to_lower = 0.10 * S2
        S2 = S2 - to_lower
        S3 = S3 + to_lower
        
        # 下层水箱水量平衡（基流）
        S3 = S3 - 0.2 * pet[t]
        if S3 < 0:
            S3 = 0
        
        q3 = k3 * S3
        S3 = S3 - q3
        
        # 总径流 = 三箱之和
        q[t] = q1 + q2 + q3
    
    return q
```

### 物理特性
- 快速响应（q1）：峰高、峰短
- 中等响应（q2）：滞后、衰减
- 慢速响应（q3）：基流、稳定
- 水量守恒：输入 - 输出 = 储水变化

---

## SCS-CN 模型

### 适用场景
- 干旱小流域
- 降雨径流快速推求
- 缺乏详细水文资料的流域

### 参数范围
| 参数 | 下界 | 上界 | 单位 | 说明 |
|------|------|------|------|------|
| CN   | 30   | 95   | 无量纲 | 曲线数，反映流域持水能力 |
| Smax | 0    | 500  | mm    | 最大截留量 |
| Ia   | 0    | 20   | mm    | 初始截留 |

### 代码模板
```python
def simulate_runoff(precip, pet, params):
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
    
    # CN 转 Smax: Smax = 25400/CN - 254
    Smax = 25400.0 / CN - 254.0
    if Smax < 0:
        Smax = 0
    
    Ia = params.get("Ia", 0.2 * Smax)
    
    n = len(precip)
    q = np.zeros(n)
    
    for t in range(n):
        P = precip[t]
        
        if P <= Ia:
            # 降雨不足以克服初始截留，无径流
            q[t] = 0.0
        else:
            # 超额蓄水产流公式
            P_eff = P - Ia
            q[t] = (P_eff ** 2) / (P_eff + Smax)
        
        # 蒸散发消耗（简化：按比例消耗截留量）
        # 实际实现中可考虑土壤水分状态
    
    return q
```

### 物理特性
- 降雨小于 Ia 时不产流（阈值效应）
- 产流与 (P - Ia)^2 成正比（非线性）
- 适合干旱小流域，不适合湿润区

---

## 新安江模型 (Xinanjiang Model)

### 适用场景
- 湿润区、湿润季节
- 需要区分蓄满产流与超渗产流
- 南方湿润流域

### 参数范围
| 参数 | 下界 | 上界 | 单位 | 说明 |
|------|------|------|------|------|
| WM   | 50   | 200  | mm    | 流域平均张力水容量 |
| UM   | 20   | 80   | mm    | 上层张力水容量 |
| LM   | 30   | 120  | mm    | 下层张力水容量 |
| B    | 0.1  | 1.0  | 无量纲 | 张力水容量分布曲线指数 |
| K    | 0.5  | 1.0  | 无量纲 | 蒸散发折减系数 |
| C    | 0.1  | 0.3  | 无量纲 | 深层蒸散发系数 |
| SM   | 10   | 50   | mm    | 自由水蓄水容量 |

### 代码模板
```python
def simulate_runoff(precip, pet, params):
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
    WM = params.get("WM", 120.0)  # 流域平均张力水容量
    UM = params.get("UM", 40.0)   # 上层
    LM = params.get("LM", 80.0)   # 下层
    B = params.get("B", 0.4)     # 分布曲线指数
    K = params.get("K", 0.9)     # 蒸散发折减
    C = params.get("C", 0.15)    # 深层系数
    SM = params.get("SM", 30.0)   # 自由水容量
    
    WM = UM + LM
    WUM = UM
    WLM = LM
    
    n = len(precip)
    q = np.zeros(n)
    
    W = WM * 0.5  # 初始张力水蓄量
    S = SM * 0.5  # 初始自由水蓄量
    
    for t in range(n):
        P = precip[t]
        PE = K * pet[t]
        
        # 蒸散发计算（三层）
        if W >= WUM:
            # 上层满足
            EU = PE
            EL = 0
            EC = 0
        elif W >= WUM - WLM:
            # 上层不足，从下层蒸发
            EU = W - (WUM - WLM)
            EL = PE - EU
            EC = 0
        else:
            # 上层下层均不足，动用深层
            EU = 0
            EL = max(W - (WUM - WLM), 0)
            EC = PE - EU - EL
            EC = min(EC, C * PE)
        
        W = W - EU - EL - EC
        if W < 0:
            W = 0
        
        # 蓄满产流计算
        P_total = P + W
        WMM = WM * (1 + B)
        
        if P_total < WMM:
            # 未全部蓄满
            WMM_W = WMM * (1 - (P_total / WMM) ** (1 / (1 + B)))
            WMM_P = WMM * (1 - ((P_total - P) / WMM) ** (1 / (1 + B)))
            
            if P_total <= WMM_W:
                # 全部未蓄满，无径流
                W = P_total
                Q = 0
            else:
                # 部分蓄满
                FR = 1 - (WMM_W / WMM) ** (1 + B)
                W = WMM_W
                Q = FR * P
        else:
            # 全部蓄满
            Q = P + W - WM
            W = WM
        
        # 自由水产流
        S = S + Q
        if S < 0:
            S = 0
        
        EX = S / SM
        if EX > 1:
            EX = 1
        
        # 线性水库调蓄（简化：直接出流）
        q[t] = EX * S * 0.1
        S = S * (1 - 0.1)
    
    return q
```

### 物理特性
- 蓄满才产流（阈值效应）
- 产流面积随降雨增加
- 适合湿润区南方流域

---

## HBV 模型 (Hydrological By Växjö)

### 适用场景
- 通用型水文模型
- 温带、寒带流域
- 需要区分土壤含水量、融雪、地下水

### 参数范围
| 参数 | 下界 | 上界 | 单位 | 说明 |
|------|------|------|------|------|
| FC   | 100  | 500  | mm    | 土壤层最大蓄水容量 |
| LP   | 0.3  | 0.9  | 无量纲 | 土壤层临界比率 |
| BETA | 1.0  | 6.0  | 无量纲 | 产流系数 |
| K0   | 0.1  | 0.5  | 无量纲 | 快速响应流速系数 |
| K1   | 0.01 | 0.1  | 无量纲 | 慢速响应流速系数 |
| UZL  | 0    | 100  | mm    | 地下水上限 |

### 代码模板
```python
def simulate_runoff(precip, pet, params):
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
    
    # 状态变量
    soil = FC * 0.5  # 土壤含水量
    U1 = 0.0         # 快速响应上箱
    U2 = 0.0         # 快速响应下箱
    L = 0.0          # 慢速响应水库
    
    for t in range(n):
        P = precip[t]
        PE = pet[t]
        
        # 土壤层计算
        soil_change = (P * (soil / FC) ** BETA) if soil < FC else P
        soil_change = max(soil_change, -soil)
        soil = soil + soil_change
        
        # 蒸散发
        if soil > LP * FC:
            E = PE
        else:
            E = PE * (soil / (LP * FC))
        soil = max(soil - E, 0)
        
        # 产流（快速和慢速）
        P_eff = soil_change - max(soil_change - (soil - LP * FC), 0)
        
        # 进入快速响应
        U1 = U1 + P_eff * 0.5
        
        # 快流出流
        Q0 = K0 * max(U1 - UZL, 0)
        U1 = max(U1 - Q0, 0)
        
        # 慢速响应
        U2 = U2 + Q0
        Q1 = K1 * U2
        U2 = max(U2 - Q1, 0)
        
        # 总出流
        q[t] = Q0 + Q1
    
    return q
```

### 物理特性
- 土壤非线性产流（BETA 参数）
- 区分快速流和慢速流
- 适用温带、寒带流域

---

## 使用说明

1. **检索方式**: 使用 `CodeTemplateRAG.retrieve_by_plan(plan_text)` 根据 Planner 方案自动匹配
2. **手动指定**: 可在用户请求中直接指定模型类型，如"使用 Tank Model"
3. **参数传递**: 模板中的 `params.get("key", default)` 会被 SCE-UA 率定覆盖
4. **优化建议**: 
   - 干旱区 → 线性水库 或 SCS-CN
   - 湿润区 → 新安江 或 Tank Model
   - 通用型 → HBV