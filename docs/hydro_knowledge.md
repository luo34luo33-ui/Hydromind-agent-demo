# 水文模型核心原理

## 1. 水量平衡方程 (Water Balance)
dS/dt = P - ET - Q
其中 P 为降雨，ET 为蒸散发，Q 为径流，S 为流域储水量。

## 2. 产流模型 (Runoff Generation)
- **SCS-CN 模型**: Q = (P - Ia)^2 / (P - Ia + S)，适用于降雨径流快速推求。
- **降雨径流系数法**: Q = C * P，简单线性关系。
- **Tank Model**: 串联水箱模型，通过多个出流孔模拟不同速度的径流组分。

## 3. 汇流模型 (Routing)
- **线性水库模型**: Q_out = k * S，流量与储水量成正比。
- **单位线法 (Unit Hydrograph)**: 考虑降雨在时间上的分布和延迟。
- **Muskingum 法**: 用于河道汇流计算。

## 4. 蒸散发计算
- **潜在蒸散发 (PET)**: 考虑气温和日照。
- **实际蒸散发 (AET)**: AET = min(PET, 可用土壤水)。

## 5. Python 代码示例

### 线性水库模型:
```python
def simulate_runoff(precip, pet, params):
    k = params.get("k", 0.3)
    storage = np.zeros_like(precip)
    q = np.zeros_like(precip)
    S = params.get("S0", 50.0)
    for t in range(len(precip)):
        S += precip[t] - 0.5 * pet[t]
        if S < 0:
            S = 0
        q[t] = k * S
        S -= q[t]
    return q
```
