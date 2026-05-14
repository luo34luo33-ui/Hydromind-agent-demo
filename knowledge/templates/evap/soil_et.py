"""
土壤蒸散发机制模板

依据土壤含水量比例折算实际蒸散发
来源模型: HBV, GR4J
"""

def soil_et(pet, soil_storage, fc, lp):
    """
    土壤蒸散发计算
    
    参数:
        pet: 日潜在蒸散发 (mm/day)
        soil_storage: 当前土壤含水量 (mm)
        fc: 田间持水量 (mm)
        lp: 土壤蒸发限制系数 (ratio)
    
    返回:
        actual_et: 实际蒸散发 (mm/day)
    """
    if soil_storage > lp * fc:
        actual_et = pet
    else:
        actual_et = pet * (soil_storage / (lp * fc))
    
    actual_et = min(actual_et, soil_storage)
    actual_et = max(actual_et, 0)
    return actual_et