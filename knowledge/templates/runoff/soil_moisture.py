"""
土壤水分账机制模板

通过土壤含水量动态平衡计算产流
来源模型: HBV, GR4J
"""

def soil_moisture_accounting(precip, pet, soil_storage, fc, lp, beta, params):
    """
    土壤水分账计算
    
    参数:
        precip: 日降雨量 (mm/day)
        pet: 日潜在蒸散发 (mm/day)
        soil_storage: 当前土壤含水量 (mm)
        fc: 田间持水量 (mm)
        lp: 土壤蒸发限制系数 (ratio)
        beta: 形状参数
        params: 额外参数字典
    
    返回:
        runoff: 产流量 (mm/day)
        actual_et: 实际蒸散发 (mm/day)
        soil_storage_after: 计算后土壤含水量
    """
    p_eff = precip * (soil_storage / fc) ** beta if soil_storage < fc else precip
    p_eff = max(p_eff, -soil_storage)
    soil_storage = soil_storage + p_eff
    
    if soil_storage > lp * fc:
        actual_et = pet
    else:
        actual_et = pet * (soil_storage / (lp * fc))
    
    actual_et = min(actual_et, soil_storage)
    soil_storage = soil_storage - actual_et
    
    p_eff_for_runoff = max(p_eff - max(soil_storage - lp * fc, 0), 0)
    runoff = p_eff_for_runoff
    
    runoff = max(runoff, 0)
    return runoff, actual_et, soil_storage