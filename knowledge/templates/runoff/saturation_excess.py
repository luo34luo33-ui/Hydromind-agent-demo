"""
蓄满产流机制模板

当土壤含水量超过田间持水量时产生径流
来源模型: XAJ, TOPMODEL, Sacramento
"""

def saturation_excess(precip, soil_storage, field_capacity, params):
    """
    蓄满产流计算
    
    参数:
        precip: 日降雨量 (mm/day)
        soil_storage: 土壤当前蓄水量 (mm)
        field_capacity: 田间持水量 (mm)
        params: 参数字典 {"b": 流域分配参数, "wm": 最大张力水}
    
    返回:
        runoff: 产流量 (mm/day)
        soil_storage_after: 产流后土壤蓄水量
    """
    b = params.get("b", 0.4)
    wm = params.get("wm", field_capacity * 1.4)
    wmm = field_capacity * (1 + b)
    
    if soil_storage >= field_capacity:
        soil_storage_after = min(soil_storage + precip, wm)
        runoff = precip - (soil_storage_after - soil_storage)
    else:
        w_temp = soil_storage + precip
        if w_temp >= wmm:
            runoff = max(precip + soil_storage - field_capacity, 0)
            soil_storage_after = field_capacity
        else:
            wmm_w = wmm * (1 - (w_temp / wmm) ** (1 / (1 + b)))
            fr_new = 1 - (wmm_w / wmm) ** (1 + b)
            runoff = max(fr_new * precip, 0)
            soil_storage_after = wmm_w
    
    runoff = max(runoff, 0)
    return runoff, soil_storage_after