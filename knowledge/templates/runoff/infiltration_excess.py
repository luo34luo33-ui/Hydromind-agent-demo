"""
超渗产流机制模板

降雨强度超过入渗能力时产生地表径流
来源模型: SCS-CN, Green-Ampt
"""

def infiltration_excess(precip, infiltration_capacity, params):
    """
    超渗产流计算
    
    参数:
        precip: 日降雨量 (mm/day)
        infiltration_capacity: 入渗能力 (mm/day)
        params: 参数字典 {"cn": 曲线数, "smax": 最大截留}
    
    返回:
        runoff: 产流量 (mm/day)
        infiltration: 入渗量 (mm/day)
    """
    cn = params.get("cn", 75)
    smax = max(25400.0 / cn - 254.0, 0)
    ia = params.get("ia_factor", 0.2) * smax
    
    if precip <= ia:
        return 0.0, precip
    
    p_eff = precip - ia
    runoff = (p_eff ** 2) / (p_eff + smax)
    infiltration = precip - runoff
    
    runoff = max(runoff, 0)
    infiltration = max(infiltration, 0)
    return runoff, infiltration