"""
分层蒸散发机制模板

三层结构：上层优先蒸发，中层次之，下层最慢
来源模型: XAJ, Sacramento
"""

def layered_et(precip, pet, w, um, lm, c):
    """
    三层蒸散发计算
    
    参数:
        precip: 日降雨量 (mm/day)
        pet: 日潜在蒸散发 (mm/day)
        w: 当前张力水蓄量 (mm)
        um: 上层张力水容量 (mm)
        lm: 中层张力水容量 (mm)
        c: 深层蒸发系数
    
    返回:
        eu: 上层蒸散发 (mm/day)
        el: 中层蒸散发 (mm/day)
        ec: 深层蒸散发 (mm/day)
        w_after: 蒸散后张力水蓄量
    """
    wum = um
    wlm = lm
    wm = wum + wlm
    
    p = precip
    pe = pet
    
    if w >= wum:
        eu = pe
        el = 0
        ec = 0
    elif w >= wum - wlm:
        eu = w - (wum - wlm)
        el = pe - eu
        ec = 0
    else:
        eu = 0
        el = max(w - (wum - wlm), 0)
        ec = pe - eu - el
        ec = min(ec, c * pe)
    
    w_after = w - eu - el - ec
    w_after = max(w_after, 0)
    
    return eu, el, ec, w_after