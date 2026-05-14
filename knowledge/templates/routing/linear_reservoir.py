"""
线性水库汇流机制模板

蓄水量与出流量呈线性关系 Q = k * S
来源模型: XAJ, HBV, Tank, SAC
"""

def linear_reservoir(inflow, storage, k):
    """
    线性水库汇流计算
    
    参数:
        inflow: 入流量 (mm/day)
        storage: 当前蓄水量 (mm)
        k: 消退系数 (0 < k < 1)
    
    返回:
        outflow: 出流量 (mm/day)
        storage_after: 汇流后蓄水量
    """
    storage_after = storage + inflow
    outflow = k * storage_after
    storage_after = storage_after - outflow
    
    outflow = max(outflow, 0)
    storage_after = max(storage_after, 0)
    return outflow, storage_after