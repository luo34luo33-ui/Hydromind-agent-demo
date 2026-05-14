"""
线性水库串联汇流机制模板

多个线性水库串联，模拟延迟效应
来源模型: HBV, Tank, Sacramento
"""

def cascade(inflow, k1, k2, k3):
    """
    三库串联汇流计算
    
    参数:
        inflow: 入流量 (mm/day)
        k1, k2, k3: 各水库消退系数
    
    返回:
        outflow: 最终出流量 (mm/day)
    """
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    
    s1 = s1 + inflow
    q1 = k1 * s1
    s1 = s1 - q1
    
    s2 = s2 + q1
    q2 = k2 * s2
    s2 = s2 - q2
    
    s3 = s3 + q2
    outflow = k3 * s3
    s3 = s3 - outflow
    
    return max(outflow, 0)