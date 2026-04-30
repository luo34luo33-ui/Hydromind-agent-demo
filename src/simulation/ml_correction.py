import xgboost as xgb
import numpy as np


class ResidualCorrector:
    """
    基于 XGBoost 的误差残差校正器。
    训练: 根据 LLM 模拟结果和降雨特征预测误差
    推理: 将预测的误差加回到原始模拟结果上
    """

    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        )

    def _build_features(self, q_sim, precip):
        """构建特征矩阵: 当前模拟流量、当前降雨、滞后1-3天降雨"""
        n = len(q_sim)
        p = precip.copy()
        p_lag1 = np.roll(p, 1)
        p_lag1[0] = p_lag1[1]
        p_lag2 = np.roll(p, 2)
        p_lag2[0] = p_lag2[2]
        p_lag3 = np.roll(p, 3)
        p_lag3[0] = p_lag3[3]
        return np.column_stack([q_sim, p, p_lag1, p_lag2, p_lag3])

    def train(self, q_obs, q_sim, precip):
        """
        训练误差校正模型。
        q_obs: 观测径流序列
        q_sim: LLM 模拟径流序列
        precip: 降雨序列
        """
        q_obs = np.asarray(q_obs, dtype=np.float64)
        q_sim = np.asarray(q_sim, dtype=np.float64)
        precip = np.asarray(precip, dtype=np.float64)
        
        valid_mask = (
            np.isfinite(q_obs) & 
            np.isfinite(q_sim) & 
            np.isfinite(precip) &
            (q_obs >= 0) &
            (q_sim >= 0)
        )
        
        if valid_mask.sum() < 10:
            return
        
        q_obs_valid = q_obs[valid_mask]
        q_sim_valid = q_sim[valid_mask]
        precip_valid = precip[valid_mask]
        
        error = q_obs_valid - q_sim_valid
        
        error_valid_mask = np.isfinite(error)
        if error_valid_mask.sum() < 10:
            return
        
        error = error[error_valid_mask]
        q_sim_valid = q_sim_valid[error_valid_mask]
        precip_valid = precip_valid[error_valid_mask]
        
        X = self._build_features(q_sim_valid, precip_valid)
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        error = np.nan_to_num(error, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.model.fit(X, error)

    def predict(self, q_sim, precip):
        """
        对模拟结果进行校正。
        返回 q_corrected = q_sim + predicted_error
        """
        q_sim = np.asarray(q_sim, dtype=np.float64)
        precip = np.asarray(precip, dtype=np.float64)
        
        valid_mask = np.isfinite(q_sim) & np.isfinite(precip) & (q_sim >= 0)
        
        q_corrected = q_sim.copy()
        
        if valid_mask.sum() > 0:
            X = self._build_features(q_sim[valid_mask], precip[valid_mask])
            correction = self.model.predict(X)
            q_corrected[valid_mask] = q_sim[valid_mask] + correction
        
        return q_corrected
