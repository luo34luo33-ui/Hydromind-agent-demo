import pandas as pd
import numpy as np
from pathlib import Path


def get_base_dir():
    """获取项目根目录"""
    return Path(__file__).resolve().parent.parent.parent


DATE_COLS_CANDIDATES = ["date", "Date", "DATE", "datetime", "time", "timestamp"]

YMD_CANDIDATES = [
    ["year", "month", "day"],
    ["Year", "Month", "Day"],
    ["YEAR", "MONTH", "DAY"],
]


def compute_pet_hargreaves(tmin, tmax, lat, doy):
    """
    使用 Hargreaves 公式从温度计算日 PET (mm/day)。

    参数:
        tmin: 日最低温 (℃)
        tmax: 日最高温 (℃)
        lat: 纬度 (度)
        doy: 年积日 (1-365/366)
    返回:
        pet: 日潜在蒸散发 (mm/day)
    """
    tmean = (tmax + tmin) / 2.0
    delta = np.radians(lat) * np.sin(2 * np.pi * doy / 365 - 1.39)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    ws = np.arccos(-np.tan(np.radians(lat)) * np.tan(delta))
    ra = (24 * 60 / np.pi) * 0.082 * dr * (ws * np.sin(np.radians(lat)) * np.sin(delta)
           + np.cos(np.radians(lat)) * np.cos(delta) * np.sin(ws))
    pet = 0.0023 * ra * (tmean + 17.8) * np.sqrt(np.maximum(tmax - tmin, 0))
    return np.maximum(pet, 0)


def compute_pet_oudin(tmean, lat, doy):
    """
    使用 Oudin 公式从平均温度计算日 PET (mm/day)。
    比 Hargreaves 更简单，仅需 Tmean 和纬度。

    参数:
        tmean: 日平均温 (℃)，可以由 (tmax+tmin)/2 得出
        lat: 纬度 (度)
        doy: 年积日
    返回:
        pet: 日潜在蒸散发 (mm/day)
    """
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    ws = np.arccos(-np.tan(np.radians(lat)) * np.tan(delta))
    ra = (24 * 60 / np.pi) * 0.082 * dr * (ws * np.sin(np.radians(lat)) * np.sin(delta)
           + np.cos(np.radians(lat)) * np.cos(delta) * np.sin(ws))
    pet = ra * np.maximum(tmean + 5, 0) / 2450
    return pet


def load_from_csv(csv_path, column_mapping=None, lat=None):
    """
    从用户 CSV 加载并转换为系统标准格式。

    标准输出列: date, precip, pet, q_obs

    参数:
        csv_path: CSV 文件路径
        column_mapping: 列名映射字典，键为系统列名，值为 CSV 中的列名:
            {
                "date": "date",         # 日期列 (可选: 用 year/month/day 替代)
                "precip": "P",          # 降雨列 (mm/day)
                "pet": "PET",           # PET 列 (mm/day)，可选 —— 若无则用温度估算
                "q_obs": "Qobs",        # 观测径流 (mm/day)
                "tmax": None,           # 最高温 (℃)，可选
                "tmin": None,           # 最低温 (℃)，可选
                "area_km2": None,       # 流域面积，用于单位换算: m³/s → mm/day
            }
        lat: 流域纬度，用于从温度估算 PET (若 pet 缺失但 tmax/tmin 存在)。
            若为 None 则尝试从 basin_attributes 读取。

    返回:
        DataFrame 包含 date, precip, pet, q_obs 列
    """
    if column_mapping is None:
        column_mapping = {}

    df = pd.read_csv(csv_path)

    date_col = None
    if column_mapping.get("date"):
        date_col = column_mapping["date"]
    else:
        for c in DATE_COLS_CANDIDATES:
            if c in df.columns:
                date_col = c
                break

    if date_col and date_col in df.columns:
        df["date"] = pd.to_datetime(df[date_col])
    else:
        ym = None
        for candidate in YMD_CANDIDATES:
            if all(c in df.columns for c in candidate):
                ym = candidate
                break
        if ym:
            y, m, d = ym
            df["date"] = pd.to_datetime(
                df[y].astype(str) + "-" + df[m].astype(str).str.zfill(2)
                + "-" + df[d].astype(str).str.zfill(2),
                errors="coerce",
            )
        else:
            raise ValueError(
                f"未找到日期列。请通过 column_mapping 指定 date 列名，"
                f"或确保 CSV 包含 Year/Month/Day 三列。"
                f"找到的列: {list(df.columns)}"
            )

    result = pd.DataFrame({"date": df["date"]})

    prec_col = column_mapping.get("precip", "precip")
    if prec_col in df.columns:
        result["precip"] = df[prec_col].astype(float)
    else:
        for name in ["P", "precip", "Precip", "prcp", "PRCP", "rainfall", "Rainfall",
                     "Rainfall_mm", "precip_mm", "P_mm"]:
            if name in df.columns:
                result["precip"] = df[name].astype(float)
                break
        else:
            raise ValueError(f"未找到降雨列。找到的列: {list(df.columns)}")

    pet_col = column_mapping.get("pet", "pet")
    if pet_col and pet_col in df.columns:
        result["pet"] = df[pet_col].astype(float)
    else:
        pet_found = False
        for name in ["pet", "PET", "Pet", "PET_mm", "pet_mm", "evap", "Evap"]:
            if name in df.columns:
                result["pet"] = df[name].astype(float)
                pet_found = True
                break
        if not pet_found:
            tmax_col = column_mapping.get("tmax", None)
            tmin_col = column_mapping.get("tmin", None)
            if tmax_col is None:
                for name in ["tmax", "TMAX", "Tmax", "temp_max", "tmax(C)"]:
                    if name in df.columns:
                        tmax_col = name
                        break
            if tmin_col is None:
                for name in ["tmin", "TMIN", "Tmin", "temp_min", "tmin(C)"]:
                    if name in df.columns:
                        tmin_col = name
                        break

            if tmax_col and tmin_col:
                if lat is None:
                    lat = _try_get_lat()
                doy = df["date"].dt.dayofyear.values
                tmax = df[tmax_col].astype(float).values
                tmin = df[tmin_col].astype(float).values
                result["pet"] = compute_pet_hargreaves(tmin, tmax, lat, doy)
            else:
                result["pet"] = 0.0

    q_col = column_mapping.get("q_obs", "q_obs")
    if q_col in df.columns:
        result["q_obs"] = df[q_col].astype(float)
    else:
        for name in ["Q", "q_obs", "Qobs", "qobs", "streamflow", "Streamflow",
                     "Flow", "discharge", "Discharge", "runoff", "Runoff",
                     "q_obs(mm)", "QObs", "QObs(mm/d)", "QObs(mm/day)",
                     "Streamflow_mm", "streamflow_mm"]:
            if name in df.columns:
                result["q_obs"] = df[name].astype(float)
                break
        else:
            raise ValueError(f"未找到径流列。找到的列: {list(df.columns)}")

    are_col = column_mapping.get("area_km2", None)
    if are_col is None:
        for name in ["area", "Area", "area_km2"]:
            if name in df.columns:
                are_col = name
                break
    if are_col and are_col in df.columns:
        area = df[are_col].iloc[0] if len(df) > 0 else 0
        if result["q_obs"].mean() > 100 and area > 0:
            result["q_obs"] = result["q_obs"] * 86400 / (area * 1e6) * 1000

    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
    return result[["date", "precip", "pet", "q_obs"]]


def _try_get_lat():
    """尝试从 basin_attributes 读一个默认纬度"""
    try:
        attrs = get_basin_attributes()
        if "lat" in attrs.columns:
            return float(attrs["lat"].iloc[0])
    except Exception:
        pass
    return 40.0


def find_basin_csvs(data_dir=None):
    """
    扫描 data 目录下所有 CSV，返回可用流域列表。
    若没有真实数据，退回内置样例。
    """
    if data_dir is None:
        data_dir = get_base_dir() / "data"

    csv_files = sorted(Path(data_dir).glob("*.csv"))
    basin_files = [f for f in csv_files if f.name not in ["basin_attributes.csv", "basin_metadata_merged.csv"]]

    ids = []
    for f in basin_files:
        stem = f.stem
        ids.append(stem)
    return ids


def load_basin_data(gauge_id, data_dir=None):
    """
    统一数据加载入口:
    1. 若 data/{gauge_id}.csv 存在 → 用 load_from_csv 加载
    2. 否则 → 降级到 generate_sample_data 生成模拟数据
    """
    if data_dir is None:
        data_dir = get_base_dir() / "data"

    csv_path = Path(data_dir) / f"{gauge_id}.csv"
    if csv_path.exists():
        return load_from_csv(str(csv_path))

    return generate_sample_data(gauge_id)


def generate_sample_data(gauge_id):
    """
    生成模拟的 CAMELS 时间序列数据。
    为 01013500 (湿润) 和 09513700 (干旱) 生成不同特征的合成数据。
    """
    dates = pd.date_range(start="2020-01-01", periods=365)

    if "01013500" in str(gauge_id):
        rng = np.random.default_rng(13500)
        precip = rng.gamma(2.0, 5.0, 365)
        temp_base = 15
        temp_amp = 10
    else:
        rng = np.random.default_rng(13700)
        precip = rng.gamma(0.5, 2.0, 365)
        temp_base = 25
        temp_amp = 15

    precip[precip < 1] = 0

    day_of_year = np.arange(365)
    temp = temp_base + temp_amp * np.sin(2 * np.pi * day_of_year / 365)
    pet = 0.1 * (temp + 10)

    q_obs = np.zeros(365)
    storage = 50.0
    for t in range(365):
        storage += precip[t] - 0.3 * pet[t]
        if storage < 0:
            storage = 0
        outflow = 0.2 * storage
        storage -= outflow
        q_obs[t] = max(outflow + rng.normal(0, 0.1), 0)

    df = pd.DataFrame({
        "date": dates,
        "precip": np.round(precip, 2),
        "pet": np.round(pet, 2),
        "q_obs": np.round(q_obs.clip(min=0), 3),
    })
    return df


def get_basin_attributes():
    """读取流域属性表"""
    root = get_base_dir()
    csv_path = root / "data" / "basin_attributes.csv"
    return pd.read_csv(csv_path, dtype={"gauge_id": str})


def get_basin_info_dict(gauge_id):
    """获取单个流域的属性字典"""
    df = get_basin_attributes()
    row = df[df["gauge_id"] == gauge_id]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def get_available_basin_ids():
    """返回所有可选的流域 ID 列表（属性表 + data 目录下 CSV）"""
    attrs = get_basin_attributes()
    ids_from_attrs = attrs["gauge_id"].tolist()
    ids_from_csvs = find_basin_csvs()
    all_ids = ids_from_attrs.copy()
    for cid in ids_from_csvs:
        if cid not in all_ids:
            all_ids.append(cid)
    if not all_ids:
        all_ids = ["01013500", "09513700"]
    return all_ids
