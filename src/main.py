import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import io
import zipfile
from pathlib import Path
from datetime import datetime

from utils.data_loader import (
    load_basin_data,
    get_basin_attributes,
    get_basin_info_dict,
    get_available_basin_ids,
)
from utils.rag_engine import HydroKnowledgeBase, CodeTemplateRAG
from agents.graph import run_agent_loop
from agents.planner import Planner, ModelingPlan
from agents.executer import Executer
from agents.validator import CodeValidator, execute_with_fallback, FALLBACK_CODE
from simulation.ml_correction import ResidualCorrector
from simulation.sceua import (
    SCEUA, extract_params_from_code, get_bounds_for_params,
    build_calibration_objective,
)


def compute_nse(obs, sim):
    """计算 Nash-Sutcliffe 效率系数"""
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs = obs[mask]
    sim = sim[mask]
    
    if len(obs) == 0:
        return 0.0
    
    eps = 1e-10
    numerator = np.sum((sim - obs) ** 2)
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    
    if denominator < eps or np.isnan(denominator):
        return 0.0
    result = 1.0 - numerator / denominator
    return 0.0 if np.isnan(result) else result


def compute_kge(obs, sim):
    """计算 Kling-Gupta 效率系数"""
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs = obs[mask]
    sim = sim[mask]
    
    if len(obs) == 0:
        return 0.0
    
    r = np.corrcoef(obs, sim)[0, 1]
    if np.isnan(r):
        r = 0.0
    alpha = np.std(sim) / (np.std(obs) + 1e-10)
    beta = np.mean(sim) / (np.mean(obs) + 1e-10)
    
    if np.isnan(alpha):
        alpha = 1.0
    if np.isnan(beta):
        beta = 1.0
    
    result = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return 0.0 if np.isnan(result) else result


def build_export_dataframe(data, q_sim, q_corrected):
    """构建可导出的完整结果 DataFrame"""
    df = data[["date", "precip", "pet", "q_obs"]].copy()
    df["q_sim"] = q_sim
    df["q_corrected"] = q_corrected
    df["error_raw"] = df["q_obs"] - df["q_sim"]
    df["error_corrected"] = df["q_obs"] - df["q_corrected"]
    return df


def split_calibration_validation(data, calib_years=25):
    """按时间划分率定期和验证期"""
    data = data.sort_values("date").reset_index(drop=True)
    
    if "date" not in data.columns:
        n = len(data)
        split_idx = int(n * 0.75)
        return data.iloc[:split_idx], data.iloc[split_idx:]
    
    years = data["date"].dt.year
    unique_years = sorted(years.unique())
    n_years = len(unique_years)
    
    if n_years < calib_years:
        n = len(data)
        split_idx = int(n * 0.75)
        return data.iloc[:split_idx], data.iloc[split_idx:]
    
    calib_year_list = unique_years[:calib_years]
    
    calib_data = data[years.isin(calib_year_list)].copy()
    valid_data = data[~years.isin(calib_year_list)].copy()
    
    return calib_data, valid_data


st.set_page_config(
    page_title="Hydromind流域水文模型开发智能体v1.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 Hydromind 流域水文模型开发智能体 v1.0")
st.markdown(
    "Agent 根据流域特征自动生成产流/汇流模型 → SCE-UA 全局率定 → "
    "XGBoost 误差校正 → 对比实观测流"
)

if "generated_code" not in st.session_state:
    st.session_state["generated_code"] = None
if "plan" not in st.session_state:
    st.session_state["plan"] = None
if "q_sim" not in st.session_state:
    st.session_state["q_sim"] = None
if "q_corrected" not in st.session_state:
    st.session_state["q_corrected"] = None
if "used_fallback" not in st.session_state:
    st.session_state["used_fallback"] = False
if "sceua_optimal_params" not in st.session_state:
    st.session_state["sceua_optimal_params"] = None
if "sceua_optimal_nse" not in st.session_state:
    st.session_state["sceua_optimal_nse"] = None
if "sceua_history" not in st.session_state:
    st.session_state["sceua_history"] = None
if "_prev_basin" not in st.session_state:
    st.session_state["_prev_basin"] = None
if "user_request" not in st.session_state:
    st.session_state["user_request"] = ""
if "_prev_user_request" not in st.session_state:
    st.session_state["_prev_user_request"] = ""
if "trigger_run" not in st.session_state:
    st.session_state["trigger_run"] = False

sidebar = st.sidebar

sidebar.header("🔑 系统配置")
api_key = sidebar.text_input("OpenAI API Key", value="", type="password",
                             placeholder="sk-...")
if api_key and api_key.startswith("sk-"):
    os.environ["OPENAI_API_KEY"] = api_key

selected_model = sidebar.selectbox(
    "选择 LLM 模型",
    ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    index=0
)
sidebar.caption(f"🤖 当前模型: {selected_model}")



basin_ids = get_available_basin_ids()
selected_basin_id = sidebar.selectbox("选择流域", basin_ids)

# 切流域时清空 AI 缓存
if st.session_state["_prev_basin"] != selected_basin_id:
    st.session_state["plan"] = None
    st.session_state["generated_code"] = None
    st.session_state["q_sim"] = None
    st.session_state["q_corrected"] = None
    st.session_state["sceua_optimal_params"] = None
    st.session_state["sceua_optimal_nse"] = None
    st.session_state["_prev_basin"] = selected_basin_id

basin_info = get_basin_info_dict(selected_basin_id)
if basin_info:
    info_lines = []
    for key, label, unit in [("name", "流域名称", ""), ("climate", "气候类型", ""),
                              ("area", "流域面积", "km²"), ("slope", "平均坡度", "m/km"),
                              ("permeability", "土壤渗透性", ""), ("lat", "纬度", "°")]:
        val = basin_info.get(key)
        if val is not None and val != "":
            if unit and unit != "":
                info_lines.append(f"**{label}**: {val} {unit}")
            else:
                info_lines.append(f"**{label}**: {val}")
    sidebar.info("\n\n".join(info_lines))

use_ml_correction = sidebar.checkbox("🤖 使用 XGBoost 误差校正", value=False)

data = load_basin_data(selected_basin_id)

left_col, right_col = st.columns([1, 1.2])

with left_col:
    st.subheader("🤖 模型生成与执行")

    st.markdown("#### 💬 告诉 AI 你的建模需求（可选）")
    user_request = st.text_area(
        "例如：'用 SCS-CN 做产流'、'加一个慢速地下水箱'",
        value=st.session_state.get("user_request", ""),
        key="user_request_input",
        height=68,
        label_visibility="collapsed",
    )
    st.session_state["user_request"] = user_request

    if user_request != st.session_state.get("_prev_user_request", ""):
        st.session_state["plan"] = None
        st.session_state["generated_code"] = None
        st.session_state["q_sim"] = None
        st.session_state["q_corrected"] = None
        st.session_state["sceua_optimal_params"] = None
        st.session_state["sceua_optimal_nse"] = None
        st.session_state["_prev_user_request"] = user_request

    st.markdown("")

    def run_modeling_flow():
        progress = st.status("Agent 工作中...", expanded=True)

        # ===== Phase 1: Planner =====
        if st.session_state["plan"] is None:
            progress.write("📋 Planner: 正在检索知识库并制定方案...")
            kb = HydroKnowledgeBase()
            user_request = st.session_state.get("user_request", "")
            if user_request.strip():
                rag_query = user_request
            else:
                rag_query = "水文模型 产流 汇流"
            context = kb.retrieve(rag_query, k=5)
            planner = Planner(api_key, selected_model)
            
            planner_use_structured = selected_model in ["gpt-4o", "gpt-4o-mini"]
            plan_obj = planner.plan(
                str(basin_info), context, st.session_state.get("user_request", ""),
                use_structured=planner_use_structured
            )
            
            if planner_use_structured and isinstance(plan_obj, ModelingPlan):
                st.session_state["plan"] = plan_obj
                progress.write(f"✅ 方案生成完成 (结构化: {plan_obj.runoff_type} + {plan_obj.flow_routing})")
            else:
                st.session_state["plan"] = plan_obj
                progress.write("✅ 方案生成完成")
        else:
            plan_obj = st.session_state["plan"]
            progress.write("📋 使用已缓存的建模方案")

        # ===== Phase 2: Code Generation & Validation (LangGraph) =====
        need_codegen = (st.session_state["generated_code"] is None)

        if need_codegen:
            progress.write("📚 正在检索代码模板...")
            code_rag = CodeTemplateRAG()
            
            if isinstance(plan_obj, ModelingPlan):
                plan_str = plan_obj.description
                param_constraints = plan_obj.param_suggestions
                code_template = code_rag.get_templates_by_ids(plan_obj.template_ids)
            else:
                plan_str = str(plan_obj)
                param_constraints = None
                code_template = code_rag.retrieve_by_plan(plan_str)
            
            progress.write(f"✅ 已匹配代码模板")

            test_params = {"k": 0.40, "S0": 100.0, "CN": 75.0}
            test_inputs = {
                "precip": data["precip"].values,
                "pet": data["pet"].values,
                "params": test_params,
            }

            progress.write("⚙️ LangGraph Agent 运行中...")
            
            use_structured = selected_model in ["gpt-4o", "gpt-4o-mini"]
            
            generated_params_config = None
            
            if use_structured:
                executer = Executer(api_key, selected_model)
                code_result = {"code": None, "parameters_config": {}}
                error_msg = ""
                for attempt in range(3):
                    if attempt == 0:
                        progress.write(f"💻 Executer: 正在生成代码 (结构化输出)...")
                        code_result = executer.generate_code(plan_str, code_template=code_template, param_constraints=param_constraints, use_structured=True)
                    else:
                        progress.write(f"🔁 重试: 将错误反馈给 LLM...")
                        code_result = executer.retry_with_error(code_result, error_msg, use_structured=True)
                    
                    code = code_result.get("code", "") if isinstance(code_result, dict) else code_result
                    
                    syntax_ok, syntax_err = CodeValidator.validate_syntax(code)
                    if not syntax_ok:
                        error_msg = f"语法错误: {syntax_err}"
                        progress.write(f"⚠️ {error_msg}")
                        continue
                    
                    exec_ok, _, exec_err = CodeValidator.execute_safe(code, test_inputs)
                    if exec_ok:
                        st.session_state["generated_code"] = code
                        generated_params_config = code_result.get("parameters_config", {})
                        st.session_state["used_fallback"] = False
                        progress.write("✅ 代码验证通过 (结构化输出)")
                        break
                    else:
                        error_msg = exec_err or ""
                        progress.write(f"⚠️ 执行错误: {error_msg[:200]}...")
                
                if st.session_state["generated_code"] is None:
                    progress.write("🛟 结构化输出失败，尝试普通模式...")
            
            if st.session_state["generated_code"] is None:
                result = run_agent_loop(
                    plan=plan_str,
                    code_template=code_template,
                    inputs=test_inputs,
                    openai_api_key=api_key,
                    model_name=selected_model,
                    max_attempts=3
                )
                
                if result["validated"] and result["code"]:
                    st.session_state["generated_code"] = result["code"]
                    st.session_state["used_fallback"] = False
                    progress.write(f"✅ LangGraph Agent 完成 (尝试次数: {result['attempts']})")
                else:
                    progress.write(f"⚠️ Agent 执行失败: {result.get('error', '未知错误')[:100]}")
                    progress.write("🛟 使用兜底线性水库模型...")
                    st.session_state["generated_code"] = FALLBACK_CODE
                    st.session_state["used_fallback"] = True
        else:
            progress.write("💻 使用已缓存的模型代码")
            generated_params_config = st.session_state.get("generated_params_config", {})

        # ===== Phase 3: SCE-UA Calibration =====
        code = st.session_state["generated_code"]

        calib_data, valid_data = split_calibration_validation(data, calib_years=25)
        precip_vals = calib_data["precip"].values
        pet_vals = calib_data["pet"].values
        q_obs_vals = calib_data["q_obs"].values

        if generated_params_config:
            param_names = list(generated_params_config.keys())
            bounds_list = [generated_params_config[k] for k in param_names]
            st.session_state["generated_params_config"] = generated_params_config
        else:
            param_names = extract_params_from_code(code)
            if not param_names:
                param_names = ["k", "S0"]
            bounds_list = get_bounds_for_params(param_names)
        
        n_params = len(param_names)
        progress.write(f"🎯 SCE-UA 率定: {n_params} 个参数 ({', '.join(param_names)})")

        obj_func = build_calibration_objective(
            code, precip_vals, pet_vals, q_obs_vals, param_names, compute_nse
        )

        sceua = SCEUA(bounds_list, obj_func, maxn=3000)
        best_x, best_nse, history = sceua.calibrate()

        ratio = (sceua.neval / sceua.maxn) * 100
        progress.write(f"✅ 率定完成: {sceua.neval} 次评估 ({ratio:.0f}%), 最优 NSE = {best_nse:.4f}")

        optimal_params = dict(zip(param_names, np.atleast_1d(best_x)))
        for k, v in optimal_params.items():
            progress.write(f"   → 最优 {k} = {v:.4f}")

        st.session_state["sceua_optimal_params"] = optimal_params
        st.session_state["sceua_optimal_nse"] = float(best_nse)
        st.session_state["sceua_history"] = history

        # ===== Phase 4: Execute with Optimal Params on Full Data =====
        full_inputs = {
            "precip": data["precip"].values,
            "pet": data["pet"].values,
            "params": optimal_params,
        }
        exec_ok, q_sim_full, exec_err = CodeValidator.execute_safe(code, full_inputs)

        if exec_ok:
            st.session_state["q_sim"] = q_sim_full
            progress.write("✅ 全数据模拟执行成功")
        else:
            err_msg = exec_err[:200] if exec_err else "Unknown error"
            progress.write(f"⚠️ 全数据模拟失败: {err_msg}...")
            progress.write("🛟 尝试兜底模型...")
            fb_code = FALLBACK_CODE
            fb_inputs = {"precip": data["precip"].values, "pet": data["pet"].values, "params": optimal_params}
            _, q_sim_full, fb_msg, used_fb = execute_with_fallback(code, fb_inputs)
            st.session_state["q_sim"] = q_sim_full
            st.session_state["generated_code"] = fb_code
            st.session_state["used_fallback"] = True
            if fb_msg:
                progress.write(f"⚠️ {fb_msg[:200]}")

        # ===== Phase 5: ML Correction =====
        if st.session_state["q_sim"] is not None and use_ml_correction:
            progress.write("🤖 ML 校正: XGBoost 残差学习...")
            corrector = ResidualCorrector()
            corrector.train(
                calib_data["q_obs"].values,
                st.session_state["q_sim"][:len(calib_data)],
                calib_data["precip"].values,
            )
            q_corrected = corrector.predict(
                st.session_state["q_sim"], data["precip"].values
            )
            st.session_state["q_corrected"] = q_corrected
            progress.write("✅ 全部流程完成")
        elif st.session_state["q_sim"] is not None and not use_ml_correction:
            st.session_state["q_corrected"] = None
            progress.write("✅ 模拟完成（未使用XGBoost校正）")

        progress.update(label="流程完成", state="complete", expanded=False)

    start_col1, start_col2 = st.columns([2, 1])
    with start_col1:
        if st.button("🚀 开始智能建模", type="primary", use_container_width=True):
            if not api_key or not api_key.startswith("sk-"):
                st.error("请先在侧边栏输入有效的 OpenAI API Key（以 sk- 开头）")
            else:
                st.session_state["user_request"] = user_request
                run_modeling_flow()
    with start_col2:
        if st.button("🔄 重新规划", use_container_width=True,
                     help="清空缓存的方案，重新调用 Planner"):
            st.session_state["plan"] = None
            st.session_state["generated_code"] = None
            st.session_state["q_sim"] = None
            st.session_state["q_corrected"] = None
            st.session_state["sceua_optimal_params"] = None
            st.session_state["sceua_optimal_nse"] = None
            st.rerun()

    if st.session_state.get("trigger_run", False):
        st.session_state["trigger_run"] = False
        if not api_key or not api_key.startswith("sk-"):
            st.error("请先在侧边栏输入有效的 OpenAI API Key（以 sk- 开头）")
        else:
            run_modeling_flow()

    if st.session_state.get("plan"):
        with st.expander("📋 Planner 建模方案", expanded=False):
            st.markdown(st.session_state["plan"])

    if st.session_state.get("generated_code"):
        with st.expander("💻 Executer 生成的代码", expanded=False):
            st.code(st.session_state["generated_code"], language="python")
            if st.session_state.get("used_fallback"):
                st.warning("⚠️ LLM 生成的代码未能成功执行，当前使用内置兜底模型。")

with right_col:
    st.subheader("📈 模拟与观测对比")

    calib_data_display, _ = split_calibration_validation(data, calib_years=25)

    if st.session_state["q_sim"] is not None:
        q_sim = st.session_state["q_sim"]
        q_obs = data["q_obs"].values
        calib_q_obs = calib_data_display["q_obs"].values
        calib_q_sim = q_sim[:len(calib_data_display)]

        dates = pd.to_datetime(data["date"]).dt.strftime('%Y-%m-%d').tolist()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=q_obs,
            name="观测径流", line=dict(color="black", width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=q_sim,
            name="SCEUA 率定模拟", line=dict(color="#1f77b4", width=1.5),
        ))

        if st.session_state["q_corrected"] is not None:
            q_corrected = st.session_state["q_corrected"]
            fig.add_trace(go.Scatter(
                x=dates, y=q_corrected,
                name="AI+ML 融合模拟", line=dict(color="#d62728", width=1.5, dash="dot"),
            ))

        fig.update_layout(
            title="日径流过程线对比 (Daily Runoff Comparison)",
            xaxis_title="日期",
            yaxis_title="径流量 (mm/day)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            height=420,
            margin=dict(l=0, r=0, t=50, b=0),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        nse_raw = compute_nse(calib_q_obs, calib_q_sim)
        kge_raw = compute_kge(calib_q_obs, calib_q_sim)

        m1, m2 = st.columns(2)
        m1.metric("率定期 NSE", f"{nse_raw:.3f}")
        m2.metric("率定期 KGE", f"{kge_raw:.3f}")

        if st.session_state["q_corrected"] is not None:
            q_corrected = st.session_state["q_corrected"]
            calib_q_corr = q_corrected[:len(calib_data_display)]
            nse_corr = compute_nse(calib_q_obs, calib_q_corr)
            kge_corr = compute_kge(calib_q_obs, calib_q_corr)
            m3, m4 = st.columns(2)
            m3.metric("AI+ML 校正 NSE", f"{nse_corr:.3f}", delta=f"{nse_corr - nse_raw:.3f}")
            m4.metric("AI+ML 校正 KGE", f"{kge_corr:.3f}", delta=f"{kge_corr - kge_raw:.3f}")

        sceua_history = st.session_state.get("sceua_history")
        if sceua_history:
            with st.expander("📈 SCE-UA 收敛曲线", expanded=False):
                hist_nse = [h[1] for h in sceua_history]
                hist_iter = [h[0] for h in sceua_history]
                fig_conv = go.Figure()
                fig_conv.add_trace(go.Scatter(
                    x=hist_iter, y=hist_nse,
                    mode="lines+markers",
                    name="NSE",
                    line=dict(color="#2ca02c", width=2),
                ))
                fig_conv.update_layout(
                    title="SCE-UA 率定收敛过程",
                    xaxis_title="评估次数",
                    yaxis_title="NSE",
                    height=250,
                    margin=dict(l=40, r=20, t=40, b=30),
                )
                st.plotly_chart(fig_conv, use_container_width=True)

        st.divider()

        q_corrected_export = st.session_state.get("q_corrected")
        export_df = build_export_dataframe(data, q_sim, q_corrected_export)
        csv_buffer = io.BytesIO()
        export_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="📥 导出完整结果 (CSV)",
            data=csv_data,
            file_name=f"simulation_{selected_basin_id}.csv",
            mime="text/csv",
        )

        with st.expander("📋 结果数据预览", expanded=False):
            st.dataframe(export_df.head(30), use_container_width=True)

        st.caption(
            "NSE (Nash-Sutcliffe Efficiency): 越接近 1 越好。"
            "KGE (Kling-Gupta Efficiency): 综合考虑相关性、偏差和变异性。"
        )
    else:
        st.info("👈 点击 '开始智能建模' 按钮启动 Agent 流程")

st.divider()

st.subheader("📥 代码下载")

if st.session_state.get("generated_code") and st.session_state.get("sceua_optimal_params"):
    generated_model_code = st.session_state["generated_code"]
    optimal_params = st.session_state["sceua_optimal_params"]
    basin_id = selected_basin_id
    
    param_comments = []
    for k, v in optimal_params.items():
        if k == "k":
            param_comments.append("k: 出流系数 (0.05-0.80, 无量纲)")
        elif k == "k1":
            param_comments.append("k1: 上层水箱出流系数 (0.10-0.80, 无量纲)")
        elif k == "k2":
            param_comments.append("k2: 中层水箱出流系数 (0.01-0.30, 无量纲)")
        elif k == "k3":
            param_comments.append("k3: 下层水箱出流系数 (0.001-0.10, 无量纲)")
        elif k == "S0":
            param_comments.append("S0: 初始储水量 (0-200 mm)")
        elif k == "S1":
            param_comments.append("S1: 上层水箱初始储水量 (0-100 mm)")
        elif k == "S2":
            param_comments.append("S2: 中层水箱初始储水量 (0-150 mm)")
        elif k == "S3":
            param_comments.append("S3: 下层水箱初始储水量 (0-200 mm)")
        elif k == "CN":
            param_comments.append("CN: SCS曲线数 (30-95, 无量纲)")
        elif k == "Smax":
            param_comments.append("Smax: 最大截留量 (0-500 mm)")
        elif k == "Ia":
            param_comments.append("Ia: 初始截留 (0-20 mm)")
        else:
            param_comments.append(f"{k}: 率定参数")
    
    full_script = f'''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
流域水文模型率定工具 (Agent开发模型 + SCE-UA + XGBoost)
=====================================================
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
目标流域: {basin_id}

使用方法:
    python run_calibration.py --input your_data.csv
    python run_calibration.py --input your_data.csv --maxn 5000

输入数据格式 (CSV):
    date,precip,pet,q_obs
    2020-01-01,5.2,2.1,1.3
    2020-01-02,0.0,2.3,1.1
    ...

说明:
    - precip: 日降雨量 (mm/day)
    - pet: 日潜在蒸散发 (mm/day)  
    - q_obs: 日观测径流 (mm/day)
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import argparse
import sys
import subprocess
from pathlib import Path


def check_and_install_deps():
    \"\"\"检查并安装依赖\"\"\"
    required = {"numpy", "pandas", "xgboost"}
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"正在安装依赖: {{', '.join(missing)}} ...")
        for pkg in missing:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
                print(f"  ✓ {{pkg}} 安装成功")
            except Exception as e:
                print(f"  ✗ {{pkg}} 安装失败: {{e}}")
                sys.exit(1)
        print("依赖安装完成！")


check_and_install_deps()


# =============================================================================
# 第一部分: 数据加载 (支持用户上传)
# =============================================================================

def load_user_data(csv_path):
    """加载用户上传的CSV数据"""
    df = pd.read_csv(csv_path)
    
    # 自动识别列名
    date_col = None
    for c in ["date", "Date", "DATE"]:
        if c in df.columns:
            date_col = c
            break
    
    precip_col = None
    for c in ["precip", "P", "prcp", "rainfall", "Rainfall"]:
        if c in df.columns:
            precip_col = c
            break
            
    pet_col = None
    for c in ["pet", "PET", "ETp"]:
        if c in df.columns:
            pet_col = c
            break
            
    q_obs_col = None
    for c in ["q_obs", "Q", "qobs", "runoff", "streamflow"]:
        if c in df.columns:
            q_obs_col = c
            break
    
    if not all([precip_col, q_obs_col]):
        print("错误: CSV必须包含降雨(precip)和径流(q_obs)列")
        sys.exit(1)
    
    result = pd.DataFrame()
    if date_col:
        result["date"] = pd.to_datetime(df[date_col])
    result["precip"] = df[precip_col].astype(float)
    
    if pet_col:
        result["pet"] = df[pet_col].astype(float)
    else:
        # 默认PET为0
        result["pet"] = 0.0
    
    result["q_obs"] = df[q_obs_col].astype(float)
    
    return result


# =============================================================================
# 第二部分: Agent开发的水文模型代码
# =============================================================================

def simulate_runoff(precip, pet, params):
    \"\"\"
    水文模型模拟函数
    
    参数:
        precip: 日降雨量序列 (mm/day)
        pet: 日潜在蒸散发序列 (mm/day)
        params: 模型参数字典
            {chr(10).join(param_comments)}
    
    返回:
        q: 日径流序列 (mm/day)
    \"\"\"
{generated_model_code}


# =============================================================================
# SCE-UA全局优化
# =============================================================================

def extract_params_from_code(code_str):
    import re
    params = set()
    for m in re.finditer(r'params\.get\\s*\\(\\s*["\\']([\\w]+)["\\']', code_str):
        params.add(m.group(1))
    return sorted(params)


DEFAULT_BOUNDS = {{
    "k": (0.05, 0.8), "k1": (0.1, 0.8), "k2": (0.01, 0.3), "k3": (0.001, 0.1),
    "S0": (0, 200), "S1": (0, 100), "S2": (0, 150), "S3": (0, 200), "CN": (30, 95)
}}


def get_bounds(param_names):
    return [DEFAULT_BOUNDS.get(name, (0.01, 100.0)) for name in param_names]


def compute_nse(obs, sim):
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs, sim = obs[mask], sim[mask]
    if len(obs) == 0:
        return 0.0
    eps = 1e-10
    num = np.sum((sim - obs) ** 2)
    den = np.sum((obs - np.mean(obs)) ** 2)
    return 1.0 - num / den if den > eps else 0.0


def split_calibration_validation(data, calib_years=25):
    """按时间划分率定期和验证期"""
    data = data.sort_values("date").reset_index(drop=True)
    years = data["date"].dt.year
    unique_years = sorted(years.unique())
    n_years = len(unique_years)
    if n_years < calib_years:
        n = len(data)
        split_idx = int(n * 0.75)
        return data.iloc[:split_idx], data.iloc[split_idx:]
    calib_year_list = unique_years[:calib_years]
    calib_data = data[years.isin(calib_year_list)].copy()
    valid_data = data[~years.isin(calib_year_list)].copy()
    return calib_data, valid_data


class SCEUA:
    def __init__(self, bounds, objective_func, maxn=3000, p=2, seed=21):
        self.n = len(bounds)
        self.bounds = np.array(bounds, dtype=float)
        self.objective = objective_func
        self.maxn = maxn
        self.p = p
        self.m = max(2 * self.n + 1, 4)
        self.rng = np.random.default_rng(seed)
        self.s = self.p * self.m
        self.neval = 0
        self.history = []

    def _rand(self):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return lo + self.rng.random(self.n) * (hi - lo)

    def _eval(self, x):
        self.neval += 1
        score = self.objective(np.array(x))
        return float(score) if np.isfinite(score) else -1e10

    def _sort(self, pop):
        pop.sort(key=lambda pt: -pt["score"])

    def _partition(self, pop):
        complexes = [[] for _ in range(self.p)]
        for i, pt in enumerate(pop):
            complexes[i % self.p].append(pt)
        return complexes

    def _prob(self, n):
        p_i = np.array([2.0 * (n + 1 - i) / (n * (n + 1)) for i in range(1, n + 1)])
        return p_i / p_i.sum()

    def _cce(self, points):
        if len(points) < 3:
            return points
        q, beta = min(self.n + 1, len(points)), max(2 * self.n + 1, len(points))
        
        for _ in range(beta):
            probs = self._prob(len(points))
            sel = np.sort(self.rng.choice(len(points), size=min(q, len(points)), replace=False, p=probs))
            sel_s = sorted(sel, key=lambda i: -points[i]["score"])
            idx_w, x_w, f_w = sel_s[-1], points[sel_s[-1]]["x"].copy(), points[sel_s[-1]]["score"]
            cent = np.mean([points[i]["x"] for i in sel_s[:-1]], axis=0)
            
            x_ref = 2 * cent - x_w
            if np.all((x_ref >= self.bounds[:, 0]) & (x_ref <= self.bounds[:, 1])):
                f_ref = self._eval(x_ref)
                if f_ref > f_w:
                    points[idx_w] = {{"x": x_ref, "score": f_ref}}
                    self._sort(points)
                    continue
            
            x_con = (cent + x_w) / 2
            if np.all((x_con >= self.bounds[:, 0]) & (x_con <= self.bounds[:, 1])):
                f_con = self._eval(x_con)
                if f_con > f_w:
                    points[idx_w] = {{"x": x_con, "score": f_con}}
                    self._sort(points)
                    continue
            
            points[idx_w] = {{"x": self._rand(), "score": self._eval(self._rand())}}
            self._sort(points)
        return points

    def calibrate(self):
        pop = [{{"x": self._rand(), "score": self._eval(self._rand())}} for _ in range(self.s)]
        self._sort(pop)
        best_sofar, best_x = pop[0]["score"], pop[0]["x"].copy()
        
        while self.neval < self.maxn:
            complexes = self._partition(pop)
            evolved = [self._cce(c) for c in complexes]
            pop = []
            for c in evolved:
                pop.extend(c)
            self._sort(pop)
            
            if pop[0]["score"] > best_sofar:
                best_sofar, best_x = pop[0]["score"], pop[0]["x"].copy()
            self.history.append((self.neval, best_sofar))
        
        return best_x, best_sofar, self.history


# =============================================================================
# XGBoost误差校正
# =============================================================================

class ResidualCorrector:
    def __init__(self):
        self.model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    def _features(self, q_sim, precip):
        p = precip.copy()
        p_lag1 = np.roll(p, 1); p_lag1[0] = p_lag1[1]
        p_lag2 = np.roll(p, 2); p_lag2[0] = p_lag2[2]
        return np.column_stack([q_sim, p, p_lag1, p_lag2])

    def train(self, q_obs, q_sim, precip):
        self.model.fit(self._features(q_sim, precip), q_obs - q_sim)

    def predict(self, q_sim, precip):
        return q_sim + self.model.predict(self._features(q_sim, precip))


# =============================================================================
# 主流程
# =============================================================================

def run(input_file, maxn=3000, calib_years=25, output_file=None):
    print(f"\\n{{'='*50}}")
    print(f"输入文件: {{input_file}} | 率定期: {{calib_years}}年 | maxn={{maxn}}")
    print('='*50)
    
    print("[1/6] 加载数据...")
    data = load_user_data(input_file)
    print(f"  原始数据: {{len(data)}}天")
    
    print("[2/6] 数据划分 (率定期/验证期)...")
    calib_data, valid_data = split_calibration_validation(data, calib_years)
    print(f"  率定期: {{len(calib_data)}}天 ({{calib_data['date'].min().date()}} ~ {{calib_data['date'].max().date()}})")
    print(f"  验证期: {{len(valid_data)}}天 ({{valid_data['date'].min().date()}} ~ {{valid_data['date'].max().date()}})")
    
    precip_calib = calib_data["precip"].values
    pet_calib = calib_data["pet"].values
    q_obs_calib = calib_data["q_obs"].values
    
    print("[3/6] SCE-UA参数率定 (率定期)...")
    code = simulate_runoff.__doc__ or ""
    param_names = extract_params_from_code(code) if code else ["k", "S0"]
    if not param_names:
        param_names = ["k", "S0"]
    bounds = get_bounds(param_names)
    sceua = SCEUA(bounds, lambda x: compute_nse(q_obs_calib, simulate_runoff(precip_calib, pet_calib, dict(zip(param_names, x)))), maxn)
    best_x, best_nse, history = sceua.calibrate()
    optimal_params = dict(zip(param_names, best_x))
    print(f"  评估次数: {{sceua.neval}}, 率定期NSE: {{best_nse:.4f}}")
    print(f"  最优参数: " + ", ".join([f"{{k}}={{v:.4f}}" for k, v in optimal_params.items()]))
    
    q_sim_calib = simulate_runoff(precip_calib, pet_calib, optimal_params)
    nse_calib = compute_nse(q_obs_calib, q_sim_calib)
    print(f"  率定期模拟NSE: {{nse_calib:.4f}}")
    
    print("[4/6] XGBoost误差校正 (率定期)...")
    corrector = ResidualCorrector()
    corrector.train(q_obs_calib, q_sim_calib, precip_calib)
    q_corrected_calib = corrector.predict(q_sim_calib, precip_calib)
    nse_calib_corr = compute_nse(q_obs_calib, q_corrected_calib)
    print(f"  校正后NSE: {{nse_calib_corr:.4f}}, 提升: {{nse_calib_corr - nse_calib:+.4f}}")
    
    print("[5/6] 验证期评估...")
    precip_valid = valid_data["precip"].values
    pet_valid = valid_data["pet"].values
    q_obs_valid = valid_data["q_obs"].values
    
    q_sim_valid = simulate_runoff(precip_valid, pet_valid, optimal_params)
    nse_valid = compute_nse(q_obs_valid, q_sim_valid)
    print(f"  验证期NSE: {{nse_valid:.4f}}")
    
    q_corrected_valid = corrector.predict(q_sim_valid, precip_valid)
    nse_valid_corr = compute_nse(q_obs_valid, q_corrected_valid)
    print(f"  验证期校正NSE: {{nse_valid_corr:.4f}}, 提升: {{nse_valid_corr - nse_valid:+.4f}}")
    
    print("[6/6] 保存结果...")
    result = data.copy()
    
    calib_data = calib_data.copy()
    calib_data["q_sim"] = q_sim_calib
    calib_data["q_corrected"] = q_corrected_calib
    calib_data["period"] = "calibration"
    
    valid_data = valid_data.copy()
    valid_data["q_sim"] = q_sim_valid
    valid_data["q_corrected"] = q_corrected_valid
    valid_data["period"] = "validation"
    
    result = pd.concat([calib_data, valid_data], ignore_index=True)
    result["error_raw"] = result["q_obs"] - result["q_sim"]
    result["error_corrected"] = result["q_obs"] - result["q_corrected"]
    
    output_file = output_file or "simulation_result.csv"
    result.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"  已保存: {{output_file}}")
    
    return {{
        "data": result, 
        "optimal_params": optimal_params, 
        "nse_calib": nse_calib,
        "nse_calib_corr": nse_calib_corr,
        "nse_valid": nse_valid,
        "nse_valid_corr": nse_valid_corr
    }}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="流域水文模型率定工具")
    parser.add_argument("--input", type=str, required=True, help="输入CSV文件")
    parser.add_argument("--calib_years", type=int, default=25, help="率定期年数 (默认25年)")
    parser.add_argument("--maxn", type=int, default=3000, help="SCE-UA最大评估次数")
    parser.add_argument("--output", type=str, default=None, help="输出CSV文件")
    args = parser.parse_args()
    
    result = run(args.input, args.maxn, args.calib_years, args.output)
    print(f"\\n{{'='*50}}")
    print("率定完成!")
    print(f"  率定期NSE: {{result['nse_calib']:.4f}} (原始), {{result['nse_calib_corr']:.4f}} (校正)")
    print(f"  验证期NSE: {{result['nse_valid']:.4f}} (原始), {{result['nse_valid_corr']:.4f}} (校正)")
    print('='*50)
'''
    
    readme_content = f'''# Hydromind-Agent生成模型使用指南

## 简介
本工具包含 Agent 开发的水文模型代码，配合 SCE-UA 全局优化进行参数率定，并使用 XGBoost 进行误差校正。

## 生成信息
- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 目标流域: {basin_id}
- 率定参数: {str(list(optimal_params.keys()))}
- 最优参数值: {optimal_params}

## 快速开始

### 1. 解压文件
将压缩包解压到任意目录

### 2. 运行模型
首次运行会自动安装依赖（需要联网）
```bash
python HydroMind-Agent-*.py --input sample_data.csv
```

### 3. 使用自己的数据
```bash
python HydroMind-Agent-*.py --input your_data.csv
```

---

## 详细说明

### 环境配置（可选）
如需手动配置环境：
```bash
# 创建虚拟环境
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 准备输入数据
创建 CSV 文件，格式如下：
```csv
date,precip,pet,q_obs
2020-01-01,5.2,2.1,1.3
2020-01-02,0.0,2.3,1.1
2020-01-03,10.5,1.8,2.8
...
```
列说明：
- date: 日期 (可选)
- precip: 日降雨量 (mm/day)
- pet: 日潜在蒸散发 (mm/day，可选，默认0)
- q_obs: 日观测径流 (mm/day)

### 3. 运行率定
```bash
python run_calibration.py --input your_data.csv
```

### 4. 可选参数
- `--input`: 输入CSV文件 (必需)
- `--calib_years`: 率定期年数 (默认: 25年，剩余为验证期)
- `--maxn`: SCE-UA最大评估次数 (默认: 3000)
- `--output`: 输出CSV文件名 (默认: simulation_result.csv)

### 5. 输出结果
- 仿真结果 CSV 文件 - 包含日期、降雨、PET、观测径流、模拟径流、校正后径流、误差、时期(calibration/validation)等列

## 流程说明
1. **数据加载** - 读取用户上传的CSV数据
2. **数据划分** - 按时间划分率定期(默认25年)和验证期(剩余年份)
3. **SCE-UA率定** - 使用率定期数据优化模型参数
4. **验证期评估** - 用率定参数在验证期进行模拟评估
5. **XGBoost校正** - 学习模型残差，提升模拟精度
6. **结果输出** - 保存率定期和验证期的完整结果

## Agent开发的水文模型代码
```python
{generated_model_code}
```
'''
    
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    py_filename = f"HydroMind-Agent-{timestamp}.py"
    zip_filename = f"HydroMind-Agent-{timestamp}.zip"
    
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(py_filename, full_script)
        zf.writestr("README.md", readme_content.replace("HydroMind-Agent-*.py", py_filename))
        zf.writestr("requirements.txt", "numpy>=1.24.0\npandas>=2.0.0\nxgboost>=2.0.0\n")
        
        sample_data = data.copy()
        sample_csv = sample_data.to_csv(index=False, encoding="utf-8-sig")
        zf.writestr("sample_data.csv", sample_csv)
    
    buffer.seek(0)
    st.download_button(
        label="📦 导出模型",
        data=buffer.getvalue(),
        file_name=zip_filename,
        mime="application/zip",
    )
    
    st.success("✅ 模型已生成！点击上方按钮导出完整包")
else:
    st.info("👆 请先点击「开始智能建模」运行完整流程，然后导出模型")

st.caption(
    "Hydromind v1.0 · Agent开发模型 · SCE-UA全局优化率定 · XGBoost误差校正"
)
