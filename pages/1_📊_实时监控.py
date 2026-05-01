"""实时监控页：从数据库取通道数据 → 模拟流式推送 → 在线检测 → Plotly 实时绘图。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from algorithms.registry import ALGORITHM_LABELS, build, list_algorithms
from core.database import (
    count_signals,
    create_alert,
    init_db,
    list_channels,
    list_devices,
    query_signals,
)
from core.health import health_score, health_status
from streaming.stream_engine import StreamEngine

st.set_page_config(page_title="实时监控", page_icon="📊", layout="wide")
init_db()

st.title("📊 实时监控")

# ---- 选择通道 ----
devices = list_devices()
if not devices:
    st.warning("暂无设备，请先到「数据管理」页生成示例数据。")
    st.stop()

col_a, col_b, col_c = st.columns([1, 1, 1])
with col_a:
    device_options = {f"{d.id} · {d.name}": d.id for d in devices}
    sel_dev_label = st.selectbox("设备", list(device_options.keys()))
    sel_dev_id = device_options[sel_dev_label]

channels = list_channels(sel_dev_id)
if not channels:
    st.warning("当前设备暂无通道。")
    st.stop()

with col_b:
    channel_options = {f"{c.id} · {c.name} ({c.unit})": c.id for c in channels}
    sel_ch_label = st.selectbox("通道", list(channel_options.keys()))
    sel_ch_id = channel_options[sel_ch_label]

with col_c:
    algo = st.selectbox(
        "检测算法",
        list_algorithms(),
        format_func=lambda x: ALGORITHM_LABELS.get(x, x),
    )

# 算法参数
with st.expander("⚙️ 算法参数"):
    if algo in ("three_sigma", "mad"):
        k = st.slider("阈值 k", 1.0, 5.0, 3.0, 0.1)
        det_kwargs = {"k": k}
    elif algo == "iqr":
        k = st.slider("阈值 k", 0.5, 3.0, 1.5, 0.1)
        det_kwargs = {"k": k}
    elif algo == "iforest":
        contamination = st.slider("污染比例", 0.01, 0.2, 0.05, 0.01)
        det_kwargs = {"contamination": contamination}
    elif algo == "lof":
        n_neighbors = st.slider("近邻数", 5, 50, 20, 1)
        contamination = st.slider("污染比例", 0.01, 0.2, 0.05, 0.01)
        det_kwargs = {"n_neighbors": n_neighbors, "contamination": contamination}
    else:
        det_kwargs = {}

# ---- 加载数据 ----
df = query_signals(sel_ch_id)
if df.empty:
    st.warning("该通道暂无数据，请到「批量检测」或「数据管理」页导入。")
    st.stop()

st.info(f"该通道共 {len(df)} 个采样点。")

# ---- 模式切换 ----
mode = st.radio("监控模式", ["📷 静态视图", "🌊 流式仿真"], horizontal=True)

if mode == "📷 静态视图":
    # 全量训练 + 预测
    detector = build(algo, **det_kwargs)
    detector.fit(df["value"].values)
    pred = detector.predict(df["value"].values)
    score = detector.score(df["value"].values)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df["value"], mode="lines", name="信号", line=dict(color="#3b82f6")
        )
    )
    anom_mask = pred == 1
    if anom_mask.any():
        fig.add_trace(
            go.Scatter(
                x=df.loc[anom_mask, "timestamp"],
                y=df.loc[anom_mask, "value"],
                mode="markers",
                name="异常点",
                marker=dict(color="#dc2626", size=8, symbol="x"),
            )
        )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="时间 (s)", yaxis_title="值")
    st.plotly_chart(fig, use_container_width=True)

    n_total = int(len(pred))
    n_anom = int(anom_mask.sum())
    mean_sev = float(np.mean(score[anom_mask])) if n_anom else 0.0
    hs = health_score(n_total, n_anom, mean_sev)
    label, color = health_status(hs)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("样本数", n_total)
    c2.metric("异常数", n_anom)
    c3.metric("异常率", f"{(n_anom/max(1,n_total))*100:.2f}%")
    c4.markdown(
        f"<div style='font-size:14px;color:#888'>健康分</div>"
        f"<div style='font-size:28px;font-weight:600;color:{color}'>{hs:.1f} <span style='font-size:14px'>({label})</span></div>",
        unsafe_allow_html=True,
    )

    # 一键创建告警
    if n_anom > 0:
        if st.button("🚨 生成告警事件"):
            create_alert(
                channel_id=sel_ch_id,
                start_ts=float(df["timestamp"].iloc[0]),
                end_ts=float(df["timestamp"].iloc[-1]),
                severity="critical" if hs < 60 else "warning",
                algorithm=algo,
                note=f"检测出 {n_anom} 个异常点 (率 {(n_anom/n_total)*100:.2f}%)",
            )
            st.success("告警已写入数据库，可到「告警中心」查看。")

else:
    # 流式仿真
    if "stream_engine" not in st.session_state:
        st.session_state.stream_engine = None
        st.session_state.stream_ch = None

    speed = st.select_slider("仿真速率 (倍速)", options=[0.5, 1, 2, 5, 10, 20, 50], value=10)
    base_sr = float(channels[0].sample_rate) if channels else 1000.0
    push_rate = base_sr * speed / 100  # 不要太快，避免 UI 卡顿

    bcol1, bcol2, bcol3 = st.columns([1, 1, 4])
    with bcol1:
        start_btn = st.button("▶️ 开始")
    with bcol2:
        stop_btn = st.button("⏸️ 停止 / 重置")

    if start_btn:
        # 重置已有引擎
        if st.session_state.stream_engine is not None:
            st.session_state.stream_engine.stop()
        engine = StreamEngine(df["value"].values, sample_rate=push_rate, buffer_size=2000)
        engine.start()
        st.session_state.stream_engine = engine
        st.session_state.stream_ch = sel_ch_id

    if stop_btn and st.session_state.stream_engine is not None:
        st.session_state.stream_engine.stop()
        st.session_state.stream_engine = None
        st.rerun()

    chart_slot = st.empty()
    metrics_slot = st.empty()
    progress_slot = st.empty()

    engine: StreamEngine | None = st.session_state.stream_engine
    if engine is not None and engine.is_running():
        # 用全量数据训练检测器（先验阈值），然后对滑动窗口预测
        detector = build(algo, **det_kwargs)
        detector.fit(df["value"].values)

        # 实时刷新循环（最多 200 帧，避免无限）
        for _ in range(600):
            ts, vs = engine.snapshot()
            if len(vs) < 5:
                time.sleep(0.1)
                continue
            pred = detector.predict(vs)
            score_arr = detector.score(vs)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts, y=vs, mode="lines", name="实时信号", line=dict(color="#3b82f6")))
            mask = pred == 1
            if mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=ts[mask],
                        y=vs[mask],
                        mode="markers",
                        name="异常",
                        marker=dict(color="#dc2626", size=10, symbol="x"),
                    )
                )
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10), uirevision="live")
            chart_slot.plotly_chart(fig, use_container_width=True)

            n_anom = int(mask.sum())
            n_total = int(len(vs))
            hs = health_score(n_total, n_anom, float(np.mean(score_arr[mask])) if n_anom else 0.0)
            label, color = health_status(hs)
            with metrics_slot.container():
                m1, m2, m3 = st.columns(3)
                m1.metric("窗口样本", n_total)
                m2.metric("异常数", n_anom)
                m3.markdown(
                    f"<div style='font-size:14px;color:#888'>健康分</div>"
                    f"<div style='font-size:28px;font-weight:600;color:{color}'>{hs:.1f} ({label})</div>",
                    unsafe_allow_html=True,
                )
            progress_slot.progress(engine.progress, text=f"播放进度 {engine.progress*100:.1f}%")

            if not engine.is_running() or engine.progress >= 1.0:
                progress_slot.success("✅ 流式仿真完成")
                break
            time.sleep(0.3)
    else:
        st.info("点击「开始」启动流式仿真。")
