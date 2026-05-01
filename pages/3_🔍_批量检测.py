"""批量检测页：选通道 + 选算法 → 检测全量 → 写入异常标记 + 创建告警 + 记录运行历史。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from algorithms.registry import ALGORITHM_LABELS, build, list_algorithms
from core.database import (
    create_alert,
    create_detection_run,
    init_db,
    list_channels,
    list_devices,
    list_detection_runs,
    query_signals,
    update_anomaly_flags,
)

st.set_page_config(page_title="批量检测", page_icon="🔍", layout="wide")
init_db()

st.title("🔍 批量检测")

devices = list_devices()
if not devices:
    st.warning("请先到「数据管理」页导入数据。")
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    dev_sel = st.selectbox("设备", [f"{d.id} · {d.name}" for d in devices])
    did = int(dev_sel.split(" · ")[0])
chs = list_channels(did)
if not chs:
    st.warning("该设备暂无通道。")
    st.stop()
with c2:
    ch_sel = st.selectbox("通道", [f"{c.id} · {c.name}" for c in chs])
    cid = int(ch_sel.split(" · ")[0])
with c3:
    algo = st.selectbox("算法", list_algorithms(), format_func=lambda x: ALGORITHM_LABELS.get(x, x))

with st.expander("⚙️ 参数"):
    if algo in ("three_sigma", "mad"):
        det_kwargs = {"k": st.slider("k", 1.0, 5.0, 3.0, 0.1)}
    elif algo == "iqr":
        det_kwargs = {"k": st.slider("k", 0.5, 3.0, 1.5, 0.1)}
    elif algo == "iforest":
        det_kwargs = {
            "contamination": st.slider("contamination", 0.01, 0.2, 0.05, 0.01),
            "use_features": st.checkbox("使用滑窗特征", value=False),
            "window": st.slider("窗口", 10, 200, 50, 10),
        }
    elif algo == "lof":
        det_kwargs = {
            "n_neighbors": st.slider("n_neighbors", 5, 50, 20, 1),
            "contamination": st.slider("contamination", 0.01, 0.2, 0.05, 0.01),
            "use_features": st.checkbox("使用滑窗特征", value=False),
            "window": st.slider("窗口", 10, 200, 50, 10),
        }
    else:
        det_kwargs = {}

if st.button("🚀 开始检测", type="primary"):
    df = query_signals(cid)
    if df.empty:
        st.error("该通道无数据")
        st.stop()

    with st.spinner("检测中..."):
        det = build(algo, **det_kwargs)
        det.fit(df["value"].values)
        pred = det.predict(df["value"].values)
        score = det.score(df["value"].values)

    n_total, n_anom = int(len(pred)), int(pred.sum())
    st.success(f"检测完成：{n_anom}/{n_total} 个点被标为异常 ({(n_anom/n_total)*100:.2f}%)")

    # 写回数据库
    update_anomaly_flags(cid, df["id"].tolist(), pred.tolist(), score.tolist())

    # 自动创建告警（如果异常比例 > 1%）
    if n_anom / n_total > 0.01:
        sev = "critical" if n_anom / n_total > 0.05 else "warning"
        create_alert(
            channel_id=cid,
            start_ts=float(df["timestamp"].iloc[0]),
            end_ts=float(df["timestamp"].iloc[-1]),
            severity=sev,
            algorithm=algo,
            note=f"批量检测：{n_anom} 异常点 ({(n_anom/n_total)*100:.2f}%)",
        )

    # 记录运行历史
    create_detection_run(
        channel_id=cid,
        algorithm=algo,
        params=det.get_params(),
        n_total=n_total,
        n_anomaly=n_anom,
    )

    # 绘图
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["value"], mode="lines", name="信号", line=dict(color="#3b82f6")))
    mask = pred == 1
    if mask.any():
        fig.add_trace(
            go.Scatter(
                x=df.loc[mask, "timestamp"],
                y=df.loc[mask, "value"],
                mode="markers",
                name="异常",
                marker=dict(color="#dc2626", size=8, symbol="x"),
            )
        )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # 异常分数热力曲线
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["timestamp"], y=score, mode="lines", name="异常分", line=dict(color="#f97316")))
    fig2.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), title="异常分曲线")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.subheader("🕓 该通道的检测运行历史")
runs = list_detection_runs(cid)
if runs:
    import pandas as pd

    rows = [
        {
            "id": r.id,
            "run_at": r.run_at,
            "algorithm": r.algorithm,
            "n_total": r.n_total,
            "n_anomaly": r.n_anomaly,
            "rate(%)": round(r.n_anomaly / max(1, r.n_total) * 100, 2),
            "params": str(r.params),
        }
        for r in runs
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.caption("暂无历史")
