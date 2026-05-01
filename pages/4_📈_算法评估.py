"""算法评估页：在合成 ground-truth 数据上对所有算法做横向对比。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from algorithms.evaluation import compare
from algorithms.registry import ALGORITHM_LABELS, build, list_algorithms
from ingestion.simulator import generate_signal

st.set_page_config(page_title="算法评估", page_icon="📈", layout="wide")
st.title("📈 算法评估对比")
st.caption("使用注入了已知异常的合成信号（ground truth），对所有算法做横向 P/R/F1/ROC 对比")

# ---- 数据生成参数 ----
with st.sidebar:
    st.subheader("合成数据参数")
    n_points = st.slider("样本数", 1000, 20000, 5000, 1000)
    base_freq = st.slider("基频 Hz", 1.0, 50.0, 5.0)
    noise_std = st.slider("噪声标准差", 0.05, 0.5, 0.1, 0.05)
    anomaly_ratio = st.slider("异常比例", 0.005, 0.1, 0.02, 0.005)
    seed = st.number_input("随机种子", value=42)

t, x, y = generate_signal(
    n_points=int(n_points),
    base_freq=float(base_freq),
    noise_std=float(noise_std),
    anomaly_ratio=float(anomaly_ratio),
    seed=int(seed),
)

st.write(f"生成了 {len(x)} 点，含 {int(y.sum())} 个真实异常点")

# 显示信号
fig0 = go.Figure()
fig0.add_trace(go.Scatter(x=t, y=x, mode="lines", name="signal", line=dict(color="#3b82f6")))
mask = y == 1
fig0.add_trace(
    go.Scatter(x=t[mask], y=x[mask], mode="markers", name="ground truth", marker=dict(color="#dc2626", size=6))
)
fig0.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig0, use_container_width=True)

# ---- 选择算法 ----
selected = st.multiselect(
    "选择参评算法",
    list_algorithms(),
    default=list_algorithms(),
    format_func=lambda x: ALGORITHM_LABELS.get(x, x),
)

if st.button("🏁 开始评估", type="primary") and selected:
    detectors = [build(name) for name in selected]
    with st.spinner("评估中..."):
        results = compare(detectors, x, y)

    # 评估表
    rows = [
        {
            "算法": ALGORITHM_LABELS.get(r.algorithm, r.algorithm),
            "Precision": round(r.precision, 3),
            "Recall": round(r.recall, 3),
            "F1": round(r.f1, 3),
            "ROC-AUC": round(r.roc_auc, 3),
            "PR-AUC": round(r.pr_auc, 3),
            "预测异常数": r.n_pred_anomaly,
            "真实异常数": r.n_true_anomaly,
        }
        for r in results
    ]
    df = pd.DataFrame(rows).sort_values("F1", ascending=False)
    st.subheader("📊 评估指标")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # 柱状对比
    fig_bar = go.Figure()
    for metric in ["Precision", "Recall", "F1"]:
        fig_bar.add_trace(go.Bar(name=metric, x=df["算法"], y=df[metric]))
    fig_bar.update_layout(barmode="group", height=400, title="P / R / F1 对比")
    st.plotly_chart(fig_bar, use_container_width=True)

    # ROC 曲线对比
    fig_roc = go.Figure()
    for r in results:
        fig_roc.add_trace(
            go.Scatter(
                x=r.fpr,
                y=r.tpr,
                mode="lines",
                name=f"{ALGORITHM_LABELS.get(r.algorithm, r.algorithm)} (AUC={r.roc_auc:.3f})",
            )
        )
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="random"))
    fig_roc.update_layout(
        title="ROC 曲线", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=500
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    best = df.iloc[0]
    st.success(f"🏆 当前最佳算法: **{best['算法']}** (F1 = {best['F1']:.3f})")
