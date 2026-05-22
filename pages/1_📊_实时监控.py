"""实时监控页：选择设备通道，自动循环流式仿真，多算法投票检测。"""
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

from algorithms.features import power_spectral_density
from algorithms.registry import ALGORITHM_LABELS, build, list_algorithms
from core.database import (
    create_alert,
    create_realtime_anomalies,
    init_db,
    list_channels,
    list_devices,
    list_realtime_anomalies,
    query_signals,
)
from core.health import health_score, health_status
from streaming.stream_engine import StreamEngine

st.set_page_config(page_title="实时监控", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1500px;
        padding-top: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
init_db()


def _default_algorithms(available: list[str]) -> list[str]:
    preferred = ["three_sigma", "mad", "iforest"]
    selected = [name for name in preferred if name in available]
    return selected or available[: min(2, len(available))]


def _stop_current_engine() -> None:
    engine = st.session_state.get("stream_engine")
    if engine is not None:
        engine.stop()


def _fit_detectors(channel_id: int, values: np.ndarray, algorithms: list[str]) -> dict:
    key = (channel_id, tuple(algorithms), len(values))
    if st.session_state.get("stream_detector_key") == key:
        return st.session_state.stream_detectors

    detectors = {}
    errors: list[str] = []
    for name in algorithms:
        if name == "lof":
            continue
        try:
            detector = build(name)
            detector.fit(values)
            detectors[name] = detector
        except Exception as exc:  # pragma: no cover - surfaced in UI
            errors.append(f"{ALGORITHM_LABELS.get(name, name)}：{exc}")

    st.session_state.stream_detector_key = key
    st.session_state.stream_detectors = detectors
    st.session_state.stream_detector_errors = errors
    return detectors


def _predict_votes(
    values: np.ndarray,
    detectors: dict,
    algorithms: list[str],
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    masks: dict[str, np.ndarray] = {}
    scores = np.zeros(len(values), dtype=float)
    errors: list[str] = []

    for name in algorithms:
        try:
            detector = build(name) if name == "lof" else detectors.get(name)
            if detector is None:
                continue
            if name == "lof":
                detector.fit(values)
            pred = np.asarray(detector.predict(values), dtype=int)
            score = np.asarray(detector.score(values), dtype=float)
            if len(pred) != len(values) or len(score) != len(values):
                errors.append(f"{ALGORITHM_LABELS.get(name, name)}：输出长度不一致")
                continue
            mask = pred == 1
            masks[name] = mask
            scores = np.maximum(scores, np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0))
        except Exception as exc:  # pragma: no cover - surfaced in UI
            errors.append(f"{ALGORITHM_LABELS.get(name, name)}：{exc}")

    return masks, scores, errors


def _algorithm_label_text(names: list[str]) -> str:
    return ", ".join(ALGORITHM_LABELS.get(name, name) for name in names)


st.title("📊 实时监控")

devices = list_devices()
if not devices:
    st.warning("暂无设备，请先到「数据管理」页生成或导入示例数据。")
    st.stop()

top_a, top_b, top_c = st.columns([1.2, 1.2, 1])
with top_a:
    device_options = {f"{d.id} - {d.name}": d for d in devices}
    device_label = st.selectbox("设备", list(device_options.keys()), index=0)
    device = device_options[device_label]

channels = list_channels(device.id)
if not channels:
    st.warning("当前设备暂无通道，请先导入通道数据。")
    st.stop()

with top_b:
    channel_options = {f"{c.id} - {c.name} ({c.unit})": c for c in channels}
    channel_label = st.selectbox("通道", list(channel_options.keys()), index=0)
    channel = channel_options[channel_label]

with top_c:
    speed = st.slider("播放速度", 1, 50, 10, 1, format="%d%%")

df = query_signals(channel.id)
if df.empty:
    st.warning("当前通道暂无信号数据，请先到「数据管理」页生成或导入数据。")
    st.stop()

algorithm_names = list_algorithms()
base_sr = float(channel.sample_rate or 1000.0)
push_rate = max(1.0, base_sr * speed / 100)

with st.expander("检测与告警设置", expanded=True):
    cfg_a, cfg_b, cfg_c, cfg_d = st.columns([1.8, 1, 1, 1])
    with cfg_a:
        selected_algorithms = st.multiselect(
            "参与投票的算法",
            algorithm_names,
            default=_default_algorithms(algorithm_names),
            format_func=lambda x: ALGORITHM_LABELS.get(x, x),
        )
    if not selected_algorithms:
        st.warning("请至少选择一个算法。")
        st.stop()
    default_votes = min(2, len(selected_algorithms))
    with cfg_b:
        vote_threshold = st.slider(
            "异常投票阈值",
            1,
            len(selected_algorithms),
            default_votes,
            help="同一个采样点至少被多少个算法判为异常，才在实时图中标记为异常。",
        )
    with cfg_c:
        alert_min_points = st.slider("告警最少异常点", 1, 100, 5, 1)
    with cfg_d:
        alert_min_rate = st.slider("告警最小异常率", 0.0, 10.0, 0.5, 0.1, format="%.1f%%")

st.caption(
    f"当前运行：设备 {device.id} - {device.name} / 通道 {channel.id} - {channel.name} / "
    f"采样 {len(df)} 点 / 投票规则 {vote_threshold}/{len(selected_algorithms)}"
)

if "stream_engine" not in st.session_state:
    st.session_state.stream_engine = None
    st.session_state.stream_ch = None
    st.session_state.stream_rate = None
if "stream_alerted_channels" not in st.session_state:
    st.session_state.stream_alerted_channels = set()
if "stream_paused" not in st.session_state:
    st.session_state.stream_paused = False

engine_needs_restart = (
    st.session_state.stream_engine is None
    or st.session_state.stream_ch != channel.id
    or st.session_state.stream_rate != push_rate
    or not st.session_state.stream_engine.is_running()
)
if engine_needs_restart and not st.session_state.stream_paused:
    _stop_current_engine()
    engine = StreamEngine(df["value"].values, sample_rate=push_rate, buffer_size=2500, loop=True)
    engine.start()
    st.session_state.stream_engine = engine
    st.session_state.stream_ch = channel.id
    st.session_state.stream_rate = push_rate

ctl_a, ctl_b, ctl_c = st.columns([1, 1, 6])
with ctl_a:
    if st.button("暂停" if not st.session_state.stream_paused else "继续"):
        st.session_state.stream_paused = not st.session_state.stream_paused
        if st.session_state.stream_paused:
            _stop_current_engine()
        else:
            st.session_state.stream_engine = None
        st.rerun()
with ctl_b:
    if st.button("重启流式"):
        _stop_current_engine()
        st.session_state.stream_engine = None
        st.session_state.stream_alerted_channels.discard(channel.id)
        st.rerun()

engine: StreamEngine | None = st.session_state.stream_engine
if engine is None:
    st.info("流式监控已暂停。")
    st.stop()

detectors = _fit_detectors(channel.id, df["value"].values, selected_algorithms)
fit_errors = st.session_state.get("stream_detector_errors", [])
for err in fit_errors:
    st.warning(f"算法初始化失败，已跳过：{err}")

ts, vs = engine.snapshot()
if len(vs) < 20:
    st.info("正在积累流式窗口数据...")
    time.sleep(0.3)
    st.rerun()

algorithm_masks, combined_score, predict_errors = _predict_votes(vs, detectors, selected_algorithms)
for err in predict_errors:
    st.warning(f"实时检测失败，已跳过：{err}")

vote_count = np.zeros(len(vs), dtype=int)
for mask in algorithm_masks.values():
    vote_count += mask.astype(int)
combined_mask = vote_count >= vote_threshold
hit_algorithms = [
    name for name, mask in algorithm_masks.items() if np.any(mask & combined_mask)
]

n_anom = int(combined_mask.sum())
n_total = int(len(vs))
anom_rate = n_anom / max(1, n_total) * 100
mean_sev = float(np.mean(combined_score[combined_mask])) if n_anom else 0.0
hs = health_score(n_total, n_anom, mean_sev)
label, color = health_status(hs)

metrics_slot = st.container()
with metrics_slot:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("窗口样本", n_total)
    m2.metric("异常数", n_anom, f"{anom_rate:.2f}%")
    m3.metric("触发算法", len(hit_algorithms))
    m4.metric("投票阈值", f"{vote_threshold}/{len(selected_algorithms)}")
    m5.markdown(
        f"<div style='font-size:14px;color:#64748b'>健康分</div>"
        f"<div style='font-size:30px;font-weight:700;color:{color}'>{hs:.1f} ({label})</div>",
        unsafe_allow_html=True,
    )

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=ts, y=vs, mode="lines", name="实时信号", line=dict(color="#2563eb", width=1.8))
)
if combined_mask.any():
    fig.add_trace(
        go.Scatter(
            x=ts[combined_mask],
            y=vs[combined_mask],
            mode="markers",
            name="投票异常",
            marker=dict(color="#dc2626", size=10, symbol="x"),
        )
    )
fig.update_layout(
    height=520,
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", y=1.08, x=0),
    uirevision="live",
)
st.plotly_chart(fig, use_container_width=True)

psd_col, vote_col = st.columns([1.4, 1])
with psd_col:
    freqs, psd = power_spectral_density(vs, sample_rate=push_rate)
    psd_fig = go.Figure()
    psd_fig.add_trace(
        go.Scatter(x=freqs, y=psd, mode="lines", name="PSD", line=dict(color="#16a34a"))
    )
    psd_fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power/Frequency",
        uirevision="live-psd",
    )
    st.plotly_chart(psd_fig, use_container_width=True)
with vote_col:
    vote_fig = go.Figure()
    vote_fig.add_trace(
        go.Scatter(
            x=ts,
            y=vote_count,
            mode="lines",
            name="异常投票数",
            line=dict(color="#f97316", shape="hv"),
        )
    )
    vote_fig.add_hline(y=vote_threshold, line_dash="dash", line_color="#dc2626")
    vote_fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(dtick=1, range=[0, max(len(selected_algorithms), vote_threshold) + 0.5]),
        uirevision="live-vote",
    )
    st.plotly_chart(vote_fig, use_container_width=True)

st.progress(engine.progress, text=f"循环流式位置 {engine.progress*100:.1f}%")

current_rows = pd.DataFrame(
    {
        "stream_ts": ts[combined_mask],
        "value": vs[combined_mask],
        "score": combined_score[combined_mask],
        "vote": vote_count[combined_mask],
    }
)
if not current_rows.empty:
    current_rows = current_rows.sort_values("stream_ts", ascending=False).head(100)
    current_rows["stream_ts"] = current_rows["stream_ts"].round(3)
    current_rows["value"] = current_rows["value"].round(6)
    current_rows["score"] = current_rows["score"].round(4)

with st.expander("当前窗口异常点明细（最近缓冲区）", expanded=not current_rows.empty):
    st.caption("这里只显示实时流式缓冲区里的最近异常点；需要查已保存的历史点，请到「告警中心 → 历史异常点查询」。")
    if current_rows.empty:
        st.caption("当前窗口没有达到投票阈值的异常点。")
    else:
        st.dataframe(current_rows, use_container_width=True, hide_index=True, height=240)

should_alert = n_anom >= alert_min_points and anom_rate >= alert_min_rate
if should_alert and channel.id not in st.session_state.stream_alerted_channels:
    alert = create_alert(
        channel_id=channel.id,
        start_ts=float(ts[combined_mask][0]),
        end_ts=float(ts[combined_mask][-1]),
        severity="critical" if hs < 60 else "warning",
        algorithm=",".join(hit_algorithms),
        note=(
            f"实时流式检测发现 {n_anom} 个投票异常点 ({anom_rate:.2f}%)，"
            f"规则 {vote_threshold}/{len(selected_algorithms)}，触发算法："
            f"{_algorithm_label_text(hit_algorithms)}"
        ),
    )
    saved_points = create_realtime_anomalies(
        alert_id=alert.id,
        channel_id=channel.id,
        timestamps=ts[combined_mask],
        values=vs[combined_mask],
        scores=combined_score[combined_mask],
        vote_counts=vote_count[combined_mask],
        vote_threshold=vote_threshold,
        algorithms=_algorithm_label_text(hit_algorithms),
    )
    st.session_state.stream_alerted_channels.add(channel.id)
    st.warning(
        f"已生成实时告警并保存 {saved_points} 个异常点明细。"
        "页面将继续停留在实时监控，可到「告警中心」查看告警。"
    )
elif should_alert:
    st.warning("当前窗口达到告警阈值，已记录过该通道的实时告警。")
else:
    st.success("当前窗口未达到告警阈值。")

with st.expander("实时历史回看", expanded=False):
    hist_limit = st.slider("历史明细条数", 50, 1000, 300, 50)
    hist = list_realtime_anomalies(channel_id=channel.id, limit=hist_limit)
    if hist.empty:
        st.caption("当前通道暂无已保存的实时异常明细。")
    else:
        display_hist = hist.copy()
        display_hist["stream_ts"] = display_hist["stream_ts"].round(3)
        display_hist["value"] = display_hist["value"].round(6)
        display_hist["score"] = display_hist["score"].round(4)
        display_hist["vote"] = (
            display_hist["vote_count"].astype(str)
            + "/"
            + display_hist["vote_threshold"].astype(str)
        )
        st.dataframe(
            display_hist[
                [
                    "created_at",
                    "alert_id",
                    "stream_ts",
                    "value",
                    "score",
                    "vote",
                    "algorithms",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            height=260,
        )

        hist_plot = hist.sort_values("stream_ts")
        history_fig = go.Figure()
        history_fig.add_trace(
            go.Scatter(
                x=hist_plot["stream_ts"],
                y=hist_plot["value"],
                mode="markers",
                name="历史异常点",
                marker=dict(
                    color=hist_plot["vote_count"],
                    colorscale="Reds",
                    size=9,
                    showscale=True,
                    colorbar=dict(title="投票数"),
                ),
            )
        )
        history_fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="流式时间 (s)",
            yaxis_title=f"值 ({channel.unit})" if channel.unit else "值",
        )
        st.plotly_chart(history_fig, use_container_width=True)

if engine.is_running() and not st.session_state.stream_paused:
    time.sleep(0.35)
    st.rerun()
