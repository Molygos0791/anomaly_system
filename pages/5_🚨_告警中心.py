"""告警中心：告警处理 + 实时异常点历史查询。"""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.database import (
    delete_alert,
    init_db,
    list_alerts,
    list_channels,
    list_realtime_anomalies,
    update_alert_status,
)

st.set_page_config(page_title="告警中心", page_icon="🚨", layout="wide")
init_db()

st.title("🚨 告警中心")

alerts_tab, points_tab = st.tabs(["告警列表", "历史异常点查询"])

with alerts_tab:
    c1, c2 = st.columns(2)
    with c1:
        status_filter = st.selectbox("状态过滤", ["全部", "new", "ack", "closed"])
    with c2:
        chs = list_channels()
        ch_map = {c.id: f"{c.name}" for c in chs}
        ch_options = ["全部"] + [f"{c.id} · {c.name}" for c in chs]
        ch_filter = st.selectbox("通道过滤", ch_options)

    alerts = list_alerts(
        channel_id=int(ch_filter.split(" · ")[0]) if ch_filter != "全部" else None,
        status=None if status_filter == "全部" else status_filter,
    )

    if not alerts:
        st.info("暂无告警")
    else:
        rows = [
            {
                "id": a.id,
                "channel": ch_map.get(a.channel_id, a.channel_id),
                "severity": a.severity,
                "status": a.status,
                "algorithm": a.algorithm,
                "start_ts": round(a.start_ts, 3),
                "end_ts": round(a.end_ts, 3),
                "created_at": a.created_at,
                "note": a.note,
            }
            for a in alerts
        ]
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True, height=400)

        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button("⬇️ 导出告警 CSV", buf.getvalue(), file_name="alerts.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("操作")
        sel = st.selectbox("选择告警", [f"{a.id} · {a.severity} · {a.status}" for a in alerts])
        aid = int(sel.split(" · ")[0])
        cur = next(a for a in alerts if a.id == aid)
        new_status = st.selectbox(
            "新状态",
            ["new", "ack", "closed"],
            index=["new", "ack", "closed"].index(cur.status),
        )
        new_note = st.text_area("备注", cur.note)
        op1, op2 = st.columns(2)
        with op1:
            if st.button("💾 更新"):
                update_alert_status(aid, new_status, new_note)
                st.success("已更新")
                st.rerun()
        with op2:
            if st.button("🗑️ 删除", type="primary"):
                delete_alert(aid)
                st.warning("已删除")
                st.rerun()

with points_tab:
    chs = list_channels()
    ch_map = {c.id: f"{c.name}" for c in chs}
    q1, q2, q3 = st.columns([1.4, 1, 1])
    with q1:
        ch_options = ["全部"] + [f"{c.id} · {c.name}" for c in chs]
        point_ch_filter = st.selectbox("通道", ch_options, key="history_point_channel")
    with q2:
        alert_id_text = st.text_input("告警 ID", placeholder="留空查询全部")
    with q3:
        point_limit = st.slider("最多显示", 50, 3000, 500, 50)

    channel_id = (
        int(point_ch_filter.split(" · ")[0]) if point_ch_filter != "全部" else None
    )
    alert_id: int | None = None
    if alert_id_text.strip():
        try:
            alert_id = int(alert_id_text.strip())
        except ValueError:
            st.error("告警 ID 必须是数字。")
            st.stop()

    points = list_realtime_anomalies(
        channel_id=channel_id,
        alert_id=alert_id,
        limit=point_limit,
    )

    if points.empty:
        st.info("没有查到历史实时异常点。实时页触发告警后，异常点会保存到这里。")
    else:
        display = points.copy()
        display["channel"] = display["channel_id"].map(ch_map).fillna(display["channel_id"])
        display["stream_ts"] = display["stream_ts"].round(3)
        display["value"] = display["value"].round(6)
        display["score"] = display["score"].round(4)
        display["vote"] = (
            display["vote_count"].astype(str)
            + "/"
            + display["vote_threshold"].astype(str)
        )

        st.dataframe(
            display[
                [
                    "created_at",
                    "alert_id",
                    "channel",
                    "stream_ts",
                    "value",
                    "score",
                    "vote",
                    "algorithms",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            height=420,
        )

        csv_buf = io.StringIO()
        display.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ 导出异常点 CSV",
            csv_buf.getvalue(),
            file_name="realtime_anomaly_points.csv",
            mime="text/csv",
        )

        plot_df = points.sort_values("stream_ts")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=plot_df["stream_ts"],
                y=plot_df["value"],
                mode="markers",
                name="历史异常点",
                marker=dict(
                    color=plot_df["vote_count"],
                    colorscale="Reds",
                    size=9,
                    showscale=True,
                    colorbar=dict(title="投票数"),
                ),
                text=[
                    f"告警 {row.alert_id}<br>分数 {row.score:.4f}<br>{row.algorithms}"
                    for row in plot_df.itertuples()
                ],
                hovertemplate="时间 %{x:.3f}<br>值 %{y}<br>%{text}<extra></extra>",
            )
        )
        fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="流式时间 (s)",
            yaxis_title="异常值",
        )
        st.plotly_chart(fig, use_container_width=True)
