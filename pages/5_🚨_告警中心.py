"""告警中心：列表 / 状态确认 / 导出。"""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from core.database import (
    delete_alert,
    init_db,
    list_alerts,
    list_channels,
    update_alert_status,
)

st.set_page_config(page_title="告警中心", page_icon="🚨", layout="wide")
init_db()

st.title("🚨 告警中心")

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

    # 导出 CSV
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button("⬇️ 导出 CSV", buf.getvalue(), file_name="alerts.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("操作")
    sel = st.selectbox("选择告警", [f"{a.id} · {a.severity} · {a.status}" for a in alerts])
    aid = int(sel.split(" · ")[0])
    cur = next(a for a in alerts if a.id == aid)
    new_status = st.selectbox("新状态", ["new", "ack", "closed"], index=["new", "ack", "closed"].index(cur.status))
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
