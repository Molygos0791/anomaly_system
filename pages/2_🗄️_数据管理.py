"""数据管理页：设备 / 通道 / 信号 的 CRUD。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from core.config import SAMPLE_TDMS
from core.database import (
    count_signals,
    create_channel,
    create_device,
    delete_channel,
    delete_device,
    delete_signal,
    delete_signals_by_channel,
    import_channel_from_array,
    init_db,
    list_channels,
    list_devices,
    query_signals,
    update_channel,
    update_device,
)
from ingestion.simulator import write_sample_tdms
from ingestion.tdms_loader import list_tdms_metadata, load_tdms

st.set_page_config(page_title="数据管理", page_icon="🗄️", layout="wide")
init_db()

st.title("🗄️ 数据管理 (CRUD)")

tab_dev, tab_ch, tab_sig, tab_import = st.tabs(["🏭 设备", "📡 通道", "📈 信号", "📥 数据导入"])

# ===== 设备 =====
with tab_dev:
    st.subheader("设备列表")
    devs = list_devices()
    if devs:
        df = pd.DataFrame(
            [{"id": d.id, "name": d.name, "location": d.location, "created_at": d.created_at} for d in devs]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("暂无设备")

    st.markdown("---")
    with st.expander("➕ 新增设备"):
        with st.form("add_dev"):
            n = st.text_input("设备名")
            l = st.text_input("位置")
            if st.form_submit_button("创建"):
                if n.strip():
                    create_device(n.strip(), l.strip())
                    st.success("已创建")
                    st.rerun()

    if devs:
        with st.expander("✏️ 编辑 / 删除设备"):
            sel = st.selectbox("选择设备", [f"{d.id} · {d.name}" for d in devs], key="edit_dev_sel")
            did = int(sel.split(" · ")[0])
            cur = next(d for d in devs if d.id == did)
            new_name = st.text_input("新名称", cur.name, key="edit_dev_name")
            new_loc = st.text_input("新位置", cur.location, key="edit_dev_loc")
            cu1, cu2 = st.columns(2)
            with cu1:
                if st.button("保存修改"):
                    update_device(did, name=new_name, location=new_loc)
                    st.success("已更新")
                    st.rerun()
            with cu2:
                if st.button("删除设备 (级联)", type="primary"):
                    delete_device(did)
                    st.warning("已删除")
                    st.rerun()

# ===== 通道 =====
with tab_ch:
    st.subheader("通道列表")
    chs = list_channels()
    if chs:
        df = pd.DataFrame(
            [
                {
                    "id": c.id,
                    "device_id": c.device_id,
                    "name": c.name,
                    "unit": c.unit,
                    "sample_rate": c.sample_rate,
                    "detector": c.detector_type,
                }
                for c in chs
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("暂无通道")

    st.markdown("---")
    devs = list_devices()
    if devs:
        with st.expander("➕ 新增通道"):
            with st.form("add_ch"):
                dev_sel = st.selectbox("所属设备", [f"{d.id} · {d.name}" for d in devs])
                did = int(dev_sel.split(" · ")[0])
                cn = st.text_input("通道名")
                cu = st.text_input("单位", "")
                cr = st.number_input("采样率 Hz", value=1000.0)
                if st.form_submit_button("创建"):
                    if cn.strip():
                        create_channel(did, cn.strip(), unit=cu, sample_rate=cr)
                        st.success("已创建")
                        st.rerun()

        if chs:
            with st.expander("✏️ 编辑 / 删除通道"):
                sel = st.selectbox("选择通道", [f"{c.id} · {c.name}" for c in chs], key="edit_ch_sel")
                cid = int(sel.split(" · ")[0])
                cur = next(c for c in chs if c.id == cid)
                new_name = st.text_input("名称", cur.name, key="edit_ch_name")
                new_unit = st.text_input("单位", cur.unit, key="edit_ch_unit")
                new_sr = st.number_input("采样率", value=float(cur.sample_rate), key="edit_ch_sr")
                cu1, cu2 = st.columns(2)
                with cu1:
                    if st.button("保存"):
                        update_channel(cid, name=new_name, unit=new_unit, sample_rate=new_sr)
                        st.success("已更新")
                        st.rerun()
                with cu2:
                    if st.button("删除通道", type="primary"):
                        delete_channel(cid)
                        st.warning("已删除")
                        st.rerun()

# ===== 信号 =====
with tab_sig:
    st.subheader("信号查询 / 删除")
    chs = list_channels()
    if not chs:
        st.info("暂无通道")
    else:
        sel = st.selectbox("通道", [f"{c.id} · {c.name}" for c in chs], key="sig_sel_ch")
        cid = int(sel.split(" · ")[0])
        total, anom = count_signals(cid)
        st.write(f"总数: **{total}** · 异常: **{anom}** · 异常率: **{(anom/max(1,total))*100:.2f}%**")

        col1, col2 = st.columns(2)
        with col1:
            limit = st.number_input("显示前 N 条", 10, 5000, 200, 10)
        with col2:
            if st.button("🗑️ 清空该通道全部信号", type="primary"):
                n = delete_signals_by_channel(cid)
                st.warning(f"已删除 {n} 条")
                st.rerun()

        df = query_signals(cid, limit=int(limit))
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True, height=400)

            st.markdown("**单条删除**")
            del_id = st.number_input("信号 ID", min_value=0, value=0, step=1)
            if st.button("删除该 ID"):
                if delete_signal(int(del_id)):
                    st.success("已删除")
                    st.rerun()
                else:
                    st.error("ID 不存在")

# ===== 数据导入 =====
with tab_import:
    st.subheader("数据导入")

    st.markdown("#### 方式 1：一键生成示例 TDMS")
    if st.button("🎲 生成 + 导入示例数据"):
        write_sample_tdms(SAMPLE_TDMS)
        chs_data = load_tdms(SAMPLE_TDMS)
        for c in chs_data:
            import_channel_from_array(
                device_name=c.device,
                channel_name=c.channel,
                timestamps=c.timestamps,
                values=c.values,
                unit=c.unit,
                sample_rate=c.sample_rate,
            )
        st.success(f"已导入 {len(chs_data)} 个通道，共 {sum(len(c.values) for c in chs_data)} 个采样点。")

    st.markdown("---")
    st.markdown("#### 方式 2：扫描 data/ 目录中已有的 TDMS")
    from core.config import DATA_DIR

    local_tdms = sorted([p for p in DATA_DIR.glob("*.tdms") if not p.name.startswith("_")])
    if local_tdms:
        local_sel = st.selectbox(
            "选择文件", [p.name for p in local_tdms], key="local_tdms_sel"
        )
        local_path = DATA_DIR / local_sel
        meta_local = list_tdms_metadata(local_path)
        st.write(f"📋 **{len(meta_local)}** 通道")
        st.dataframe(pd.DataFrame(meta_local), use_container_width=True, hide_index=True)

        total_local = sum(m["n_points"] for m in meta_local)
        if total_local > 100_000:
            st.warning(f"⚠️ 总点数 {total_local:,}，强烈建议降采样")
        cl1, cl2 = st.columns(2)
        with cl1:
            local_max = st.number_input(
                "每通道最多读取点数（0=全部）",
                min_value=0,
                value=200_000,
                step=10_000,
                key="local_max",
            )
        with cl2:
            local_ds = st.number_input(
                "降采样步长", min_value=1, value=10, step=1, key="local_ds"
            )
        est = (local_max or max(m["n_points"] for m in meta_local)) // local_ds
        st.caption(
            f"预计每通道入库 **{est:,}** 点，等效采样率 ≈ {meta_local[0]['sample_rate']/local_ds:.1f} Hz"
        )
        if st.button("📥 导入此本地文件"):
            with st.spinner("解析 + 写入..."):
                chs_data = load_tdms(
                    local_path,
                    max_points_per_channel=int(local_max) if local_max > 0 else None,
                    downsample=int(local_ds),
                )
                progress = st.progress(0.0)
                for i, c in enumerate(chs_data):
                    import_channel_from_array(
                        device_name=c.device or local_sel,
                        channel_name=c.channel,
                        timestamps=c.timestamps,
                        values=c.values,
                        unit=c.unit,
                        sample_rate=c.sample_rate,
                    )
                    progress.progress((i + 1) / len(chs_data))
            st.success(
                f"✅ 导入完成：{len(chs_data)} 通道，"
                f"共 {sum(len(c.values) for c in chs_data):,} 个点"
            )
    else:
        st.caption("data/ 目录无 .tdms 文件")

    st.markdown("---")
    st.markdown("#### 方式 3：上传 TDMS 文件")
    up = st.file_uploader("选择 .tdms 文件", type=["tdms"])
    if up is not None:
        tmp_path = Path(SAMPLE_TDMS).parent / f"_uploaded_{up.name}"
        tmp_path.write_bytes(up.read())

        # 先读元数据（不加载数据）
        meta = list_tdms_metadata(tmp_path)
        st.write(f"📋 解析到 **{len(meta)}** 个通道：")
        meta_df = pd.DataFrame(meta)
        st.dataframe(meta_df, use_container_width=True, hide_index=True)

        total_pts = sum(m["n_points"] for m in meta)
        if total_pts > 100_000:
            st.warning(
                f"⚠️ 总采样点 **{total_pts:,}**，全量入库会非常慢（SQLite 单点逐行）。"
                f"建议使用「降采样」或「截取」。"
            )

        col_a, col_b = st.columns(2)
        with col_a:
            max_pts = st.number_input(
                "每通道最多读取点数（0=全部）", min_value=0, value=200_000, step=10_000
            )
        with col_b:
            downsample = st.number_input(
                "降采样步长（每 N 点取 1 点）", min_value=1, value=10, step=1
            )

        est_per_ch = (max_pts or max(m["n_points"] for m in meta))
        est_per_ch = est_per_ch // downsample
        st.caption(
            f"预计每通道入库约 **{est_per_ch:,}** 个点，"
            f"等效采样率 ≈ {meta[0]['sample_rate']/downsample:.1f} Hz"
        )

        if st.button("✅ 写入数据库"):
            with st.spinner("正在解析 + 写入..."):
                chs_data = load_tdms(
                    tmp_path,
                    max_points_per_channel=int(max_pts) if max_pts > 0 else None,
                    downsample=int(downsample),
                )
                progress = st.progress(0.0)
                for i, c in enumerate(chs_data):
                    import_channel_from_array(
                        device_name=c.device,
                        channel_name=c.channel,
                        timestamps=c.timestamps,
                        values=c.values,
                        unit=c.unit,
                        sample_rate=c.sample_rate,
                    )
                    progress.progress((i + 1) / len(chs_data))
            st.success(
                f"✅ 导入完成：{len(chs_data)} 通道，"
                f"共 {sum(len(c.values) for c in chs_data):,} 个点"
            )
