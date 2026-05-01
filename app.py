"""Streamlit 主入口（首页）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from core.database import (
    init_db,
    list_alerts,
    list_channels,
    list_devices,
)
from core.health import health_score, health_status

st.set_page_config(page_title="工业时序异常检测系统", page_icon="🏭", layout="wide")

init_db()

st.title("🏭 工业时序数据异常检测系统")
st.caption("MVP for 设备健康监控 · TDMS 解析 + 多算法异常检测 + 实时监控")

st.markdown("---")

# ---- 概览 KPI ----
devices = list_devices()
channels = list_channels()
alerts_new = list_alerts(status="new")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("设备数", len(devices))
with col2:
    st.metric("通道数", len(channels))
with col3:
    st.metric("待处理告警", len(alerts_new))
with col4:
    # 全局健康分（所有通道异常率均值）
    if channels:
        from core.database import count_signals

        rates = []
        for ch in channels:
            t, a = count_signals(ch.id)
            if t > 0:
                rates.append(a / t)
        if rates:
            avg_rate_pct = float(sum(rates) / len(rates) * 100)
            score = health_score(100, int(avg_rate_pct), mean_severity=1.0)
        else:
            score = 100.0
    else:
        score = 100.0
    label, color = health_status(score)
    st.markdown(
        f"<div style='font-size:14px;color:#888'>系统健康分</div>"
        f"<div style='font-size:32px;font-weight:600;color:{color}'>{score:.1f} <span style='font-size:14px'>({label})</span></div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

st.markdown(
    """
### 📖 使用指南

请使用左侧导航栏进入各功能页面：

| 页面 | 功能 |
|------|------|
| 📊 **实时监控** | 加载 TDMS / 模拟流式数据 / 实时异常检测 / 健康评分 |
| 🗄️ **数据管理** | 设备 · 通道 · 信号的增删改查 (CRUD) |
| 🔍 **批量检测** | 上传或选择已有数据 → 选择算法 → 全量检测 + 写入数据库 |
| 📈 **算法评估** | 多算法横向对比：Precision / Recall / F1 / ROC 曲线 |
| 🚨 **告警中心** | 告警列表 · 状态确认 · 导出 CSV |

### 🧠 已集成的算法

- **统计法**：3-Sigma · IQR · MAD（鲁棒）
- **机器学习**：Isolation Forest · LOF
- 支持 **滑窗特征工程**（mean / std / rms / 峰峰值 / 峭度 / 偏度）

### 🚀 快速开始

1. 进入「数据管理」点「**生成示例 TDMS 数据**」一键导入
2. 进入「批量检测」选算法运行
3. 进入「实时监控」启动流式仿真，查看异常实时标记
"""
)

st.markdown("---")
st.caption("研究生 Python 课程作业 · MVP 版本")
