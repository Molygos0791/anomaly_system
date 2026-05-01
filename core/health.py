"""设备健康评分。基于异常率与严重度计算 0-100 健康分。"""
from __future__ import annotations

import numpy as np

from core.config import HEALTH_BASE, HEALTH_PENALTY_PER_ANOMALY_PCT


def health_score(n_total: int, n_anomaly: int, mean_severity: float = 1.0) -> float:
    """
    n_total: 样本总数
    n_anomaly: 异常数
    mean_severity: 平均异常严重度（0-3 范围；3-Sigma 检测器分数即可）
    """
    if n_total == 0:
        return HEALTH_BASE
    rate_pct = (n_anomaly / n_total) * 100.0
    severity_factor = max(1.0, mean_severity / 3.0)
    score = HEALTH_BASE - rate_pct * HEALTH_PENALTY_PER_ANOMALY_PCT * severity_factor
    return float(np.clip(score, 0.0, 100.0))


def health_status(score: float) -> tuple[str, str]:
    """返回 (等级, 颜色)。"""
    if score >= 90:
        return "优秀", "#16a34a"
    if score >= 75:
        return "良好", "#65a30d"
    if score >= 60:
        return "注意", "#ca8a04"
    if score >= 40:
        return "警告", "#ea580c"
    return "严重", "#dc2626"
