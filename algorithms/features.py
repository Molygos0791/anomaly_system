"""滑窗特征工程：mean / std / rms / 峰峰值 / 峭度 / 偏度。"""
from __future__ import annotations

import numpy as np
from scipy import stats


def sliding_windows(x: np.ndarray, window: int, step: int = 1) -> np.ndarray:
    """返回二维滑窗矩阵 (n_windows, window)。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < window:
        return np.empty((0, window))
    n_win = (n - window) // step + 1
    idx = np.arange(window)[None, :] + step * np.arange(n_win)[:, None]
    return x[idx]


def extract_features(x: np.ndarray, window: int = 50, step: int = 1) -> np.ndarray:
    """对每个滑窗提取 6 个统计特征。

    Returns:
        (n_windows, 6) 矩阵：[mean, std, rms, peak_to_peak, kurtosis, skew]
    """
    win = sliding_windows(x, window, step)
    if win.size == 0:
        return np.empty((0, 6))

    mean = win.mean(axis=1)
    std = win.std(axis=1)
    rms = np.sqrt((win**2).mean(axis=1))
    p2p = win.max(axis=1) - win.min(axis=1)
    # kurtosis/skew：避免常数窗导致 nan
    with np.errstate(invalid="ignore", divide="ignore"):
        kurt = stats.kurtosis(win, axis=1, bias=False)
        skew = stats.skew(win, axis=1, bias=False)
    kurt = np.nan_to_num(kurt, nan=0.0)
    skew = np.nan_to_num(skew, nan=0.0)

    return np.column_stack([mean, std, rms, p2p, kurt, skew])


def expand_window_labels(window_labels: np.ndarray, n: int, window: int, step: int = 1) -> np.ndarray:
    """将基于窗口的预测扩展回原长度（取窗口最后一个位置）。前 window-1 个点补 0。"""
    out = np.zeros(n, dtype=int)
    if len(window_labels) == 0:
        return out
    for i, lbl in enumerate(window_labels):
        end = window - 1 + i * step
        if end < n:
            out[end] = int(lbl)
    return out
