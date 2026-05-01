"""统计法异常检测：3-Sigma / IQR / MAD。"""
from __future__ import annotations

import numpy as np

from algorithms.base import BaseDetector


class ThreeSigmaDetector(BaseDetector):
    name = "three_sigma"

    def __init__(self, k: float = 3.0):
        self.k = k
        self.mean_: float = 0.0
        self.std_: float = 1.0

    def fit(self, x: np.ndarray) -> "ThreeSigmaDetector":
        x = np.asarray(x, dtype=float)
        self.mean_ = float(np.mean(x))
        self.std_ = float(np.std(x)) or 1e-9
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.abs(x - self.mean_) / self.std_

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.score(x) > self.k).astype(int)

    def get_params(self) -> dict:
        return {"k": self.k}


class IQRDetector(BaseDetector):
    name = "iqr"

    def __init__(self, k: float = 1.5):
        self.k = k
        self.q1_: float = 0.0
        self.q3_: float = 0.0
        self.iqr_: float = 1.0

    def fit(self, x: np.ndarray) -> "IQRDetector":
        x = np.asarray(x, dtype=float)
        self.q1_ = float(np.percentile(x, 25))
        self.q3_ = float(np.percentile(x, 75))
        self.iqr_ = (self.q3_ - self.q1_) or 1e-9
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        center = (self.q1_ + self.q3_) / 2
        return np.abs(x - center) / self.iqr_

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        low = self.q1_ - self.k * self.iqr_
        high = self.q3_ + self.k * self.iqr_
        return ((x < low) | (x > high)).astype(int)

    def get_params(self) -> dict:
        return {"k": self.k}


class MADDetector(BaseDetector):
    """基于中位数绝对偏差的鲁棒检测。"""

    name = "mad"

    def __init__(self, k: float = 3.5):
        self.k = k
        self.median_: float = 0.0
        self.mad_: float = 1.0

    def fit(self, x: np.ndarray) -> "MADDetector":
        x = np.asarray(x, dtype=float)
        self.median_ = float(np.median(x))
        self.mad_ = float(np.median(np.abs(x - self.median_))) or 1e-9
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        # 0.6745 让 MAD 在正态分布下与 std 一致
        return 0.6745 * np.abs(x - self.median_) / self.mad_

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.score(x) > self.k).astype(int)

    def get_params(self) -> dict:
        return {"k": self.k}
