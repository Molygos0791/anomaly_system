"""检测器抽象接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseDetector(ABC):
    name: str = "base"

    def fit(self, x: np.ndarray) -> "BaseDetector":  # 默认无监督，可被覆盖
        return self

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """返回 0/1 数组，1 表示异常。"""

    @abstractmethod
    def score(self, x: np.ndarray) -> np.ndarray:
        """返回每点异常分数，越大越异常。"""

    def fit_predict(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).predict(x)

    def get_params(self) -> dict:
        return {}
