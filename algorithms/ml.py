"""机器学习异常检测：Isolation Forest / LOF。可在原始值或滑窗特征上运行。"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from algorithms.base import BaseDetector
from algorithms.features import expand_window_labels, extract_features


def _ensure_2d(x: np.ndarray, use_features: bool, window: int) -> tuple[np.ndarray, int]:
    """根据是否启用特征工程，返回 (X矩阵, 用于扩展回原序列的偏移参考)。"""
    x = np.asarray(x, dtype=float)
    if use_features:
        feats = extract_features(x, window=window, step=1)
        if feats.size == 0:
            # 序列过短，回退到原始值
            return x.reshape(-1, 1), 0
        return feats, window - 1
    return x.reshape(-1, 1), 0


class IsolationForestDetector(BaseDetector):
    name = "iforest"

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        use_features: bool = False,
        window: int = 50,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.use_features = use_features
        self.window = window
        self.random_state = random_state
        self.model_: IsolationForest | None = None
        self._n_orig: int = 0

    def fit(self, x: np.ndarray) -> "IsolationForestDetector":
        x = np.asarray(x, dtype=float)
        self._n_orig = len(x)
        X, _ = _ensure_2d(x, self.use_features, self.window)
        self.model_ = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        self.model_.fit(X)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        assert self.model_ is not None, "must fit first"
        x = np.asarray(x, dtype=float)
        X, _ = _ensure_2d(x, self.use_features, self.window)
        # decision_function 越小越异常 → 取反作为异常分
        s = -self.model_.decision_function(X)
        if self.use_features and len(s) != len(x):
            full = np.zeros(len(x))
            for i, v in enumerate(s):
                end = self.window - 1 + i
                if end < len(x):
                    full[end] = v
            return full
        return s

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.model_ is not None, "must fit first"
        x = np.asarray(x, dtype=float)
        X, _ = _ensure_2d(x, self.use_features, self.window)
        raw = self.model_.predict(X)  # -1 异常, 1 正常
        win_labels = (raw == -1).astype(int)
        if self.use_features and len(win_labels) != len(x):
            return expand_window_labels(win_labels, len(x), self.window, step=1)
        return win_labels

    def get_params(self) -> dict:
        return {
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "use_features": self.use_features,
            "window": self.window,
        }


class LOFDetector(BaseDetector):
    name = "lof"

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.05,
        use_features: bool = False,
        window: int = 50,
    ):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.use_features = use_features
        self.window = window
        self.model_: LocalOutlierFactor | None = None
        self._labels_: np.ndarray | None = None
        self._scores_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "LOFDetector":
        x = np.asarray(x, dtype=float)
        X, _ = _ensure_2d(x, self.use_features, self.window)
        # LOF 在无 novelty 模式下，fit_predict 返回标签
        self.model_ = LocalOutlierFactor(
            n_neighbors=min(self.n_neighbors, max(2, len(X) - 1)),
            contamination=self.contamination,
            novelty=False,
        )
        raw = self.model_.fit_predict(X)
        win_labels = (raw == -1).astype(int)
        win_scores = -self.model_.negative_outlier_factor_

        if self.use_features and len(win_labels) != len(x):
            self._labels_ = expand_window_labels(win_labels, len(x), self.window, step=1)
            full_score = np.zeros(len(x))
            for i, v in enumerate(win_scores):
                end = self.window - 1 + i
                if end < len(x):
                    full_score[end] = v
            self._scores_ = full_score
        else:
            self._labels_ = win_labels
            self._scores_ = win_scores
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        if self._scores_ is None:
            self.fit(x)
        return self._scores_  # type: ignore[return-value]

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._labels_ is None:
            self.fit(x)
        return self._labels_  # type: ignore[return-value]

    def get_params(self) -> dict:
        return {
            "n_neighbors": self.n_neighbors,
            "contamination": self.contamination,
            "use_features": self.use_features,
            "window": self.window,
        }
