"""算法评估：P/R/F1/ROC + 算法对比。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from algorithms.base import BaseDetector


@dataclass
class EvaluationResult:
    algorithm: str
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    n_total: int
    n_pred_anomaly: int
    n_true_anomaly: int
    fpr: np.ndarray
    tpr: np.ndarray


def evaluate(detector: BaseDetector, x: np.ndarray, y_true: np.ndarray) -> EvaluationResult:
    """对单个检测器进行评估。"""
    x = np.asarray(x, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    detector.fit(x)
    y_pred = detector.predict(x)
    scores = detector.score(x)

    p = float(precision_score(y_true, y_pred, zero_division=0))
    r = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # ROC / PR (需要分数)
    try:
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = float(auc(fpr, tpr))
    except Exception:
        fpr, tpr, roc_auc = np.array([0, 1]), np.array([0, 1]), 0.5

    try:
        pr_p, pr_r, _ = precision_recall_curve(y_true, scores)
        pr_auc = float(auc(pr_r, pr_p))
    except Exception:
        pr_auc = 0.0

    return EvaluationResult(
        algorithm=detector.name,
        precision=p,
        recall=r,
        f1=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        n_total=int(len(x)),
        n_pred_anomaly=int(y_pred.sum()),
        n_true_anomaly=int(y_true.sum()),
        fpr=fpr,
        tpr=tpr,
    )


def compare(detectors: list[BaseDetector], x: np.ndarray, y_true: np.ndarray) -> list[EvaluationResult]:
    return [evaluate(d, x, y_true) for d in detectors]
