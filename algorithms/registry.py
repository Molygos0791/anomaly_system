"""检测器工厂：根据名称构建检测器实例。"""
from __future__ import annotations

from algorithms.base import BaseDetector
from algorithms.ml import IsolationForestDetector, LOFDetector
from algorithms.statistical import IQRDetector, MADDetector, ThreeSigmaDetector

REGISTRY = {
    "three_sigma": ThreeSigmaDetector,
    "iqr": IQRDetector,
    "mad": MADDetector,
    "iforest": IsolationForestDetector,
    "lof": LOFDetector,
}

ALGORITHM_LABELS = {
    "three_sigma": "3-Sigma (统计)",
    "iqr": "IQR (统计)",
    "mad": "MAD (鲁棒统计)",
    "iforest": "Isolation Forest (机器学习)",
    "lof": "LOF (机器学习)",
}


def build(name: str, **kwargs) -> BaseDetector:
    if name not in REGISTRY:
        raise ValueError(f"未知算法: {name}; 可用: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)


def list_algorithms() -> list[str]:
    return list(REGISTRY.keys())
