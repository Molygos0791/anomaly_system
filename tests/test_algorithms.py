"""算法测试：合成数据 + 注入异常验证检出能力。"""
from __future__ import annotations

import numpy as np
import pytest

from algorithms.evaluation import evaluate
from algorithms.features import extract_features, sliding_windows
from algorithms.ml import IsolationForestDetector, LOFDetector
from algorithms.registry import build, list_algorithms
from algorithms.statistical import IQRDetector, MADDetector, ThreeSigmaDetector


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(0)
    n = 2000
    x = np.sin(np.linspace(0, 40 * np.pi, n)) + rng.normal(0, 0.1, n)
    y = np.zeros(n, dtype=int)
    idx = rng.choice(n, 60, replace=False)
    x[idx] += rng.choice([-1, 1], 60) * rng.uniform(4, 6, 60)
    y[idx] = 1
    return x, y


def test_three_sigma(synthetic):
    x, y = synthetic
    d = ThreeSigmaDetector(k=3.0)
    d.fit(x)
    pred = d.predict(x)
    assert pred.shape == x.shape
    # 至少检出大部分注入异常
    recall = (pred & y).sum() / max(1, y.sum())
    assert recall > 0.5


def test_iqr(synthetic):
    x, y = synthetic
    d = IQRDetector(k=1.5)
    d.fit(x)
    pred = d.predict(x)
    assert pred.sum() > 0


def test_mad(synthetic):
    x, y = synthetic
    d = MADDetector(k=3.5)
    d.fit(x)
    pred = d.predict(x)
    recall = (pred & y).sum() / max(1, y.sum())
    assert recall > 0.5


def test_iforest(synthetic):
    x, y = synthetic
    d = IsolationForestDetector(contamination=0.05)
    d.fit(x)
    pred = d.predict(x)
    assert pred.shape == x.shape
    assert pred.sum() > 0


def test_lof(synthetic):
    x, y = synthetic
    d = LOFDetector(contamination=0.05)
    d.fit(x)
    pred = d.predict(x)
    assert pred.shape == x.shape


def test_features():
    x = np.random.randn(500)
    feats = extract_features(x, window=50, step=1)
    assert feats.shape == (451, 6)


def test_sliding_window_short():
    x = np.array([1.0, 2.0])
    win = sliding_windows(x, window=10)
    assert win.shape == (0, 10)


def test_evaluation(synthetic):
    x, y = synthetic
    d = ThreeSigmaDetector()
    res = evaluate(d, x, y)
    assert 0 <= res.precision <= 1
    assert 0 <= res.recall <= 1
    assert 0 <= res.f1 <= 1
    assert res.algorithm == "three_sigma"


def test_registry():
    names = list_algorithms()
    assert "three_sigma" in names
    assert "iforest" in names
    d = build("three_sigma", k=2.5)
    assert isinstance(d, ThreeSigmaDetector)
    assert d.k == 2.5
