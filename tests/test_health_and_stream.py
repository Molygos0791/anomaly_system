"""健康评分与流式引擎测试。"""
from __future__ import annotations

import time

import numpy as np

from core.health import health_score, health_status
from streaming.stream_engine import StreamEngine


def test_health_score_zero():
    assert health_score(0, 0) == 100.0


def test_health_score_normal():
    s = health_score(1000, 10, mean_severity=1.0)  # 1% anomaly
    assert 90 < s <= 100


def test_health_score_critical():
    s = health_score(100, 50, mean_severity=3.0)
    assert s < 50


def test_health_status_levels():
    assert health_status(95)[0] == "优秀"
    assert health_status(80)[0] == "良好"
    assert health_status(65)[0] == "注意"
    assert health_status(50)[0] == "警告"
    assert health_status(20)[0] == "严重"


def test_stream_engine():
    vals = np.arange(50, dtype=float)
    eng = StreamEngine(vals, sample_rate=200.0, buffer_size=100)  # ~5ms / 点
    eng.start()
    time.sleep(0.5)  # 应足够 push 完
    eng.stop()
    ts, vs = eng.snapshot()
    assert len(vs) > 0
    # 数据顺序正确
    assert np.all(np.diff(ts) >= 0)


def test_stream_engine_reset():
    vals = np.arange(20, dtype=float)
    eng = StreamEngine(vals, sample_rate=100.0)
    eng.start()
    time.sleep(0.1)
    eng.reset()
    ts, vs = eng.snapshot()
    assert len(ts) == 0 and len(vs) == 0
    assert eng.progress == 0.0
