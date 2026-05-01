"""TDMS 解析测试。"""
from __future__ import annotations

import tempfile
from pathlib import Path

from ingestion.simulator import write_sample_tdms
from ingestion.tdms_loader import load_tdms, to_dataframe


def test_load_sample_tdms():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.tdms"
        write_sample_tdms(path, n_points=500, channels=("a", "b"))
        chs = load_tdms(path)
        assert len(chs) == 2
        names = {c.channel for c in chs}
        assert names == {"a", "b"}
        for c in chs:
            assert len(c.values) == 500
            assert c.sample_rate == 1000.0


def test_to_dataframe():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.tdms"
        write_sample_tdms(path, n_points=100, channels=("x",))
        df = to_dataframe(load_tdms(path))
        assert {"device", "channel", "timestamp", "value"}.issubset(df.columns)
        assert len(df) == 100


def test_load_missing_file():
    import pytest

    with pytest.raises(FileNotFoundError):
        load_tdms("/nonexistent/path.tdms")
