"""数据库 CRUD 测试。"""
from __future__ import annotations

import numpy as np
import pytest

from core import database as db


@pytest.fixture(autouse=True)
def fresh_db():
    db.reset_db()
    yield
    db.reset_db()


def test_device_crud():
    d = db.create_device("M1", "Lab A")
    assert d.id is not None
    devs = db.list_devices()
    assert len(devs) == 1 and devs[0].name == "M1"

    assert db.update_device(d.id, location="Lab B") is True
    assert db.list_devices()[0].location == "Lab B"

    assert db.delete_device(d.id) is True
    assert db.list_devices() == []


def test_channel_crud():
    d = db.create_device("M1")
    c = db.create_channel(d.id, "vib", unit="mm/s", sample_rate=2000.0)
    assert c.id is not None

    chs = db.list_channels(d.id)
    assert len(chs) == 1
    assert db.update_channel(c.id, detector_type="iforest") is True
    assert db.list_channels(d.id)[0].detector_type == "iforest"

    assert db.delete_channel(c.id) is True


def test_signal_bulk_insert_and_query():
    d = db.create_device("M1")
    c = db.create_channel(d.id, "vib")
    n = 1000
    ts = np.arange(n) / 1000.0
    vs = np.random.randn(n)
    inserted = db.insert_signals_bulk(c.id, ts, vs)
    assert inserted == n

    df = db.query_signals(c.id, start_ts=0.1, end_ts=0.5)
    assert len(df) > 0
    assert df["timestamp"].min() >= 0.1
    assert df["timestamp"].max() <= 0.5

    total, anom = db.count_signals(c.id)
    assert total == n
    assert anom == 0


def test_signal_update_anomaly_flags():
    d = db.create_device("M1")
    c = db.create_channel(d.id, "vib")
    db.insert_signals_bulk(c.id, [0.0, 0.1, 0.2], [1.0, 2.0, 3.0])
    df = db.query_signals(c.id)
    db.update_anomaly_flags(c.id, df["id"].tolist(), [0, 1, 0], [0.1, 0.9, 0.2])
    _, anom = db.count_signals(c.id)
    assert anom == 1


def test_signal_delete():
    d = db.create_device("M1")
    c = db.create_channel(d.id, "vib")
    db.insert_signals_bulk(c.id, [0.0, 0.1], [1.0, 2.0])
    df = db.query_signals(c.id)
    assert db.delete_signal(int(df["id"].iloc[0])) is True
    assert len(db.query_signals(c.id)) == 1


def test_alert_crud():
    d = db.create_device("M1")
    c = db.create_channel(d.id, "vib")
    a = db.create_alert(c.id, 0.0, 0.5, severity="critical", algorithm="3sigma")
    assert a.id is not None

    alerts = db.list_alerts(c.id)
    assert len(alerts) == 1
    assert db.update_alert_status(a.id, "ack", note="已查看") is True
    assert db.list_alerts(c.id)[0].status == "ack"
    assert db.delete_alert(a.id) is True


def test_realtime_anomaly_detail():
    d = db.create_device("M1")
    c = db.create_channel(d.id, "vib")
    a = db.create_alert(c.id, 0.0, 0.5, severity="warning", algorithm="three_sigma,mad")

    inserted = db.create_realtime_anomalies(
        alert_id=a.id,
        channel_id=c.id,
        timestamps=[0.1, 0.2],
        values=[10.0, 12.0],
        scores=[3.2, 4.1],
        vote_counts=[2, 3],
        vote_threshold=2,
        algorithms="3-Sigma (统计), MAD (鲁棒统计)",
    )

    assert inserted == 2
    rows = db.list_realtime_anomalies(channel_id=c.id)
    assert len(rows) == 2
    assert set(rows["alert_id"]) == {a.id}
    assert rows["vote_threshold"].tolist() == [2, 2]


def test_detection_run():
    d = db.create_device("M1")
    c = db.create_channel(d.id, "vib")
    r = db.create_detection_run(
        c.id, "iforest", {"contamination": 0.05}, 100, 5, 0.8, 0.6, 0.69
    )
    assert r.id is not None
    runs = db.list_detection_runs(c.id)
    assert len(runs) == 1
    assert runs[0].algorithm == "iforest"
