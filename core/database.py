"""数据库 Session + CRUD Repository 层。"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, delete, event, select
from sqlalchemy.orm import Session, sessionmaker

from core.config import DB_PATH
from core.models import Alert, Base, Channel, DetectionRun, Device, Signal

_engine = create_engine(
    f"sqlite:///{DB_PATH}",
    future=True,
    echo=False,
    connect_args={"check_same_thread": False, "timeout": 30},
)


@event.listens_for(_engine, "connect")
def _sqlite_pragmas(dbapi_conn, _):
    """SQLite 性能优化：WAL + 关闭同步 + 内存缓存。"""
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA cache_size=-64000")
    cur.execute("PRAGMA temp_store=MEMORY")
    cur.close()


_SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)


def init_db() -> None:
    """创建所有表。"""
    Base.metadata.create_all(_engine)


def reset_db() -> None:
    """清空数据库（测试用）。"""
    Base.metadata.drop_all(_engine)
    Base.metadata.create_all(_engine)


@contextmanager
def get_session() -> Iterator[Session]:
    s = _SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


# ---------- Device ----------
def create_device(name: str, location: str = "") -> Device:
    with get_session() as s:
        d = Device(name=name, location=location)
        s.add(d)
        s.flush()
        s.refresh(d)
        s.expunge(d)
        return d


def list_devices() -> list[Device]:
    with get_session() as s:
        objs = list(s.scalars(select(Device).order_by(Device.id)))
        for o in objs:
            s.expunge(o)
        return objs


def update_device(device_id: int, **fields) -> bool:
    with get_session() as s:
        d = s.get(Device, device_id)
        if not d:
            return False
        for k, v in fields.items():
            if hasattr(d, k):
                setattr(d, k, v)
        return True


def delete_device(device_id: int) -> bool:
    with get_session() as s:
        d = s.get(Device, device_id)
        if not d:
            return False
        s.delete(d)
        return True


# ---------- Channel ----------
def create_channel(
    device_id: int,
    name: str,
    unit: str = "",
    sample_rate: float = 1000.0,
    detector_type: str = "three_sigma",
    threshold_config: dict | None = None,
) -> Channel:
    with get_session() as s:
        c = Channel(
            device_id=device_id,
            name=name,
            unit=unit,
            sample_rate=sample_rate,
            detector_type=detector_type,
            threshold_config=threshold_config or {},
        )
        s.add(c)
        s.flush()
        s.refresh(c)
        s.expunge(c)
        return c


def list_channels(device_id: int | None = None) -> list[Channel]:
    with get_session() as s:
        stmt = select(Channel).order_by(Channel.id)
        if device_id is not None:
            stmt = stmt.where(Channel.device_id == device_id)
        objs = list(s.scalars(stmt))
        for o in objs:
            s.expunge(o)
        return objs


def update_channel(channel_id: int, **fields) -> bool:
    with get_session() as s:
        c = s.get(Channel, channel_id)
        if not c:
            return False
        for k, v in fields.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return True


def delete_channel(channel_id: int) -> bool:
    with get_session() as s:
        c = s.get(Channel, channel_id)
        if not c:
            return False
        s.delete(c)
        return True


# ---------- Signal (批量优先) ----------
def insert_signals_bulk(
    channel_id: int,
    timestamps: Iterable[float],
    values: Iterable[float],
    is_anomaly: Iterable[int] | None = None,
    scores: Iterable[float] | None = None,
    chunk_size: int = 20000,
) -> int:
    """批量插入时序点（分块 + INSERT），返回插入条数。"""
    ts = np.asarray(list(timestamps), dtype=float)
    vs = np.asarray(list(values), dtype=float)
    n = len(ts)
    if n == 0:
        return 0
    if is_anomaly is None:
        ia = np.zeros(n, dtype=int)
    else:
        ia = np.asarray(list(is_anomaly), dtype=int)
    if scores is None:
        sc = np.zeros(n, dtype=float)
    else:
        sc = np.asarray(list(scores), dtype=float)

    from sqlalchemy import text

    stmt = text(
        "INSERT INTO signals (channel_id, timestamp, value, is_anomaly, score) "
        "VALUES (:channel_id, :timestamp, :value, :is_anomaly, :score)"
    )
    total = 0
    with _engine.begin() as conn:
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            rows = [
                {
                    "channel_id": channel_id,
                    "timestamp": float(ts[i]),
                    "value": float(vs[i]),
                    "is_anomaly": int(ia[i]),
                    "score": float(sc[i]),
                }
                for i in range(start, end)
            ]
            conn.execute(stmt, rows)
            total += len(rows)
    return total


def query_signals(
    channel_id: int,
    start_ts: float | None = None,
    end_ts: float | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    with get_session() as s:
        stmt = select(
            Signal.id, Signal.timestamp, Signal.value, Signal.is_anomaly, Signal.score
        ).where(Signal.channel_id == channel_id)
        if start_ts is not None:
            stmt = stmt.where(Signal.timestamp >= start_ts)
        if end_ts is not None:
            stmt = stmt.where(Signal.timestamp <= end_ts)
        stmt = stmt.order_by(Signal.timestamp)
        if limit:
            stmt = stmt.limit(limit)
        rows = s.execute(stmt).all()
    if not rows:
        return pd.DataFrame(columns=["id", "timestamp", "value", "is_anomaly", "score"])
    return pd.DataFrame(rows, columns=["id", "timestamp", "value", "is_anomaly", "score"])


def update_signal(signal_id: int, **fields) -> bool:
    with get_session() as s:
        sig = s.get(Signal, signal_id)
        if not sig:
            return False
        for k, v in fields.items():
            if hasattr(sig, k):
                setattr(sig, k, v)
        return True


def delete_signal(signal_id: int) -> bool:
    with get_session() as s:
        sig = s.get(Signal, signal_id)
        if not sig:
            return False
        s.delete(sig)
        return True


def delete_signals_by_channel(channel_id: int) -> int:
    with get_session() as s:
        result = s.execute(delete(Signal).where(Signal.channel_id == channel_id))
        return result.rowcount or 0


def update_anomaly_flags(
    channel_id: int, ids: Iterable[int], is_anomaly: Iterable[int], scores: Iterable[float]
) -> int:
    """批量更新异常标记与分数。"""
    ids = list(ids)
    flags = list(is_anomaly)
    sc = list(scores)
    rows = [
        {"id": int(ids[i]), "is_anomaly": int(flags[i]), "score": float(sc[i])}
        for i in range(len(ids))
    ]
    if not rows:
        return 0
    with get_session() as s:
        s.bulk_update_mappings(Signal, rows)
    return len(rows)


def count_signals(channel_id: int) -> tuple[int, int]:
    """返回 (总数, 异常数)。"""
    with get_session() as s:
        total = s.execute(
            select(Signal.id).where(Signal.channel_id == channel_id)
        ).all()
        anom = s.execute(
            select(Signal.id).where(
                Signal.channel_id == channel_id, Signal.is_anomaly == 1
            )
        ).all()
        return len(total), len(anom)


# ---------- Alert ----------
def create_alert(
    channel_id: int,
    start_ts: float,
    end_ts: float,
    severity: str = "warning",
    algorithm: str = "",
    note: str = "",
) -> Alert:
    with get_session() as s:
        a = Alert(
            channel_id=channel_id,
            start_ts=start_ts,
            end_ts=end_ts,
            severity=severity,
            algorithm=algorithm,
            note=note,
        )
        s.add(a)
        s.flush()
        s.refresh(a)
        s.expunge(a)
        return a


def list_alerts(
    channel_id: int | None = None, status: str | None = None
) -> list[Alert]:
    with get_session() as s:
        stmt = select(Alert).order_by(Alert.created_at.desc())
        if channel_id is not None:
            stmt = stmt.where(Alert.channel_id == channel_id)
        if status is not None:
            stmt = stmt.where(Alert.status == status)
        objs = list(s.scalars(stmt))
        for o in objs:
            s.expunge(o)
        return objs


def update_alert_status(alert_id: int, status: str, note: str | None = None) -> bool:
    with get_session() as s:
        a = s.get(Alert, alert_id)
        if not a:
            return False
        a.status = status
        if note is not None:
            a.note = note
        return True


def delete_alert(alert_id: int) -> bool:
    with get_session() as s:
        a = s.get(Alert, alert_id)
        if not a:
            return False
        s.delete(a)
        return True


# ---------- DetectionRun ----------
def create_detection_run(
    channel_id: int,
    algorithm: str,
    params: dict,
    n_total: int,
    n_anomaly: int,
    precision: float = 0.0,
    recall: float = 0.0,
    f1: float = 0.0,
) -> DetectionRun:
    with get_session() as s:
        r = DetectionRun(
            channel_id=channel_id,
            algorithm=algorithm,
            params=params,
            n_total=n_total,
            n_anomaly=n_anomaly,
            precision=precision,
            recall=recall,
            f1=f1,
        )
        s.add(r)
        s.flush()
        s.refresh(r)
        s.expunge(r)
        return r


def list_detection_runs(channel_id: int | None = None) -> list[DetectionRun]:
    with get_session() as s:
        stmt = select(DetectionRun).order_by(DetectionRun.run_at.desc())
        if channel_id is not None:
            stmt = stmt.where(DetectionRun.channel_id == channel_id)
        objs = list(s.scalars(stmt))
        for o in objs:
            s.expunge(o)
        return objs


# ---------- 高层辅助 ----------
def import_channel_from_array(
    device_name: str,
    channel_name: str,
    timestamps: np.ndarray,
    values: np.ndarray,
    unit: str = "",
    sample_rate: float = 1000.0,
) -> int:
    """便捷：device 不存在则创建，channel 不存在则创建，批量导入数据。返回 channel_id。"""
    init_db()
    # 复用或创建 device
    devices = [d for d in list_devices() if d.name == device_name]
    if devices:
        device_id = devices[0].id
    else:
        device_id = create_device(device_name).id

    chs = [c for c in list_channels(device_id) if c.name == channel_name]
    if chs:
        channel_id = chs[0].id
        delete_signals_by_channel(channel_id)  # 重新导入清空旧数据
    else:
        channel_id = create_channel(
            device_id, channel_name, unit=unit, sample_rate=sample_rate
        ).id

    insert_signals_bulk(channel_id, timestamps, values)
    return channel_id
