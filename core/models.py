"""SQLAlchemy ORM 模型：devices / channels / signals / alerts / detections。"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Device(Base):
    __tablename__ = "devices"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    location: Mapped[str] = mapped_column(String(200), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    channels: Mapped[list["Channel"]] = relationship(
        back_populates="device", cascade="all, delete-orphan"
    )


class Channel(Base):
    __tablename__ = "channels"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_id: Mapped[int] = mapped_column(
        ForeignKey("devices.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    unit: Mapped[str] = mapped_column(String(20), default="")
    sample_rate: Mapped[float] = mapped_column(Float, default=1000.0)
    detector_type: Mapped[str] = mapped_column(String(50), default="three_sigma")
    threshold_config: Mapped[dict] = mapped_column(JSON, default=dict)

    device: Mapped[Device] = relationship(back_populates="channels")
    signals: Mapped[list["Signal"]] = relationship(
        back_populates="channel", cascade="all, delete-orphan"
    )
    alerts: Mapped[list["Alert"]] = relationship(
        back_populates="channel", cascade="all, delete-orphan"
    )
    realtime_anomalies: Mapped[list["RealtimeAnomaly"]] = relationship(
        back_populates="channel", cascade="all, delete-orphan"
    )


class Signal(Base):
    __tablename__ = "signals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    channel_id: Mapped[int] = mapped_column(
        ForeignKey("channels.id", ondelete="CASCADE"), nullable=False, index=True
    )
    timestamp: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    is_anomaly: Mapped[int] = mapped_column(Integer, default=0)  # 0/1
    score: Mapped[float] = mapped_column(Float, default=0.0)

    channel: Mapped[Channel] = relationship(back_populates="signals")


class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    channel_id: Mapped[int] = mapped_column(
        ForeignKey("channels.id", ondelete="CASCADE"), nullable=False
    )
    start_ts: Mapped[float] = mapped_column(Float, nullable=False)
    end_ts: Mapped[float] = mapped_column(Float, nullable=False)
    severity: Mapped[str] = mapped_column(String(20), default="warning")  # info/warning/critical
    algorithm: Mapped[str] = mapped_column(String(50), default="")
    status: Mapped[str] = mapped_column(String(20), default="new")  # new/ack/closed
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    channel: Mapped[Channel] = relationship(back_populates="alerts")
    realtime_anomalies: Mapped[list["RealtimeAnomaly"]] = relationship(
        back_populates="alert", cascade="all, delete-orphan"
    )


class RealtimeAnomaly(Base):
    __tablename__ = "realtime_anomalies"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alert_id: Mapped[int | None] = mapped_column(
        ForeignKey("alerts.id", ondelete="CASCADE"), nullable=True, index=True
    )
    channel_id: Mapped[int] = mapped_column(
        ForeignKey("channels.id", ondelete="CASCADE"), nullable=False, index=True
    )
    stream_ts: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    vote_count: Mapped[int] = mapped_column(Integer, default=0)
    vote_threshold: Mapped[int] = mapped_column(Integer, default=1)
    algorithms: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    channel: Mapped[Channel] = relationship(back_populates="realtime_anomalies")
    alert: Mapped[Alert | None] = relationship(back_populates="realtime_anomalies")


class DetectionRun(Base):
    __tablename__ = "detection_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    channel_id: Mapped[int] = mapped_column(
        ForeignKey("channels.id", ondelete="CASCADE"), nullable=False
    )
    run_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    algorithm: Mapped[str] = mapped_column(String(50), nullable=False)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    n_total: Mapped[int] = mapped_column(Integer, default=0)
    n_anomaly: Mapped[int] = mapped_column(Integer, default=0)
    precision: Mapped[float] = mapped_column(Float, default=0.0)
    recall: Mapped[float] = mapped_column(Float, default=0.0)
    f1: Mapped[float] = mapped_column(Float, default=0.0)
