"""TDMS 文件解析器。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from nptdms import TdmsFile


@dataclass
class ChannelData:
    device: str
    channel: str
    unit: str
    sample_rate: float
    timestamps: np.ndarray  # 秒（相对起点）
    values: np.ndarray


def _infer_sample_rate(props: dict) -> float:
    """从 TDMS 通道属性推断采样率。优先级：sample_rate > 1/wf_increment > 默认 1000."""
    if "sample_rate" in props:
        try:
            return float(props["sample_rate"])
        except (TypeError, ValueError):
            pass
    inc = props.get("wf_increment")
    if inc is not None:
        try:
            inc_f = float(inc)
            if inc_f > 0:
                return 1.0 / inc_f
        except (TypeError, ValueError):
            pass
    return 1000.0


def _infer_unit(props: dict) -> str:
    for key in ("unit", "unit_string", "NI_UnitDescription"):
        v = props.get(key)
        if v:
            return str(v)
    return ""


def load_tdms(
    path: str | Path,
    max_points_per_channel: int | None = None,
    downsample: int = 1,
) -> list[ChannelData]:
    """解析 TDMS 文件，返回所有通道的数据列表。

    Args:
        path: TDMS 文件路径
        max_points_per_channel: 每通道最多读取的原始点数（None=全部）
        downsample: 等间隔降采样步长（>=1，1 表示不降采样）
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TDMS 文件不存在: {path}")

    downsample = max(1, int(downsample))
    tdms = TdmsFile.read(str(path))
    result: list[ChannelData] = []

    for group in tdms.groups():
        for channel in group.channels():
            props = dict(channel.properties or {})
            n_total = len(channel)
            n_read = min(n_total, max_points_per_channel) if max_points_per_channel else n_total
            raw = np.asarray(channel[:n_read], dtype=float)
            if downsample > 1:
                raw = raw[::downsample]
            sr = _infer_sample_rate(props)
            unit = _infer_unit(props)
            effective_sr = sr / downsample
            timestamps = np.arange(len(raw)) / effective_sr
            # 通道名清洗：替换斜杠避免后续路径混淆（保留原名仅做显示）
            ch_name = str(props.get("NI_ChannelName") or channel.name)
            result.append(
                ChannelData(
                    device=group.name,
                    channel=ch_name,
                    unit=unit,
                    sample_rate=effective_sr,
                    timestamps=timestamps,
                    values=raw,
                )
            )

    return result


def list_tdms_metadata(path: str | Path) -> list[dict]:
    """仅读取元数据（不加载数据值），返回每通道的 size / sample_rate / unit 信息。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TDMS 文件不存在: {path}")
    # read_metadata 不加载实际数据，速度极快
    tdms = TdmsFile.read_metadata(str(path))
    info: list[dict] = []
    for group in tdms.groups():
        for channel in group.channels():
            props = dict(channel.properties or {})
            info.append(
                {
                    "device": group.name,
                    "channel": str(props.get("NI_ChannelName") or channel.name),
                    "n_points": len(channel),
                    "sample_rate": _infer_sample_rate(props),
                    "unit": _infer_unit(props),
                }
            )
    return info


def to_dataframe(channels: list[ChannelData]) -> pd.DataFrame:
    """合并所有通道为长表 DataFrame。"""
    frames = []
    for ch in channels:
        frames.append(
            pd.DataFrame(
                {
                    "device": ch.device,
                    "channel": ch.channel,
                    "unit": ch.unit,
                    "sample_rate": ch.sample_rate,
                    "timestamp": ch.timestamps,
                    "value": ch.values,
                }
            )
        )
    if not frames:
        return pd.DataFrame(
            columns=["device", "channel", "unit", "sample_rate", "timestamp", "value"]
        )
    return pd.concat(frames, ignore_index=True)
