"""生成模拟 TDMS 文件：正弦波 + 噪声 + 注入异常。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from nptdms import ChannelObject, GroupObject, RootObject, TdmsWriter


def generate_signal(
    n_points: int = 5000,
    sample_rate: float = 1000.0,
    base_freq: float = 5.0,
    noise_std: float = 0.1,
    anomaly_ratio: float = 0.02,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成一段时序信号。

    Returns:
        timestamps: 秒
        values: 信号值
        labels: 0/1 异常标签（ground truth）
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_points) / sample_rate
    base = np.sin(2 * np.pi * base_freq * t)
    noise = rng.normal(0, noise_std, n_points)
    values = base + noise
    labels = np.zeros(n_points, dtype=np.int8)

    # 注入若干尖峰异常
    n_anom = max(1, int(n_points * anomaly_ratio))
    anom_idx = rng.choice(n_points, size=n_anom, replace=False)
    spikes = rng.choice([-1, 1], size=n_anom) * rng.uniform(3.0, 6.0, n_anom)
    values[anom_idx] += spikes
    labels[anom_idx] = 1

    return t, values, labels


def write_sample_tdms(
    output_path: Path,
    n_points: int = 5000,
    sample_rate: float = 1000.0,
    channels: tuple[str, ...] = ("vibration", "temperature", "current"),
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """生成包含多通道的示例 TDMS 文件。

    Returns labels dict: channel_name -> ground truth labels
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_map: dict[str, np.ndarray] = {}

    root = RootObject(properties={"description": "Industrial sample TDMS"})
    group = GroupObject("DeviceA", properties={"sample_rate": sample_rate})

    channel_objs = []
    for i, name in enumerate(channels):
        # 不同通道用不同频率/噪声
        _, vals, lbls = generate_signal(
            n_points=n_points,
            sample_rate=sample_rate,
            base_freq=5.0 + i * 2.0,
            noise_std=0.1 + i * 0.05,
            anomaly_ratio=0.02,
            seed=seed + i,
        )
        labels_map[name] = lbls
        channel_objs.append(
            ChannelObject(
                "DeviceA",
                name,
                vals,
                properties={
                    "unit": ["mm/s", "°C", "A"][i % 3],
                    "sample_rate": sample_rate,
                },
            )
        )

    with TdmsWriter(str(output_path)) as w:
        w.write_segment([root, group, *channel_objs])

    return labels_map


if __name__ == "__main__":
    from core.config import SAMPLE_TDMS

    labels = write_sample_tdms(SAMPLE_TDMS)
    print(f"已生成: {SAMPLE_TDMS}")
    for name, lbl in labels.items():
        print(f"  通道 {name}: {len(lbl)} 点, {lbl.sum()} 个异常")
