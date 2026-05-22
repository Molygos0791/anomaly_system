"""流式仿真引擎：后台线程从已加载的信号中按节奏推送数据点。"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class StreamPoint:
    timestamp: float
    value: float


class StreamEngine:
    """模拟实时数据源：按指定速率从给定数组推送数据点。"""

    def __init__(
        self,
        values: np.ndarray,
        sample_rate: float = 100.0,
        buffer_size: int = 2000,
        loop: bool = False,
    ):
        self.values = np.asarray(values, dtype=float)
        self.sample_rate = sample_rate
        self.loop = loop
        self.buffer: deque[StreamPoint] = deque(maxlen=buffer_size)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._idx = 0
        self._emitted = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        interval = 1.0 / max(1.0, self.sample_rate)
        while not self._stop_event.is_set():
            if len(self.values) == 0:
                break
            if self._idx >= len(self.values):
                if not self.loop:
                    break
                self._idx = 0
            t = self._emitted / self.sample_rate
            with self._lock:
                self.buffer.append(StreamPoint(timestamp=t, value=float(self.values[self._idx])))
            self._idx += 1
            self._emitted += 1
            time.sleep(interval)

    def stop(self, wait: bool = True) -> None:
        self._stop_event.set()
        if (
            wait
            and self._thread is not None
            and self._thread.is_alive()
            and threading.current_thread() is not self._thread
        ):
            self._thread.join(timeout=1.0)

    def reset(self) -> None:
        self.stop()
        with self._lock:
            self.buffer.clear()
            self._idx = 0
            self._emitted = 0

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """返回当前缓冲区内 (timestamps, values)。"""
        with self._lock:
            ts = np.array([p.timestamp for p in self.buffer])
            vs = np.array([p.value for p in self.buffer])
        return ts, vs

    @property
    def progress(self) -> float:
        if len(self.values) == 0:
            return 1.0
        if self.loop:
            return (self._idx % len(self.values)) / len(self.values)
        return min(1.0, self._idx / len(self.values))
