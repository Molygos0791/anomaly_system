# 系统架构说明

## 模块分层

```
┌─────────────────────────────────────────────────────────┐
│                      UI 层                                │
│  Streamlit 多页应用：5 个独立页面                          │
│  ├─ 实时监控  ├─ 数据管理  ├─ 批量检测  ├─ 评估  ├─ 告警  │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼────────┐ ┌──▼──────────┐ ┌▼───────────────┐
│  算法层         │ │ 流式仿真     │ │  健康评分      │
│  BaseDetector  │ │ StreamEngine│ │  health_score  │
│  ├─ 统计法     │ │ 后台线程     │ │                │
│  ├─ ML         │ │ deque缓冲   │ │                │
│  └─ 评估指标   │ └─────────────┘ └────────────────┘
└────────┬───────┘
         │
┌────────▼─────────────────────────────────────────────┐
│  存储层 (SQLite + SQLAlchemy ORM)                      │
│                                                        │
│   devices ──< channels ──< signals                     │
│                  │                                     │
│                  ├──< alerts                           │
│                  └──< detection_runs                   │
└────────▲─────────────────────────────────────────────┘
         │
┌────────┴───────────────────────┐
│  数据接入层                     │
│  ├─ tdms_loader (nptdms)       │
│  └─ simulator (合成数据 + GT)  │
└────────────────────────────────┘
```

## 数据流

**离线分析流**：
```
TDMS 文件 → 解析 → 批量入库 → 选算法检测 → 异常标记 + 告警 → UI 可视化
```

**实时监控流**：
```
入库数据 → StreamEngine 后台推送 → 滑动窗口缓冲 → 在线检测 → Plotly 实时绘图
```

## 数据库 ER 图

```
┌──────────┐       ┌──────────┐       ┌──────────┐
│ devices  │ 1   N │ channels │ 1   N │ signals  │
│──────────│───────│──────────│───────│──────────│
│ id (PK)  │       │ id (PK)  │       │ id (PK)  │
│ name     │       │ device_id│       │ channel_id│
│ location │       │ name     │       │ timestamp│
│created_at│       │ unit     │       │ value    │
└──────────┘       │sample_rate│      │is_anomaly│
                   │detector  │       │ score    │
                   │threshold │       └──────────┘
                   └──────────┘
                       │ 1
                       │
                  ┌────┴────┐
                  │ N      N│
                  ▼         ▼
            ┌─────────┐ ┌──────────────┐
            │ alerts  │ │detection_runs│
            │─────────│ │──────────────│
            │ id      │ │ id           │
            │ severity│ │ algorithm    │
            │ status  │ │ params (JSON)│
            │ note    │ │ n_total      │
            └─────────┘ │ n_anomaly    │
                        │ precision/r/f1│
                        └──────────────┘
```

## 算法接口设计

所有检测器遵循统一接口：

```python
class BaseDetector(ABC):
    name: str

    def fit(self, x: np.ndarray) -> "BaseDetector": ...
    def predict(self, x: np.ndarray) -> np.ndarray:    # 0/1
    def score(self, x: np.ndarray) -> np.ndarray:      # 异常分
    def get_params(self) -> dict: ...
```

通过 `algorithms/registry.py` 的工厂方法 `build(name, **kwargs)` 统一构造。

## 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 数据库 | SQLite | 零配置，作业演示足够 |
| ORM | SQLAlchemy 2.0 | 类型安全，规范化 CRUD |
| UI | Streamlit | 单文件页面，无需前端工程 |
| 流式 | threading + deque | 标准库，无需 Redis/Kafka |
| 评估基准 | 合成数据 + 注入异常 | 有 ground truth 才能算 P/R/F1 |
| 算法注册 | 工厂模式 | UI 上动态切换算法、参数 |
