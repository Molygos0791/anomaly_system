# 工业时序数据异常检测系统

这是一个面向工业设备健康监控的 Python MVP 项目，提供 TDMS 数据接入、SQLite 数据管理、异常检测算法、流式仿真监控和告警管理等功能。

## 功能概览

- TDMS 数据解析：基于 `nptdms` 读取工业时序数据，支持多设备、多通道数据导入。
- 数据库管理：使用 SQLite + SQLAlchemy 管理设备、通道、信号、告警、检测记录和实时异常记录。
- 异常检测算法：内置 3-Sigma、IQR、MAD、Isolation Forest、LOF。
- 特征工程：支持滑动窗口特征提取，包括 mean、std、rms、peak-to-peak、kurtosis、skew。
- 批量检测：对已导入通道数据执行全量检测，写入异常标记并生成告警。
- 实时监控：通过后台线程和缓冲队列模拟实时数据流，展示动态曲线和健康分。
- 算法评估：支持 Precision、Recall、F1、ROC-AUC、PR-AUC 等指标对比。
- 告警中心：支持告警查询、状态确认、关闭、备注和 CSV 导出。

## 技术栈

- Python 3.10+
- Streamlit
- Plotly
- NumPy / pandas / SciPy
- scikit-learn
- nptdms
- SQLite
- SQLAlchemy 2.x
- pytest / pytest-cov

## 项目结构

```text
anomaly_system/
├── app.py                         # Streamlit 启动入口
├── pages/                         # Streamlit 页面
│   ├── 1_📊_实时监控.py
│   ├── 2_🗄️_数据管理.py
│   ├── 3_🔍_批量检测.py
│   └── 5_🚨_告警中心.py
├── core/                          # 配置、ORM、数据库访问、健康评分
│   ├── config.py
│   ├── database.py
│   ├── health.py
│   └── models.py
├── ingestion/                     # TDMS 解析与模拟数据生成
│   ├── simulator.py
│   └── tdms_loader.py
├── algorithms/                    # 异常检测算法与评估
│   ├── base.py
│   ├── evaluation.py
│   ├── features.py
│   ├── ml.py
│   ├── registry.py
│   └── statistical.py
├── streaming/                     # 流式仿真引擎
│   └── stream_engine.py
├── tests/                         # 自动化测试
├── docs/                          # 项目文档
├── data/                          # 本地数据目录，存放 SQLite 和 TDMS 文件
├── requirements.txt
├── pytest.ini
└── README.md
```

## 快速开始

### 1. 创建虚拟环境

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动应用

```bash
streamlit run app.py
```

启动后访问：

```text
http://localhost:8501
```

如果默认端口被占用，可以指定其他端口：

```bash
streamlit run app.py --server.port 8502
```

## 推荐使用流程

1. 进入“数据管理”页面，创建设备和通道，或导入 TDMS / 示例数据。
2. 进入“批量检测”页面，选择通道和算法，执行全量异常检测。
3. 进入“实时监控”页面，选择通道和检测算法，查看静态曲线或启动流式仿真。
4. 进入“告警中心”页面，查看、确认、关闭或导出告警记录。
5. 使用测试数据时，可在算法评估模块中对比不同检测算法的效果。

## 内置算法

| 算法 | 类型 | 适用场景 |
| --- | --- | --- |
| 3-Sigma | 统计方法 | 数据接近正态分布，异常点偏离均值明显 |
| IQR | 统计方法 | 数据分布偏斜或存在极端值 |
| MAD | 鲁棒统计 | 噪声较大、希望降低离群点对阈值的影响 |
| Isolation Forest | 机器学习 | 高维特征或复杂工业振动信号 |
| LOF | 机器学习 | 局部密度异常、簇状分布数据 |

## 数据模型

系统主要包含以下数据表：

- `devices`：设备信息。
- `channels`：设备通道和采样配置。
- `signals`：时序信号点、异常标记和异常分数。
- `alerts`：告警事件、等级、状态和备注。
- `realtime_anomalies`：实时检测到的异常点。
- `detection_runs`：批量检测运行记录和指标。

## 健康评分

健康分基于异常率和异常严重程度计算，默认满分 100：

```text
score = 100 - (anomaly_rate_percent * 2.0 * max(1, mean_severity / 3.0))
```

分数越低表示设备状态越差。页面中会按健康分展示不同风险等级。

## 运行测试

```bash
pytest
```

查看覆盖率：

```bash
pytest --cov=core --cov=algorithms --cov-report=term
```

## 开发说明

### 添加新算法

1. 在 `algorithms/` 下新增检测器类，并继承 `BaseDetector`。
2. 实现 `fit`、`predict`、`score`、`get_params` 方法。
3. 在 `algorithms/registry.py` 中注册算法名称和显示标签。
4. 重新启动 Streamlit 应用，在页面中选择新算法进行检测。

### 数据目录

`data/` 目录用于存放本地 SQLite 数据库和 TDMS 示例文件。默认数据库路径由 `core/config.py` 中的 `DB_PATH` 定义。

如需重新初始化本地数据，可以停止应用后删除：

```bash
data/anomaly.db
```

再次启动应用时会重新创建数据库表。

## 文档

更多说明可以查看 `docs/` 目录：

- `docs/architecture.md`
- `docs/algorithms.md`
- `docs/设计说明书.md`
- `docs/使用说明书.md`
