# 🏭 工业时序数据异常检测系统

> 研究生 Python 课程作业 · 工业设备健康监控 MVP

一个端到端的工业时序数据异常检测系统：**TDMS 解析** + **SQLite CRUD** + **5 种异常检测算法** + **Streamlit 实时监控界面**。

## ✨ 功能特性

- 📥 **TDMS 数据解析**：基于 `nptdms`，支持多通道、多设备
- 🗄️ **完整 CRUD**：设备 / 通道 / 信号 / 告警 / 检测运行 5 张表，SQLAlchemy ORM
- 🧠 **多算法异常检测**
  - 统计法：3-Sigma、IQR、MAD（中位数绝对偏差，鲁棒）
  - 机器学习：Isolation Forest、LOF
  - 支持滑窗特征工程（mean / std / rms / 峰峰值 / 峭度 / 偏度）
- 📊 **算法评估**：Precision / Recall / F1 / ROC-AUC / PR-AUC 横向对比
- 🌊 **流式仿真**：后台线程 + 滑动窗口缓冲，模拟实时数据采集
- ❤️ **健康评分**：0–100 分等级（优秀 / 良好 / 注意 / 警告 / 严重）
- 🚨 **告警中心**：异常事件记录、状态确认、CSV 导出
- ✅ **测试覆盖率 85%**（25 个测试用例）

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────┐
│  UI 层 (Streamlit)                                  │
│  实时监控 / 数据管理 / 批量检测 / 算法评估 / 告警中心 │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼─────────┐  ┌──────────────────────┐
│  算法层                 │  │  流式仿真引擎         │
│  统计法 + ML + 评估     │◄─┤  threading + Queue   │
└──────────────┬─────────┘  └──────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│  存储层 (SQLite + SQLAlchemy ORM)                    │
│  devices / channels / signals / alerts / detections │
└──────────────▲──────────────────────────────────────┘
               │
┌──────────────┴──────────────────────────────────────┐
│  数据接入层                                          │
│  TDMS 解析 (nptdms) + 模拟器 (合成数据)              │
└─────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
anomaly_system/
├── app.py                          # Streamlit 入口（首页 + KPI）
├── pages/                          # 5 个功能页面
│   ├── 1_📊_实时监控.py
│   ├── 2_🗄️_数据管理.py
│   ├── 3_🔍_批量检测.py
│   ├── 4_📈_算法评估.py
│   └── 5_🚨_告警中心.py
├── core/
│   ├── config.py                   # 全局配置
│   ├── models.py                   # SQLAlchemy ORM 模型
│   ├── database.py                 # CRUD repository
│   └── health.py                   # 健康评分
├── ingestion/
│   ├── tdms_loader.py              # TDMS 解析
│   └── simulator.py                # 合成数据生成（含 ground truth）
├── algorithms/
│   ├── base.py                     # BaseDetector 抽象类
│   ├── statistical.py              # 3-Sigma / IQR / MAD
│   ├── ml.py                       # IsolationForest / LOF
│   ├── features.py                 # 滑窗特征工程
│   ├── evaluation.py               # P/R/F1/ROC 评估
│   └── registry.py                 # 算法注册表
├── streaming/
│   └── stream_engine.py            # 后台线程流式引擎
├── tests/                          # 25 个测试，覆盖率 85%
│   ├── test_database.py
│   ├── test_tdms.py
│   ├── test_algorithms.py
│   └── test_health_and_stream.py
├── data/                           # SQLite + 示例 TDMS
├── requirements.txt
└── README.md
```

## 🚀 快速开始

### 1. 创建虚拟环境并安装依赖

```bash
cd anomaly_system
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 2. 启动 Web 界面

```bash
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`。

### 3. 推荐演示流程

1. **数据管理 → 数据导入 → 一键生成示例 TDMS** —— 自动生成 3 通道（vibration / temperature / current）共 15000 个采样点，含人为注入的尖峰异常
2. **批量检测 → 选通道 + 选算法 → 开始检测** —— 全量检测，标记异常入库，自动生成告警
3. **实时监控 → 选通道 + 选算法 → 流式仿真** —— 启动后台线程实时推送数据，看异常红点 + 健康分动态变化
4. **算法评估** —— 在合成 ground truth 数据上对 5 种算法横向对比
5. **告警中心** —— 查看 / 确认 / 导出告警

## 🧪 运行测试

```bash
pytest                                                    # 跑全部测试
pytest --cov=core --cov=algorithms --cov-report=term      # 带覆盖率
```

当前 25 个用例，**85% 覆盖率**。

## 📊 算法说明

| 算法 | 类型 | 原理 | 适用场景 |
|------|------|------|---------|
| **3-Sigma** | 统计法 | 偏离均值超 k×σ 即异常 | 数据近似正态分布 |
| **IQR** | 统计法 | 超出 [Q1−k·IQR, Q3+k·IQR] | 数据有偏分布 |
| **MAD** | 鲁棒统计 | 基于中位数偏差，抗离群点污染 | 噪声大、有少量极值 |
| **Isolation Forest** | 集成学习 | 用随机森林孤立异常点 | 高维 / 大数据 |
| **LOF** | 密度法 | 局部离群因子，对比邻居密度 | 局部异常、簇结构数据 |

后两种支持 **滑窗特征工程**：将原始信号转为 6 维统计特征（mean / std / rms / peak-to-peak / kurtosis / skew），更适合工业振动信号。

## ❤️ 健康评分公式

```
score = 100 - (anomaly_rate_% × 2.0 × max(1, mean_severity / 3.0))
```

| 分数 | 等级 | 颜色 |
|------|------|------|
| ≥ 90 | 优秀 | 绿 |
| 75–90 | 良好 | 黄绿 |
| 60–75 | 注意 | 黄 |
| 40–60 | 警告 | 橙 |
| < 40 | 严重 | 红 |

## 🛠️ 技术栈

- **语言**：Python 3.10+
- **数据**：nptdms · NumPy · pandas · SciPy
- **存储**：SQLite + SQLAlchemy 2.0
- **算法**：scikit-learn
- **UI**：Streamlit + Plotly
- **测试**：pytest + pytest-cov

## 📝 课程作业说明

本项目对应作业要求：

| 要求 | 实现 |
|------|------|
| 数据库（CRUD） | SQLite + 5 张表 + 完整增删改查接口 |
| 异常检测算法 | 5 种算法 + 评估指标 + 滑窗特征 |
| 人机交互界面 + 实时监控 | Streamlit 5 页 + 流式仿真引擎 |
| TDMS 数据格式 | nptdms 解析 + 模拟器生成 |
| 业务目标 | 设备健康分 + 多级告警 + 历史追溯 |
