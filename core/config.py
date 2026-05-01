"""全局配置：路径、默认参数。"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "anomaly.db"
SAMPLE_TDMS = DATA_DIR / "sample.tdms"

DATA_DIR.mkdir(exist_ok=True)

# 默认算法参数
DEFAULT_SIGMA_K = 3.0
DEFAULT_IQR_K = 1.5
DEFAULT_IFOREST_CONTAMINATION = 0.05
DEFAULT_LOF_NEIGHBORS = 20
DEFAULT_WINDOW_SIZE = 50  # 滑窗特征长度

# 健康评分
HEALTH_BASE = 100.0
HEALTH_PENALTY_PER_ANOMALY_PCT = 2.0  # 每 1% 异常率扣 2 分
