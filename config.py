import os
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 基础路径
DATA_CLASS = 611
DATA_DIR = os.path.join(BASE_DIR, "..", "..", "Data", "DAGM", f"Class{DATA_CLASS}")  # 基础路径

# MODEL_PATH = os.path.join(BASE_DIR, "model", "CheapNet_G64.pkl")
MODEL_PATH = None
SAVE_MODEL = True

DAGM6_MEAN = [0.348]    # DAGM_class6上的均值
DAGM6_STD = [0.278]     # DAGM_class6上的方差

MAX_EPOCH = 3           # 跑多少轮
BATCH_SIZE = 4          # 每次载入多少图片
GROUP = 64              # cheapnet中的group数
SLOPE = 0.2             # 激活函数
DATALOADER_WORKERS = 1  # dataloader线程数

TIME_STR = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M')  # 时间格式化
TAG = "cheapNetV1"      # 备注

LR = 0.01               # 学习率
MILESTONES = [1]        # 学习率在第多少个epoch下降
GAMMA = 0.1             # 下降参数

LOG_DIR = os.path.join(BASE_DIR, "..", "results", f"C{DATA_CLASS}_B{BATCH_SIZE}_P{MAX_EPOCH}_G{GROUP}_{TIME_STR}")  # 结果保存路径
log_name = f'{TIME_STR}.log'
