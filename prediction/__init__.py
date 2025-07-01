"""
STA-BiLSTM船舶流量预测模型包

基于图注意力网络和双向LSTM的时空预测模型
用于长江浔江流域船舶交通流量预测
"""

__version__ = '1.0.0'
__author__ = 'AI船舶流量分析团队'

import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("初始化STA-BiLSTM船舶流量预测模型") 