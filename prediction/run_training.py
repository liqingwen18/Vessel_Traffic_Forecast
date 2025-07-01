"""
STA-BiLSTM模型训练运行脚本

此脚本用于启动模型训练流程，是整个预测系统的主入口
"""

import os
import sys
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt

# 将当前工作目录设置为项目根目录
if os.path.basename(os.getcwd()) == 'prediction':
    os.chdir('..')
    sys.path.append('prediction')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction/training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("run_training")

# 导入模型模块
from config import Config
from data_loader import DataProcessor
from model import STABiLSTMModel
from train import train_model

def main():
    """主函数"""
    logger.info("开始运行STA-BiLSTM模型训练流程")
    
    try:
        # 创建必要的目录
        os.makedirs("prediction/saved_models", exist_ok=True)
        os.makedirs("prediction/results", exist_ok=True)
        os.makedirs("prediction/logs", exist_ok=True)
        
        # 初始化配置
        config = Config()
        config.print_config()
        
        # 数据处理
        logger.info("开始数据处理...")
        data_processor = DataProcessor(config)
        train_loader, val_loader, edge_index, h3_indices = data_processor.prepare_training_data()
        
        logger.info(f"数据加载完成: {len(h3_indices)} 个H3网格")
        logger.info(f"训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}")
        
        # 训练模型
        logger.info("开始模型训练...")
        training_history = train_model(config, train_loader, val_loader, edge_index, data_processor.scaler)
        
        logger.info("模型训练完成!")
        
        # 绘制训练历史
        plt.figure(figsize=config.FIGURE_SIZE)
        
        # 损失曲线
        plt.subplot(2, 1, 1)
        plt.plot(training_history['train_loss'], label='训练损失')
        plt.plot(training_history['val_loss'], label='验证损失')
        plt.title('训练和验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        
        # 指标曲线
        plt.subplot(2, 1, 2)
        plt.plot(training_history['val_rmse'], label='验证RMSE')
        plt.plot(training_history['val_mae'], label='验证MAE')
        plt.title('验证指标')
        plt.xlabel('Epoch')
        plt.ylabel('指标值')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("prediction/results/training_summary.png", dpi=300)
        
        logger.info(f"训练结果已保存至 prediction/results/training_summary.png")
        
        # 完成
        logger.info("训练流程完成")
        
    except Exception as e:
        logger.error(f"训练过程发生错误: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 