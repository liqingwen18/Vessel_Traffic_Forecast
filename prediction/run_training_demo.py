"""
STA-BiLSTM模型训练运行脚本（演示版）

使用合成数据集进行模型训练测试
"""

import os
import sys
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# 将当前工作目录设置为项目根目录
if os.path.basename(os.getcwd()) == 'prediction':
    os.chdir('..')
    sys.path.append('prediction')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction/training_demo.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("run_training_demo")

# 导入模型模块
from config import Config
from data_loader import ShipFlowDataset
from model import STABiLSTMModel
from train import train_model
from utils import build_h3_adjacency_graph, create_time_series_data, normalize_data

class DemoDataProcessor:
    """
    演示数据处理器
    使用生成的演示数据进行处理
    """
    
    def __init__(self, config: Config):
        """
        初始化数据处理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        
        # 数据存储
        self.h3_indices = None  # H3网格索引列表
        self.edge_index = None  # 图的边索引
        self.scaler = None      # 数据标准化器
        self.timestamps = []    # 时间戳列表
        
    def prepare_demo_data(self):
        """
        准备演示数据
        
        Returns:
            train_loader, val_loader, edge_index, h3_indices
        """
        # 检查演示数据是否存在，不存在则创建
        demo_dir = "prediction/demo_data"
        demo_grid_file = os.path.join(demo_dir, "demo_grid_data.csv")
        
        if not os.path.exists(demo_grid_file):
            logger.info("演示数据不存在，创建演示数据集...")
            from create_demo_dataset import main as create_demo_data
            create_demo_data()
        
        # 1. 加载H3网格和邻接关系
        logger.info("加载H3网格和邻接关系...")
        h3_file = "h3_grid_analysis_res7/h3_grid_coverage.csv"
        h3_df = pd.read_csv(h3_file)
        self.h3_indices = h3_df['h3_index'].tolist()
        logger.info(f"加载了 {len(self.h3_indices)} 个H3网格")
        
        # 2. 构建图结构
        logger.info("构建H3网格邻接图...")
        self.edge_index = build_h3_adjacency_graph(self.h3_indices)
        logger.info(f"构建了 {self.edge_index.shape[1]} 条边的邻接图")
        
        # 3. 加载聚合后的网格数据
        logger.info(f"加载聚合网格数据: {demo_grid_file}")
        grid_data = pd.read_csv(demo_grid_file)
        
        # 转换时间戳
        grid_data['timestamp'] = pd.to_datetime(grid_data['timestamp'])
        
        # 记录时间戳
        self.timestamps = sorted(grid_data['timestamp'].unique())
        logger.info(f"数据集包含 {len(self.timestamps)} 个时间步")
        
        # 4. 创建时间序列数据
        logger.info("创建时间序列数据...")
        X, y = create_time_series_data(
            grid_data, 
            self.h3_indices,
            self.config.SEQUENCE_LENGTH,
            self.config.PREDICTION_LENGTH,
            time_col='timestamp',
            value_col='ship_count'
        )
        
        # 添加特征维度
        X = X[..., np.newaxis]  # [num_samples, seq_len, num_nodes, 1]
        y = y[..., np.newaxis]  # [num_samples, pred_len, num_nodes, 1]
        
        logger.info(f"时间序列数据形状: X={X.shape}, y={y.shape}")
        
        # 5. 数据分割
        split_idx = int(len(X) * (1 - self.config.VALIDATION_SPLIT))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"训练集大小: {X_train.shape[0]}个样本, 验证集大小: {X_val.shape[0]}个样本")
        
        # 6. 数据标准化
        logger.info("标准化数据...")
        X_train_scaled, scaler, X_val_scaled, y_train_scaled, y_val_scaled = normalize_data(
            X_train, X_val, y_train, y_val, scaler_type='standard'
        )
        self.scaler = scaler
        
        # 7. 创建数据加载器
        logger.info("创建数据加载器...")
        train_dataset = ShipFlowDataset(X_train_scaled, y_train_scaled, self.edge_index)
        val_dataset = ShipFlowDataset(X_val_scaled, y_val_scaled, self.edge_index)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.config.DEVICE.type == 'cuda' else False
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.config.DEVICE.type == 'cuda' else False
        )
        
        logger.info("演示数据准备完成")
        return train_loader, val_loader, self.edge_index, self.h3_indices

def main():
    """主函数"""
    logger.info("开始运行STA-BiLSTM模型训练流程（演示版）")
    
    try:
        # 创建必要的目录
        os.makedirs("prediction/saved_models", exist_ok=True)
        os.makedirs("prediction/results", exist_ok=True)
        os.makedirs("prediction/logs", exist_ok=True)
        
        # 初始化配置
        config = Config()
        
        # 修改配置，减少训练时间
        config.NUM_EPOCHS = 10
        config.EARLY_STOPPING_PATIENCE = 3
        config.BATCH_SIZE = 16
        
        config.print_config()
        
        # 数据处理
        logger.info("开始数据处理...")
        data_processor = DemoDataProcessor(config)
        train_loader, val_loader, edge_index, h3_indices = data_processor.prepare_demo_data()
        
        logger.info(f"数据加载完成: {len(h3_indices)} 个H3网格")
        logger.info(f"训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}")
        
        # 训练模型
        logger.info("开始模型训练...")
        trainer = train_model(config, train_loader, val_loader, edge_index, data_processor.scaler)
        
        logger.info("模型训练完成!")
        
        # 获取训练历史数据
        training_history = trainer.training_history
        
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
        plt.plot(training_history['val_smape'], label='验证sMAPE(%)')
        plt.title('验证指标')
        plt.xlabel('Epoch')
        plt.ylabel('指标值')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("prediction/results/training_demo_summary.png", dpi=300)
        
        logger.info(f"训练结果已保存至 prediction/results/training_demo_summary.png")
        
        # 完成
        logger.info("演示训练流程完成")
        
    except Exception as e:
        logger.error(f"训练过程发生错误: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 