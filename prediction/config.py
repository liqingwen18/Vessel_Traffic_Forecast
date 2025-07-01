"""
STA-BiLSTM模型配置文件
定义了模型架构、训练参数和数据处理配置
"""

import torch
import os
from pathlib import Path

class Config:
    """模型配置类"""
    
    # 项目路径
    BASE_DIR = Path().absolute()
    
    # 数据路径配置
    DATA_DIR = "data"
    AIS_DATA_PATH = "data/active_ships_cleaned.csv"  # 预处理好的AIS数据
    H3_GRID_DIR = "h3_grid_analysis_res7"            # H3网格分析目录
    H3_GRID_CSV_PATH = "h3_grid_analysis_res7/h3_grid_coverage.csv"  # H3网格覆盖文件
    
    # STA-BiLSTM模型架构参数
    SEQUENCE_LENGTH = 12        # 历史时间窗口长度
    PREDICTION_LENGTH = 6       # 预测时间窗口长度
    HIDDEN_DIM = 128            # LSTM隐藏层维度（增加）
    GAT_HEADS = 8              # GAT注意力头数（增加）
    GAT_HIDDEN_DIM = 64        # GAT隐藏层维度（增加）
    ATTENTION_HEADS = 8        # 时间注意力头数（增加）
    NUM_STA_BLOCKS = 3         # STA-BiLSTM块数量（增加）
    DROPOUT_RATE = 0.1         # Dropout比例（减小）
    USE_ATTENTION = True       # 是否使用时间注意力
    
    # 训练参数
    BATCH_SIZE = 32            # 批次大小
    LEARNING_RATE = 0.001      # 学习率
    WEIGHT_DECAY = 1e-4        # 权重衰减（增加）
    NUM_EPOCHS = 100           # 训练轮数
    EARLY_STOPPING_PATIENCE = 15  # 早停耐心值
    VALIDATION_SPLIT = 0.2     # 验证集比例
    TEACHER_FORCING_RATIO = 0.6  # 教师强制比例（增加）

    # 损失函数权重
    MSE_WEIGHT = 1.0           # MSE损失权重
    SPATIAL_WEIGHT = 0.2       # 空间一致性权重（增加）
    TEMPORAL_WEIGHT = 0.1      # 时间平滑性权重（增加）
    CONSERVATION_WEIGHT = 0.05 # 流量守恒权重（增加）
    USE_MAE = True             # 是否使用MAE损失
    
    # 数据处理参数
    TIME_INTERVAL = 5          # 时间间隔（分钟）
    MIN_SHIP_COUNT = 0         # 最小船舶数量阈值
    MAX_SHIP_COUNT = 100       # 最大船舶数量阈值（用于异常值处理）

    # H3网格参数
    H3_RESOLUTION = 7          # H3网格分辨率（与现有网格一致）
    
    # 设备配置
    DEVICE = torch.device('cuda')  # 强制使用CUDA设备
    
    # 模型保存和结果路径
    MODEL_SAVE_DIR = "prediction/saved_models"
    RESULTS_DIR = "prediction/results"
    LOG_DIR = "prediction/logs"
    BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
    
    # 日志配置
    LOG_LEVEL = "INFO"
    LOG_FILE = os.path.join(LOG_DIR, "training.log")
    
    # 可视化参数
    FIGURE_SIZE = (12, 8)
    DPI = 300
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        os.makedirs(cls.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.LOG_FILE), exist_ok=True)
    
    @classmethod
    def get_model_config(cls):
        """获取STA-BiLSTM模型配置字典"""
        return {
            'sequence_length': cls.SEQUENCE_LENGTH,
            'prediction_length': cls.PREDICTION_LENGTH,
            'hidden_dim': cls.HIDDEN_DIM,
            'gat_heads': cls.GAT_HEADS,
            'gat_hidden_dim': cls.GAT_HIDDEN_DIM,
            'attention_heads': cls.ATTENTION_HEADS,
            'num_sta_blocks': cls.NUM_STA_BLOCKS,
            'dropout_rate': cls.DROPOUT_RATE,
            'use_attention': cls.USE_ATTENTION,
            'device': cls.DEVICE
        }
    
    @classmethod
    def get_training_config(cls):
        """获取训练配置字典"""
        return {
            'batch_size': cls.BATCH_SIZE,
            'learning_rate': cls.LEARNING_RATE,
            'weight_decay': cls.WEIGHT_DECAY,
            'num_epochs': cls.NUM_EPOCHS,
            'early_stopping_patience': cls.EARLY_STOPPING_PATIENCE,
            'validation_split': cls.VALIDATION_SPLIT,
            'teacher_forcing_ratio': cls.TEACHER_FORCING_RATIO,
            'device': cls.DEVICE
        }

    @classmethod
    def get_loss_config(cls):
        """获取损失函数配置字典"""
        return {
            'mse_weight': cls.MSE_WEIGHT,
            'spatial_weight': cls.SPATIAL_WEIGHT,
            'temporal_weight': cls.TEMPORAL_WEIGHT,
            'conservation_weight': cls.CONSERVATION_WEIGHT,
            'use_mae': cls.USE_MAE
        }
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("=" * 60)
        print("STA-BiLSTM船舶流量预测模型配置")
        print("=" * 60)
        print("模型架构:")
        print(f"  设备: {cls.DEVICE}")
        print(f"  历史窗口长度: {cls.SEQUENCE_LENGTH}")
        print(f"  预测窗口长度: {cls.PREDICTION_LENGTH}")
        print(f"  LSTM隐藏维度: {cls.HIDDEN_DIM}")
        print(f"  GAT注意力头数: {cls.GAT_HEADS}")
        print(f"  时间注意力头数: {cls.ATTENTION_HEADS}")
        print(f"  STA-BiLSTM块数: {cls.NUM_STA_BLOCKS}")
        print(f"  使用时间注意力: {cls.USE_ATTENTION}")
        print(f"  Dropout比例: {cls.DROPOUT_RATE}")
        print("\n训练参数:")
        print(f"  批次大小: {cls.BATCH_SIZE}")
        print(f"  学习率: {cls.LEARNING_RATE}")
        print(f"  权重衰减: {cls.WEIGHT_DECAY}")
        print(f"  训练轮数: {cls.NUM_EPOCHS}")
        print(f"  早停耐心值: {cls.EARLY_STOPPING_PATIENCE}")
        print(f"  教师强制比例: {cls.TEACHER_FORCING_RATIO}")
        print("\n损失函数权重:")
        print(f"  MSE权重: {cls.MSE_WEIGHT}")
        print(f"  空间一致性权重: {cls.SPATIAL_WEIGHT}")
        print(f"  时间平滑性权重: {cls.TEMPORAL_WEIGHT}")
        print(f"  流量守恒权重: {cls.CONSERVATION_WEIGHT}")
        print(f"  使用MAE损失: {cls.USE_MAE}")
        print("=" * 60) 