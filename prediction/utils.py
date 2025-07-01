"""
STA-BiLSTM模型工具函数

包含数据处理、H3网格图构建、评估指标、可视化等功能
"""

import numpy as np
import pandas as pd
import torch
import h3
import geopandas as gpd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Union
import logging
import os
from datetime import datetime, timedelta
import json

# 导入H3兼容性包装器
try:
    from .h3_compat import latlng_to_h3_compat, h3_to_latlng_compat, grid_disk_compat
except ImportError:
    from h3_compat import latlng_to_h3_compat, h3_to_latlng_compat, grid_disk_compat

def setup_logging(log_file: str, log_level: str = "INFO"):
    """
    设置日志配置
    
    Args:
        log_file: 日志文件路径
        log_level: 日志级别
    
    Returns:
        logger实例
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_h3_grid_data(csv_path: str) -> pd.DataFrame:
    """
    加载H3网格数据
    
    Args:
        csv_path: H3网格CSV文件路径
    
    Returns:
        包含H3网格信息的DataFrame
    """
    try:
        # 尝试不同的编码格式
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
        h3_df = None
        
        for encoding in encodings:
            try:
                h3_df = pd.read_csv(csv_path, encoding=encoding)
                logging.info(f"成功使用{encoding}编码加载H3网格数据")
                break
            except UnicodeDecodeError:
                continue
                
        if h3_df is None:
            raise ValueError(f"无法加载H3网格数据文件 {csv_path}")
            
        # 检查必要的列
        if 'h3_index' not in h3_df.columns:
            raise ValueError("H3网格数据必须包含'h3_index'列")
            
        logging.info(f"成功加载H3网格数据: {len(h3_df)} 个网格")
        return h3_df
        
    except Exception as e:
        logging.error(f"加载H3网格数据失败: {e}")
        raise

def build_h3_adjacency_graph(h3_indices: List[str]) -> torch.Tensor:
    """
    构建H3网格邻接图
    
    Args:
        h3_indices: H3网格索引列表
    
    Returns:
        邻接矩阵的边索引 [2, num_edges]
    """
    logging.info("构建H3网格邻接图...")
    n_nodes = len(h3_indices)
    h3_to_idx = {h3_idx: i for i, h3_idx in enumerate(h3_indices)}
    
    edges = []
    
    # 遍历每个H3网格，获取其邻居
    for i, h3_idx in enumerate(h3_indices):
        try:
            # 获取直接相邻的网格（1-ring）
            neighbors = grid_disk_compat(h3_idx, 1)
            # 移除自身
            if isinstance(neighbors, list):
                if h3_idx in neighbors:
                    neighbors.remove(h3_idx)
            elif isinstance(neighbors, set):
                neighbors.discard(h3_idx)
            
            # 添加边
            for neighbor in neighbors:
                if neighbor in h3_to_idx:
                    j = h3_to_idx[neighbor]
                    edges.append([i, j])
        except Exception as e:
            logging.warning(f"处理H3网格{h3_idx}的邻居时出错: {e}")
    
    # 如果没有边，基于坐标计算K近邻
    if len(edges) == 0:
        logging.warning("没有找到H3网格间的邻接关系，将基于坐标计算最近邻")
        coords = []
        for h3_idx in h3_indices:
            lat, lng = h3_to_latlng_compat(h3_idx)
            coords.append([lat, lng])
        
        coords = np.array(coords)
        distances = cdist(coords, coords)
        k_neighbors = min(6, n_nodes - 1)  # 默认6个邻居，或者所有其他节点
        
        for i in range(n_nodes):
            # 找到最近的K个邻居（排除自身）
            nearest = np.argsort(distances[i])[1:k_neighbors+1]
            for j in nearest:
                edges.append([i, j])
    
    # 去重
    edges = list(set(tuple(edge) for edge in edges))
    logging.info(f"构建了{len(edges)}条边的H3网格邻接图")
    
    # 转换为PyTorch张量
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

def create_time_series_data(
    df: pd.DataFrame, 
    h3_indices: List[str], 
    sequence_length: int, 
    prediction_length: int,
    time_col: str = 'timestamp', 
    value_col: str = 'ship_count'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建时间序列训练数据
    
    Args:
        df: 包含时间序列数据的DataFrame
        h3_indices: H3网格索引列表
        sequence_length: 历史序列长度
        prediction_length: 预测序列长度
        time_col: 时间列名
        value_col: 数值列名
    
    Returns:
        (X, y): 输入序列和目标序列
    """
    # 确保时间列是datetime类型
    df[time_col] = pd.to_datetime(df[time_col])
    
    # 创建完整的时间网格
    time_range = pd.date_range(
        start=df[time_col].min(), 
        end=df[time_col].max(), 
        freq='5T'  # 5分钟间隔
    )
    
    # 构建节点-时间网格
    grid_data = {}
    node_indices = {h3_idx: idx for idx, h3_idx in enumerate(h3_indices)}
    
    # 为每个时间步创建一个空的数据数组
    for t in time_range:
        grid_data[t] = np.zeros(len(h3_indices))
    
    # 填充数据
    for _, row in df.iterrows():
        h3_idx = row['h3_index']
        if h3_idx in node_indices:
            t = row[time_col]
            node_idx = node_indices[h3_idx]
            grid_data[t][node_idx] = row[value_col]
    
    # 转换为时间序列
    time_series = np.array([grid_data[t] for t in time_range])
    
    # 创建滑动窗口
    X, y = [], []
    total_steps = len(time_series)
    
    for i in range(total_steps - sequence_length - prediction_length + 1):
        X.append(time_series[i:i+sequence_length])
        y.append(time_series[i+sequence_length:i+sequence_length+prediction_length])
    
    return np.array(X), np.array(y)

def normalize_data(
    X_train: np.ndarray, 
    X_val: np.ndarray = None,
    y_train: np.ndarray = None, 
    y_val: np.ndarray = None,
    scaler_type: str = 'standard'
) -> Tuple:
    """
    标准化数据
    
    Args:
        X_train: 训练输入序列 [num_samples, seq_len, num_nodes, feature_dim]
        X_val: 验证输入序列
        y_train: 训练目标序列
        y_val: 验证目标序列
        scaler_type: 缩放器类型 ('standard' 或 'minmax')
    
    Returns:
        (X_train_scaled, scaler, X_val_scaled, y_train_scaled, y_val_scaled)
    """
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"不支持的缩放器类型: {scaler_type}")
    
    # 获取形状
    num_train_samples, seq_len, num_nodes, feature_dim = X_train.shape
    
    # 重塑数据以进行缩放
    X_train_reshaped = X_train.reshape(-1, feature_dim)
    scaler.fit(X_train_reshaped)
    
    # 缩放训练数据
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    
    results = [X_train_scaled, scaler]
    
    # 缩放验证数据（如果提供）
    if X_val is not None:
        X_val_reshaped = X_val.reshape(-1, feature_dim)
        X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape)
        results.append(X_val_scaled)
    else:
        results.append(None)
    
    # 缩放训练目标数据（如果提供）
    if y_train is not None:
        y_train_reshaped = y_train.reshape(-1, feature_dim)
        y_train_scaled = scaler.transform(y_train_reshaped).reshape(y_train.shape)
        results.append(y_train_scaled)
    else:
        results.append(None)
    
    # 缩放验证目标数据（如果提供）
    if y_val is not None:
        y_val_reshaped = y_val.reshape(-1, feature_dim)
        y_val_scaled = scaler.transform(y_val_reshaped).reshape(y_val.shape)
        results.append(y_val_scaled)
    else:
        results.append(None)
    
    return tuple(results)

def inverse_transform(scaler, data: np.ndarray) -> np.ndarray:
    """
    反向变换缩放的数据
    
    Args:
        scaler: 缩放器实例
        data: 缩放后的数据
    
    Returns:
        原始尺度的数据
    """
    original_shape = data.shape
    data_reshaped = data.reshape(-1, original_shape[-1])
    data_inverse = scaler.inverse_transform(data_reshaped)
    return data_inverse.reshape(original_shape)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
    
    Returns:
        包含各项评估指标的字典
    """
    # 处理维度
    if y_true.ndim == 4:
        # [batch, seq, nodes, features] -> [batch*seq*nodes, features]
        y_true_flat = y_true.reshape(-1, y_true.shape[-1])
        y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
    else:
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
    
    # 计算指标
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    r2 = r2_score(y_true_flat, y_pred_flat)
    
    # 计算相对误差 - 改进MAPE计算
    # 1. 过滤掉真实值为零或接近零的数据点
    # 2. 使用sMAPE（对称平均绝对百分比误差）计算方法
    # 3. 限制最大MAPE值
    
    # 过滤有效数据点（非零且非极小值）
    threshold = 0.5  # 设置阈值，低于此值的视为极小值
    valid_indices = y_true_flat.flatten() > threshold
    
    if np.sum(valid_indices) > 0:
        # 有效数据点上计算sMAPE
        y_true_valid = y_true_flat.flatten()[valid_indices]
        y_pred_valid = y_pred_flat.flatten()[valid_indices]
        
        # sMAPE公式: 2 * |y_true - y_pred| / (|y_true| + |y_pred|)
        smape = 200.0 * np.mean(np.abs(y_true_valid - y_pred_valid) / 
                              (np.abs(y_true_valid) + np.abs(y_pred_valid) + 1e-6))
        
        # 传统MAPE（添加保护措施）
        mape = np.mean(np.abs((y_true_valid - y_pred_valid) / (y_true_valid + 1e-6))) * 100
        
        # 限制最大值
        mape = min(mape, 500.0)  # 限制在500%以内
    else:
        # 如果没有有效数据点，设置为默认值
        smape = 0.0
        mape = 0.0
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'sMAPE': smape
    }

def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    绘制训练历史
    
    Args:
        history: 包含训练历史的字典
        save_path: 图表保存路径（可选）
    """
    plt.figure(figsize=(15, 10))
    
    # 绘制损失
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.ylabel('损失')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # 绘制指标
    metrics = [key for key in history.keys() if key not in ['train_loss', 'val_loss']]
    if metrics:
        plt.subplot(2, 1, 2)
        for metric in metrics:
            plt.plot(history[metric], label=metric)
        plt.title('模型评估指标')
        plt.ylabel('指标值')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def save_predictions_to_csv(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    h3_indices: List[str],
    timestamps: List[datetime],
    save_path: str
):
    """
    将预测结果保存为CSV文件
    
    Args:
        y_true: 真实值 [batch_size, pred_len, num_nodes, feature_dim]
        y_pred: 预测值 [batch_size, pred_len, num_nodes, feature_dim]
        h3_indices: H3网格索引列表
        timestamps: 对应的时间戳列表
        save_path: 保存路径
    """
    # 准备数据框
    results = []
    
    # 遍历每个批次、时间步和网格
    for batch in range(y_pred.shape[0]):
        for t in range(y_pred.shape[1]):
            timestamp = timestamps[batch * y_pred.shape[1] + t]
            for node in range(y_pred.shape[2]):
                h3_index = h3_indices[node]
                true_value = y_true[batch, t, node, 0]
                pred_value = y_pred[batch, t, node, 0]
                
                results.append({
                    'timestamp': timestamp,
                    'h3_index': h3_index,
                    'true_value': true_value,
                    'predicted_value': pred_value,
                    'error': true_value - pred_value
                })
    
    # 创建数据框并保存
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results_df.to_csv(save_path, index=False)
    
    logging.info(f"预测结果已保存至: {save_path}")

def visualize_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_idx: int = 0,
    node_idx: int = 0,
    save_path: Optional[str] = None
):
    """
    可视化单个时间序列的预测结果
    
    Args:
        y_true: 真实值 [batch_size, pred_len, num_nodes, feature_dim]
        y_pred: 预测值 [batch_size, pred_len, num_nodes, feature_dim]
        sample_idx: 样本索引
        node_idx: 节点索引
        save_path: 图表保存路径（可选）
    """
    # 提取指定样本和节点的时间序列
    true_series = y_true[sample_idx, :, node_idx, 0]
    pred_series = y_pred[sample_idx, :, node_idx, 0]
    
    # 创建时间步
    time_steps = list(range(len(true_series)))
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, true_series, 'b-', marker='o', label='真实值')
    plt.plot(time_steps, pred_series, 'r-', marker='x', label='预测值')
    plt.title(f'样本 {sample_idx} 节点 {node_idx} 的预测结果')
    plt.xlabel('时间步')
    plt.ylabel('流量')
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def save_model_config(config: Dict, save_path: str):
    """
    保存模型配置
    
    Args:
        config: 配置字典
        save_path: 保存路径
    """
    # 转换所有NumPy和PyTorch类型为Python原生类型
    def convert_to_serializable(obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    # 转换配置
    serializable_config = convert_to_serializable(config)
    
    # 保存为JSON
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_config, f, ensure_ascii=False, indent=2)
    
    logging.info(f"模型配置已保存至: {save_path}") 