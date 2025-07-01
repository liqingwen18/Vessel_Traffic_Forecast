"""
数据加载和预处理模块
处理AIS数据和H3网格数据，生成训练用的时空序列
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h3
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional
import logging
import os

# 导入H3兼容性包装器和工具函数
try:
    from .h3_compat import latlng_to_h3_compat, h3_to_latlng_compat, grid_disk_compat
    from .utils import load_h3_grid_data, build_h3_adjacency_graph, create_time_series_data, normalize_data
    from .config import Config
except ImportError:
    from h3_compat import latlng_to_h3_compat, h3_to_latlng_compat, grid_disk_compat
    from utils import load_h3_grid_data, build_h3_adjacency_graph, create_time_series_data, normalize_data
    from config import Config

# 获取模块级别的logger
logger = logging.getLogger(__name__)

class ShipFlowDataset(Dataset):
    """
    船舶流量数据集
    
    处理AIS数据并转换为H3网格时间序列格式
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, edge_index: torch.Tensor):
        """
        初始化数据集
        
        Args:
            X: 输入序列 [num_samples, seq_len, num_nodes, input_size]
            y: 目标序列 [num_samples, pred_len, num_nodes, input_size]
            edge_index: 图的边索引 [2, num_edges]
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.edge_index = edge_index
        
        self.num_samples = X.shape[0]
        self.seq_len = X.shape[1]
        self.num_nodes = X.shape[2]
        
        logger.info(f"创建数据集: {self.num_samples} 个样本, {self.seq_len} 个时间步, {self.num_nodes} 个节点")
        
    def __len__(self):
        """返回数据集大小"""
        return self.num_samples
    
    def __getitem__(self, idx):
        """获取单个样本"""
        return self.X[idx], self.y[idx], self.edge_index


class DataProcessor:
    """
    数据处理器
    
    负责加载、预处理和转换数据
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
        
    def load_ais_data(self, file_path: str = None) -> pd.DataFrame:
        """
        加载预处理好的AIS数据

        Args:
            file_path: AIS数据文件路径（可选，默认使用配置中的路径）

        Returns:
            预处理好的AIS数据DataFrame
        """
        if file_path is None:
            file_path = self.config.AIS_DATA_PATH
            
        logger.info(f"加载AIS数据: {file_path}")
        
        try:
            # 尝试不同的编码方式
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"成功使用{encoding}编码加载AIS数据")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError(f"无法读取文件 {file_path}")
            
            # 检查必要的列
            lat_cols = [col for col in df.columns if 'lat' in col.lower()]
            lon_cols = [col for col in df.columns if 'lon' in col.lower()]
            
            logger.info(f"可能的纬度列: {lat_cols}")
            logger.info(f"可能的经度列: {lon_cols}")
            
            # 使用正确的列
            if 'lat' in df.columns and 'lon' in df.columns:
                lat_col = 'lat'
                lon_col = 'lon'
            elif 'lat_decimal' in df.columns and 'lon_decimal' in df.columns:
                lat_col = 'lat_decimal'
                lon_col = 'lon_decimal'
            else:
                # 如果没有找到标准名称，尝试使用第一个带lat/lon的列
                if lat_cols and lon_cols:
                    lat_col = lat_cols[0]
                    lon_col = lon_cols[0]
                else:
                    raise ValueError("无法找到经纬度列")
            
            logger.info(f"使用纬度列: {lat_col}, 经度列: {lon_col}")
            
            # 检查坐标范围并转换单位（如果需要）
            lat_min = df[lat_col].min()
            lat_max = df[lat_col].max()
            lon_min = df[lon_col].min()
            lon_max = df[lon_col].max()
            
            logger.info(f"原始坐标范围: 纬度 [{lat_min}, {lat_max}], 经度 [{lon_min}, {lon_max}]")
            
            # 判断坐标是否需要转换
            need_conversion = (lat_min < -90 or lat_max > 90 or lon_min < -180 or lon_max > 180 or 
                              lat_min > 1000 or lon_min > 1000)  # 可能是投影坐标系
            
            if need_conversion:
                logger.info("坐标需要转换，尝试将投影坐标转换为WGS84经纬度")
                
                # 创建LAT和LON列用于后续计算
                # 这里假设坐标是某种投影系统的值，我们暂时将它们直接赋值
                # 实际应用中应该使用适当的坐标转换方法
                df['LAT'] = df[lat_col]
                df['LON'] = df[lon_col]
                
                # 由于没有明确的坐标系信息，我们使用一个适当的缩放方法
                # 这只是一个临时解决方案，实际中应根据具体投影系统进行正确转换
                if lat_min > 1000:  # 大数值可能是坐标系统单位的问题
                    # 对于某些国家的坐标系统，可能需要这样的转换
                    # 仅作为示例，实际转换需要根据具体投影系统
                    scale_factor = 0.00001  # 缩放因子，根据实际坐标系统调整
                    df['LAT'] = (df['LAT'] - 4000000) * scale_factor
                    df['LON'] = (df['LON'] - 500000) * scale_factor
                else:
                    # 简单缩放
                    df['LAT'] = df['LAT'] * 0.000001
                    df['LON'] = df['LON'] * 0.000001
            else:
                # 使用原始列
                df['LAT'] = df[lat_col]
                df['LON'] = df[lon_col]
            
            # 检查转换后的坐标
            logger.info(f"转换后坐标范围: 纬度 [{df['LAT'].min()}, {df['LAT'].max()}], " 
                      f"经度 [{df['LON'].min()}, {df['LON'].max()}]")
            
            # 检查MMSI列
            mmsi_col = None
            for col in df.columns:
                if col.upper() == 'MMSI':
                    mmsi_col = col
                    break
            
            if mmsi_col is None:
                raise ValueError("无法找到MMSI列")
            
            df['MMSI'] = df[mmsi_col]
            
            # 检查时间列
            datetime_col = None
            for col in df.columns:
                if 'time' in col.lower() or 'date' in col.lower():
                    datetime_col = col
                    break
            
            if datetime_col is None:
                raise ValueError("无法找到时间列")
            
            # 转换时间格式
            df['BaseDateTime'] = pd.to_datetime(df[datetime_col])
            
            # 数据清洗
            df = df.dropna(subset=['MMSI', 'LAT', 'LON', 'BaseDateTime'])
            
            # 过滤异常坐标
            df = df[(df['LAT'] >= -90) & (df['LAT'] <= 90)]
            df = df[(df['LON'] >= -180) & (df['LON'] <= 180)]
            
            logger.info(f"加载AIS数据: {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"加载AIS数据失败: {str(e)}")
            raise
    
    def aggregate_to_h3_grid(self, ais_df: pd.DataFrame, h3_indices: List[str]) -> pd.DataFrame:
        """
        将AIS数据聚合到H3网格
        
        Args:
            ais_df: AIS数据DataFrame
            h3_indices: H3网格索引列表
        
        Returns:
            聚合后的网格数据DataFrame
        """
        logger.info("开始将AIS数据聚合到H3网格...")
        
        # 为每个AIS点分配H3网格索引
        def assign_h3_index(row):
            """为AIS点分配H3索引"""
            try:
                return latlng_to_h3_compat(row['LAT'], row['LON'], self.config.H3_RESOLUTION)
            except Exception as e:
                logger.warning(f"无法为坐标 ({row['LAT']}, {row['LON']}) 分配H3索引: {e}")
                return None

        ais_df['h3_index'] = ais_df.apply(assign_h3_index, axis=1)
        ais_df = ais_df.dropna(subset=['h3_index'])
        
        # 只保留在指定H3网格内的数据
        ais_df = ais_df[ais_df['h3_index'].isin(h3_indices)]
        
        if len(ais_df) == 0:
            raise ValueError("没有AIS数据落在指定的H3网格内")
        
        # 时间窗口聚合
        ais_df['time_window'] = ais_df['BaseDateTime'].dt.floor(f'{self.config.TIME_INTERVAL}T')
        
        # 按H3网格和时间窗口聚合船舶数量
        aggregated = ais_df.groupby(['h3_index', 'time_window']).agg({
            'MMSI': 'nunique'  # 统计唯一船舶数量
        }).reset_index()
        
        aggregated = aggregated.rename(columns={
            'MMSI': 'ship_count',
            'time_window': 'timestamp'
        })
        
        # 记录时间戳
        self.timestamps = sorted(aggregated['timestamp'].unique())
        
        logger.info(f"聚合完成: {len(aggregated)} 条时空记录, 跨越 {len(self.timestamps)} 个时间步")
        return aggregated
    
    def prepare_training_data(self, ais_file_path: str = None) -> Tuple[DataLoader, DataLoader, torch.Tensor, List[str]]:
        """
        准备训练数据
        
        Args:
            ais_file_path: AIS数据文件路径（可选，默认使用配置中的路径）
        
        Returns:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            edge_index: 图的边索引
            h3_indices: H3网格索引列表
        """
        if ais_file_path is None:
            ais_file_path = self.config.AIS_DATA_PATH
            
        # 1. 加载现有的H3网格数据
        logger.info(f"加载H3网格数据: {self.config.H3_GRID_CSV_PATH}")
        h3_df = load_h3_grid_data(self.config.H3_GRID_CSV_PATH)
        self.h3_indices = h3_df['h3_index'].tolist()
        logger.info(f"加载了 {len(self.h3_indices)} 个H3网格")
        
        # 2. 构建图结构
        logger.info("构建H3网格邻接图...")
        self.edge_index = build_h3_adjacency_graph(self.h3_indices)
        
        # 3. 加载和处理AIS数据
        logger.info(f"加载AIS数据: {ais_file_path}")
        ais_df = self.load_ais_data(ais_file_path)
        
        # 4. 聚合到H3网格
        grid_data = self.aggregate_to_h3_grid(ais_df, self.h3_indices)
        
        # 5. 创建时间序列数据
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
        
        # 6. 数据分割
        split_idx = int(len(X) * (1 - self.config.VALIDATION_SPLIT))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"训练集大小: {X_train.shape[0]}个样本, 验证集大小: {X_val.shape[0]}个样本")
        
        # 7. 数据标准化
        logger.info("标准化数据...")
        X_train_scaled, scaler, X_val_scaled, y_train_scaled, y_val_scaled = normalize_data(
            X_train, X_val, y_train, y_val, scaler_type='standard'
        )
        self.scaler = scaler
        
        # 8. 创建数据加载器
        logger.info("创建数据加载器...")
        train_dataset = ShipFlowDataset(X_train_scaled, y_train_scaled, self.edge_index)
        val_dataset = ShipFlowDataset(X_val_scaled, y_val_scaled, self.edge_index)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,  # 避免Windows平台的多进程问题
            pin_memory=True if self.config.DEVICE.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.config.DEVICE.type == 'cuda' else False
        )
        
        logger.info("数据准备完成")
        return train_loader, val_loader, self.edge_index, self.h3_indices
    
    def inverse_transform(self, data: torch.Tensor) -> np.ndarray:
        """
        反向变换缩放的数据
        
        Args:
            data: 缩放后的数据 (torch.Tensor 或 numpy.ndarray)
        
        Returns:
            原始尺度的数据 (numpy.ndarray)
        """
        if self.scaler is None:
            logger.warning("尚未初始化scaler，无法进行反向转换")
            return data.numpy() if isinstance(data, torch.Tensor) else data
            
        # 转换为numpy
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
            
        # 获取原始形状
        original_shape = data.shape
        
        # 重塑为2D进行转换
        data_reshaped = data.reshape(-1, original_shape[-1])
        
        # 反向转换
        data_inverse = self.scaler.inverse_transform(data_reshaped)
        
        # 重塑回原始形状
        return data_inverse.reshape(original_shape)
    
    def get_data_info(self) -> Dict:
        """
        获取数据集信息
        
        Returns:
            包含数据集信息的字典
        """
        return {
            'num_nodes': len(self.h3_indices) if self.h3_indices else 0,
            'num_edges': self.edge_index.shape[1] // 2 if self.edge_index is not None else 0,
            'num_timestamps': len(self.timestamps),
            'h3_resolution': self.config.H3_RESOLUTION,
            'time_interval': self.config.TIME_INTERVAL
        } 