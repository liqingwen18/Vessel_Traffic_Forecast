"""
创建演示数据集
生成一个小型的合成数据集用于测试模型训练功能
"""

import os
import sys
import pandas as pd
import numpy as np
import h3
import logging
from datetime import datetime, timedelta
import random
from typing import List, Tuple

# 将当前工作目录设置为项目根目录
if os.path.basename(os.getcwd()) == 'prediction':
    os.chdir('..')
    sys.path.append('prediction')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("create_demo_dataset")

def load_h3_grid() -> List[str]:
    """
    加载H3网格索引
    """
    h3_file = "h3_grid_analysis_res7/h3_grid_coverage.csv"
    
    if not os.path.exists(h3_file):
        logger.error(f"H3网格文件不存在: {h3_file}")
        return []
    
    df = pd.read_csv(h3_file)
    logger.info(f"加载了 {len(df)} 个H3网格")
    
    return df['h3_index'].tolist()

def generate_synthetic_data(h3_indices: List[str], days: int = 7, time_interval_minutes: int = 5) -> pd.DataFrame:
    """
    生成合成的AIS数据
    
    Args:
        h3_indices: H3网格索引列表
        days: 生成多少天的数据
        time_interval_minutes: 时间间隔（分钟）
        
    Returns:
        合成的AIS数据DataFrame
    """
    logger.info(f"生成 {days} 天的合成数据，时间间隔 {time_interval_minutes} 分钟")
    
    # 创建时间戳序列
    start_date = datetime(2024, 5, 1)
    timestamps = []
    
    for day in range(days):
        current_date = start_date + timedelta(days=day)
        for hour in range(24):
            for minute in range(0, 60, time_interval_minutes):
                timestamps.append(current_date.replace(hour=hour, minute=minute))
    
    total_timestamps = len(timestamps)
    logger.info(f"生成了 {total_timestamps} 个时间戳")
    
    # 为每个网格生成随机船舶数量
    rows = []
    
    for h3_idx in h3_indices:
        # 获取该H3单元的中心点经纬度（用于生成更真实的数据）
        lat, lon = h3.h3_to_geo(h3_idx)
        
        for ts in timestamps:
            # 生成船舶数量（高峰时段有更多船舶）
            base_count = random.randint(0, 20)  # 基础船舶数量
            
            # 添加一些时间模式
            hour_factor = 1.0
            if 6 <= ts.hour < 9:  # 早高峰
                hour_factor = 1.5
            elif 17 <= ts.hour < 20:  # 晚高峰
                hour_factor = 1.8
            elif 0 <= ts.hour < 4:  # 深夜
                hour_factor = 0.3
                
            # 添加一些空间模式
            space_factor = 0.5 + random.random()  # 0.5-1.5之间的随机系数
            
            # 生成最终船舶数量（保证非负整数）
            ship_count = max(0, int(base_count * hour_factor * space_factor))
            
            # 创建一条记录
            row = {
                'MMSI': random.randint(100000000, 999999999),  # 随机MMSI
                'LAT': lat,
                'LON': lon,
                'BaseDateTime': ts,
                'h3_index': h3_idx,
                'ship_count': ship_count
            }
            rows.append(row)
            
    # 创建DataFrame
    df = pd.DataFrame(rows)
    logger.info(f"生成了 {len(df)} 条记录")
    
    return df

def split_and_save_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    分割数据为原始AIS数据和聚合后的网格数据，并保存
    
    Args:
        df: 合成数据DataFrame
    
    Returns:
        元组 (ais_df, grid_df)
    """
    # 原始AIS数据 - 每条记录代表一个船舶信号
    ais_df = df[['MMSI', 'LAT', 'LON', 'BaseDateTime']].copy()
    
    # 网格聚合数据 - 每条记录代表一个时间点一个网格的船舶数量
    grid_df = df[['h3_index', 'BaseDateTime', 'ship_count']].rename(columns={'BaseDateTime': 'timestamp'})
    
    # 保存数据
    demo_dir = "prediction/demo_data"
    os.makedirs(demo_dir, exist_ok=True)
    
    ais_path = os.path.join(demo_dir, "demo_ais_data.csv")
    grid_path = os.path.join(demo_dir, "demo_grid_data.csv")
    
    ais_df.to_csv(ais_path, index=False)
    grid_df.to_csv(grid_path, index=False)
    
    logger.info(f"保存AIS数据到: {ais_path}")
    logger.info(f"保存网格数据到: {grid_path}")
    
    return ais_df, grid_df

def create_h3_adjacency_file(h3_indices: List[str]):
    """
    创建H3邻接关系文件
    """
    from utils import build_h3_adjacency_graph
    
    demo_dir = "prediction/demo_data"
    os.makedirs(demo_dir, exist_ok=True)
    
    # 构建邻接关系
    edge_index = build_h3_adjacency_graph(h3_indices)
    
    # 将边索引转换为可读的CSV格式
    edges = []
    for i in range(edge_index.shape[1]):
        source = int(edge_index[0, i])
        target = int(edge_index[1, i])
        edges.append({
            'source_idx': source,
            'target_idx': target,
            'source_h3': h3_indices[source],
            'target_h3': h3_indices[target]
        })
    
    edges_df = pd.DataFrame(edges)
    edges_path = os.path.join(demo_dir, "demo_h3_adjacency.csv")
    edges_df.to_csv(edges_path, index=False)
    
    logger.info(f"创建了 {len(edges)} 条边的邻接关系，保存到: {edges_path}")

def main():
    """主函数"""
    logger.info("开始创建演示数据集")
    
    # 加载H3网格索引
    h3_indices = load_h3_grid()
    if not h3_indices:
        logger.error("无法加载H3网格索引，退出")
        return 1
    
    # 生成合成数据
    df = generate_synthetic_data(h3_indices)
    
    # 分割和保存数据
    ais_df, grid_df = split_and_save_data(df)
    
    # 创建邻接关系
    create_h3_adjacency_file(h3_indices)
    
    logger.info("演示数据集创建完成")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 