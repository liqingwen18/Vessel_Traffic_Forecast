#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
船舶AIS数据预处理脚本
功能：
1. 重复记录删除（完全重复 + 高频刷屏）
2. 剔除静止船舶
3. 时间重采样 & SG滤波
4. H3网格化处理

"""

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from math import radians, cos, sin, asin, sqrt
import h3
import warnings
import os
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

class AISDataPreprocessor:
    """AIS数据预处理器"""
    
    def __init__(self, data_path='data', output_path='processed_data'):
        """
        初始化预处理器
        
        Args:
            data_path (str): 输入数据路径
            output_path (str): 输出数据路径
        """
        self.data_path = data_path
        self.output_path = output_path
        
        # 创建输出目录
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        计算两点间的球面距离（公里）

        Args:
            lat1, lon1: 第一个点的纬度和经度
            lat2, lon2: 第二个点的纬度和经度

        Returns:
            float: 距离（公里）
        """
        # 转换为弧度
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # haversine公式
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))

        # 地球半径（公里）
        r = 6371
        return c * r

    def haversine_distance_vectorized(self, lat1, lon1, lat2, lon2):
        """
        向量化计算多个点对间的球面距离（公里）

        Args:
            lat1, lon1, lat2, lon2: numpy数组，包含多个点的坐标

        Returns:
            numpy数组: 距离（公里）
        """
        # 转换为弧度
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

        # haversine公式
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        # 地球半径（公里）
        r = 6371
        return c * r
    
    def load_data(self, filename='final_ships_in_xunjiang.csv'):
        """
        加载AIS数据
        
        Args:
            filename (str): 数据文件名
            
        Returns:
            pd.DataFrame: 加载的数据
        """
        file_path = os.path.join(self.data_path, filename)
        print(f"正在加载数据: {file_path}")
        
        df = pd.read_csv(file_path)
        print(f"原始数据形状: {df.shape}")
        
        # 数据类型转换
        df['last_time'] = pd.to_datetime(df['last_time'])
        df['BaseDateTime'] = df['last_time']  # 统一字段名
        
        # 坐标转换（从度分格式转换为十进制度）
        df['lat'] = df['lat'] / 100000  # 根据数据格式调整
        df['lon'] = df['lon'] / 100000
        
        # SOG转换（从0.1节转换为节）
        df['SOG'] = df['sog'] / 100  # 根据数据格式调整
        df['COG'] = df['cog'] / 10   # 根据数据格式调整
        
        # MMSI字段
        df['MMSI'] = df['mmsi']
        
        # 导航状态映射
        nav_status_map = {
            0: 'Under way using engine',
            1: 'At anchor',
            2: 'Not under command',
            3: 'Restricted manoeuvrability',
            4: 'Constrained by her draught',
            5: 'Moored',
            6: 'Aground',
            7: 'Engaged in fishing',
            8: 'Under way sailing',
            15: 'Undefined'
        }
        df['NavStatus'] = df['navi_stat'].map(nav_status_map).fillna('Undefined')
        
        return df
    
    def remove_duplicates(self, df):
        """
        删除重复记录

        Args:
            df (pd.DataFrame): 输入数据

        Returns:
            pd.DataFrame: 去重后的数据
        """
        print("开始删除重复记录...")
        original_count = len(df)

        # 1. 删除完全重复的记录
        df = df.drop_duplicates(subset=['MMSI', 'BaseDateTime', 'lat', 'lon', 'SOG', 'COG'])
        after_exact_dup = len(df)
        print(f"删除完全重复记录: {original_count - after_exact_dup} 条")

        # 2. 删除高频刷屏记录 - 优化版本
        print("处理高频刷屏记录...")
        df = df.sort_values(['MMSI', 'BaseDateTime'])

        # 分批处理以提高效率
        batch_size = 100000
        processed_dfs = []

        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size].copy()
            print(f"处理批次 {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")

            # 计算时间差
            batch_df['time_diff'] = batch_df.groupby('MMSI')['BaseDateTime'].diff().dt.total_seconds()

            # 计算距离差（向量化）
            batch_df['lat_prev'] = batch_df.groupby('MMSI')['lat'].shift(1)
            batch_df['lon_prev'] = batch_df.groupby('MMSI')['lon'].shift(1)

            # 向量化距离计算
            mask_valid = ~(batch_df['lat_prev'].isna() | batch_df['lon_prev'].isna())
            if mask_valid.any():
                batch_df.loc[mask_valid, 'distance_diff'] = self.haversine_distance_vectorized(
                    batch_df.loc[mask_valid, 'lat_prev'].values,
                    batch_df.loc[mask_valid, 'lon_prev'].values,
                    batch_df.loc[mask_valid, 'lat'].values,
                    batch_df.loc[mask_valid, 'lon'].values
                )

            # 标记高频刷屏记录
            high_freq_mask = (batch_df['time_diff'] < 5) & (batch_df['distance_diff'] < 0.01)
            batch_df = batch_df[~high_freq_mask]

            # 清理临时列
            batch_df = batch_df.drop(['time_diff', 'lat_prev', 'lon_prev', 'distance_diff'], axis=1)
            processed_dfs.append(batch_df)

        # 合并所有批次
        df = pd.concat(processed_dfs, ignore_index=True)

        after_high_freq = len(df)
        print(f"删除高频刷屏记录: {after_exact_dup - after_high_freq} 条")
        print(f"去重后数据形状: {df.shape}")

        return df
    
    def remove_static_ships(self, df):
        """
        剔除静止船舶
        
        Args:
            df (pd.DataFrame): 输入数据
            
        Returns:
            pd.DataFrame: 剔除静止船舶后的数据
        """
        print("开始剔除静止船舶...")
        original_count = len(df)
        
        # 基于船速和导航状态判定静止
        static_by_speed = df['SOG'] <= 0.5
        static_by_status = df['NavStatus'].isin(['Moored', 'At anchor', 'Aground'])
        
        df['is_static'] = static_by_speed | static_by_status
        
        # 统计静止船舶数量
        static_count = df['is_static'].sum()
        print(f"静止船舶记录数: {static_count}")
        
        # 保存静止船舶数据（用于后续分析）
        static_ships = df[df['is_static']].copy()
        static_output_path = os.path.join(self.output_path, 'static_ships.csv')
        static_ships.to_csv(static_output_path, index=False, encoding='utf-8-sig')
        print(f"静止船舶数据已保存至: {static_output_path}")
        
        # 返回活跃船舶数据
        df_active = df[~df['is_static']].copy()
        df_active = df_active.drop('is_static', axis=1)
        
        print(f"剔除静止船舶后数据形状: {df_active.shape}")
        return df_active
    
    def add_h3_grid(self, df, resolution=8):
        """
        添加H3网格ID
        
        Args:
            df (pd.DataFrame): 输入数据
            resolution (int): H3网格分辨率
            
        Returns:
            pd.DataFrame: 添加H3网格ID的数据
        """
        print(f"添加H3网格ID (分辨率: {resolution})...")
        
        # 计算H3网格ID
        df['h3_id'] = df.apply(lambda row: h3.geo_to_h3(row['lat'], row['lon'], resolution), axis=1)
        
        print(f"生成的唯一网格数量: {df['h3_id'].nunique()}")
        return df
    
    def time_resampling_and_sg_filter(self, df, interval='5min', window_length=5, polyorder=2):
        """
        时间重采样和SG滤波
        
        Args:
            df (pd.DataFrame): 输入数据
            interval (str): 重采样间隔
            window_length (int): SG滤波窗口长度
            polyorder (int): SG滤波多项式阶数
            
        Returns:
            pd.DataFrame: 处理后的流量数据
        """
        print(f"开始时间重采样 (间隔: {interval}) 和SG滤波...")
        
        # 时间重采样到指定间隔
        df['time_interval'] = df['BaseDateTime'].dt.floor(interval)
        
        # 按时间间隔和网格统计唯一MMSI数量（流量）
        traffic_data = df.groupby(['time_interval', 'h3_id'])['MMSI'].nunique().reset_index()
        traffic_data.rename(columns={'MMSI': 'traffic_count'}, inplace=True)
        
        # 创建完整的时间-网格矩阵
        time_range = pd.date_range(
            start=df['BaseDateTime'].min().floor(interval),
            end=df['BaseDateTime'].max().ceil(interval),
            freq=interval
        )
        
        all_grids = df['h3_id'].unique()
        
        # 创建完整的时间-网格组合
        full_index = pd.MultiIndex.from_product(
            [time_range, all_grids],
            names=['time_interval', 'h3_id']
        )
        
        # 重新索引并填充缺失值
        traffic_matrix = traffic_data.set_index(['time_interval', 'h3_id']).reindex(
            full_index, fill_value=0
        ).reset_index()
        
        # 对每个网格应用SG滤波
        print("应用Savitzky-Golay滤波...")
        
        def apply_sg_filter(group):
            if len(group) >= window_length:
                group['traffic_smoothed'] = savgol_filter(
                    group['traffic_count'], 
                    window_length, 
                    polyorder
                )
            else:
                group['traffic_smoothed'] = group['traffic_count']
            return group
        
        traffic_matrix = traffic_matrix.groupby('h3_id').apply(apply_sg_filter).reset_index(drop=True)
        
        print(f"重采样后数据形状: {traffic_matrix.shape}")
        return traffic_matrix
    
    def process_all(self, input_filename='final_ships_in_xunjiang.csv', 
                   h3_resolution=8, resample_interval='5min'):
        """
        执行完整的数据预处理流程
        
        Args:
            input_filename (str): 输入文件名
            h3_resolution (int): H3网格分辨率
            resample_interval (str): 重采样间隔
            
        Returns:
            tuple: (原始数据, 预处理后的活跃船舶数据, 流量数据)
        """
        print("=" * 60)
        print("开始AIS数据预处理流程")
        print("=" * 60)
        
        # 1. 加载数据
        df_raw = self.load_data(input_filename)
        
        # 2. 删除重复记录
        df_dedup = self.remove_duplicates(df_raw)
        
        # 3. 剔除静止船舶
        df_active = self.remove_static_ships(df_dedup)
        
        # 4. 添加H3网格
        df_with_grid = self.add_h3_grid(df_active, h3_resolution)
        
        # 5. 时间重采样和SG滤波
        traffic_data = self.time_resampling_and_sg_filter(df_with_grid, resample_interval)
        
        # 6. 保存结果
        print("保存处理结果...")
        
        # 保存预处理后的活跃船舶数据
        active_output_path = os.path.join(self.output_path, 'active_ships_processed.csv')
        df_with_grid.to_csv(active_output_path, index=False, encoding='utf-8-sig')
        print(f"活跃船舶数据已保存至: {active_output_path}")
        
        # 保存流量数据
        traffic_output_path = os.path.join(self.output_path, 'traffic_data_processed.csv')
        traffic_data.to_csv(traffic_output_path, index=False, encoding='utf-8-sig')
        print(f"流量数据已保存至: {traffic_output_path}")
        
        # 7. 生成处理报告
        self.generate_report(df_raw, df_active, traffic_data)
        
        print("=" * 60)
        print("数据预处理完成！")
        print("=" * 60)
        
        return df_raw, df_with_grid, traffic_data
    
    def generate_report(self, df_raw, df_active, traffic_data):
        """
        生成处理报告
        
        Args:
            df_raw (pd.DataFrame): 原始数据
            df_active (pd.DataFrame): 活跃船舶数据
            traffic_data (pd.DataFrame): 流量数据
        """
        report_path = os.path.join(self.output_path, 'preprocessing_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("AIS数据预处理报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("数据统计:\n")
            f.write(f"原始记录数: {len(df_raw):,}\n")
            f.write(f"活跃船舶记录数: {len(df_active):,}\n")
            f.write(f"数据保留率: {len(df_active)/len(df_raw)*100:.2f}%\n\n")
            
            f.write(f"唯一船舶数 (MMSI): {df_active['MMSI'].nunique():,}\n")
            f.write(f"H3网格数量: {df_active['h3_id'].nunique():,}\n")
            f.write(f"时间跨度: {df_active['BaseDateTime'].min()} 至 {df_active['BaseDateTime'].max()}\n\n")
            
            f.write("流量数据统计:\n")
            f.write(f"时间点数量: {traffic_data['time_interval'].nunique():,}\n")
            f.write(f"平均流量: {traffic_data['traffic_count'].mean():.2f}\n")
            f.write(f"最大流量: {traffic_data['traffic_count'].max()}\n")
            f.write(f"流量标准差: {traffic_data['traffic_count'].std():.2f}\n")
        
        print(f"处理报告已保存至: {report_path}")


if __name__ == "__main__":
    # 创建预处理器实例
    preprocessor = AISDataPreprocessor()
    
    # 执行完整预处理流程
    df_raw, df_active, traffic_data = preprocessor.process_all()
    
    print("\n处理完成！输出文件:")
    print("- processed_data/active_ships_processed.csv: 预处理后的活跃船舶数据")
    print("- processed_data/static_ships.csv: 静止船舶数据")
    print("- processed_data/traffic_data_processed.csv: 流量数据")
    print("- processed_data/preprocessing_report.txt: 处理报告")
