#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速AIS数据预处理脚本（优化版）
专门用于处理大数据集

功能：
1. 重复记录删除
2. 剔除静止船舶
3. 基础统计分析

作者：Augment AI Assistant
日期：2025-06-22
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class FastAISPreprocessor:
    """快速AIS数据预处理器"""
    
    def __init__(self, data_path='data', output_path='processed_data'):
        self.data_path = data_path
        self.output_path = output_path
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    
    def load_data(self, filename='final_ships_in_xunjiang.csv', sample_frac=None):
        """
        加载AIS数据
        
        Args:
            filename (str): 数据文件名
            sample_frac (float): 采样比例，None表示加载全部数据
            
        Returns:
            pd.DataFrame: 加载的数据
        """
        file_path = os.path.join(self.data_path, filename)
        print(f"正在加载数据: {file_path}")
        
        # 分块读取大文件
        chunk_size = 50000
        chunks = []
        
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            if sample_frac and sample_frac < 1.0:
                chunk = chunk.sample(frac=sample_frac)
            chunks.append(chunk)
            print(f"已加载 {len(chunks) * chunk_size} 条记录...")
        
        df = pd.concat(chunks, ignore_index=True)
        print(f"数据加载完成，形状: {df.shape}")
        
        # 数据类型转换
        df['BaseDateTime'] = pd.to_datetime(df['last_time'])

        # 坐标转换：从度分格式转换为十进制度
        # 原始格式：DDMM.MMMM (如2336.8209表示23度36.8209分)
        # 转换为：DD.DDDDDD (23 + 36.8209/60 = 23.613682度)
        df['lat_decimal'] = (df['lat'] // 100) + ((df['lat'] % 100) / 60)
        df['lon_decimal'] = (df['lon'] // 100) + ((df['lon'] % 100) / 60)
        df['lat'] = df['lat_decimal'] / 100  # 还需要除以100
        df['lon'] = df['lon_decimal'] / 100

        df['SOG'] = df['sog'] / 100  # 0.1节转换为节
        df['COG'] = df['cog'] / 10   # 0.1度转换为度
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
    
    def remove_exact_duplicates(self, df):
        """
        删除完全重复的记录
        
        Args:
            df (pd.DataFrame): 输入数据
            
        Returns:
            pd.DataFrame: 去重后的数据
        """
        print("删除完全重复记录...")
        original_count = len(df)
        
        df = df.drop_duplicates(subset=['MMSI', 'BaseDateTime', 'lat', 'lon', 'SOG', 'COG'])
        
        removed_count = original_count - len(df)
        print(f"删除完全重复记录: {removed_count:,} 条")
        
        return df
    
    def remove_static_ships(self, df):
        """
        剔除静止船舶
        
        Args:
            df (pd.DataFrame): 输入数据
            
        Returns:
            tuple: (活跃船舶数据, 静止船舶数据)
        """
        print("剔除静止船舶...")
        
        # 基于船速和导航状态判定静止
        static_by_speed = df['SOG'] <= 0.5
        static_by_status = df['NavStatus'].isin(['Moored', 'At anchor', 'Aground'])
        
        df['is_static'] = static_by_speed | static_by_status
        
        static_count = df['is_static'].sum()
        print(f"静止船舶记录数: {static_count:,}")
        
        # 分离静止和活跃船舶
        static_ships = df[df['is_static']].copy()
        active_ships = df[~df['is_static']].copy()
        
        # 清理标记列
        static_ships = static_ships.drop('is_static', axis=1)
        active_ships = active_ships.drop('is_static', axis=1)
        
        print(f"活跃船舶记录数: {len(active_ships):,}")
        
        return active_ships, static_ships
    
    def basic_statistics(self, df_raw, df_active, df_static):
        """
        生成基础统计信息
        
        Args:
            df_raw: 原始数据
            df_active: 活跃船舶数据
            df_static: 静止船舶数据
        """
        stats = {
            '处理时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '原始记录数': len(df_raw),
            '活跃船舶记录数': len(df_active),
            '静止船舶记录数': len(df_static),
            '数据保留率': f"{len(df_active)/len(df_raw)*100:.2f}%",
            '唯一船舶数(MMSI)': df_active['MMSI'].nunique(),
            '时间跨度开始': df_active['BaseDateTime'].min(),
            '时间跨度结束': df_active['BaseDateTime'].max(),
            '平均船速': f"{df_active['SOG'].mean():.2f} 节",
            '最大船速': f"{df_active['SOG'].max():.2f} 节",
            '纬度范围': f"{df_active['lat'].min():.6f} ~ {df_active['lat'].max():.6f}",
            '经度范围': f"{df_active['lon'].min():.6f} ~ {df_active['lon'].max():.6f}"
        }
        
        return stats
    
    def save_results(self, df_active, df_static, stats):
        """
        保存处理结果
        
        Args:
            df_active: 活跃船舶数据
            df_static: 静止船舶数据
            stats: 统计信息
        """
        print("保存处理结果...")
        
        # 保存活跃船舶数据
        active_path = os.path.join(self.output_path, 'active_ships_cleaned.csv')
        df_active.to_csv(active_path, index=False, encoding='utf-8-sig')
        print(f"活跃船舶数据已保存: {active_path}")
        
        # 保存静止船舶数据
        static_path = os.path.join(self.output_path, 'static_ships_cleaned.csv')
        df_static.to_csv(static_path, index=False, encoding='utf-8-sig')
        print(f"静止船舶数据已保存: {static_path}")
        
        # 保存统计报告
        report_path = os.path.join(self.output_path, 'cleaning_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("AIS数据清洗报告\n")
            f.write("=" * 50 + "\n\n")
            
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        
        print(f"处理报告已保存: {report_path}")
    
    def process_fast(self, input_filename='final_ships_in_xunjiang.csv', sample_frac=None):
        """
        执行快速预处理流程
        
        Args:
            input_filename (str): 输入文件名
            sample_frac (float): 采样比例，用于测试
            
        Returns:
            tuple: (活跃船舶数据, 静止船舶数据, 统计信息)
        """
        print("=" * 60)
        print("开始快速AIS数据预处理")
        print("=" * 60)
        
        # 1. 加载数据
        df_raw = self.load_data(input_filename, sample_frac)
        
        # 2. 删除完全重复记录
        df_dedup = self.remove_exact_duplicates(df_raw)
        
        # 3. 剔除静止船舶
        df_active, df_static = self.remove_static_ships(df_dedup)
        
        # 4. 生成统计信息
        stats = self.basic_statistics(df_raw, df_active, df_static)
        
        # 5. 保存结果
        self.save_results(df_active, df_static, stats)
        
        # 6. 打印摘要
        print("\n" + "=" * 60)
        print("处理完成！")
        print("=" * 60)
        print(f"原始记录数: {len(df_raw):,}")
        print(f"活跃船舶记录数: {len(df_active):,}")
        print(f"静止船舶记录数: {len(df_static):,}")
        print(f"唯一船舶数: {df_active['MMSI'].nunique():,}")
        print(f"数据保留率: {len(df_active)/len(df_raw)*100:.2f}%")
        
        return df_active, df_static, stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='快速AIS数据预处理工具')
    parser.add_argument('--input', '-i', default='final_ships_in_xunjiang.csv',
                       help='输入CSV文件名')
    parser.add_argument('--sample', '-s', type=float, default=None,
                       help='采样比例 (0.0-1.0)，用于测试')
    parser.add_argument('--output', '-o', default='processed_data',
                       help='输出目录')
    
    args = parser.parse_args()
    
    try:
        processor = FastAISPreprocessor(output_path=args.output)
        df_active, df_static, stats = processor.process_fast(
            input_filename=args.input,
            sample_frac=args.sample
        )
        
        print("\n✅ 处理成功完成!")
        
    except Exception as e:
        print(f"❌ 处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
