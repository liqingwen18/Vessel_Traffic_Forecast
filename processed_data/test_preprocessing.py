#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据预处理功能
"""

from data_preprocessing_fast import FastAISPreprocessor
import pandas as pd

def test_small_sample():
    """测试小样本数据处理"""
    print("测试小样本数据处理...")
    
    processor = FastAISPreprocessor(output_path='test_output')
    
    # 使用0.1%的数据进行测试
    df_active, df_static, stats = processor.process_fast(
        input_filename='final_ships_in_xunjiang.csv',
        sample_frac=0.001  # 0.1%的数据
    )
    
    print("\n测试结果:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    return df_active, df_static, stats

def analyze_data_quality(df_active):
    """分析数据质量"""
    print("\n数据质量分析:")
    print(f"数据形状: {df_active.shape}")
    print(f"时间范围: {df_active['BaseDateTime'].min()} 到 {df_active['BaseDateTime'].max()}")
    print(f"坐标范围: 纬度 {df_active['lat'].min():.6f} ~ {df_active['lat'].max():.6f}")
    print(f"坐标范围: 经度 {df_active['lon'].min():.6f} ~ {df_active['lon'].max():.6f}")
    print(f"船速范围: {df_active['SOG'].min():.2f} ~ {df_active['SOG'].max():.2f} 节")
    print(f"航向范围: {df_active['COG'].min():.2f} ~ {df_active['COG'].max():.2f} 度")
    
    # 检查异常值
    high_speed = df_active[df_active['SOG'] > 50]  # 超过50节的记录
    if len(high_speed) > 0:
        print(f"\n⚠️  发现 {len(high_speed)} 条高速记录 (>50节):")
        print(high_speed[['MMSI', 'BaseDateTime', 'SOG', 'lat', 'lon']].head())
    
    # 检查坐标异常
    if df_active['lat'].min() < 0 or df_active['lat'].max() > 90:
        print("⚠️  纬度数据可能存在格式问题")
    
    if df_active['lon'].min() < 0 or df_active['lon'].max() > 180:
        print("⚠️  经度数据可能存在格式问题")

def create_time_series_sample(df_active):
    """创建时间序列样本"""
    print("\n创建时间序列样本...")
    
    # 按小时统计船舶数量
    df_active['hour'] = df_active['BaseDateTime'].dt.floor('H')
    hourly_traffic = df_active.groupby('hour')['MMSI'].nunique().reset_index()
    hourly_traffic.columns = ['hour', 'unique_ships']
    
    print(f"时间序列数据点数: {len(hourly_traffic)}")
    print("前5个时间点的流量:")
    print(hourly_traffic.head())
    
    # 保存时间序列数据
    hourly_traffic.to_csv('test_output/hourly_traffic_sample.csv', index=False)
    print("时间序列样本已保存: test_output/hourly_traffic_sample.csv")
    
    return hourly_traffic

if __name__ == "__main__":
    try:
        # 1. 测试小样本处理
        df_active, df_static, stats = test_small_sample()
        
        # 2. 分析数据质量
        analyze_data_quality(df_active)
        
        # 3. 创建时间序列样本
        hourly_traffic = create_time_series_sample(df_active)
        
        print("\n✅ 测试完成！")
        print("\n生成的文件:")
        print("- test_output/active_ships_cleaned.csv")
        print("- test_output/static_ships_cleaned.csv") 
        print("- test_output/cleaning_report.txt")
        print("- test_output/hourly_traffic_sample.csv")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
