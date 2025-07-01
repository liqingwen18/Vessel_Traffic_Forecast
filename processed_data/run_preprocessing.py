#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速运行AIS数据预处理脚本

使用方法:
python run_preprocessing.py

可选参数:
--input: 输入文件名 (默认: final_ships_in_xunjiang.csv)
--h3_resolution: H3网格分辨率 (默认: 8)
--interval: 重采样间隔 (默认: 5min)
--output: 输出目录 (默认: processed_data)
"""

import argparse
import sys
import os
from data_preprocessing import AISDataPreprocessor

def main():
    parser = argparse.ArgumentParser(description='AIS数据预处理工具')
    
    parser.add_argument('--input', '-i', 
                       default='final_ships_in_xunjiang.csv',
                       help='输入CSV文件名 (默认: final_ships_in_xunjiang.csv)')
    
    parser.add_argument('--h3_resolution', '-r', 
                       type=int, default=8,
                       help='H3网格分辨率 (默认: 8)')
    
    parser.add_argument('--interval', '-t', 
                       default='5min',
                       help='时间重采样间隔 (默认: 5min)')
    
    parser.add_argument('--output', '-o', 
                       default='processed_data',
                       help='输出目录 (默认: processed_data)')
    
    parser.add_argument('--data_path', '-d', 
                       default='data',
                       help='输入数据目录 (默认: data)')
    
    args = parser.parse_args()
    
    try:
        print("AIS数据预处理工具")
        print("=" * 50)
        print(f"输入文件: {args.input}")
        print(f"H3分辨率: {args.h3_resolution}")
        print(f"重采样间隔: {args.interval}")
        print(f"输出目录: {args.output}")
        print("=" * 50)
        
        # 检查输入文件是否存在
        input_file_path = os.path.join(args.data_path, args.input)
        if not os.path.exists(input_file_path):
            print(f"错误: 输入文件不存在 - {input_file_path}")
            sys.exit(1)
        
        # 创建预处理器
        preprocessor = AISDataPreprocessor(
            data_path=args.data_path,
            output_path=args.output
        )
        
        # 执行预处理
        df_raw, df_active, traffic_data = preprocessor.process_all(
            input_filename=args.input,
            h3_resolution=args.h3_resolution,
            resample_interval=args.interval
        )
        
        print("\n✅ 预处理成功完成!")
        print(f"\n📊 数据统计:")
        print(f"   原始记录数: {len(df_raw):,}")
        print(f"   活跃船舶记录数: {len(df_active):,}")
        print(f"   唯一船舶数: {df_active['MMSI'].nunique():,}")
        print(f"   H3网格数: {df_active['h3_id'].nunique():,}")
        print(f"   流量数据点数: {len(traffic_data):,}")
        
        print(f"\n📁 输出文件:")
        print(f"   - {args.output}/active_ships_processed.csv")
        print(f"   - {args.output}/static_ships.csv")
        print(f"   - {args.output}/traffic_data_processed.csv")
        print(f"   - {args.output}/preprocessing_report.txt")
        
    except Exception as e:
        print(f"❌ 处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
