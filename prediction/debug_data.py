"""
数据调试脚本
检查AIS数据加载和H3网格映射问题
"""

import os
import sys
import pandas as pd
import logging
import numpy as np

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

logger = logging.getLogger("debug_data")

def inspect_ais_file():
    """检查AIS数据文件"""
    file_path = "data/active_ships_cleaned.csv"
    
    logger.info(f"检查AIS数据文件: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return
    
    logger.info(f"文件大小: {os.path.getsize(file_path) / (1024 * 1024):.2f} MB")
    
    try:
        # 尝试仅加载前几行了解结构
        logger.info("尝试读取前5行数据:")
        df_head = pd.read_csv(file_path, nrows=5)
        logger.info(f"列名: {df_head.columns.tolist()}")
        logger.info(f"数据样例:\n{df_head}")
        
        # 尝试计算总行数
        logger.info("计算总行数...")
        with open(file_path, 'r', encoding='utf-8') as f:
            # 跳过标题行
            next(f)
            line_count = sum(1 for _ in f)
        logger.info(f"文件总行数: {line_count}")
    except Exception as e:
        logger.error(f"读取文件出错: {str(e)}")

def check_h3_grid():
    """检查H3网格文件"""
    file_path = "h3_grid_analysis_res7/h3_grid_coverage.csv"
    
    logger.info(f"检查H3网格文件: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"H3网格总数: {len(df)}")
        logger.info(f"列名: {df.columns.tolist()}")
        logger.info(f"数据样例:\n{df.head()}")
        
        # 检查是否有h3_index列
        if 'h3_index' not in df.columns:
            logger.error("没有找到h3_index列")
            # 尝试查找可能的替代列
            possible_columns = [col for col in df.columns if 'h3' in col.lower() or 'index' in col.lower()]
            if possible_columns:
                logger.info(f"可能的替代列: {possible_columns}")
    except Exception as e:
        logger.error(f"读取H3网格文件出错: {str(e)}")

def debug_data_mapping():
    """调试数据到网格的映射"""
    from h3_compat import latlng_to_h3_compat
    
    ais_file = "data/active_ships_cleaned.csv"
    h3_file = "h3_grid_analysis_res7/h3_grid_coverage.csv"
    
    logger.info("调试数据到网格的映射")
    
    try:
        # 加载H3网格文件
        h3_df = pd.read_csv(h3_file)
        h3_indices = h3_df['h3_index'].tolist() if 'h3_index' in h3_df.columns else []
        
        logger.info(f"加载了 {len(h3_indices)} 个H3网格")
        
        # 加载小样本AIS数据
        ais_df = pd.read_csv(ais_file, nrows=1000)
        logger.info(f"加载了 {len(ais_df)} 条AIS记录")
        
        # 检查列名
        logger.info(f"AIS数据列名: {ais_df.columns.tolist()}")
        
        # 识别经纬度列
        lat_col = 'lat_decimal'
        lon_col = 'lon_decimal'
        
        # 检查坐标范围
        lat_min = ais_df[lat_col].min()
        lat_max = ais_df[lat_col].max()
        lon_min = ais_df[lon_col].min()
        lon_max = ais_df[lon_col].max()
        
        logger.info(f"纬度范围: {lat_min} 到 {lat_max}")
        logger.info(f"经度范围: {lon_min} 到 {lon_max}")
        
        # 进行坐标转换
        logger.info("执行坐标转换...")
        
        # 简单的缩放方法 - 实际应该使用更精确的投影转换
        # 中国经纬度大致在东经73-135度，北纬4-54度
        # 尝试不同的缩放因子
        scale_factors = [0.00001, 0.0001, 0.001]
        
        for scale in scale_factors:
            # 创建临时经纬度列
            ais_df['lat_scaled'] = (ais_df[lat_col] - 4000000) * scale
            ais_df['lon_scaled'] = ais_df[lon_col] * scale
            
            logger.info(f"尝试缩放因子 {scale}: 纬度 [{ais_df['lat_scaled'].min():.4f}, {ais_df['lat_scaled'].max():.4f}], "
                       f"经度 [{ais_df['lon_scaled'].min():.4f}, {ais_df['lon_scaled'].max():.4f}]")
        
        # 使用对比航海图或其他参考数据调整缩放因子
        # 选择最佳的缩放因子应该使得坐标落在合理的地理区域内
        best_scale = 0.0001  # 基于上面的测试选择最佳缩放因子
        ais_df['LAT'] = (ais_df[lat_col] - 4000000) * best_scale  # 假设有一个北偏移量
        ais_df['LON'] = ais_df[lon_col] * best_scale
        
        logger.info(f"使用缩放因子 {best_scale} 后: 纬度 [{ais_df['LAT'].min():.4f}, {ais_df['LAT'].max():.4f}], "
                   f"经度 [{ais_df['LON'].min():.4f}, {ais_df['LON'].max():.4f}]")
        
        # 为每个点分配H3网格索引
        h3_resolution = 7  # 与模型配置相同
        
        ais_df['h3_index'] = ais_df.apply(
            lambda row: latlng_to_h3_compat(row['LAT'], row['LON'], h3_resolution),
            axis=1
        )
        
        # 检查有多少点落在指定网格内
        h3_set = set(h3_indices)  # 转换为集合以提高查找效率
        in_grid = ais_df[ais_df['h3_index'].isin(h3_set)]
        
        logger.info(f"在指定H3网格内的记录数: {len(in_grid)} ({len(in_grid)/len(ais_df)*100:.2f}%)")
        
        if len(in_grid) == 0:
            logger.error("没有记录落在指定网格内!")
            
            # 检查几个样本点的H3索引
            sample_indices = ais_df['h3_index'].unique()[:5].tolist()
            logger.info(f"AIS数据中的部分H3索引: {sample_indices}")
            
            # 检查网格索引的字符串格式
            logger.info(f"H3网格中索引示例: {h3_indices[:3]}")
            logger.info(f"AIS数据中索引示例: {sample_indices[:3]}")
            
            # 检查是否有任何交集
            ais_indices_set = set(ais_df['h3_index'].unique())
            common_indices = h3_set.intersection(ais_indices_set)
            logger.info(f"H3网格和AIS数据共同的索引数量: {len(common_indices)}")
            
            if len(common_indices) > 0:
                logger.info(f"共同索引示例: {list(common_indices)[:3]}")
        else:
            # 显示一些匹配上的记录
            logger.info("匹配上的部分记录示例:")
            logger.info(in_grid.head())
    except Exception as e:
        logger.error(f"调试数据映射时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logger.info("开始数据调试...")
    
    # 检查文件位置和大小
    inspect_ais_file()
    
    # 检查H3网格文件
    check_h3_grid()
    
    # 调试数据映射
    debug_data_mapping()
    
    logger.info("数据调试完成") 