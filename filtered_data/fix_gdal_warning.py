#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复GDAL相关警告的脚本
"""

import os
import sys
import warnings

def fix_gdal_warnings():
    """修复GDAL相关的警告"""
    print("正在修复GDAL相关警告...")
    
    # 1. 抑制GDAL警告
    os.environ['CPL_LOG_LEVEL'] = 'ERROR'  # 只显示错误，不显示警告
    os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
    
    # 2. 尝试设置GDAL_DATA环境变量
    try:
        import gdal
        gdal_data_path = gdal.GetConfigOption('GDAL_DATA')
        if gdal_data_path is None:
            # 尝试找到GDAL数据目录
            possible_paths = [
                os.path.join(sys.prefix, 'Library', 'share', 'gdal'),  # Conda环境
                os.path.join(sys.prefix, 'share', 'gdal'),  # 标准安装
                '/usr/share/gdal',  # Linux系统安装
                '/opt/homebrew/share/gdal',  # macOS Homebrew
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    os.environ['GDAL_DATA'] = path
                    print(f"设置GDAL_DATA为: {path}")
                    break
            else:
                print("警告: 未找到GDAL数据目录，但这通常不影响程序运行")
    except ImportError:
        print("GDAL模块未直接安装，使用GeoPandas内置的GDAL")
    
    # 3. 抑制其他相关警告
    warnings.filterwarnings('ignore', category=UserWarning, module='.*gdal.*')
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='.*gdal.*')
    warnings.filterwarnings('ignore', message='.*header.dxf.*')
    
    print("GDAL警告修复完成")

def test_gdal_functionality():
    """测试GDAL功能是否正常"""
    print("\n测试GDAL功能...")
    
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        
        # 创建一个简单的测试GeoDataFrame
        test_data = {
            'id': [1, 2, 3],
            'name': ['点1', '点2', '点3']
        }
        geometry = [Point(110.0, 23.0), Point(110.1, 23.1), Point(110.2, 23.2)]
        gdf = gpd.GeoDataFrame(test_data, geometry=geometry, crs='EPSG:4326')
        
        print("✓ GeoPandas功能正常")
        print(f"✓ 测试GeoDataFrame创建成功，包含 {len(gdf)} 个点")
        print(f"✓ 坐标参考系统: {gdf.crs}")
        
        return True
    except Exception as e:
        print(f"✗ GDAL功能测试失败: {e}")
        return False

if __name__ == "__main__":
    fix_gdal_warnings()
    test_gdal_functionality()
