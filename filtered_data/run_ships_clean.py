#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清洁版船舶数据筛选程序启动脚本
完全抑制GDAL警告
"""

import os
import sys
import subprocess
import tempfile
from datetime import datetime

def create_clean_filter_script():
    """创建一个清洁版本的筛选脚本"""
    
    script_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 在导入任何地理库之前设置环境
import os
import sys
import warnings

# 完全抑制所有输出到stderr的警告
class DevNull:
    def write(self, msg): pass
    def flush(self): pass

# 保存原始stderr
original_stderr = sys.stderr

# 临时重定向stderr
sys.stderr = DevNull()

# 设置环境变量
os.environ.update({
    'CPL_LOG_LEVEL': 'ERROR',
    'GDAL_DISABLE_READDIR_ON_OPEN': 'TRUE',
    'GDAL_PAM_ENABLED': 'NO',
    'VSI_CACHE': 'FALSE',
    'GDAL_DATA': '',
    'PROJ_LIB': ''
})

# 抑制所有警告
warnings.filterwarnings('ignore')

try:
    # 导入所需库
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime
    import matplotlib.font_manager as fm
    
    # 恢复stderr用于正常输出
    sys.stderr = original_stderr
    
    # 配置matplotlib支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    def read_xunjiang_boundary():
        """读取浔江水域边界数据"""
        # 临时抑制输出
        temp_stderr = sys.stderr
        sys.stderr = DevNull()
        
        try:
            xunjiang_area = gpd.read_file('浔江数据/浔江_面数据.shp')
            xunjiang_line = gpd.read_file('浔江数据/浔江_线数据.shp')
            xunjiang_boundary = gpd.read_file('浔江数据/浔江_水面边界线.shp')
        finally:
            sys.stderr = temp_stderr
        
        print("浔江面数据CRS:", xunjiang_area.crs)
        
        # 获取浔江区域的坐标范围
        x_min, y_min, x_max, y_max = xunjiang_area.total_bounds
        print(f"浔江区域坐标范围: 经度({x_min}, {x_max}), 纬度({y_min}, {y_max})")
        
        return xunjiang_area, x_min, y_min, x_max, y_max

    def read_ship_data(file_path, chunksize=100000):
        """读取船舶数据(CSV格式)，按块处理大文件"""
        try:
            return pd.read_csv(
                file_path, 
                chunksize=chunksize, 
                encoding='utf-8',
                quotechar="'",
                escapechar='\\\\',
                on_bad_lines='skip',
                low_memory=False
            )
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            return None

    def filter_ships_in_xunjiang(ship_data_chunk, xunjiang_area, lon_col='lon', lat_col='lat', coord_factor=1000000):
        """筛选出在浔江区域内的船舶数据，只返回原始数据，不添加任何新列"""
        if lon_col not in ship_data_chunk.columns or lat_col not in ship_data_chunk.columns:
            print(f"缺少必要的列: {lon_col} 或 {lat_col}")
            return pd.DataFrame()
        
        ship_data_chunk = ship_data_chunk.dropna(subset=[lon_col, lat_col])
        
        try:
            geometry = [Point(lon/coord_factor, lat/coord_factor) for lon, lat in zip(ship_data_chunk[lon_col], ship_data_chunk[lat_col])]
            gdf = gpd.GeoDataFrame(ship_data_chunk, geometry=geometry, crs=xunjiang_area.crs)
            
            within_mask = gdf.geometry.apply(lambda point: xunjiang_area.geometry.contains(point).any())
            ships_in_xunjiang = ship_data_chunk[within_mask]
            return ships_in_xunjiang
        except Exception as e:
            print(f"空间筛选时发生错误: {e}")
            return pd.DataFrame()

    def export_results(ships_data, output_file='final_ships_in_xunjiang.csv', excel_file='final_ships_in_xunjiang.xlsx'):
        """导出筛选结果为CSV和Excel格式"""
        ships_data.to_csv(output_file, index=False, encoding='utf-8')
        print(f"已保存筛选结果至CSV文件: {output_file}")
        
        try:
            if len(ships_data) <= 1000000:
                ships_data.to_excel(excel_file, index=False)
                print(f"已保存筛选结果至Excel文件: {excel_file}")
            else:
                ships_data.head(10000).to_excel('sample_' + excel_file, index=False)
                print(f"数据太大，已保存10000行样本至Excel文件: sample_{excel_file}")
        except Exception as e:
            print(f"保存Excel文件时出错: {e}")

    def main():
        start_time = datetime.now()
        print(f"开始处理时间: {start_time}")
        
        print("读取浔江边界数据...")
        xunjiang_area, x_min, y_min, x_max, y_max = read_xunjiang_boundary()
        
        ship_data_files = [os.path.join('data', f) for f in os.listdir('data') if f.startswith('shipinfo_data_')]
        print(f"找到 {len(ship_data_files)} 个船舶数据文件")
        
        all_ships_in_xunjiang = pd.DataFrame()
        
        for file_path in ship_data_files:
            print(f"\\n处理文件: {file_path}")
            file_start_time = datetime.now()
            
            total_rows = 0
            file_found_count = 0
            
            try:
                for i, chunk in enumerate(read_ship_data(file_path)):
                    chunk_size = len(chunk)
                    total_rows += chunk_size
                    
                    ships_in_area = filter_ships_in_xunjiang(chunk, xunjiang_area, coord_factor=1000000)
                    
                    if len(ships_in_area) > 0:
                        all_ships_in_xunjiang = pd.concat([all_ships_in_xunjiang, ships_in_area])
                        file_found_count += len(ships_in_area)
                    
                    if (i+1) % 10 == 0:
                        elapsed = (datetime.now() - file_start_time).total_seconds()
                        print(f"已处理 {i+1} 个块, 约 {total_rows} 条记录, 找到 {file_found_count} 条符合条件的记录 (用时: {elapsed:.1f}秒)")
                
                file_elapsed = (datetime.now() - file_start_time).total_seconds()
                print(f"文件 {file_path} 处理完成, 共处理 {total_rows} 条记录, 找到 {file_found_count} 条符合条件的记录 (用时: {file_elapsed:.1f}秒)")
            except Exception as e:
                print(f"处理文件 {file_path} 时发生错误: {e}")
        
        if not all_ships_in_xunjiang.empty:
            total_records = len(all_ships_in_xunjiang)
            unique_ships = all_ships_in_xunjiang['mmsi'].nunique()
            
            print(f"\\n筛选完成, 共找到 {total_records} 条符合条件的记录, {unique_ships} 艘唯一船舶")
            export_results(all_ships_in_xunjiang)
        else:
            print("\\n未找到浔江区域内的船舶数据")
        
        end_time = datetime.now()
        total_elapsed = (end_time - start_time).total_seconds()
        print(f"\\n处理结束时间: {end_time}")
        print(f"总用时: {total_elapsed:.1f}秒 ({total_elapsed/60:.1f}分钟)")

    if __name__ == "__main__":
        main()

except Exception as e:
    # 恢复stderr以显示错误
    sys.stderr = original_stderr
    print(f"程序运行出错: {e}")
    import traceback
    traceback.print_exc()
'''
    
    return script_content

def main():
    """主函数"""
    print("船舶数据筛选程序 - 清洁版")
    print("="*40)
    print(f"启动时间: {datetime.now()}")
    print()
    
    # 检查必要文件
    required_dirs = ['浔江数据', 'data']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"错误: 找不到目录 {dir_name}")
            return
    
    print("✓ 数据目录检查完成")
    print("✓ 正在启动清洁版筛选程序...")
    print()
    
    # 创建临时脚本文件
    script_content = create_clean_filter_script()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(script_content)
        temp_script = f.name
    
    try:
        # 运行脚本
        result = subprocess.run([sys.executable, temp_script], 
                              capture_output=False, 
                              text=True)
        
        if result.returncode == 0:
            print("\\n程序执行完成!")
        else:
            print(f"\\n程序执行失败，返回码: {result.returncode}")
            
    except Exception as e:
        print(f"运行脚本时出错: {e}")
    finally:
        # 清理临时文件
        try:
            os.unlink(temp_script)
        except:
            pass

if __name__ == "__main__":
    main()
