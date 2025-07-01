import geopandas as gpd
import pandas as pd
import os
from shapely.geometry import Point
import numpy as np
import csv

def read_xunjiang_boundary():
    """读取浔江水域边界数据"""
    # 读取面数据
    xunjiang_area = gpd.read_file('浔江数据/浔江_面数据.shp')
    # 读取线数据
    xunjiang_line = gpd.read_file('浔江数据/浔江_线数据.shp')
    # 读取水面边界线
    xunjiang_boundary = gpd.read_file('浔江数据/浔江_水面边界线.shp')
    
    print("浔江面数据CRS:", xunjiang_area.crs)
    print("浔江线数据CRS:", xunjiang_line.crs)
    print("浔江边界线数据CRS:", xunjiang_boundary.crs)
    
    # 获取浔江区域的坐标范围
    x_min, y_min, x_max, y_max = xunjiang_area.total_bounds
    print(f"浔江区域坐标范围: 经度({x_min}, {x_max}), 纬度({y_min}, {y_max})")
    
    return xunjiang_area, x_min, y_min, x_max, y_max

def read_ship_data(file_path, chunksize=100000):
    """读取船舶数据(CSV格式)，按块处理大文件"""
    try:
        # 使用更健壮的方法读取CSV
        return pd.read_csv(
            file_path, 
            chunksize=chunksize, 
            encoding='utf-8',
            quotechar="'",  # 处理引号问题
            escapechar='\\',  # 处理转义字符
            error_bad_lines=False,  # 跳过有问题的行
            warn_bad_lines=True,    # 显示警告
            low_memory=False        # 更多内存用于解析
        )
    except TypeError:  # pandas更新版本可能不支持error_bad_lines参数
        return pd.read_csv(
            file_path, 
            chunksize=chunksize, 
            encoding='utf-8',
            quotechar="'",  # 处理引号问题
            escapechar='\\',  # 处理转义字符
            on_bad_lines='skip',  # 新版本用这个替代error_bad_lines
            low_memory=False      # 更多内存用于解析
        )

def filter_ships_in_xunjiang(ship_data_chunk, xunjiang_area, lon_col='lon', lat_col='lat'):
    """筛选出在浔江区域内的船舶数据"""
    # 确保经纬度列存在
    if lon_col not in ship_data_chunk.columns or lat_col not in ship_data_chunk.columns:
        print(f"缺少必要的列: {lon_col} 或 {lat_col}")
        return pd.DataFrame()
    
    # 过滤无效数据
    ship_data_chunk = ship_data_chunk.dropna(subset=[lon_col, lat_col])
    
    try:
        # 创建点几何对象
        geometry = [Point(lon/600000, lat/600000) for lon, lat in zip(ship_data_chunk[lon_col], ship_data_chunk[lat_col])]
        gdf = gpd.GeoDataFrame(ship_data_chunk, geometry=geometry, crs=xunjiang_area.crs)
        
        # 空间查询：判断点是否在浔江区域内
        ships_in_xunjiang = gpd.sjoin(gdf, xunjiang_area, how='inner', predicate='within')
        return ships_in_xunjiang
    except Exception as e:
        print(f"空间筛选时发生错误: {e}")
        return pd.DataFrame()

def filter_ships_by_bounds(ship_data_chunk, x_min, y_min, x_max, y_max, lon_col='lon', lat_col='lat'):
    """根据边界框筛选船舶数据"""
    # 确保经纬度列存在
    if lon_col not in ship_data_chunk.columns or lat_col not in ship_data_chunk.columns:
        return pd.DataFrame()
    
    # 过滤无效数据
    ship_data_chunk = ship_data_chunk.dropna(subset=[lon_col, lat_col])
    if len(ship_data_chunk) == 0:
        return pd.DataFrame()
    
    try:
        # 尝试多种坐标转换方法
        # 方法1：除以600000（根据数据格式猜测）
        lon1 = ship_data_chunk[lon_col] / 600000
        lat1 = ship_data_chunk[lat_col] / 600000
        
        # 方法2：除以1000000 (另一种常见转换)
        lon2 = ship_data_chunk[lon_col] / 1000000
        lat2 = ship_data_chunk[lat_col] / 1000000
        
        # 方法3：假设已经是度数
        lon3 = ship_data_chunk[lon_col]
        lat3 = ship_data_chunk[lat_col]
        
        # 筛选符合条件的数据
        mask1 = (lon1 >= x_min) & (lon1 <= x_max) & (lat1 >= y_min) & (lat1 <= y_max)
        mask2 = (lon2 >= x_min) & (lon2 <= x_max) & (lat2 >= y_min) & (lat2 <= y_max)
        mask3 = (lon3 >= x_min) & (lon3 <= x_max) & (lat3 >= y_min) & (lat3 <= y_max)
        
        # 合并结果
        mask = mask1 | mask2 | mask3
        return ship_data_chunk[mask]
    except Exception as e:
        print(f"边界筛选时发生错误: {e}")
        return pd.DataFrame()

def search_by_keywords(ship_data_chunk):
    """通过关键词搜索与浔江相关的船舶"""
    keywords = ['浔江', '梧州', '桂平', '贵港']
    
    try:
        # 检查船舶名称或目的地是否包含相关关键词
        mask = pd.Series(False, index=ship_data_chunk.index)
        
        for col in ['name', 'dest', 'dest_name', 'dest_std']:
            if col in ship_data_chunk.columns:
                for keyword in keywords:
                    # 确保列数据是字符串类型
                    if ship_data_chunk[col].dtype == 'object':
                        mask = mask | ship_data_chunk[col].str.contains(keyword, na=False, case=False)
        
        return ship_data_chunk[mask]
    except Exception as e:
        print(f"关键词筛选时发生错误: {e}")
        return pd.DataFrame()

def main():
    # 读取浔江边界数据
    print("读取浔江边界数据...")
    xunjiang_area, x_min, y_min, x_max, y_max = read_xunjiang_boundary()
    
    # 获取船舶数据文件列表
    ship_data_files = [os.path.join('data', f) for f in os.listdir('data') if f.startswith('shipinfo_data_')]
    print(f"找到 {len(ship_data_files)} 个船舶数据文件")
    
    # 初始化结果DataFrame
    all_ships_in_xunjiang = pd.DataFrame()
    
    # 处理船舶数据
    for file_path in ship_data_files[:1]:  # 先处理第一个文件测试
        print(f"处理文件: {file_path}")
        total_rows = 0
        found_rows = 0
        
        try:
            # 按块读取和处理数据
            for i, chunk in enumerate(read_ship_data(file_path)):
                chunk_size = len(chunk)
                total_rows += chunk_size
                
                # 尝试多种筛选方法
                # 方法1：基于精确空间几何的筛选
                ships_in_area = filter_ships_in_xunjiang(chunk, xunjiang_area)
                if len(ships_in_area) > 0:
                    all_ships_in_xunjiang = pd.concat([all_ships_in_xunjiang, ships_in_area])
                    found_rows += len(ships_in_area)
                
                # 方法2：基于边界框的筛选
                ships_in_bounds = filter_ships_by_bounds(chunk, x_min, y_min, x_max, y_max)
                if len(ships_in_bounds) > 0:
                    # 避免重复添加
                    new_ships = ships_in_bounds[~ships_in_bounds.index.isin(all_ships_in_xunjiang.index)] if not all_ships_in_xunjiang.empty else ships_in_bounds
                    all_ships_in_xunjiang = pd.concat([all_ships_in_xunjiang, new_ships])
                    found_rows += len(new_ships)
                    
                # 方法3：基于关键词的筛选
                ships_by_keywords = search_by_keywords(chunk)
                if len(ships_by_keywords) > 0:
                    # 避免重复添加
                    new_ships = ships_by_keywords[~ships_by_keywords.index.isin(all_ships_in_xunjiang.index)] if not all_ships_in_xunjiang.empty else ships_by_keywords
                    all_ships_in_xunjiang = pd.concat([all_ships_in_xunjiang, new_ships])
                    found_rows += len(new_ships)
                
                # 每处理10个块输出进度
                if (i+1) % 10 == 0:
                    print(f"已处理 {i+1} 个块, 约 {total_rows} 条记录, 找到 {len(all_ships_in_xunjiang)} 条符合条件的记录")
        except Exception as e:
            print(f"处理文件 {file_path} 时发生错误: {e}")
    
    # 保存结果
    if not all_ships_in_xunjiang.empty:
        output_file = 'ships_in_xunjiang.csv'
        all_ships_in_xunjiang.to_csv(output_file, index=False)
        print(f"已将筛选结果保存至 {output_file}, 共 {len(all_ships_in_xunjiang)} 条记录")
    else:
        print("未找到浔江区域内的船舶数据")
        
    # 使用更宽松的边界尝试
    print("尝试使用更宽松的边界范围...")
    # 浔江大致位于广西壮族自治区境内，使用更广泛的坐标范围
    wider_x_min = 104.0  # 广西西部边界
    wider_x_max = 113.0  # 广西东部边界
    wider_y_min = 20.0   # 广西南部边界
    wider_y_max = 27.0   # 广西北部边界
    
    all_ships_wider_range = pd.DataFrame()
    
    # 处理船舶数据
    for file_path in ship_data_files[:1]:  # 先处理第一个文件测试
        print(f"处理文件(宽松边界): {file_path}")
        total_rows = 0
        
        try:
            # 按块读取和处理数据
            for i, chunk in enumerate(read_ship_data(file_path)):
                total_rows += len(chunk)
                
                # 使用多种坐标转换尝试匹配
                ships_in_bounds = filter_ships_by_bounds(
                    chunk, wider_x_min, wider_y_min, wider_x_max, wider_y_max
                )
                
                if len(ships_in_bounds) > 0:
                    all_ships_wider_range = pd.concat([all_ships_wider_range, ships_in_bounds])
                
                # 每处理10个块输出进度
                if (i+1) % 10 == 0:
                    print(f"已处理 {i+1} 个块, 约 {total_rows} 条记录, 找到 {len(all_ships_wider_range)} 条符合条件的记录")
        except Exception as e:
            print(f"处理文件 {file_path} 时发生错误: {e}")
    
    # 保存广泛范围的结果
    if not all_ships_wider_range.empty:
        output_file = 'ships_in_wider_range.csv'
        all_ships_wider_range.to_csv(output_file, index=False)
        print(f"已将广泛范围筛选结果保存至 {output_file}, 共 {len(all_ships_wider_range)} 条记录")
    else:
        print("在更宽松的范围内仍未找到船舶数据")

if __name__ == "__main__":
    main() 