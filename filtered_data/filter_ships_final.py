# 强力抑制GDAL警告 - 必须在最开始
import os
import sys
import warnings
import logging

# 设置所有GDAL相关环境变量
gdal_env_settings = {
    'CPL_LOG_LEVEL': 'ERROR',
    'GDAL_DISABLE_READDIR_ON_OPEN': 'TRUE',
    'GDAL_PAM_ENABLED': 'NO',
    'VSI_CACHE': 'FALSE',
    'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif,.tiff,.vrt,.ovr'
}
for key, value in gdal_env_settings.items():
    os.environ[key] = value

# 抑制所有警告
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

# 重定向stderr来抑制GDAL警告
class SuppressGDALWarnings:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.font_manager as fm

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def read_xunjiang_boundary():
    """读取浔江水域边界数据"""
    # 使用上下文管理器抑制GDAL警告
    with SuppressGDALWarnings():
        # 读取面数据
        xunjiang_area = gpd.read_file('浔江数据/浔江_面数据.shp')
        # 读取线数据
        xunjiang_line = gpd.read_file('浔江数据/浔江_线数据.shp')
        # 读取水面边界线
        xunjiang_boundary = gpd.read_file('浔江数据/浔江_水面边界线.shp')

    print("浔江面数据CRS:", xunjiang_area.crs)

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
            on_bad_lines='skip',  # 跳过有问题的行
            low_memory=False      # 更多内存用于解析
        )
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return None

def filter_ships_in_xunjiang(ship_data_chunk, xunjiang_area, lon_col='lon', lat_col='lat', coord_factor=1000000):
    """筛选出在浔江区域内的船舶数据，只返回原始数据，不添加任何新列"""
    # 确保经纬度列存在
    if lon_col not in ship_data_chunk.columns or lat_col not in ship_data_chunk.columns:
        print(f"缺少必要的列: {lon_col} 或 {lat_col}")
        return pd.DataFrame()

    # 过滤无效数据
    ship_data_chunk = ship_data_chunk.dropna(subset=[lon_col, lat_col])

    try:
        # 创建点几何对象，使用1000000作为坐标转换因子
        geometry = [Point(lon/coord_factor, lat/coord_factor) for lon, lat in zip(ship_data_chunk[lon_col], ship_data_chunk[lat_col])]
        gdf = gpd.GeoDataFrame(ship_data_chunk, geometry=geometry, crs=xunjiang_area.crs)

        # 空间查询：判断点是否在浔江区域内
        # 使用within方法判断每个点是否在任何一个浔江区域多边形内
        within_mask = gdf.geometry.apply(lambda point: xunjiang_area.geometry.contains(point).any())

        # 只返回原始数据（不包含geometry列和其他新增列）
        ships_in_xunjiang = ship_data_chunk[within_mask]
        return ships_in_xunjiang
    except Exception as e:
        print(f"空间筛选时发生错误: {e}")
        return pd.DataFrame()

def plot_results(xunjiang_area, ships_data, output_file='ships_in_xunjiang_plot.png'):
    """绘制结果地图，显示浔江区域和船舶位置"""
    try:
        # 设置中文字体（如果默认设置不生效）
        try:
            # 尝试使用系统中文字体
            chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
            available_fonts = [f.name for f in fm.fontManager.ttflist]

            for font in chinese_fonts:
                if font in available_fonts:
                    plt.rcParams['font.sans-serif'] = [font]
                    print(f"使用中文字体: {font}")
                    break
            else:
                print("警告: 未找到合适的中文字体，中文可能显示为方块")
        except Exception as font_error:
            print(f"字体设置警告: {font_error}")

        # 创建点几何对象，使用1000000作为坐标转换因子
        sample_size = min(5000, len(ships_data))  # 限制样本大小，避免绘图太慢
        sample_data = ships_data.sample(sample_size) if len(ships_data) > sample_size else ships_data

        geometry = [Point(lon/1000000, lat/1000000) for lon, lat in zip(sample_data['lon'], sample_data['lat'])]
        ships_gdf = gpd.GeoDataFrame(sample_data, geometry=geometry, crs=xunjiang_area.crs)

        # 绘图
        fig, ax = plt.subplots(figsize=(12, 10))

        # 绘制浔江区域
        xunjiang_area.plot(ax=ax, color='lightblue', alpha=0.5, edgecolor='blue', linewidth=1.5)

        # 绘制船舶位置
        ships_gdf.plot(ax=ax, color='red', markersize=2, alpha=0.7)

        # 设置标题和标签（使用fontsize参数确保显示）
        plt.title('船舶位置与浔江区域分布图', fontsize=16, pad=20)
        plt.xlabel('经度', fontsize=12)
        plt.ylabel('纬度', fontsize=12)

        # 添加网格和图例
        ax.grid(True, alpha=0.3)
        ax.legend(['浔江区域', '船舶位置'], loc='upper right', fontsize=10)

        # 保存图片（使用bbox_inches='tight'确保文字不被截断）
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()  # 关闭图形释放内存
        print(f"已保存结果图片: {output_file}")

        # 显示统计信息
        print(f"绘图统计: 显示了 {len(sample_data)} 个船舶位置点（总数: {len(ships_data)}）")

    except Exception as e:
        print(f"绘图过程中出错: {e}")
        print("尝试使用英文标签重新绘制...")
        try:
            # 备用方案：使用英文标签
            fig, ax = plt.subplots(figsize=(12, 10))
            xunjiang_area.plot(ax=ax, color='lightblue', alpha=0.5, edgecolor='blue')

            sample_size = min(5000, len(ships_data))
            sample_data = ships_data.sample(sample_size) if len(ships_data) > sample_size else ships_data
            geometry = [Point(lon/1000000, lat/1000000) for lon, lat in zip(sample_data['lon'], sample_data['lat'])]
            ships_gdf = gpd.GeoDataFrame(sample_data, geometry=geometry, crs=xunjiang_area.crs)
            ships_gdf.plot(ax=ax, color='red', markersize=2, alpha=0.7)

            plt.title('Ships Distribution in Xunjiang Area', fontsize=16)
            plt.xlabel('Longitude', fontsize=12)
            plt.ylabel('Latitude', fontsize=12)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            backup_file = output_file.replace('.png', '_english.png')
            plt.savefig(backup_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"已保存备用图片（英文标签）: {backup_file}")
        except Exception as backup_error:
            print(f"备用绘图也失败: {backup_error}")

def generate_summary(ships_data):
    """生成数据摘要统计"""
    try:
        # 统计船舶类型分布
        ship_type_counts = ships_data['ship_type'].value_counts()

        # 统计不同时间段的船舶数量
        ships_data['hour'] = pd.to_datetime(ships_data['last_time']).dt.hour
        hour_counts = ships_data['hour'].value_counts().sort_index()

        # 统计船舶速度分布
        speed_stats = ships_data['sog'].describe()

        # 保存摘要统计
        with open('summary_statistics.txt', 'w', encoding='utf-8') as f:
            f.write("浔江区域船舶数据统计摘要\n")
            f.write("======================\n\n")

            f.write(f"总记录数: {len(ships_data)}\n")
            f.write(f"唯一船舶数 (基于MMSI): {ships_data['mmsi'].nunique()}\n\n")

            f.write("船舶类型分布:\n")
            for ship_type, count in ship_type_counts.items():
                f.write(f"  类型 {ship_type}: {count} 条记录 ({count*100/len(ships_data):.2f}%)\n")

            f.write("\n时间分布:\n")
            for hour, count in hour_counts.items():
                f.write(f"  {hour}时: {count} 条记录 ({count*100/len(ships_data):.2f}%)\n")

            f.write("\n速度统计 (SOG):\n")
            for stat, value in speed_stats.items():
                f.write(f"  {stat}: {value}\n")

        print("已保存统计摘要至 summary_statistics.txt")
    except Exception as e:
        print(f"生成统计摘要时出错: {e}")

def export_results(ships_data, output_file='final_ships_in_xunjiang.csv', excel_file='final_ships_in_xunjiang.xlsx'):
    """导出筛选结果为CSV和Excel格式"""
    # 保存为CSV
    ships_data.to_csv(output_file, index=False, encoding='utf-8')
    print(f"已保存筛选结果至CSV文件: {output_file}")

    try:
        # 保存为Excel (有大小限制，可能会失败)
        if len(ships_data) <= 1000000:  # Excel限制约为104万行
            ships_data.to_excel(excel_file, index=False)
            print(f"已保存筛选结果至Excel文件: {excel_file}")
        else:
            # 如果数据太大，保存前10000行作为样本
            ships_data.head(10000).to_excel('sample_' + excel_file, index=False)
            print(f"数据太大，已保存10000行样本至Excel文件: sample_{excel_file}")
    except Exception as e:
        print(f"保存Excel文件时出错: {e}")

def main():
    # 记录处理开始时间
    start_time = datetime.now()
    print(f"开始处理时间: {start_time}")

    # 读取浔江边界数据
    print("读取浔江边界数据...")
    xunjiang_area, x_min, y_min, x_max, y_max = read_xunjiang_boundary()

    # 获取船舶数据文件列表
    ship_data_files = [os.path.join('data', f) for f in os.listdir('data') if f.startswith('shipinfo_data_')]
    print(f"找到 {len(ship_data_files)} 个船舶数据文件")

    # 初始化结果DataFrame
    all_ships_in_xunjiang = pd.DataFrame()

    # 处理所有船舶数据文件
    for file_path in ship_data_files:
        print(f"\n处理文件: {file_path}")
        file_start_time = datetime.now()

        total_rows = 0
        file_found_count = 0

        try:
            # 按块读取和处理数据
            for i, chunk in enumerate(read_ship_data(file_path)):
                chunk_size = len(chunk)
                total_rows += chunk_size

                # 使用正确的坐标转换方法筛选船舶
                ships_in_area = filter_ships_in_xunjiang(chunk, xunjiang_area, coord_factor=1000000)

                if len(ships_in_area) > 0:
                    all_ships_in_xunjiang = pd.concat([all_ships_in_xunjiang, ships_in_area])
                    file_found_count += len(ships_in_area)

                # 每处理10个块输出进度
                if (i+1) % 10 == 0:
                    elapsed = (datetime.now() - file_start_time).total_seconds()
                    print(f"已处理 {i+1} 个块, 约 {total_rows} 条记录, 找到 {file_found_count} 条符合条件的记录 (用时: {elapsed:.1f}秒)")

            # 文件处理完成统计
            file_elapsed = (datetime.now() - file_start_time).total_seconds()
            print(f"文件 {file_path} 处理完成, 共处理 {total_rows} 条记录, 找到 {file_found_count} 条符合条件的记录 (用时: {file_elapsed:.1f}秒)")
        except Exception as e:
            print(f"处理文件 {file_path} 时发生错误: {e}")

    # 保存和分析结果
    if not all_ships_in_xunjiang.empty:
        total_records = len(all_ships_in_xunjiang)
        unique_ships = all_ships_in_xunjiang['mmsi'].nunique()

        print(f"\n筛选完成, 共找到 {total_records} 条符合条件的记录, {unique_ships} 艘唯一船舶")

        # 导出结果
        export_results(all_ships_in_xunjiang)

        # 生成统计摘要
        generate_summary(all_ships_in_xunjiang)

        # 绘制结果图
        plot_results(xunjiang_area, all_ships_in_xunjiang)
    else:
        print("\n未找到浔江区域内的船舶数据")

    # 记录处理结束时间和总用时
    end_time = datetime.now()
    total_elapsed = (end_time - start_time).total_seconds()
    print(f"\n处理结束时间: {end_time}")
    print(f"总用时: {total_elapsed:.1f}秒 ({total_elapsed/60:.1f}分钟)")

if __name__ == "__main__":
    main()