import h3
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
import numpy as np
import os

# 设置输出目录
output_dir = "h3_grid_analysis_res7_complete"
os.makedirs(output_dir, exist_ok=True)

# 设置中文字体支持
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def shapely_polygon_to_h3(polygon, resolution):
    """将Shapely多边形转换为H3索引列表，确保完整覆盖"""
    # 为多边形添加更大的缓冲区，确保边界区域也被覆盖
    buffered_polygon = polygon.buffer(0.008)  # 约800米的缓冲区
    
    if buffered_polygon.geom_type == 'Polygon':
        polygons = [buffered_polygon]
    else:  # MultiPolygon
        polygons = list(buffered_polygon.geoms)
    
    h3_indexes = []
    for poly in polygons:
        # 获取外环坐标
        coords = list(poly.exterior.coords)
        
        # 确保坐标是[经度, 纬度]格式
        geo_boundary = [[coord[0], coord[1]] for coord in coords[:-1]]  # 去掉最后一个点（与第一个点重复）
        
        # 使用H3的polyfill函数，num_aperture_iterations参数控制覆盖精度
        h3_indices = h3.polyfill_geojson(
            {"type": "Polygon", "coordinates": [geo_boundary]}, 
            resolution
        )
        h3_indexes.extend(h3_indices)
    
    # 检查所有网格，确保完全覆盖
    # 对于边界情况，获取更多层相邻网格（使用更大的k值）
    edge_cells = set()
    for h3_idx in h3_indexes:
        neighbors = h3.k_ring(h3_idx, 3)  # 获取自身及其三层邻居的网格
        edge_cells.update(neighbors)
        
    # 针对细长区域，进行额外的填补
    for h3_idx in h3_indexes:
        # 获取六个方向上的邻居
        direct_neighbors = h3.hex_ring(h3_idx, 1)
        for n_idx in direct_neighbors:
            # 获取这些邻居的二环邻居
            extended_neighbors = h3.k_ring(n_idx, 2)
            edge_cells.update(extended_neighbors)
    
    # 合并原始网格和边缘网格
    all_cells = set(h3_indexes).union(edge_cells)
    
    return list(all_cells)  # 去重

def h3_to_polygon(h3_index):
    """将H3索引转换为Shapely多边形"""
    boundary = h3.h3_to_geo_boundary(h3_index, geo_json=True)
    return Polygon(boundary)

def main():
    print("开始浔江流域H3网格分析...")
    
    # 读取浔江流域边界数据
    watershed_file = r"c:\Users\27015\Desktop\船舶流量\浔江数据\浔江_面数据.shp"
    watershed = gpd.read_file(watershed_file)
    
    # 确保使用EPSG:4326坐标系统
    if watershed.crs is None or watershed.crs.to_string() != 'EPSG:4326':
        watershed = watershed.to_crs('EPSG:4326')
    
    print(f"读取了浔江流域数据，包含 {len(watershed)} 个多边形")
    
    # 设置H3网格分辨率
    h3_resolution = 7
    
    # 为每个流域多边形生成H3网格
    all_h3_indices = []
    for idx, row in watershed.iterrows():
        geom = row.geometry
        indices = shapely_polygon_to_h3(geom, h3_resolution)
        all_h3_indices.extend(indices)
    
    # 去除重复的H3索引
    all_h3_indices = list(set(all_h3_indices))
    print(f"生成了 {len(all_h3_indices)} 个H3网格（分辨率 {h3_resolution}）")
    
    # 将H3索引转换为几何对象
    h3_polygons = []
    for h3_idx in all_h3_indices:
        polygon = h3_to_polygon(h3_idx)
        h3_polygons.append({
            'h3_index': h3_idx,
            'geometry': polygon,
            'resolution': h3_resolution,
            'area_km2': h3.cell_area(h3_idx, unit='km^2')
        })
    
    # 创建GeoDataFrame
    h3_gdf = gpd.GeoDataFrame(h3_polygons, crs='EPSG:4326')
    
    # 筛选与流域相交、包含或接近的网格
    print("筛选与流域相交或接近的网格...")
    # 创建一个包含所有流域多边形的几何对象
    watershed_union = watershed.unary_union
    
    # 为流域创建缓冲区，确保捕获接近但不相交的网格
    watershed_buffer = watershed_union.buffer(0.005)  # 约500米的缓冲区
    
    # 检查每个H3网格是否与流域或其缓冲区相交
    h3_gdf['intersects_watershed'] = h3_gdf.geometry.intersects(watershed_buffer)
    h3_gdf = h3_gdf[h3_gdf['intersects_watershed'] == True].copy()
    
    print(f"筛选后保留了 {len(h3_gdf)} 个与流域相交的H3网格")
    
    # 验证覆盖的完整性
    print("验证覆盖完整性...")
    
    # 1. 创建网格的并集
    h3_union = h3_gdf.unary_union
    
    # 2. 检查流域是否完全被覆盖
    coverage_check = watershed_union.difference(h3_union)
    
    # 3. 如果有未覆盖的区域，添加额外的网格
    if not coverage_check.is_empty:
        print("检测到未完全覆盖的区域，添加额外网格...")
        # 为未覆盖区域添加更多的网格，使用更大的缓冲区
        buffered_missing = coverage_check.buffer(0.01)  # 约1公里的缓冲区
        extra_indices = shapely_polygon_to_h3(buffered_missing, h3_resolution)
        print(f"添加了 {len(extra_indices)} 个额外的网格")
        
        # 将这些额外的网格添加到GeoDataFrame
        extra_polygons = []
        for h3_idx in extra_indices:
            if h3_idx not in h3_gdf['h3_index'].values:  # 避免重复
                polygon = h3_to_polygon(h3_idx)
                extra_polygons.append({
                    'h3_index': h3_idx,
                    'geometry': polygon,
                    'resolution': h3_resolution,
                    'area_km2': h3.cell_area(h3_idx, unit='km^2'),
                    'intersects_watershed': True
                })
        
        if extra_polygons:
            extra_gdf = gpd.GeoDataFrame(extra_polygons, crs='EPSG:4326')
            h3_gdf = pd.concat([h3_gdf, extra_gdf], ignore_index=True)
    
    # 4. 再次验证覆盖完整性
    h3_union_final = h3_gdf.unary_union
    final_check = watershed_union.difference(h3_union_final)
    
    if not final_check.is_empty:
        print("仍有未覆盖区域，执行最终补充...")
        # 获取流域中所有点
        if isinstance(watershed_union, MultiPolygon):
            all_points = []
            for geom in watershed_union.geoms:
                boundary_points = list(geom.exterior.coords)
                all_points.extend(boundary_points)
        else:
            all_points = list(watershed_union.exterior.coords)
        
        # 为每个点创建H3单元格
        point_indices = []
        for point in all_points:
            # 获取点的经纬度
            lon, lat = point
            # 将点转换为H3索引
            h3_idx = h3.geo_to_h3(lat, lon, h3_resolution)
            point_indices.append(h3_idx)
        
        # 将这些额外的点网格添加到GeoDataFrame
        extra_point_polygons = []
        for h3_idx in set(point_indices):  # 去重
            if h3_idx not in h3_gdf['h3_index'].values:  # 避免重复
                polygon = h3_to_polygon(h3_idx)
                extra_point_polygons.append({
                    'h3_index': h3_idx,
                    'geometry': polygon,
                    'resolution': h3_resolution,
                    'area_km2': h3.cell_area(h3_idx, unit='km^2'),
                    'intersects_watershed': True
                })
        
        if extra_point_polygons:
            print(f"基于边界点添加了 {len(extra_point_polygons)} 个额外网格")
            extra_point_gdf = gpd.GeoDataFrame(extra_point_polygons, crs='EPSG:4326')
            h3_gdf = pd.concat([h3_gdf, extra_point_gdf], ignore_index=True)
    
    print(f"最终保留了 {len(h3_gdf)} 个H3网格，确保完全覆盖流域")
    
    # 计算覆盖统计信息
    total_area_km2 = h3_gdf['area_km2'].sum()
    mean_area_km2 = h3_gdf['area_km2'].mean()
    
    # 计算流域原始面积
    watershed_area_km2 = watershed.area.sum() * 111 * 111 * np.cos(np.radians(watershed.centroid.y.mean()))  # 近似计算
    
    # 计算覆盖率
    coverage_percentage = (total_area_km2 / watershed_area_km2) * 100 if watershed_area_km2 > 0 else 0
    
    # 保存统计信息
    stats_info = f"""浔江流域H3网格覆盖统计信息：
    H3分辨率：{h3_resolution}
    总网格数：{len(h3_gdf)}
    总覆盖面积：{total_area_km2:.2f} 平方公里
    流域原始面积：{watershed_area_km2:.2f} 平方公里
    覆盖率：{coverage_percentage:.2f}%
    平均网格面积：{mean_area_km2:.2f} 平方公里
    网格边长（近似值）：1.22 公里
    坐标系统：EPSG:4326 (WGS84)
    """
    
    with open(os.path.join(output_dir, 'grid_statistics.txt'), 'w', encoding='utf-8') as f:
        f.write(stats_info)
    
    print(stats_info)
    
    # 保存网格数据为CSV
    csv_file = os.path.join(output_dir, 'h3_grid_coverage.csv')
    h3_gdf[['h3_index', 'resolution', 'area_km2']].to_csv(csv_file, index=False, encoding='utf-8')
    print(f"已保存网格数据到 {csv_file}")
    
    # 保存为Shapefile
    shp_file = os.path.join(output_dir, 'h3_grid_coverage.shp')
    h3_gdf.to_file(shp_file, driver='ESRI Shapefile', encoding='utf-8')
    print(f"已保存网格形状文件到 {shp_file}")
    
    # 创建可视化
    fig, ax = plt.subplots(figsize=(16, 14), dpi=300)
    
    # 获取所有网格的边界
    all_bounds = h3_gdf.total_bounds
    x_min_grid, y_min_grid, x_max_grid, y_max_grid = all_bounds
    
    # 获取流域边界
    watershed_bounds = watershed.total_bounds
    x_min_ws, y_min_ws, x_max_ws, y_max_ws = watershed_bounds
    
    # 取最大范围确保全部显示
    x_min = min(x_min_grid, x_min_ws)
    y_min = min(y_min_grid, y_min_ws)
    x_max = max(x_max_grid, x_max_ws)
    y_max = max(y_max_grid, y_max_ws)
    
    # 添加较大边距以确保完整显示
    margin = 0.05
    plot_x_min = x_min - margin
    plot_x_max = x_max + margin
    plot_y_min = y_min - margin
    plot_y_max = y_max + margin
    
    # 绘制所有网格
    h3_gdf.plot(ax=ax, color='pink', alpha=0.3, edgecolor='red', linewidth=0.8)
    
    # 绘制流域区域（半透明填充）
    watershed.plot(ax=ax, color='lightblue', alpha=0.3)
    
    # 绘制流域边界
    watershed.boundary.plot(ax=ax, color='blue', linewidth=2, label='浔江流域边界')
    
    # 绘制H3网格边界
    h3_gdf.boundary.plot(ax=ax, color='red', linewidth=0.7, alpha=0.8, label=f'H3网格 (分辨率{h3_resolution})')
    
    # 设置图表属性
    ax.set_title(f'浔江流域H3网格覆盖分析 (分辨率 {h3_resolution})', fontsize=16)
    ax.set_xlabel('经度', fontsize=12)
    ax.set_ylabel('纬度', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 设置坐标轴范围
    ax.set_xlim(plot_x_min, plot_x_max)
    ax.set_ylim(plot_y_min, plot_y_max)
    
    # 保存图片
    plt.tight_layout()
    fig_file = os.path.join(output_dir, f'h3_grid_coverage_res{h3_resolution}_full.png')
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"已保存可视化图片到 {fig_file}")
    
    plt.close()
    
    print("H3网格覆盖分析完成！")

if __name__ == "__main__":
    main() 