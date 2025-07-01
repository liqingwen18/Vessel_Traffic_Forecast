import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point

def read_xunjiang_boundary():
    """读取浔江水域边界数据"""
    # 读取面数据
    xunjiang_area = gpd.read_file('浔江数据/浔江_面数据.shp')
    print("浔江面数据CRS:", xunjiang_area.crs)
    
    # 获取浔江区域的坐标范围
    x_min, y_min, x_max, y_max = xunjiang_area.total_bounds
    print(f"浔江区域坐标范围: 经度({x_min}, {x_max}), 纬度({y_min}, {y_max})")
    
    return xunjiang_area, x_min, y_min, x_max, y_max

def analyze_coordinates(ship_data_sample, lon_col='lon', lat_col='lat'):
    """分析船舶坐标数据的分布情况，以确定正确的坐标转换方法"""
    print("\n坐标数据分析:")
    
    # 获取经纬度数据
    lon_values = ship_data_sample[lon_col].values
    lat_values = ship_data_sample[lat_col].values
    
    # 原始值统计
    print(f"经度原始值范围: {np.min(lon_values)} - {np.max(lon_values)}")
    print(f"纬度原始值范围: {np.min(lat_values)} - {np.max(lat_values)}")
    
    # 各种转换方法
    conversion_methods = {
        "除以600000": (lon_values / 600000, lat_values / 600000),
        "除以1000000": (lon_values / 1000000, lat_values / 1000000),
        "原始值": (lon_values, lat_values)
    }
    
    for method_name, (lon_converted, lat_converted) in conversion_methods.items():
        print(f"\n{method_name}转换后:")
        print(f"  经度范围: {np.min(lon_converted)} - {np.max(lon_converted)}")
        print(f"  纬度范围: {np.min(lat_converted)} - {np.max(lat_converted)}")
        
        # 查看是否在正常经纬度范围内
        valid_lon = (np.min(lon_converted) >= -180 and np.max(lon_converted) <= 180)
        valid_lat = (np.min(lat_converted) >= -90 and np.max(lat_converted) <= 90)
        
        if valid_lon and valid_lat:
            print(f"  √ 该转换方法产生的坐标在有效的经纬度范围内")
        else:
            print(f"  × 该转换方法产生的坐标超出正常经纬度范围")

def verify_ships_in_xunjiang(ship_data, xunjiang_area, lon_col='lon', lat_col='lat'):
    """验证船舶是否真的在浔江区域内"""
    try:
        # 尝试多种坐标转换方法
        methods = [
            {"name": "除以600000", "factor": 600000},
            {"name": "除以1000000", "factor": 1000000},
            {"name": "原始值", "factor": 1}
        ]
        
        results = []
        for method in methods:
            factor = method["factor"]
            
            # 创建点几何对象
            print(f"\n尝试方法: {method['name']}...")
            geometry = [Point(lon/factor, lat/factor) for lon, lat in zip(ship_data[lon_col].head(1000), ship_data[lat_col].head(1000))]
            gdf = gpd.GeoDataFrame(ship_data.head(1000), geometry=geometry, crs=xunjiang_area.crs)
            
            # 空间查询：判断点是否在浔江区域内
            ships_within = gpd.sjoin(gdf, xunjiang_area, how='inner', predicate='within')
            ships_intersects = gpd.sjoin(gdf, xunjiang_area, how='inner', predicate='intersects')
            
            print(f"使用 {method['name']} 转换方法:")
            print(f"  完全在浔江区域内的船舶: {len(ships_within)} 条 (占样本的 {len(ships_within)*100/len(gdf):.2f}%)")
            print(f"  与浔江区域相交的船舶: {len(ships_intersects)} 条 (占样本的 {len(ships_intersects)*100/len(gdf):.2f}%)")
            
            results.append({
                "method": method["name"],
                "within_count": len(ships_within),
                "intersect_count": len(ships_intersects),
                "sample_size": len(gdf),
                "within_percentage": len(ships_within)*100/len(gdf) if len(gdf) > 0 else 0,
                "gdf": gdf,
                "ships_within": ships_within
            })
        
        # 找到最佳匹配方法
        best_method = max(results, key=lambda x: x["within_percentage"])
        print(f"\n最佳转换方法是: {best_method['method']}, 匹配率: {best_method['within_percentage']:.2f}%")
        
        # 可视化最佳方法的结果（如果有匹配）
        if best_method["within_percentage"] > 0:
            plot_verification(xunjiang_area, best_method["gdf"], best_method["ships_within"], best_method["method"])
        
        return best_method
    except Exception as e:
        print(f"验证过程中出错: {e}")
        return None

def plot_verification(xunjiang_area, ship_gdf, ships_within, method_name):
    """可视化船舶位置和浔江区域"""
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制浔江区域
        xunjiang_area.plot(ax=ax, color='lightblue', alpha=0.5, edgecolor='blue')
        
        # 绘制所有船舶位置
        ship_gdf.plot(ax=ax, color='red', markersize=5, alpha=0.3)
        
        # 绘制在区域内的船舶
        if len(ships_within) > 0:
            ships_within.plot(ax=ax, color='green', markersize=8)
        
        # 设置标题和标签
        plt.title(f'船舶位置与浔江区域对比 (坐标转换方法: {method_name})')
        plt.xlabel('经度')
        plt.ylabel('纬度')
        
        # 保存图片
        plt.tight_layout()
        plt.savefig('verification_plot.png', dpi=300)
        print("已保存验证结果图片: verification_plot.png")
    except Exception as e:
        print(f"绘图过程中出错: {e}")

def main():
    # 读取浔江边界数据
    print("读取浔江边界数据...")
    xunjiang_area, x_min, y_min, x_max, y_max = read_xunjiang_boundary()
    
    # 读取筛选后的船舶数据
    try:
        print("\n读取筛选后的船舶数据...")
        ships_data = pd.read_csv('ships_in_xunjiang.csv')
        print(f"读取到 {len(ships_data)} 条记录")
        
        # 查看样本数据
        print("\n数据样本:")
        print(ships_data.head())
        
        # 分析坐标数据
        analyze_coordinates(ships_data)
        
        # 验证船舶是否在浔江区域内
        print("\n验证船舶是否在浔江区域内...")
        verify_ships_in_xunjiang(ships_data, xunjiang_area)
    except Exception as e:
        print(f"处理筛选数据时出错: {e}")
    
    # 读取广范围筛选的数据
    try:
        print("\n读取广范围筛选的船舶数据...")
        wider_ships_data = pd.read_csv('ships_in_wider_range.csv')
        print(f"读取到 {len(wider_ships_data)} 条记录")
        
        # 验证是否包含浔江区域的船舶
        print("\n验证广范围数据中是否包含浔江区域的船舶...")
        verify_ships_in_xunjiang(wider_ships_data.sample(1000), xunjiang_area)
    except Exception as e:
        print(f"处理广范围数据时出错: {e}")

if __name__ == "__main__":
    main() 