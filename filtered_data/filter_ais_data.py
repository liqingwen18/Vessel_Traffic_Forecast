import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
import csv
import time
from datetime import datetime
import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

def load_shapefiles():
    """加载浔江流域的地理数据文件"""
    print("加载浔江流域地理数据...")
    
    try:
        # 加载面数据
        area_file = os.path.join('浔江数据', '浔江_面数据.shp')
        area_gdf = gpd.read_file(area_file)
        
        print(f"坐标参考系统: {area_gdf.crs}")
        print(f"已加载浔江流域面数据，包含 {len(area_gdf)} 个水面区域")
        
        # 合并所有的多边形为一个，这样可以加快点在区域内的检查
        area_union = area_gdf.geometry.unary_union
        return area_union
    except Exception as e:
        print(f"加载地理数据时出错: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

def is_point_in_area(point, area_union):
    """检查点是否在指定区域内"""
    return area_union.contains(point)

def convert_coordinates(lon, lat):
    """将经纬度坐标转换为Point对象"""
    try:
        # 将坐标转换为浮点数
        lon_float = float(lon)
        lat_float = float(lat)
        # AIS数据中的坐标换算（除以600000获取实际经纬度）
        real_lon = lon_float / 600000
        real_lat = lat_float / 600000
        return Point(real_lon, real_lat)
    except (ValueError, TypeError):
        return None

def process_file_line_by_line(file_path, area_union, output_file_path, start_line=0):
    """逐行处理CSV文件并筛选位于浔江流域内的数据"""
    columns = [
        'timestamp', 'type', 'mmsi', 'status', 'turn', 'shipname', 'callsign',
        'length', 'width', 'draft', 'shiptype', 'cargo', 'destination', 'eta',
        'imo', 'port_name', 'port_code', 'accuracy', 'lon', 'lat', 'sog', 'cog',
        'heading', 'rot', 'foreign_mmsi', 'foreign_shipname'
    ]
    
    # 估计总行数
    line_count = 0
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for _ in f:
            line_count += 1
    
    print(f"文件共有 {line_count} 行数据")
    
    # 准备处理文件
    filtered_count = 0
    processed_count = 0
    last_time = time.time()
    buffer_size = 10000 # 缓冲区大小，积累多少条记录就写入文件
    records_buffer = []
    
    # 创建或打开输出文件，如果是从头开始则新建，否则追加
    mode = 'w' if start_line == 0 else 'a'
    file_exists = os.path.exists(output_file_path)
    
    try:
        # 打开CSV文件进行读取
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # 跳过已经处理过的行
            if start_line > 0:
                print(f"跳过前 {start_line} 行...")
                for _ in range(start_line):
                    next(f, None)
                processed_count = start_line
            
            # 创建CSV读取器，处理单引号引用的数据
            csv_reader = csv.reader(f, quotechar="'", delimiter=',')
            
            # 打开输出文件，准备写入
            with open(output_file_path, mode, encoding='utf-8', newline='') as out_f:
                csv_writer = csv.writer(out_f)
                
                # 如果是新文件，写入表头
                if mode == 'w':
                    csv_writer.writerow(columns)
                
                # 处理数据
                for row in tqdm(csv_reader, total=line_count-start_line, 
                                desc=f"处理 {os.path.basename(file_path)}"):
                    processed_count += 1
                    
                    # 检查行是否有足够的列
                    if len(row) >= 20:  # 至少要有经纬度数据
                        try:
                            lon = row[18]  # 假设第19列是经度
                            lat = row[19]  # 假设第20列是纬度
                            
                            if lon and lat:  # 经纬度不为空
                                point = convert_coordinates(lon, lat)
                                if point and is_point_in_area(point, area_union):
                                    filtered_count += 1
                                    records_buffer.append(row)
                                    
                                    # 当缓冲区达到指定大小时，写入文件
                                    if len(records_buffer) >= buffer_size:
                                        csv_writer.writerows(records_buffer)
                                        records_buffer = []
                        except Exception as e:
                            continue  # 忽略单行处理错误
                    
                    # 每处理10万行输出一次进度
                    if processed_count % 100000 == 0:
                        now = time.time()
                        elapsed = now - last_time
                        speed = 100000 / elapsed if elapsed > 0 else 0
                        last_time = now
                        print(f"已处理 {processed_count:,} 行，已筛选 {filtered_count:,} 条记录，处理速度 {speed:.2f} 行/秒")
                        
                        # 保存进度
                        progress_file = f"{os.path.basename(file_path)}.progress"
                        with open(progress_file, 'w') as pf:
                            pf.write(str(processed_count))
                
                # 写入剩余的缓冲区数据
                if records_buffer:
                    csv_writer.writerows(records_buffer)
        
        return filtered_count
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        traceback.print_exc()
        
        # 保存当前进度
        progress_file = f"{os.path.basename(file_path)}.progress"
        with open(progress_file, 'w') as pf:
            pf.write(str(processed_count))
        
        print(f"已处理 {processed_count:,} 行，已筛选 {filtered_count:,} 条记录")
        print(f"进度已保存到 {progress_file}，可以从该行继续处理")
        return filtered_count

def main():
    # 创建输出目录
    output_dir = 'filtered_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载浔江流域地理数据
    area_union = load_shapefiles()
    
    # 获取所有AIS数据文件
    data_dir = 'data'
    ais_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('shipinfo_data_') and f.endswith('.csv')]
    ais_files.sort()  # 按名称排序
    print(f"发现 {len(ais_files)} 个AIS数据文件")
    
    # 显示文件列表
    print("文件列表:")
    for i, f in enumerate(ais_files):
        print(f"{i+1}. {os.path.basename(f)}")
    
    # 询问用户是否只处理部分文件
    try:
        selection = input("请输入要处理的文件序号（多个文件用逗号分隔，全部处理请输入'all'）: ")
        if selection.lower() != 'all':
            indices = [int(idx.strip()) - 1 for idx in selection.split(',') if idx.strip()]
            ais_files = [ais_files[i] for i in indices if 0 <= i < len(ais_files)]
    except:
        print("输入无效，将处理所有文件")
    
    print(f"将处理以下 {len(ais_files)} 个文件:")
    for f in ais_files:
        print(f"- {os.path.basename(f)}")
    
    # 总筛选计数
    total_filtered = 0
    
    # 处理每个AIS数据文件
    for file_path in ais_files:
        file_name = os.path.basename(file_path)
        output_file = os.path.join(output_dir, f"filtered_{file_name}")
        
        # 检查是否有进度文件
        progress_file = f"{file_name}.progress"
        start_line = 0
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as pf:
                try:
                    start_line = int(pf.read().strip())
                    print(f"发现进度文件，将从第 {start_line} 行继续处理")
                    resume = input(f"是否继续处理文件 {file_name}？(y/n): ")
                    if resume.lower() != 'y':
                        start_line = 0
                        print("将重新处理文件")
                except:
                    start_line = 0
        
        print(f"\n开始处理文件: {file_name}")
        start_time = time.time()
        
        # 处理文件
        filtered_count = process_file_line_by_line(file_path, area_union, output_file, start_line)
        total_filtered += filtered_count
        
        # 输出统计信息
        elapsed_time = time.time() - start_time
        print(f"文件 {file_name} 处理完成")
        print(f"筛选出 {filtered_count:,} 条记录")
        print(f"处理用时: {elapsed_time:.2f} 秒\n")
        
    # 输出总结果
    print(f"\n所有文件处理完成")
    print(f"总共筛选出 {total_filtered:,} 条记录")
    print(f"结果保存在 {output_dir} 目录下")
    
    # 询问是否合并结果
    try:
        merge = input("是否将所有筛选结果合并为一个文件？(y/n): ")
        if merge.lower() == 'y':
            merge_results(output_dir)
    except:
        print("用户取消合并操作")

def merge_results(output_dir):
    """将所有筛选结果合并为一个文件"""
    print("开始合并筛选结果...")
    
    # 获取所有筛选结果文件
    filtered_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith('filtered_') and f.endswith('.csv')]
    filtered_files.sort()
    
    if not filtered_files:
        print("没有找到筛选结果文件")
        return
    
    # 合并文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_file = f"merged_filtered_ais_data_{timestamp}.csv"
    
    # 读取第一个文件的表头
    with open(filtered_files[0], 'r', encoding='utf-8') as f:
        header = f.readline().strip()
    
    # 写入合并文件
    with open(merged_file, 'w', encoding='utf-8', newline='') as outfile:
        # 写入表头
        outfile.write(header + '\n')
        
        # 依次写入各个文件的内容（跳过表头）
        for file in tqdm(filtered_files, desc="合并文件"):
            with open(file, 'r', encoding='utf-8') as infile:
                # 跳过表头
                next(infile)
                
                # 写入文件内容
                for line in infile:
                    outfile.write(line)
    
    print(f"合并完成，结果保存在 {merged_file}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        traceback.print_exc() 