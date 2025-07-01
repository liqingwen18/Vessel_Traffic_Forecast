import os
import pandas as pd
import csv
from tqdm import tqdm
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def extract_records_by_coordinates(file_path, min_lon, max_lon, min_lat, max_lat, 
                                  output_file_path, start_line=0, coordinate_divisor=1000000):
    """基于坐标范围从AIS数据中提取记录"""
    columns = [
        'timestamp', 'type', 'mmsi', 'status', 'turn', 'shipname', 'callsign',
        'length', 'width', 'draft', 'shiptype', 'cargo', 'destination', 'eta',
        'imo', 'port_name', 'port_code', 'accuracy', 'lon', 'lat', 'sog', 'cog',
        'heading', 'rot', 'foreign_mmsi', 'foreign_shipname'
    ]
    
    # 估计总行数
    line_count = 0
    print(f"计算文件 {os.path.basename(file_path)} 的行数...")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in f:
                line_count += 1
            print(f"文件共有 {line_count} 行数据")
    except:
        print("计算文件行数时出错，将使用估计值")
        line_count = 10000000  # 使用一个估计值
    
    # 准备处理文件
    filtered_count = 0
    processed_count = 0
    last_time = time.time()
    buffer_size = 10000  # 缓冲区大小，积累多少条记录就写入文件
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
                            # 提取经纬度
                            lon = float(row[18])  # 第19列是经度
                            lat = float(row[19])  # 第20列是纬度
                            
                            # 转换坐标
                            real_lon = lon / coordinate_divisor
                            real_lat = lat / coordinate_divisor
                            
                            # 检查坐标是否在指定范围内
                            if (min_lon <= real_lon <= max_lon and 
                                min_lat <= real_lat <= max_lat):
                                filtered_count += 1
                                records_buffer.append(row)
                                
                                # 当缓冲区达到指定大小时，写入文件
                                if len(records_buffer) >= buffer_size:
                                    csv_writer.writerows(records_buffer)
                                    records_buffer = []
                        except:
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
        
        # 保存当前进度
        progress_file = f"{os.path.basename(file_path)}.progress"
        with open(progress_file, 'w') as pf:
            pf.write(str(processed_count))
        
        print(f"已处理 {processed_count:,} 行，已筛选 {filtered_count:,} 条记录")
        print(f"进度已保存到 {progress_file}，可以从该行继续处理")
        return filtered_count

def main():
    # 设置浔江流域的坐标范围（WGS84）
    # 我们已经知道浔江流域的坐标范围大约是：
    # 经度：110.0515629 ~ 111.40781206373563
    # 纬度：23.3595 ~ 23.5945363
    # 但为了确保不遗漏，我们可以稍微扩大范围
    min_lon = 109.5  # 往西扩展半度
    max_lon = 112.0  # 往东扩展半度
    min_lat = 23.0   # 往南扩展半度
    max_lat = 24.0   # 往北扩展半度
    
    # 坐标转换因子
    coordinate_divisor = 1000000
    
    print(f"将筛选以下坐标范围内的AIS数据:")
    print(f"经度: {min_lon} ~ {max_lon}")
    print(f"纬度: {min_lat} ~ {max_lat}")
    print(f"坐标转换因子: 除以 {coordinate_divisor}")
    
    # 创建输出目录
    output_dir = 'filtered_coordinate_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
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
        filtered_count = extract_records_by_coordinates(
            file_path, min_lon, max_lon, min_lat, max_lat,
            output_file, start_line, coordinate_divisor
        )
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
    merged_file = f"浔江流域AIS数据_{timestamp}.csv"
    
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
    main() 