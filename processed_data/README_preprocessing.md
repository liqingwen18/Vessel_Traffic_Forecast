# AIS数据预处理工具

## 功能概述

本工具实现了完整的AIS（船舶自动识别系统）数据预处理流程，包括：

1. **重复记录删除**
   - 完全重复记录删除
   - 高频刷屏记录删除（5秒内且移动距离<10米）

2. **静止船舶剔除**
   - 基于船速判定（SOG ≤ 0.5节）
   - 基于导航状态判定（锚泊、靠泊、搁浅等）

3. **时间重采样**
   - 默认5分钟间隔重采样
   - 统计每个时间窗口内的唯一船舶数量

4. **Savitzky-Golay滤波**
   - 窗口长度：5
   - 多项式阶数：2
   - 平滑流量数据，去除噪声

5. **H3网格化**
   - 默认分辨率8（约0.7km²网格）
   - 支持自定义分辨率

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本使用

```bash
python run_preprocessing.py
```

### 自定义参数

```bash
python run_preprocessing.py --input your_data.csv --h3_resolution 9 --interval 10min --output results
```

### 参数说明

- `--input, -i`: 输入CSV文件名（默认：final_ships_in_xunjiang.csv）
- `--h3_resolution, -r`: H3网格分辨率（默认：8）
- `--interval, -t`: 时间重采样间隔（默认：5min）
- `--output, -o`: 输出目录（默认：processed_data）
- `--data_path, -d`: 输入数据目录（默认：data）

## 输出文件

处理完成后会生成以下文件：

1. **active_ships_processed.csv**: 预处理后的活跃船舶数据
   - 包含H3网格ID
   - 已去重和去除静止船舶

2. **static_ships.csv**: 静止船舶数据
   - 用于后续泊位分析等

3. **traffic_data_processed.csv**: 流量数据
   - 时间间隔 × H3网格的流量矩阵
   - 包含原始流量和SG滤波后的平滑流量

4. **preprocessing_report.txt**: 处理报告
   - 数据统计信息
   - 处理结果摘要

## 数据格式要求

输入CSV文件应包含以下字段：
- `last_time`: 时间戳
- `mmsi`: 船舶MMSI号
- `lat`, `lon`: 纬度、经度（度分格式）
- `sog`: 对地航速（0.1节单位）
- `cog`: 对地航向（0.1度单位）
- `navi_stat`: 导航状态代码

## H3分辨率参考

| 分辨率 | 平均网格面积 | 适用场景 |
|--------|-------------|----------|
| 6      | ~36 km²     | 大区域分析 |
| 7      | ~5 km²      | 区域流量分析 |
| 8      | ~0.7 km²    | 详细流量分析（推荐）|
| 9      | ~0.1 km²    | 精细化分析 |
| 10     | ~0.015 km²  | 超精细分析 |

## 注意事项

1. 确保输入数据文件存在于指定的data目录中
2. 处理大数据集时可能需要较长时间，请耐心等待
3. 建议先用小样本数据测试参数设置
4. H3分辨率越高，计算时间越长，输出文件越大

## 错误排查

如果遇到问题，请检查：
1. 输入文件路径是否正确
2. 数据格式是否符合要求
3. 是否安装了所有依赖包
4. 磁盘空间是否充足

## 技术支持

如有问题，请检查preprocessing_report.txt中的详细信息。
