#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
船舶数据筛选程序启动脚本
自动处理GDAL警告和中文字体问题
"""

import os
import sys
import warnings
from datetime import datetime

def setup_environment():
    """设置运行环境，修复常见警告"""
    print("正在设置运行环境...")

    # 强力抑制GDAL警告
    gdal_env_vars = {
        'CPL_LOG_LEVEL': 'ERROR',
        'GDAL_DISABLE_READDIR_ON_OPEN': 'TRUE',
        'GDAL_PAM_ENABLED': 'NO',
        'VSI_CACHE': 'FALSE',
        'GDAL_DATA': '',
        'PROJ_LIB': ''
    }

    for key, value in gdal_env_vars.items():
        os.environ[key] = value

    # 抑制所有警告
    warnings.filterwarnings('ignore')

    print("✓ 环境设置完成")
    print("✓ GDAL警告已抑制")

def check_dependencies():
    """检查必要的依赖包"""
    print("\n检查依赖包...")

    required_packages = {
        'geopandas': 'GeoPandas',
        'pandas': 'Pandas',
        'shapely': 'Shapely',
        'matplotlib': 'Matplotlib',
        'numpy': 'NumPy'
    }

    missing_packages = []

    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - 未安装")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n错误: 缺少必要的包: {', '.join(missing_packages)}")
        print("请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    print("✓ 所有依赖包已安装")
    return True

def check_data_files():
    """检查必要的数据文件"""
    print("\n检查数据文件...")

    # 检查浔江数据文件
    xunjiang_files = [
        '浔江数据/浔江_面数据.shp',
        '浔江数据/浔江_线数据.shp',
        '浔江数据/浔江_水面边界线.shp'
    ]

    missing_files = []
    for file_path in xunjiang_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - 文件不存在")
            missing_files.append(file_path)

    # 检查船舶数据目录
    data_dir = 'data'
    if os.path.exists(data_dir):
        ship_files = [f for f in os.listdir(data_dir) if f.startswith('shipinfo_data_')]
        if ship_files:
            print(f"✓ 找到 {len(ship_files)} 个船舶数据文件")
        else:
            print(f"⚠ {data_dir} 目录存在但没有找到船舶数据文件")
            missing_files.append("船舶数据文件 (shipinfo_data_*.csv)")
    else:
        print(f"✗ {data_dir} - 目录不存在")
        missing_files.append(data_dir)

    if missing_files:
        print(f"\n警告: 缺少以下文件/目录:")
        for file in missing_files:
            print(f"  - {file}")
        print("\n请确保所有必要的数据文件都在正确的位置")
        return False

    print("✓ 所有数据文件检查完成")
    return True

def run_main_program():
    """运行主程序"""
    print("\n" + "="*50)
    print("开始运行船舶数据筛选程序")
    print("="*50)

    try:
        # 使用subprocess运行主程序，这样可以完全控制输出
        import subprocess

        # 设置环境变量
        env = os.environ.copy()
        env.update({
            'CPL_LOG_LEVEL': 'ERROR',
            'GDAL_DISABLE_READDIR_ON_OPEN': 'TRUE',
            'GDAL_PAM_ENABLED': 'NO',
            'VSI_CACHE': 'FALSE',
            'GDAL_DATA': '',
            'PROJ_LIB': ''
        })

        # 运行主程序
        result = subprocess.run(
            [sys.executable, 'filter_ships_final.py'],
            env=env,
            capture_output=False,
            text=True
        )

        if result.returncode == 0:
            print("\n" + "="*50)
            print("程序运行完成!")
            print("="*50)
            return True
        else:
            print(f"\n程序执行失败，返回码: {result.returncode}")
            return False

    except FileNotFoundError:
        print("错误: 找不到 filter_ships_final.py 文件")
        return False
    except Exception as e:
        print(f"运行程序时出错: {e}")
        return False

def main():
    """主函数"""
    print("船舶数据筛选程序")
    print("="*30)
    print(f"启动时间: {datetime.now()}")
    print()

    # 1. 设置环境
    setup_environment()

    # 2. 检查依赖
    if not check_dependencies():
        print("\n程序退出: 依赖包检查失败")
        return

    # 3. 检查数据文件
    if not check_data_files():
        print("\n程序退出: 数据文件检查失败")
        return

    # 4. 运行主程序
    success = run_main_program()

    if success:
        print(f"\n程序成功完成! 结束时间: {datetime.now()}")
    else:
        print(f"\n程序执行失败! 结束时间: {datetime.now()}")

if __name__ == "__main__":
    main()
