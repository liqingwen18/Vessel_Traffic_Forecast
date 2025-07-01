#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试matplotlib中文字体显示
"""

import os
import warnings

# 修复GDAL相关警告
os.environ['CPL_LOG_LEVEL'] = 'ERROR'
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
warnings.filterwarnings('ignore', category=UserWarning, module='.*gdal.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='.*gdal.*')
warnings.filterwarnings('ignore', message='.*header.dxf.*')

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def test_chinese_fonts():
    """测试中文字体显示"""
    print("正在测试中文字体显示...")

    # 检查可用的中文字体
    print("\n检查系统中可用的字体:")
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Arial Unicode MS']
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    found_fonts = []
    for font in chinese_fonts:
        if font in available_fonts:
            found_fonts.append(font)
            print(f"✓ 找到字体: {font}")
        else:
            print(f"✗ 未找到字体: {font}")

    if found_fonts:
        # 使用找到的第一个字体
        plt.rcParams['font.sans-serif'] = [found_fonts[0]]
        print(f"\n使用字体: {found_fonts[0]}")
    else:
        print("\n警告: 未找到任何中文字体，将使用默认字体")

    # 创建测试图表
    fig, ax = plt.subplots(figsize=(10, 6))

    # 生成测试数据
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # 绘制图表
    ax.plot(x, y, 'b-', linewidth=2, label='正弦波')
    ax.set_title('中文字体测试图表 - 船舶位置与浔江区域分布图', fontsize=16, pad=20)
    ax.set_xlabel('经度坐标', fontsize=12)
    ax.set_ylabel('纬度坐标', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 添加中文文本注释
    ax.text(5, 0.5, '这是中文测试文本\n船舶数据筛选结果',
            fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 保存图片
    plt.tight_layout()
    plt.savefig('chinese_font_test.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n测试完成！已保存测试图片: chinese_font_test.png")
    print("请检查图片中的中文是否正常显示。")

    return found_fonts

if __name__ == "__main__":
    test_chinese_fonts()
