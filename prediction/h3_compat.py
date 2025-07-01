"""
H3兼容性工具模块
处理不同版本H3库的函数名差异，提供统一的接口
"""

import h3
import logging
from typing import Tuple, List, Set, Union

# 获取模块级别的logger
logger = logging.getLogger(__name__)

def latlng_to_h3_compat(lat: float, lng: float, resolution: int) -> str:
    """
    经纬度转H3索引的兼容性包装器
    
    Args:
        lat: 纬度
        lng: 经度
        resolution: H3分辨率
    
    Returns:
        H3索引字符串
    """
    try:
        # 尝试使用新版本的API（h3-py v4+）
        if hasattr(h3, 'latlng_to_cell'):
            return h3.latlng_to_cell(lat, lng, resolution)
        # 尝试使用中间版本的API
        elif hasattr(h3, 'geo_to_h3'):
            return h3.geo_to_h3(lat, lng, resolution)
        # 尝试使用旧版本的API
        elif hasattr(h3, 'latlng_to_h3'):
            return h3.latlng_to_h3(lat, lng, resolution)
        else:
            logger.warning("找不到任何可用的H3 API (latlng_to_cell, geo_to_h3, latlng_to_h3)")
            # 返回一个伪造的H3索引（实际使用时应避免）
            fake_h3_idx = f"fake_h3_res{resolution}_{int(lat*1000)}_{int(lng*1000)}"
            return fake_h3_idx
    except Exception as e:
        logger.error(f"H3转换错误 ({lat}, {lng}): {e}")
        # 返回一个伪造的H3索引
        return f"error_h3_res{resolution}_{int(lat*1000)}_{int(lng*1000)}"

def h3_to_latlng_compat(h3_index: str) -> Tuple[float, float]:
    """
    H3索引转经纬度的兼容性包装器
    
    Args:
        h3_index: H3索引字符串
    
    Returns:
        (lat, lng) 经纬度元组
    """
    try:
        # 如果是伪造的H3索引，尝试解析
        if h3_index.startswith(('fake_', 'error_')):
            parts = h3_index.split('_')
            if len(parts) >= 4:
                return float(parts[2])/1000, float(parts[3])/1000
        
        # 尝试使用新版本的API（h3-py v4+）
        if hasattr(h3, 'cell_to_latlng'):
            return h3.cell_to_latlng(h3_index)
        # 尝试使用中间版本的API
        elif hasattr(h3, 'h3_to_geo'):
            return h3.h3_to_geo(h3_index)
        # 尝试使用旧版本的API
        elif hasattr(h3, 'h3_to_latlng'):
            return h3.h3_to_latlng(h3_index)
        else:
            logger.warning("找不到任何可用的H3 API (cell_to_latlng, h3_to_geo, h3_to_latlng)")
            # 返回默认位置（实际使用时应避免）
            return 0.0, 0.0
    except Exception as e:
        logger.error(f"H3转换错误 ({h3_index}): {e}")
        return 0.0, 0.0

def grid_disk_compat(h3_index: str, k: int = 1) -> Union[List[str], Set[str]]:
    """
    获取H3索引周围k环的兼容性包装器
    
    Args:
        h3_index: 中心H3索引
        k: 环数
    
    Returns:
        周围k环的H3索引列表或集合
    """
    try:
        # 尝试使用新版本的API（h3-py v4+）
        if hasattr(h3, 'grid_disk'):
            return h3.grid_disk(h3_index, k)
        # 尝试使用旧版本的API
        elif hasattr(h3, 'k_ring'):
            return h3.k_ring(h3_index, k)
        else:
            logger.warning("找不到任何可用的H3 API (grid_disk, k_ring)")
            # 返回只包含输入索引的列表（实际使用时应避免）
            return [h3_index]
    except Exception as e:
        logger.error(f"H3网格操作错误 ({h3_index}, k={k}): {e}")
        return [h3_index]

def h3_is_valid_compat(h3_index: str) -> bool:
    """
    检查H3索引是否有效的兼容性包装器
    
    Args:
        h3_index: H3索引字符串
    
    Returns:
        是否是有效的H3索引
    """
    try:
        # 检查伪造的索引
        if h3_index.startswith(('fake_', 'error_')):
            return False
            
        # 尝试使用新版本的API（h3-py v4+）
        if hasattr(h3, 'is_valid_cell'):
            return h3.is_valid_cell(h3_index)
        # 尝试使用旧版本的API
        elif hasattr(h3, 'h3_is_valid'):
            return h3.h3_is_valid(h3_index)
        else:
            logger.warning("找不到任何可用的H3 API (is_valid_cell, h3_is_valid)")
            # 尝试通过其他方式验证
            try:
                coords = h3_to_latlng_compat(h3_index)
                return -90 <= coords[0] <= 90 and -180 <= coords[1] <= 180
            except:
                return False
    except Exception as e:
        logger.error(f"H3有效性检查错误 ({h3_index}): {e}")
        return False

def h3_get_resolution_compat(h3_index: str) -> int:
    """
    获取H3索引分辨率的兼容性包装器
    
    Args:
        h3_index: H3索引字符串
    
    Returns:
        H3索引的分辨率
    """
    try:
        # 检查伪造的索引
        if h3_index.startswith(('fake_', 'error_')):
            parts = h3_index.split('_')
            if len(parts) >= 2 and parts[1].startswith('res'):
                return int(parts[1][3:])
            return -1
            
        # 尝试使用新版本的API（h3-py v4+）
        if hasattr(h3, 'get_resolution'):
            return h3.get_resolution(h3_index)
        # 尝试使用旧版本的API
        elif hasattr(h3, 'h3_get_resolution'):
            return h3.h3_get_resolution(h3_index)
        else:
            logger.warning("找不到任何可用的H3 API (get_resolution, h3_get_resolution)")
            return -1
    except Exception as e:
        logger.error(f"H3分辨率获取错误 ({h3_index}): {e}")
        return -1 