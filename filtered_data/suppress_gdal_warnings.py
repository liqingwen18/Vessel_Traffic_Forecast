#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强力抑制GDAL警告的模块
在导入GeoPandas之前调用此模块
"""

import os
import sys
import warnings
import logging

def suppress_all_gdal_warnings():
    """强力抑制所有GDAL相关警告"""
    
    # 1. 设置GDAL环境变量
    gdal_env_vars = {
        'CPL_LOG_LEVEL': 'ERROR',
        'GDAL_DISABLE_READDIR_ON_OPEN': 'TRUE',
        'GDAL_PAM_ENABLED': 'NO',
        'GDAL_PAM_PROXY_DIR': '/tmp',
        'VSI_CACHE': 'FALSE',
        'GDAL_MAX_DATASET_POOL_SIZE': '100',
        'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif,.tiff,.vrt,.ovr',
        'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR'
    }
    
    for key, value in gdal_env_vars.items():
        os.environ[key] = value
    
    # 2. 抑制Python警告
    warning_filters = [
        ('ignore', None, UserWarning, '.*gdal.*'),
        ('ignore', None, RuntimeWarning, '.*gdal.*'),
        ('ignore', None, FutureWarning, '.*gdal.*'),
        ('ignore', None, DeprecationWarning, '.*gdal.*'),
        ('ignore', '.*header.dxf.*', None, None),
        ('ignore', '.*Cannot find.*', None, None),
        ('ignore', '.*GDAL_DATA.*', None, None),
    ]
    
    for action, message, category, module in warning_filters:
        if message and category and module:
            warnings.filterwarnings(action, message=message, category=category, module=module)
        elif message:
            warnings.filterwarnings(action, message=message)
        elif category and module:
            warnings.filterwarnings(action, category=category, module=module)
        else:
            warnings.filterwarnings(action)
    
    # 3. 设置日志级别
    loggers_to_silence = [
        'fiona',
        'fiona.collection',
        'fiona.ogrext',
        'rasterio',
        'rasterio._io',
        'GDAL',
        'OGR'
    ]
    
    for logger_name in loggers_to_silence:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
    
    # 4. 重定向stderr（最后手段）
    class NullWriter:
        def write(self, txt): pass
        def flush(self): pass
    
    # 临时保存原始stderr
    original_stderr = sys.stderr
    
    def restore_stderr():
        sys.stderr = original_stderr
    
    def silence_stderr():
        sys.stderr = NullWriter()
        return restore_stderr
    
    return silence_stderr

def context_suppress_gdal():
    """上下文管理器，临时抑制GDAL警告"""
    class GDALWarningSupressor:
        def __enter__(self):
            self.restore_func = suppress_all_gdal_warnings()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if hasattr(self, 'restore_func'):
                self.restore_func()
    
    return GDALWarningSupressor()

# 自动执行抑制
suppress_all_gdal_warnings()
