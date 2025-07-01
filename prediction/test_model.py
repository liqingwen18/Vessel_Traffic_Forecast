"""
STA-BiLSTM模型测试脚本
测试模型能否正常加载和初始化
"""

import os
import sys
import logging
import torch
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("test_model")

def test_imports():
    """测试导入模块"""
    try:
        # 创建必要的目录
        os.makedirs("saved_models", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        logger.info("测试导入模块...")
        
        try:
            # 导入配置
            from config import Config
            logger.info("√ 成功导入Config")
        except Exception as e:
            logger.error(f"导入Config失败: {str(e)}")
            traceback.print_exc()
            return False
        
        try:
            # 导入模型
            from model import STABiLSTMModel, STABiLSTMLoss
            logger.info("√ 成功导入STABiLSTMModel")
        except Exception as e:
            logger.error(f"导入STABiLSTMModel失败: {str(e)}")
            traceback.print_exc()
            return False
        
        try:
            # 导入数据加载器
            from data_loader import DataProcessor, ShipFlowDataset
            logger.info("√ 成功导入DataProcessor")
        except Exception as e:
            logger.error(f"导入DataProcessor失败: {str(e)}")
            traceback.print_exc()
            return False
        
        try:
            # 导入工具函数
            from utils import inverse_transform, calculate_metrics
            logger.info("√ 成功导入utils")
        except Exception as e:
            logger.error(f"导入utils失败: {str(e)}")
            traceback.print_exc()
            return False
        
        try:
            # 导入训练函数
            from train import train_model, Trainer
            logger.info("√ 成功导入train")
        except Exception as e:
            logger.error(f"导入train失败: {str(e)}")
            traceback.print_exc()
            return False
        
        try:
            # 导入H3兼容性函数
            from h3_compat import latlng_to_h3_compat
            logger.info("√ 成功导入h3_compat")
        except Exception as e:
            logger.error(f"导入h3_compat失败: {str(e)}")
            traceback.print_exc()
            return False
        
        # 测试H3函数
        try:
            import h3
            h3_idx = latlng_to_h3_compat(30.0, 120.0, 7)
            logger.info(f"√ H3函数测试成功，示例H3索引: {h3_idx}")
        except Exception as e:
            logger.error(f"H3函数测试失败: {str(e)}")
            traceback.print_exc()
            return False
        
        logger.info("所有导入测试成功!")
        return True
    except Exception as e:
        logger.error(f"导入测试过程中出现未捕获的错误: {str(e)}")
        traceback.print_exc()
        return False

def test_model_init():
    """测试模型初始化"""
    try:
        from model import STABiLSTMModel
        from config import Config
        
        logger.info("测试模型初始化...")
        
        # 获取设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {device}")
        
        # 创建一个小型测试模型
        num_nodes = 10  # 小型测试
        input_dim = 1
        hidden_dim = 32
        model = STABiLSTMModel(
            num_nodes=num_nodes,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            gat_heads=4,
            sequence_length=12,
            prediction_length=6,
            device=device
        )
        
        logger.info(f"√ 成功创建模型: {type(model).__name__}")
        
        # 查看模型参数数量
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"√ 模型参数数量: {num_params:,}")
        
        # 测试前向传播
        batch_size = 2
        seq_len = 12
        
        # 创建测试输入
        x = torch.randn(batch_size, seq_len, num_nodes, input_dim)
        
        # 创建测试边索引
        edge_index = torch.randint(0, num_nodes, (2, 20))  # 随机连接
        
        logger.info("测试前向传播...")
        model.eval()  # 设置为评估模式
        
        with torch.no_grad():
            outputs = model(x, edge_index)
        
        logger.info(f"√ 前向传播成功，输出形状: {outputs.shape}")
        expected_shape = (batch_size, 6, num_nodes, input_dim)
        assert outputs.shape == expected_shape, f"输出形状错误：期望 {expected_shape}，实际 {outputs.shape}"
        
        logger.info("模型初始化测试完成")
        return True
        
    except Exception as e:
        logger.error(f"模型初始化测试失败: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    logger.info("开始STA-BiLSTM模型测试")
    
    # 测试导入
    if not test_imports():
        logger.error("模块导入测试失败")
        return 1
    
    # 测试模型初始化
    if not test_model_init():
        logger.error("模型初始化测试失败")
        return 1
    
    logger.info("所有测试通过！")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 