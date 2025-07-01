"""
STA-BiLSTM模型训练模块

实现模型训练、评估和保存
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import logging
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Optional

# 导入自定义模块
try:
    from .config import Config
    from .model import STABiLSTMModel, STABiLSTMLoss
    from .utils import inverse_transform, calculate_metrics, plot_training_history, save_model_config
except ImportError:
    from config import Config
    from model import STABiLSTMModel, STABiLSTMLoss
    from utils import inverse_transform, calculate_metrics, plot_training_history, save_model_config

# 获取模块级别的logger
logger = logging.getLogger(__name__)

class Trainer:
    """
    STA-BiLSTM模型训练器
    
    负责模型的训练、评估、保存和加载
    """
    
    def __init__(self, model: STABiLSTMModel, config: Config, loss_fn: STABiLSTMLoss = None):
        """
        初始化训练器
        
        Args:
            model: STA-BiLSTM模型实例
            config: 配置对象
            loss_fn: 损失函数对象（可选）
        """
        self.model = model
        self.config = config
        self.loss_fn = loss_fn if loss_fn is not None else STABiLSTMLoss(
            mse_weight=config.MSE_WEIGHT,
            spatial_weight=config.SPATIAL_WEIGHT,
            temporal_weight=config.TEMPORAL_WEIGHT,
            conservation_weight=config.CONSERVATION_WEIGHT,
            use_mae=config.USE_MAE
        )
        
        # 优化器
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # 学习率调整器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_rmse': [],
            'val_mae': [],
            'val_mape': [],
            'val_smape': []  # 添加sMAPE指标
        }
        
        # 创建必要的目录
        self.config.create_directories()
    
    def train(self, train_loader, val_loader, edge_index, scaler=None):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            edge_index: 图的边索引
            scaler: 数据缩放器（用于反向转换预测结果）
            
        Returns:
            训练历史记录
        """
        logger.info(f"开始训练STA-BiLSTM模型... 设备: {self.config.DEVICE}")
        
        # 移动模型到设备
        self.model = self.model.to(self.config.DEVICE)
        edge_index = edge_index.to(self.config.DEVICE)
        
        # 保存模型配置
        model_config_path = os.path.join(self.config.MODEL_SAVE_DIR, "model_config.json")
        save_model_config(vars(self.config), model_config_path)
        
        # 训练循环
        for epoch in range(self.current_epoch, self.config.NUM_EPOCHS):
            self.current_epoch = epoch
            start_time = time.time()
            
            # 训练一个epoch
            train_loss = self._train_epoch(train_loader, edge_index)
            
            # 在验证集上评估
            val_loss, val_metrics = self._validate(val_loader, edge_index, scaler)
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 记录训练历史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_rmse'].append(val_metrics['RMSE'])
            self.training_history['val_mae'].append(val_metrics['MAE'])
            self.training_history['val_mape'].append(val_metrics['MAPE'])
            self.training_history['val_smape'].append(val_metrics['sMAPE'])
            
            # 计算耗时
            epoch_time = time.time() - start_time
            
            # 打印训练信息
            logger.info(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} - "
                      f"Time: {epoch_time:.2f}s - "
                      f"Train Loss: {train_loss:.4f} - "
                      f"Val Loss: {val_loss:.4f} - "
                      f"Val RMSE: {val_metrics['RMSE']:.4f} - "
                      f"Val MAE: {val_metrics['MAE']:.4f} - "
                      f"Val MAPE: {val_metrics['MAPE']:.2f}% - "
                      f"Val sMAPE: {val_metrics['sMAPE']:.2f}%")
            
            # 检查是否是最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                self._save_checkpoint(is_best=True)
                logger.info(f"保存最佳模型，验证损失: {val_loss:.4f}")
            else:
                self.early_stopping_counter += 1
                logger.info(f"验证损失未改善，计数器: {self.early_stopping_counter}/{self.config.EARLY_STOPPING_PATIENCE}")
                
                # 保存常规检查点
                if (epoch + 1) % 10 == 0:  # 每10个epoch保存一次
                    self._save_checkpoint(is_best=False)
            
            # 早停检查
            if self.early_stopping_counter >= self.config.EARLY_STOPPING_PATIENCE:
                logger.info(f"早停触发！验证损失连续 {self.config.EARLY_STOPPING_PATIENCE} 个epoch未改善。")
                break
        
        # 保存训练历史
        self._save_training_history()
        
        # 绘制训练曲线
        history_plot_path = os.path.join(self.config.RESULTS_DIR, "training_history.png")
        plot_training_history(self.training_history, history_plot_path)
        
        logger.info(f"训练完成！最佳验证损失: {self.best_val_loss:.4f}")
        
        return self.training_history
    
    def _train_epoch(self, train_loader, edge_index) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}/{self.config.NUM_EPOCHS} [Training]", 
                   leave=False)
        for batch_idx, (X, y, _) in enumerate(pbar):
            # 移动数据到设备
            X = X.to(self.config.DEVICE)  # [batch_size, seq_len, num_nodes, input_dim]
            y = y.to(self.config.DEVICE)  # [batch_size, pred_len, num_nodes, input_dim]
            
            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(X, edge_index, y, self.config.TEACHER_FORCING_RATIO)
            
            # 计算损失
            loss_dict = self.loss_fn(predictions, y, edge_index)
            loss = loss_dict['loss']
            
            # 反向传播和优化
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 梯度裁剪
            self.optimizer.step()
            
            # 更新统计信息
            batch_size = X.size(0)
            total_loss += loss.item() * batch_size
            train_batches += batch_size
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # 计算平均损失
        avg_loss = total_loss / train_batches
        
        return avg_loss
    
    def _validate(self, val_loader, edge_index, scaler=None) -> Tuple[float, Dict[str, float]]:
        """在验证集上评估模型"""
        self.model.eval()
        total_loss = 0.0
        val_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch+1}/{self.config.NUM_EPOCHS} [Validation]", 
                       leave=False)
            for batch_idx, (X, y, _) in enumerate(pbar):
                # 移动数据到设备
                X = X.to(self.config.DEVICE)
                y = y.to(self.config.DEVICE)
                
                # 前向传播
                predictions = self.model(X, edge_index, teacher_forcing_ratio=0.0)
                
                # 计算损失
                loss_dict = self.loss_fn(predictions, y, edge_index)
                loss = loss_dict['loss']
                
                # 收集预测结果和目标
                if scaler is not None:
                    # 反向转换缩放的数据
                    predictions_np = predictions.cpu().numpy()
                    y_np = y.cpu().numpy()
                    
                    pred_original = inverse_transform(scaler, predictions_np)
                    targets_original = inverse_transform(scaler, y_np)
                    
                    all_predictions.append(pred_original)
                    all_targets.append(targets_original)
                else:
                    all_predictions.append(predictions.cpu().numpy())
                    all_targets.append(y.cpu().numpy())
                
                # 更新统计信息
                batch_size = X.size(0)
                total_loss += loss.item() * batch_size
                val_batches += batch_size
                
                # 更新进度条
                pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
        
        # 计算平均损失
        avg_loss = total_loss / val_batches
        
        # 合并所有批次的预测结果和目标
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # 计算评估指标
        metrics = calculate_metrics(all_targets, all_predictions)
        
        return avg_loss, metrics
    
    def predict(self, test_loader, edge_index, scaler=None) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        生成预测结果
        
        Args:
            test_loader: 测试数据加载器
            edge_index: 图的边索引
            scaler: 数据缩放器（用于反向转换预测结果）
            
        Returns:
            predictions: 预测结果
            targets: 真实目标值
            metrics: 评估指标
        """
        # 加载最佳模型
        self._load_checkpoint(self.config.BEST_MODEL_PATH)
        
        # 移动模型到设备
        self.model = self.model.to(self.config.DEVICE)
        edge_index = edge_index.to(self.config.DEVICE)
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X, y, _ in tqdm(test_loader, desc="生成预测结果"):
                # 移动数据到设备
                X = X.to(self.config.DEVICE)
                y = y.to(self.config.DEVICE)
                
                # 前向传播
                predictions = self.model.predict(X, edge_index)
                
                # 收集预测结果和目标
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        # 合并所有批次的预测结果和目标
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # 反向转换缩放的数据
        if scaler is not None:
            predictions = inverse_transform(scaler, predictions)
            targets = inverse_transform(scaler, targets)
        
        # 计算评估指标
        metrics = calculate_metrics(targets, predictions)
        
        logger.info(f"预测完成。测试集评估指标: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, MAPE={metrics['MAPE']:.2f}%")
        
        return predictions, targets, metrics
    
    def _save_checkpoint(self, is_best: bool = False):
        """保存模型检查点"""
        checkpoint_path = os.path.join(
            self.config.MODEL_SAVE_DIR,
            f"model_epoch_{self.current_epoch+1}.pth" if not is_best else "best_model.pth"
        )
        
        # 准备检查点数据
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'early_stopping_counter': self.early_stopping_counter,
            'training_history': self.training_history
        }
        
        # 保存检查点
        torch.save(checkpoint, checkpoint_path)
    
    def _load_checkpoint(self, checkpoint_path: str):
        """加载模型检查点"""
        if not os.path.exists(checkpoint_path):
            logger.warning(f"检查点文件不存在: {checkpoint_path}")
            return False
        
        try:
            # 加载检查点
            checkpoint = torch.load(checkpoint_path, map_location=self.config.DEVICE)
            
            # 恢复模型状态
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 恢复优化器状态
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 恢复学习率调整器状态
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # 恢复训练状态
            self.current_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.early_stopping_counter = checkpoint['early_stopping_counter']
            self.training_history = checkpoint['training_history']
            
            logger.info(f"成功加载检查点: {checkpoint_path}")
            logger.info(f"当前epoch: {self.current_epoch}, 最佳验证损失: {self.best_val_loss:.4f}")
            
            return True
        except Exception as e:
            logger.error(f"加载检查点失败: {str(e)}")
            return False
    
    def _save_training_history(self):
        """保存训练历史记录"""
        history_path = os.path.join(self.config.RESULTS_DIR, "training_history.json")
        
        # 转换NumPy值为Python原生类型
        history_json = {}
        for key, value in self.training_history.items():
            history_json[key] = [float(v) for v in value]
        
        with open(history_path, 'w') as f:
            json.dump(history_json, f)
            
        logger.info(f"训练历史已保存至: {history_path}")


def train_model(config: Config, train_loader, val_loader, edge_index, scaler=None):
    """
    训练模型的主函数
    
    Args:
        config: 配置对象
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        edge_index: 图的边索引
        scaler: 数据缩放器（可选）
    
    Returns:
        trainer: 训练器实例
    """
    # 获取数据集信息
    sample_batch = next(iter(train_loader))
    num_nodes = sample_batch[0].shape[2]  # 图节点数量
    input_dim = sample_batch[0].shape[3]  # 输入特征维度
    
    # 创建模型
    model = STABiLSTMModel(
        num_nodes=num_nodes,
        input_dim=input_dim,
        hidden_dim=config.HIDDEN_DIM,
        gat_heads=config.GAT_HEADS,
        gat_hidden_dim=config.GAT_HIDDEN_DIM,
        sequence_length=config.SEQUENCE_LENGTH,
        prediction_length=config.PREDICTION_LENGTH,
        dropout=config.DROPOUT_RATE,
        num_sta_blocks=config.NUM_STA_BLOCKS,
        device=config.DEVICE
    )
    
    # 创建损失函数
    loss_fn = STABiLSTMLoss(
        mse_weight=config.MSE_WEIGHT,
        spatial_weight=config.SPATIAL_WEIGHT,
        temporal_weight=config.TEMPORAL_WEIGHT,
        conservation_weight=config.CONSERVATION_WEIGHT,
        use_mae=config.USE_MAE
    )
    
    # 创建训练器
    trainer = Trainer(model, config, loss_fn)
    
    # 训练模型
    trainer.train(train_loader, val_loader, edge_index, scaler)
    
    return trainer 