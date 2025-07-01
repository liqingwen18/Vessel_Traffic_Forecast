"""
STA-BiLSTM船舶流量预测模型

基于图注意力网络和双向LSTM的时空注意力模型
用于长江浔江流域船舶交通流量预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple, Dict, Optional, Union, List
import numpy as np

# 导入自定义模块
try:
    from .sta_bilstm import STABiLSTMBlock, MultiBlockSTABiLSTM
except ImportError:
    from sta_bilstm import STABiLSTMBlock, MultiBlockSTABiLSTM

# 获取模块级别的logger
logger = logging.getLogger(__name__)

class STABiLSTMModel(nn.Module):
    """
    STA-BiLSTM船舶流量预测模型
    
    基于STA-BiLSTM架构的船舶流量预测模型
    核心特点：
    1. 空间图注意力网络（GAT）捕捉水道段空间依赖性
    2. 双向LSTM处理时间依赖性，避免误差累积
    3. 时间注意力机制聚焦关键时间步
    4. Seq2Seq框架实现端到端预测
    """
    
    def __init__(self, num_nodes: int, input_dim: int = 1,
                 hidden_dim: int = 64, gat_heads: int = 4,
                 gat_hidden_dim: int = 32, sequence_length: int = 12,
                 prediction_length: int = 6, dropout: float = 0.2,
                 num_sta_blocks: int = 2, use_teacher_forcing: bool = True,
                 device: torch.device = None):
        """
        初始化STA-BiLSTM模型
        
        Args:
            num_nodes: 图节点数量（H3网格数量）
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            gat_heads: GAT注意力头数
            gat_hidden_dim: GAT隐藏层维度
            sequence_length: 输入序列长度
            prediction_length: 预测序列长度
            dropout: Dropout比例
            num_sta_blocks: STA-BiLSTM块数量
            use_teacher_forcing: 是否使用教师强制
            device: 设备(CPU/GPU)
        """
        super(STABiLSTMModel, self).__init__()
        
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.use_teacher_forcing = use_teacher_forcing
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 输入特征投影层
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 编码器 - 多块STA-BiLSTM
        self.encoder = MultiBlockSTABiLSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_blocks=num_sta_blocks,
            gat_heads=gat_heads,
            gat_hidden_dim=gat_hidden_dim,
            attention_heads=4,
            dropout=dropout
        )
        
        # 解码器 - 用于生成预测
        self.decoder = STABiLSTMBlock(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            gat_heads=gat_heads,
            gat_hidden_dim=gat_hidden_dim,
            attention_heads=4,
            dropout=dropout
        )
        
        # 输出投影层
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, input_dim)
        )
        
        # 全局残差连接权重
        self.global_residual_weight = nn.Parameter(torch.tensor(0.1))
        
        # 最终层归一化
        self.final_layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                target: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len, num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            target: 目标序列 [batch_size, pred_len, num_nodes, input_dim] (可选)
            teacher_forcing_ratio: 教师强制比例
            
        Returns:
            输出预测 [batch_size, pred_len, num_nodes, input_dim]
        """
        # 获取形状信息
        batch_size = x.shape[0]
        
        # 存储每个批次的预测结果
        predictions = []
        
        for b in range(batch_size):
            # 提取当前批次的序列
            batch_x = x[b]  # [seq_len, num_nodes, input_dim]
            
            # 1. 输入特征投影
            batch_x_proj = batch_x.reshape(-1, self.input_dim)  # 展平为2D
            batch_x_proj = self.input_projection(batch_x_proj)  # 特征投影
            batch_x_proj = batch_x_proj.reshape(self.sequence_length, self.num_nodes, self.hidden_dim)  # 重塑回3D
            
            # 2. 编码器处理
            encoder_outputs = self.encoder(batch_x_proj, edge_index)  # [seq_len, num_nodes, hidden_dim]
            
            # 保存原始输入的最后时间步，用于全局残差连接
            last_observed = batch_x[-1, :, :]  # [num_nodes, input_dim]
            
            # 3. 增强的上下文信息 - 使用更长的上下文窗口和注意力加权
            context_length = min(6, self.sequence_length)  # 增加上下文窗口长度
            context_window = encoder_outputs[-context_length:].clone()  # [context_len, num_nodes, hidden_dim]
            
            # 计算上下文的注意力加权，重点关注最近的时间步
            context_weights = torch.softmax(torch.tensor(
                [i / context_length for i in range(1, context_length + 1)], 
                device=self.device), dim=0).view(-1, 1, 1)
            
            # 加权上下文
            weighted_context = context_weights * context_window
            decoder_input = weighted_context  # [context_len, num_nodes, hidden_dim]
            
            # 4. 存储当前批次的预测
            batch_predictions = []
            
            # 5. 增强的解码器逐步生成预测，添加增强的残差连接
            prev_prediction = last_observed  # 初始化为最后观测值
            
            for t in range(self.prediction_length):
                # 通过解码器处理
                decoder_output = self.decoder(decoder_input, edge_index)  # [context_len, num_nodes, hidden_dim]
                
                # 获取解码器最后时间步的输出
                current_hidden = decoder_output[-1]  # [num_nodes, hidden_dim]
                
                # 添加残差连接 - 直接连接编码器的最后输出
                if t == 0:
                    current_hidden = current_hidden + 0.2 * encoder_outputs[-1]
                
                # 预测下一个时间步
                next_pred_features = self.output_projection(current_hidden)  # [num_nodes, input_dim]
                
                # 多尺度残差连接
                alpha = self.global_residual_weight * 0.9**t  # 衰减系数
                if t == 0:
                    # 第一个预测步使用最后的观测值作为残差
                    next_pred_features = next_pred_features + alpha * last_observed
                else:
                    # 后续预测步使用前一个预测值和初始观测值的加权和
                    next_pred_features = next_pred_features + 0.6 * alpha * prev_prediction + 0.4 * alpha * last_observed
                
                # 应用层归一化和非负约束
                next_pred_features = self.final_layer_norm(next_pred_features)
                next_pred_features = F.relu(next_pred_features)  # 确保非负（船舶数量）
                
                # 保存当前预测
                batch_predictions.append(next_pred_features)
                prev_prediction = next_pred_features  # 更新前一预测值
                
                # 准备下一个时间步的输入
                if self.training and self.use_teacher_forcing and torch.rand(1).item() < teacher_forcing_ratio and target is not None:
                    # 教师强制：使用真实值与预测值的加权平均
                    true_value = target[b, t]  # [num_nodes, input_dim]
                    true_features = true_value.reshape(-1, self.input_dim)
                    true_features = self.input_projection(true_features).reshape(self.num_nodes, self.hidden_dim)
                    
                    # 加权平均 - 平滑过渡
                    forcing_weight = min(0.8, teacher_forcing_ratio)  # 限制最大权重
                    next_features = forcing_weight * true_features + (1 - forcing_weight) * self.input_projection(
                        next_pred_features.reshape(-1, self.input_dim)).reshape(self.num_nodes, self.hidden_dim)
                else:
                    # 自回归：使用预测值
                    pred_features = next_pred_features.reshape(-1, self.input_dim)
                    pred_features = self.input_projection(pred_features).reshape(self.num_nodes, self.hidden_dim)
                    next_features = pred_features
                
                # 更新解码器输入（滑动窗口）
                decoder_input = torch.cat([
                    decoder_input[1:],  # 移除最早的时间步
                    next_features.unsqueeze(0)  # 添加新的预测特征 [1, num_nodes, hidden_dim]
                ], dim=0)
            
            # 堆叠当前批次的所有预测
            batch_predictions = torch.stack(batch_predictions, dim=0)  # [pred_len, num_nodes, input_dim]
            predictions.append(batch_predictions)
        
        # 合并所有批次的预测
        predictions = torch.stack(predictions, dim=0)  # [batch_size, pred_len, num_nodes, input_dim]
        
        return predictions
    
    def predict(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        预测模式
        
        Args:
            x: 输入序列 [batch_size, seq_len, num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            
        Returns:
            预测结果 [batch_size, pred_len, num_nodes, input_dim]
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x, edge_index, teacher_forcing_ratio=0.0)
    
    def get_attention_weights(self, x: torch.Tensor, edge_index: torch.Tensor) -> Dict:
        """
        获取注意力权重（用于可视化和分析）
        
        Args:
            x: 输入序列 [batch_size, seq_len, num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            
        Returns:
            注意力权重字典
        """
        batch_size = x.shape[0]
        
        # 仅处理第一个批次样本
        batch_x = x[0]  # [seq_len, num_nodes, input_dim]
        
        # 特征投影
        batch_x_proj = batch_x.reshape(-1, self.input_dim)
        batch_x_proj = self.input_projection(batch_x_proj)
        batch_x_proj = batch_x_proj.reshape(self.sequence_length, self.num_nodes, self.hidden_dim)
        
        # 获取编码器注意力权重
        encoder_output = self.encoder(batch_x_proj, edge_index, return_attention=True)
        
        return {
            'encoder_attention': encoder_output['attention']
        }


class STABiLSTMLoss(nn.Module):
    """
    STA-BiLSTM模型的损失函数
    
    结合MSE/MAE损失和时空正则化项
    """
    
    def __init__(self, mse_weight: float = 1.0,
                 spatial_weight: float = 0.1,
                 temporal_weight: float = 0.05,
                 conservation_weight: float = 0.02,
                 use_mae: bool = True):
        """
        初始化损失函数
        
        Args:
            mse_weight: MSE损失权重
            spatial_weight: 空间一致性权重
            temporal_weight: 时间平滑性权重
            conservation_weight: 流量守恒权重
            use_mae: 是否使用MAE损失
        """
        super(STABiLSTMLoss, self).__init__()
        
        self.mse_weight = mse_weight
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.conservation_weight = conservation_weight
        self.use_mae = use_mae
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                edge_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算损失
        
        Args:
            predictions: 预测值 [batch_size, pred_len, num_nodes, input_dim]
            targets: 目标值 [batch_size, pred_len, num_nodes, input_dim]
            edge_index: 图的边索引 [2, num_edges]
            
        Returns:
            损失字典，包含总损失和各组成部分
        """
        # 1. 主要损失（MSE或MAE）
        if self.use_mae:
            main_loss = F.l1_loss(predictions, targets)
        else:
            main_loss = F.mse_loss(predictions, targets)
        
        # 2. 空间一致性损失
        spatial_loss = self._compute_spatial_consistency_loss(predictions, targets, edge_index)
        
        # 3. 时间平滑性损失
        temporal_loss = self._compute_temporal_smoothness_loss(predictions, targets)
        
        # 4. 流量守恒损失
        conservation_loss = self._compute_conservation_loss(predictions, targets)
        
        # 总损失
        total_loss = (
            self.mse_weight * main_loss +
            self.spatial_weight * spatial_loss +
            self.temporal_weight * temporal_loss +
            self.conservation_weight * conservation_loss
        )
        
        return {
            'loss': total_loss,
            'main_loss': main_loss,
            'spatial_loss': spatial_loss,
            'temporal_loss': temporal_loss,
            'conservation_loss': conservation_loss
        }
    
    def _compute_spatial_consistency_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                                        edge_index: torch.Tensor) -> torch.Tensor:
        """
        计算空间一致性损失
        
        邻近网格的流量变化应该相关
        """
        batch_size, pred_len, num_nodes, _ = predictions.shape
        
        # 获取连接的节点对
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        
        # 对每个时间步和批次计算
        spatial_loss = 0.0
        
        for b in range(batch_size):
            for t in range(pred_len):
                # 提取当前预测和目标
                pred_t = predictions[b, t]  # [num_nodes, input_dim]
                targ_t = targets[b, t]  # [num_nodes, input_dim]
                
                # 计算预测的相邻节点差异
                pred_diff = torch.abs(pred_t[source_nodes] - pred_t[target_nodes])
                
                # 计算目标的相邻节点差异
                targ_diff = torch.abs(targ_t[source_nodes] - targ_t[target_nodes])
                
                # 计算损失 - 预测差异应该接近目标差异
                edge_loss = F.mse_loss(pred_diff, targ_diff)
                spatial_loss += edge_loss
        
        # 归一化
        spatial_loss = spatial_loss / (batch_size * pred_len)
        
        return spatial_loss
    
    def _compute_temporal_smoothness_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算时间平滑性损失
        
        相邻时间步的流量变化应该平滑
        """
        batch_size, pred_len, num_nodes, _ = predictions.shape
        
        if pred_len <= 1:
            return torch.tensor(0.0, device=predictions.device)
        
        # 计算预测的时间差分
        pred_diff = predictions[:, 1:] - predictions[:, :-1]  # [batch, pred_len-1, num_nodes, input_dim]
        
        # 计算目标的时间差分
        targ_diff = targets[:, 1:] - targets[:, :-1]  # [batch, pred_len-1, num_nodes, input_dim]
        
        # 时间平滑性损失 - 预测变化应接近目标变化
        temporal_loss = F.mse_loss(pred_diff, targ_diff)
        
        return temporal_loss
    
    def _compute_conservation_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算流量守恒损失
        
        总体船舶数量应该大致保持一致
        """
        batch_size, pred_len, num_nodes, _ = predictions.shape
        
        # 计算每个时间步的总流量
        pred_sum = predictions.sum(dim=2)  # [batch, pred_len, input_dim]
        targ_sum = targets.sum(dim=2)  # [batch, pred_len, input_dim]
        
        # 流量守恒损失 - 总量应该接近
        conservation_loss = F.mse_loss(pred_sum, targ_sum)
        
        return conservation_loss 