"""
时间注意力模块

实现双向LSTM和注意力机制，捕捉船舶流量的时间依赖性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple, Dict, Union, List

# 获取模块级别的logger
logger = logging.getLogger(__name__)

class TemporalAttention(nn.Module):
    """
    时间注意力层
    
    用于为时间序列中的不同时间步分配注意力权重
    能够聚焦在重要的时间点上，提升预测准确性
    """
    
    def __init__(self, feature_dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        初始化时间注意力层
        
        Args:
            feature_dim: 特征维度
            num_heads: 注意力头数
            dropout: Dropout比例
        """
        super(TemporalAttention, self).__init__()
        
        # 确保特征维度可以被头数整除
        assert feature_dim % num_heads == 0, f"特征维度({feature_dim})必须能被头数({num_heads})整除"
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # 查询、键、值变换矩阵
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        # 输出投影
        self.output_linear = nn.Linear(feature_dim, feature_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(feature_dim)
        
        # 存储注意力权重
        self.attention_weights = None
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for m in [self.query, self.key, self.value, self.output_linear]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, feature_dim]
            return_attention: 是否返回注意力权重
            
        Returns:
            带有时间注意力的输出 [batch_size, seq_len, feature_dim]
            (如果return_attention=True，还会返回注意力权重)
        """
        # 保存原始输入用于残差连接
        residual = x
        
        # 获取形状信息
        batch_size, seq_len, _ = x.shape
        
        # 线性变换
        q = self.query(x)  # [batch_size, seq_len, feature_dim]
        k = self.key(x)    # [batch_size, seq_len, feature_dim]
        v = self.value(x)  # [batch_size, seq_len, feature_dim]
        
        # 重塑为多头形式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch, heads, seq, head_dim]
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch, heads, seq, head_dim]
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch, heads, seq, head_dim]
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [batch, heads, seq, seq]
        
        # 应用Softmax获取注意力权重
        attention_weights = F.softmax(scores, dim=-1)  # [batch, heads, seq, seq]
        attention_weights = self.attn_dropout(attention_weights)
        
        # 存储注意力权重（用于可视化）
        self.attention_weights = attention_weights
        
        # 加权聚合值向量
        context = torch.matmul(attention_weights, v)  # [batch, heads, seq, head_dim]
        
        # 重塑回原始形状
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.feature_dim)  # [batch, seq, feature]
        
        # 输出投影
        output = self.output_linear(context)
        output = self.output_dropout(output)
        
        # 残差连接和层归一化
        output = self.layer_norm(residual + output)  # [batch, seq, feature]
        
        if return_attention:
            # 平均所有头的注意力权重
            avg_attention = torch.mean(attention_weights, dim=1)  # [batch, seq, seq]
            return output, avg_attention
            
        return output
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        获取最近一次前向传播的注意力权重
        
        Returns:
            注意力权重 (如果可用)
        """
        return self.attention_weights


class BiLSTM(nn.Module):
    """
    双向LSTM层
    
    捕捉船舶流量的时间依赖性，同时考虑过去和未来的信息
    避免传统LSTM中的误差累积问题
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.2):
        """
        初始化双向LSTM层
        
        Args:
            input_size: 输入特征维度
            hidden_size: LSTM隐藏状态维度（每个方向）
            num_layers: LSTM层数
            dropout: Dropout比例
        """
        super(BiLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,  # seq_len在第一个维度
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 输出投影层 - 将双向输出合并
        self.output_projection = nn.Linear(hidden_size * 2, input_size)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(input_size)
    
    def forward(self, x: torch.Tensor, initial_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入序列 [seq_len, batch_size, input_size]
            initial_states: 初始隐藏状态和单元状态 (可选)
            
        Returns:
            outputs: 双向LSTM输出 [seq_len, batch_size, input_size]
            states: 最终隐藏状态和单元状态
        """
        # 保存残差连接的输入
        residual = x
        
        # 双向LSTM
        outputs, states = self.lstm(x, initial_states)
        
        # 输出投影 - 将双向输出映射回原始维度
        outputs = self.output_projection(outputs)
        
        # Dropout
        outputs = self.dropout(outputs)
        
        # 残差连接和层归一化
        outputs = self.layer_norm(residual + outputs)
        
        return outputs, states


class TemporalBiLSTMWithAttention(nn.Module):
    """
    带有注意力机制的时间双向LSTM模块
    
    结合双向LSTM和时间注意力，增强对船舶流量时间依赖性的捕捉
    """
    
    def __init__(self, 
                input_size: int, 
                hidden_size: int, 
                num_layers: int = 1,
                attention_heads: int = 4,
                dropout: float = 0.2):
        """
        初始化带注意力的时间BiLSTM模块
        
        Args:
            input_size: 输入特征维度
            hidden_size: LSTM隐藏状态维度（每个方向）
            num_layers: LSTM层数
            attention_heads: 时间注意力头数
            dropout: Dropout比例
        """
        super(TemporalBiLSTMWithAttention, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 双向LSTM
        self.bilstm = BiLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 时间注意力层
        self.temporal_attention = TemporalAttention(
            feature_dim=input_size,
            num_heads=attention_heads,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor, initial_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
                return_attention: bool = False) -> Union[torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            x: 输入序列 [seq_len, batch_size, input_size]
            initial_states: 初始隐藏状态和单元状态 (可选)
            return_attention: 是否返回注意力权重
            
        Returns:
            outputs: 处理后的序列 [seq_len, batch_size, input_size]
            或包含输出和注意力权重的字典
        """
        # 通过双向LSTM
        bilstm_outputs, states = self.bilstm(x, initial_states)
        
        # 转换形状以适应注意力层
        batch_size = bilstm_outputs.shape[1]
        seq_len = bilstm_outputs.shape[0]
        
        # [seq_len, batch, feature] -> [batch, seq, feature]
        attention_input = bilstm_outputs.permute(1, 0, 2)
        
        # 应用时间注意力
        if return_attention:
            attention_output, attention_weights = self.temporal_attention(attention_input, return_attention=True)
            
            # [batch, seq, feature] -> [seq, batch, feature]
            outputs = attention_output.permute(1, 0, 2)
            
            return {
                'output': outputs, 
                'states': states, 
                'attention': attention_weights
            }
        else:
            attention_output = self.temporal_attention(attention_input)
            
            # [batch, seq, feature] -> [seq, batch, feature]
            outputs = attention_output.permute(1, 0, 2)
            
            return outputs
            
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        获取最近一次前向传播的注意力权重
        
        Returns:
            注意力权重 (如果可用)
        """
        return self.temporal_attention.get_attention_weights() 