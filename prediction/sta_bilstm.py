"""
STA-BiLSTM核心模块

实现时空注意力双向LSTM模块，融合图注意力和时间注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple, Dict, Union, List

# 导入自定义模块
try:
    from .spatial_attention import SpatialGATLayer, MultiLayerSpatialGAT
    from .temporal_attention import TemporalAttention, BiLSTM, TemporalBiLSTMWithAttention
except ImportError:
    from spatial_attention import SpatialGATLayer, MultiLayerSpatialGAT
    from temporal_attention import TemporalAttention, BiLSTM, TemporalBiLSTMWithAttention

# 获取模块级别的logger
logger = logging.getLogger(__name__)

class STABiLSTMBlock(nn.Module):
    """
    时空注意力双向LSTM块 (STA-BiLSTM)
    
    STA-BiLSTM架构的核心组件，融合了:
    1. 空间图注意力网络(GAT)
    2. 双向LSTM
    3. 时间注意力机制
    
    以捕捉船舶流量的时空依赖关系
    """
    
    def __init__(self, 
                input_size: int, 
                hidden_size: int,
                gat_heads: int = 4, 
                gat_hidden_dim: int = 32,
                bilstm_layers: int = 1,
                attention_heads: int = 4, 
                dropout: float = 0.2):
        """
        初始化STA-BiLSTM块
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层特征维度
            gat_heads: GAT注意力头数
            gat_hidden_dim: GAT隐藏层维度
            bilstm_layers: BiLSTM层数
            attention_heads: 时间注意力头数
            dropout: Dropout比例
        """
        super(STABiLSTMBlock, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 空间GAT层 - 捕捉空间依赖性
        self.spatial_gat = MultiLayerSpatialGAT(
            input_dim=input_size,
            hidden_dim=gat_hidden_dim * gat_heads,
            num_layers=1,
            heads=gat_heads,
            dropout=dropout
        )
        
        # 时间BiLSTM+注意力层 - 捕捉时间依赖性
        self.temporal_bilstm_attn = TemporalBiLSTMWithAttention(
            input_size=input_size,
            hidden_size=hidden_size // 2,  # 双向LSTM，每个方向的隐藏层维度
            num_layers=bilstm_layers,
            attention_heads=attention_heads,
            dropout=dropout
        )
        
        # 特征整合层 - 融合空间和时间特征
        self.feature_fusion = nn.Linear(input_size * 2, input_size)
        
        # 最终处理层
        self.final_layer = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(input_size)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                return_attention: bool = False) -> Union[torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            x: 输入序列 [seq_len, num_nodes, input_size]
            edge_index: 边索引 [2, num_edges]
            return_attention: 是否返回注意力权重
            
        Returns:
            输出序列 [seq_len, num_nodes, input_size]
            或包含输出和注意力权重的字典
        """
        seq_len, num_nodes, input_size = x.shape
        
        # 空间和时间特征的存储
        spatial_features = []
        temporal_features = None
        
        # 注意力权重
        spatial_attention = {}
        temporal_attention = None
        
        # 1. 处理每个时间步的空间依赖性
        for t in range(seq_len):
            # 获取当前时间步的节点特征
            nodes_features = x[t]  # [num_nodes, input_size]
            
            # 应用GAT
            if return_attention:
                spatial_output = self.spatial_gat(nodes_features, edge_index, return_attention=True)
                spatial_features.append(spatial_output['output'])
                spatial_attention[f'time_{t}'] = spatial_output['attention']
            else:
                spatial_output = self.spatial_gat(nodes_features, edge_index)
                spatial_features.append(spatial_output['output'])
        
        # 堆叠空间特征 [seq_len, num_nodes, input_size]
        spatial_features = torch.stack(spatial_features, dim=0)
        
        # 2. 处理每个节点的时间依赖性
        # 转换为节点优先的格式 [num_nodes, seq_len, input_size]
        x_node_first = x.permute(1, 0, 2)
        
        # 节点时间特征的存储
        node_temporal_features = []
        
        for n in range(num_nodes):
            # 获取当前节点的时间序列
            node_sequence = x_node_first[n]  # [seq_len, input_size]
            
            # 添加批次维度
            node_sequence = node_sequence.unsqueeze(1)  # [seq_len, 1, input_size]
            
            # 应用BiLSTM+注意力
            if return_attention:
                temp_output = self.temporal_bilstm_attn(node_sequence, return_attention=True)
                node_temporal_features.append(temp_output['output'].squeeze(1))  # 移除批次维度
                
                # 只保存最后一个节点的注意力权重(用于演示)
                if n == num_nodes - 1:
                    temporal_attention = temp_output['attention']
            else:
                temp_output = self.temporal_bilstm_attn(node_sequence)
                node_temporal_features.append(temp_output.squeeze(1))  # 移除批次维度
        
        # 堆叠时间特征 [num_nodes, seq_len, input_size]
        node_temporal_features = torch.stack(node_temporal_features, dim=0)
        
        # 转换回序列优先 [seq_len, num_nodes, input_size]
        temporal_features = node_temporal_features.permute(1, 0, 2)
        
        # 3. 融合空间和时间特征
        concat_features = torch.cat([spatial_features, temporal_features], dim=2)  # [seq_len, num_nodes, input_size*2]
        fused_features = self.feature_fusion(concat_features)  # [seq_len, num_nodes, input_size]
        fused_features = F.relu(fused_features)
        
        # 4. 最终处理
        output = self.final_layer(fused_features)  # [seq_len, num_nodes, input_size]
        
        if return_attention:
            return {
                'output': output,
                'spatial_attention': spatial_attention,
                'temporal_attention': temporal_attention
            }
        
        return output


class MultiBlockSTABiLSTM(nn.Module):
    """
    多块堆叠的STA-BiLSTM
    
    通过堆叠多个STA-BiLSTM块，增强对复杂时空依赖关系的建模能力
    """
    
    def __init__(self, 
                input_size: int, 
                hidden_size: int,
                num_blocks: int = 2,
                gat_heads: int = 4, 
                gat_hidden_dim: int = 32,
                bilstm_layers: int = 1,
                attention_heads: int = 4, 
                dropout: float = 0.2):
        """
        初始化多块STA-BiLSTM
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层特征维度
            num_blocks: STA-BiLSTM块的数量
            gat_heads: GAT注意力头数
            gat_hidden_dim: GAT隐藏层维度
            bilstm_layers: BiLSTM层数
            attention_heads: 时间注意力头数
            dropout: Dropout比例
        """
        super(MultiBlockSTABiLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        
        # 输入投影层
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 多个STA-BiLSTM块
        self.sta_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = STABiLSTMBlock(
                input_size=hidden_size,
                hidden_size=hidden_size,
                gat_heads=gat_heads,
                gat_hidden_dim=gat_hidden_dim,
                bilstm_layers=bilstm_layers,
                attention_heads=attention_heads,
                dropout=dropout
            )
            self.sta_blocks.append(block)
        
        # 输出投影层
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.LayerNorm(input_size)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                return_attention: bool = False) -> Union[torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            x: 输入序列 [seq_len, num_nodes, input_size]
            edge_index: 边索引 [2, num_edges]
            return_attention: 是否返回注意力权重
            
        Returns:
            输出序列 [seq_len, num_nodes, input_size]
            或包含输出和注意力权重的字典
        """
        # 输入投影
        h = self.input_projection(x)
        
        # 堆叠处理
        attentions = {}
        
        for i, block in enumerate(self.sta_blocks):
            if return_attention:
                block_output = block(h, edge_index, return_attention=True)
                h = block_output['output']
                attentions[f'block_{i}'] = {
                    'spatial': block_output['spatial_attention'],
                    'temporal': block_output['temporal_attention']
                }
            else:
                h = block(h, edge_index)
        
        # 输出投影
        output = self.output_projection(h)
        
        if return_attention:
            return {'output': output, 'attention': attentions}
        
        return output 