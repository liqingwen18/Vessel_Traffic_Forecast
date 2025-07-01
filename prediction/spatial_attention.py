"""
空间图注意力模块

实现空间图注意力网络(GAT)，捕捉不同水道段之间的空间依赖性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import logging
from typing import Optional, Tuple, Dict

# 获取模块级别的logger
logger = logging.getLogger(__name__)

class SpatialGATLayer(nn.Module):
    """
    空间图注意力层
    
    专门用于捕捉水道段之间的空间依赖性
    基于GATv2实现，支持多头注意力
    """

    def __init__(self, 
                input_dim: int, 
                hidden_dim: int, 
                heads: int = 4, 
                dropout: float = 0.2,
                use_edge_feats: bool = False,
                edge_dim: Optional[int] = None):
        """
        初始化空间图注意力层
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层特征维度
            heads: 注意力头数
            dropout: Dropout比例
            use_edge_feats: 是否使用边特征
            edge_dim: 边特征维度 (如果use_edge_feats为True)
        """
        super(SpatialGATLayer, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.use_edge_feats = use_edge_feats
        self.edge_dim = edge_dim if use_edge_feats else None

        # 多头图注意力层 - 使用GATv2卷积
        self.gat_conv = GATv2Conv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=heads,
            dropout=dropout,
            concat=True,
            edge_dim=self.edge_dim
        )

        # 输出投影层 - 将多头注意力的结果投影回原始维度
        self.output_projection = nn.Linear(hidden_dim * heads, input_dim)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化网络权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征 [num_edges, edge_dim] (可选)
            return_attention: 是否返回注意力权重
            
        Returns:
            空间增强的节点特征 [num_nodes, input_dim]
        """
        # 保存残差连接的输入
        residual = x

        # GAT处理 - 使用边特征（如果提供）
        if self.use_edge_feats and edge_attr is not None:
            x_gat = self.gat_conv(x, edge_index, edge_attr=edge_attr)
        else:
            x_gat = self.gat_conv(x, edge_index)
            
        # 应用dropout
        x_gat = self.dropout(x_gat)

        # 投影回原始维度
        x_proj = self.output_projection(x_gat)

        # 残差连接和层归一化
        x_out = self.layer_norm(residual + x_proj)

        if return_attention and hasattr(self.gat_conv, '_alpha'):
            return x_out, self.gat_conv._alpha
        return x_out

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        获取最近一次前向传播的注意力权重
        
        Returns:
            注意力权重 (如果可用)
        """
        if hasattr(self.gat_conv, '_alpha'):
            return self.gat_conv._alpha
        return None


class MultiLayerSpatialGAT(nn.Module):
    """
    多层空间图注意力网络
    
    堆叠多个空间图注意力层，实现更强的表达能力
    """
    
    def __init__(self, 
                input_dim: int, 
                hidden_dim: int, 
                output_dim: Optional[int] = None,
                num_layers: int = 2,
                heads: int = 4, 
                dropout: float = 0.2):
        """
        初始化多层空间图注意力网络
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层特征维度
            output_dim: 输出特征维度 (如果为None, 则与input_dim相同)
            num_layers: 图注意力层数量
            heads: 注意力头数
            dropout: Dropout比例
        """
        super(MultiLayerSpatialGAT, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.num_layers = num_layers
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 多层图注意力网络
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = hidden_dim
            layer_output_dim = hidden_dim
            
            self.gat_layers.append(
                SpatialGATLayer(
                    input_dim=layer_input_dim,
                    hidden_dim=layer_output_dim // heads,  # 除以头数，以保持concat后的维度一致
                    heads=heads,
                    dropout=dropout
                )
            )
        
        # 输出投影层
        self.output_projection = nn.Linear(hidden_dim, self.output_dim)
        
        # 最终层归一化
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征 (可选) [num_edges, edge_dim]
            return_attention: 是否返回注意力权重
            
        Returns:
            包含输出特征和注意力权重的字典
        """
        # 输入投影
        h = self.input_projection(x)
        h = F.relu(h)
        h = self.dropout(h)
        
        # 存储各层注意力权重
        attention_weights = {}
        
        # 应用多层GAT
        for i, gat_layer in enumerate(self.gat_layers):
            if return_attention:
                h, att_weights = gat_layer(h, edge_index, edge_attr, return_attention=True)
                attention_weights[f'layer_{i}'] = att_weights
            else:
                h = gat_layer(h, edge_index, edge_attr)
                
            # 中间层激活函数
            if i < self.num_layers - 1:
                h = F.relu(h)
        
        # 输出投影
        out = self.output_projection(h)
        out = self.layer_norm(out)
        
        if return_attention:
            return {'output': out, 'attention': attention_weights}
        
        return {'output': out} 