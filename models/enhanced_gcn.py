import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, JumpingKnowledge, BatchNorm, global_add_pool, global_max_pool


class FeatureFusionLayer(nn.Module):
    """特征融合层"""
    def __init__(self, embedding_dim, centrality_dim=8):  # 由7改为8
        super(FeatureFusionLayer, self).__init__()
        self.centrality_encoder = nn.Sequential(
            nn.Linear(centrality_dim, 64),  # 直接映射到64维
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(64),
        )
        self.embedding_encoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(64),
        )
        
        # 改进的融合层：添加非线性激活和批归一化
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64, 96),  # 先扩展维度
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(96),
            nn.Linear(96, 64),  # 再压缩回原维度
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(64),
        )

    def forward(self, x):
        centrality_features = x[:, :8]  # 前8个是中心性特征
        embedding_features = x[:, 8:]
        centrality_encoded = self.centrality_encoder(centrality_features)
        embedding_encoded = self.embedding_encoder(embedding_features)
        x = torch.cat([centrality_encoded, embedding_encoded], dim=1)
        return self.fusion(x)


class ContrastEnhancementLayer(nn.Module):
    """增强版对比强化层 - 显著放大节点评分差异"""
    def __init__(self, temperature=0.05, power=3.0, top_k_ratio=0.2):
        super(ContrastEnhancementLayer, self).__init__()
        self.temperature = nn.Parameter(torch.tensor([temperature]), requires_grad=True)
        self.power = power
        self.top_k_ratio = top_k_ratio
    
    def forward(self, scores):
        # 更激进的温度缩放
        scaled_scores = scores / self.temperature
        
        # 获取排序后的分数
        sorted_scores, indices = torch.sort(scaled_scores, descending=True)
        
        # 双重增强: 先应用softmax, 再进行top-k增强
        enhanced = torch.softmax(scaled_scores, dim=0)
        
        # 计算top-k阈值
        k = max(1, int(len(scores) * self.top_k_ratio))
        threshold = sorted_scores[k-1]
        
        # 创建掩码，将非top-k的值进一步降低
        mask = (scaled_scores >= threshold).float()
        boosted = enhanced * (mask + 0.1 * (1 - mask))
        
        # 应用更高的幂函数
        return boosted ** self.power

class AdaptiveGatedResidual(nn.Module):
    """自适应门控残差连接，根据图的复杂度和层深度动态调整残差权重"""
    def __init__(self, hidden_dim, num_layers, layer_idx):
        super(AdaptiveGatedResidual, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_idx = layer_idx
        
        # 基础门控
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 图复杂度调制器
        self.graph_modulator = nn.Sequential(
            nn.Linear(3, 32),  # 输入：节点数、边数、平均度
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 层深度编码（位置嵌入）
        self.depth_encoding = nn.Parameter(
            torch.zeros(1, hidden_dim),  # 为每个层位置学习一个调制向量
            requires_grad=True
        )
        
    def forward(self, current, previous, graph_metrics):
    # 基础门控值
        base_gate = self.gate(current)
    
    # 图复杂度调制
        graph_factor = self.graph_modulator(graph_metrics)
    
    # 确保维度匹配 (更安全的方式)
        if len(graph_metrics.shape) == 2:  # 批处理情况
            if base_gate.size(0) != graph_factor.size(0):
            # 将graph_factor广播到所有节点
                graph_factor = graph_factor.repeat(base_gate.size(0), 1)
                graph_factor = graph_factor[:base_gate.size(0)]  # 安全截断
    
    # 深度调制（越深的层可能需要更多的残差连接）
        depth_factor = torch.sigmoid(
            (self.layer_idx / max(self.num_layers, 1)) *  # 防止除零
            torch.ones_like(base_gate)
        )
    
        final_gate = base_gate * graph_factor * depth_factor
        
        # 应用自适应残差
        return final_gate * current + (1 - final_gate) * (previous + self.depth_encoding)
    
class EnhancedGCN(nn.Module):
    def __init__(self, in_features, hidden_features=256, heads=8, dropout_rate=0.3, num_layers=4, 
                 contrast_temp=0.05, contrast_power=3.0, top_k_ratio=0.2):
        super(EnhancedGCN, self).__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.hidden_features = hidden_features
        
        # 特征融合层
        self.feature_fusion = FeatureFusionLayer(in_features - 8)
        
        # 添加注意力权重层
        self.feature_attention = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 64),
            nn.Sigmoid()
        )
        
        # 添加维度调整层
        self.dim_adjust = nn.Linear(64, hidden_features)
        
        # 改进的图卷积网络层
        self.convs = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # 用自适应门控残差块替代简单门控
        self.residual_blocks = nn.ModuleList()
        
        for i in range(num_layers):
            # 图卷积层
            self.convs.append(GCNConv(hidden_features, hidden_features))
            self.attentions.append(GATConv(hidden_features, hidden_features // heads, heads=heads))
            self.bns.append(BatchNorm(hidden_features))
            
            # 自适应门控残差
            self.residual_blocks.append(
                AdaptiveGatedResidual(
                    hidden_dim=hidden_features,
                    num_layers=num_layers,
                    layer_idx=i
                )
            )
        
        # 特征融合
        self.jk = JumpingKnowledge(mode='cat')
        self.lin1 = nn.Linear(hidden_features * num_layers, hidden_features)
        
        # 输出层
        self.mlp = nn.Sequential(
            nn.Linear(hidden_features, hidden_features * 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(hidden_features * 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_features * 2, hidden_features),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(hidden_features),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_features, 1),
            nn.Sigmoid()
        )
        
        # 对比增强层
        self.contrast_enhancement = ContrastEnhancementLayer(
            temperature=contrast_temp, 
            power=contrast_power,
            top_k_ratio=top_k_ratio
        )
        
    def forward(self, x, edge_index, batch=None, apply_contrast=True):
        # 计算图复杂度指标 - 添加安全检查
        num_nodes = float(x.size(0))
        num_edges = float(edge_index.size(1))
        avg_degree = num_edges / max(num_nodes, 1.0)  # 避免除零
        
        # 防止数值过大
        num_nodes = min(num_nodes, 1e6)
        num_edges = min(num_edges, 1e6)
        avg_degree = min(avg_degree, 1e3)
        
        # 构造复杂度指标 [batch_size, 3] - 使用float32
        if batch is None:
            graph_metrics = torch.tensor(
                [[num_nodes, num_edges, avg_degree]], 
                device=x.device,
                dtype=torch.float32  # 明确指定类型
            )
        else:
            batch_size = batch.max() + 1
            graph_metrics = []
            for i in range(batch_size):
                mask = (batch == i)
                sub_num_nodes = mask.sum().item()
                sub_edge_mask = (batch[edge_index[0]] == i) & (batch[edge_index[1]] == i)
                sub_num_edges = sub_edge_mask.sum().item()
                sub_avg_degree = sub_num_edges / sub_num_nodes if sub_num_nodes > 0 else 0
                graph_metrics.append([
                    sub_num_nodes, 
                    sub_num_edges, 
                    sub_avg_degree
                ])
            graph_metrics = torch.tensor(graph_metrics, device=x.device)
        
        # 特征预处理
        x = self.feature_fusion(x)
        attention_weights = self.feature_attention(x)
        x = x * attention_weights
        x = self.dim_adjust(x)
        
        # 多层GNN处理
        xs = [x]
        previous_x = x
        
        for i in range(self.num_layers):
            # GCN层
            x_gcn = self.convs[i](previous_x, edge_index)
            
            # GAT层
            x_gat = self.attentions[i](previous_x, edge_index)
            
            # 融合GCN和GAT的输出
            x = x_gcn + x_gat
            
            # 归一化和激活
            x = self.bns[i](x)
            x = F.leaky_relu(x, negative_slope=0.2)
            
            # 自适应残差连接
            x = self.residual_blocks[i](
                current=x,
                previous=previous_x,
                graph_metrics=graph_metrics
            )
            
            previous_x = x
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            xs.append(x)
        
        # 移除第一个元素（原始输入）
        xs = xs[1:]
        
        # 多尺度特征融合
        x = self.jk(xs)
        x = self.lin1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        
        # 全局池化（批处理情况）
        if batch is not None:
            batch_size = batch.max().item() + 1
            global_mean = global_add_pool(x, batch)
            global_max = global_max_pool(x, batch)
            global_representation = torch.cat([global_mean, global_max], dim=1)
            
            global_per_node = global_representation[batch]
            x = x + 0.1 * global_per_node[:, :x.size(1)]
        
        # 获取原始评分
        raw_scores = self.mlp(x).squeeze()
        
        # 应用对比强化
        if apply_contrast and not self.training:
            return self.contrast_enhancement(raw_scores)
        return raw_scores



