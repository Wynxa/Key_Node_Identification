import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, JumpingKnowledge, BatchNorm, global_add_pool, global_max_pool


class FeatureFusionLayer(nn.Module):
    """特征融合层"""
    def __init__(self, embedding_dim, centrality_dim=7):
        super(FeatureFusionLayer, self).__init__()
        # 中心性特征变换
        self.centrality_encoder = nn.Sequential(
            nn.Linear(centrality_dim, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(32),
        )
        
        # 嵌入特征变换
        self.embedding_encoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(64),
        )
        
        # 融合层
        self.fusion = nn.Linear(32 + 64, 64)
        
    def forward(self, x):
        # 分离不同类型的特征
        centrality_features = x[:, :7]  # 前7个是中心性特征
        embedding_features = x[:, 7:]  # 后面是SDNE嵌入
        
        # 独立处理
        centrality_encoded = self.centrality_encoder(centrality_features)
        embedding_encoded = self.embedding_encoder(embedding_features)
        
        # 融合
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


class EnhancedGCN(nn.Module):
    def __init__(self, in_features, hidden_features=256, heads=8, dropout_rate=0.3, num_layers=4, 
                 contrast_temp=0.05, contrast_power=3.0, top_k_ratio=0.2):
        super(EnhancedGCN, self).__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.hidden_features = hidden_features
        
        # 特征融合层
        self.feature_fusion = FeatureFusionLayer(in_features - 7)  # 减去7个中心性特征
        
        # 添加注意力权重层
        self.feature_attention = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 64),
            nn.Sigmoid()
        )
        
        # 添加维度调整层，用于输入特征到hidden_features的转换
        self.dim_adjust = nn.Linear(64, hidden_features)
        
        # GCN和GAT层
        self.convs = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(num_layers):
            # 所有层都使用hidden_features维度
            self.convs.append(GCNConv(hidden_features, hidden_features))
            self.attentions.append(GATConv(hidden_features, hidden_features // heads, heads=heads))
            self.bns.append(BatchNorm(hidden_features))
        
        # 残差连接门控
        self.residual_gates = nn.ModuleList([
            nn.Linear(hidden_features, 1) for _ in range(num_layers)
        ])
            
        # 特征融合 - 使用cat而非max
        self.jk = JumpingKnowledge(mode='cat')
        self.lin1 = nn.Linear(hidden_features * num_layers, hidden_features)
        
        # 增强的MLP输出层
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
        
        # 对比强化层
        self.contrast_enhancement = ContrastEnhancementLayer(
            temperature=contrast_temp, 
            power=contrast_power,
            top_k_ratio=top_k_ratio
        )
        
    def forward(self, x, edge_index, batch=None, apply_contrast=True):
        # 特征预处理
        x = self.feature_fusion(x)  # 输出: [N, 64]
        
        # 特征注意力权重
        attention_weights = self.feature_attention(x)
        x = x * attention_weights  # 输出: [N, 64]
        
        # 先调整维度到hidden_features
        x = self.dim_adjust(x)  # 输出: [N, hidden_features]
        
        # 多层GNN处理
        xs = [x]  # 存储每一层的输出
        previous_x = x
        
        for i in range(self.num_layers):
            # GCN层
            x = self.convs[i](x, edge_index)
            # GAT层
            x = self.attentions[i](x, edge_index)
            # 归一化+激活
            x = self.bns[i](x)
            x = F.leaky_relu(x, negative_slope=0.2)
            
            # 智能残差连接
            gate = torch.sigmoid(self.residual_gates[i](x))
            x = gate * x + (1 - gate) * previous_x
            previous_x = x
            
            # Dropout
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
            # 保存当前层输出
            xs.append(x)
        
        # 移除第一个元素，因为它是原始输入
        xs = xs[1:]
        
        # 多尺度特征融合
        x = self.jk(xs)
        x = self.lin1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        
        # 全局池化（批处理情况）
        if batch is not None:
            batch_size = batch.max().item() + 1
            # 计算全局特征
            global_mean = global_add_pool(x, batch)
            global_max = global_max_pool(x, batch)
            global_representation = torch.cat([global_mean, global_max], dim=1)
            
            # 将全局特征与节点特征结合
            global_per_node = global_representation[batch]
            x = x + 0.1 * global_per_node[:, :x.size(1)]  # 轻微结合全局信息
        
        # 获取原始评分
        raw_scores = self.mlp(x).squeeze()
        
        # 应用对比强化
        if apply_contrast and not self.training:
            return self.contrast_enhancement(raw_scores)
        return raw_scores
