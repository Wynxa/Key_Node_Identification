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
    """对比强化层 - 增加节点评分差异"""
    def __init__(self, temperature=0.1, power=2.0):
        super(ContrastEnhancementLayer, self).__init__()
        self.temperature = nn.Parameter(torch.tensor([temperature]), requires_grad=True)
        self.power = power
    
    def forward(self, scores):
        # 应用温度缩放来放大差异
        scaled_scores = scores / self.temperature
        
        # 使用softmax来放大差异同时保持值域
        enhanced = torch.softmax(scaled_scores, dim=0)
        
        # 应用幂函数进一步增强关键节点的突出性
        return enhanced ** self.power


class EnhancedGCN(nn.Module):
    def __init__(self, in_features, hidden_features=256, heads=4, dropout_rate=0.3, num_layers=4, 
                 contrast_temp=0.1, contrast_power=2.0):
        super(EnhancedGCN, self).__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        # 特征融合层
        self.feature_fusion = FeatureFusionLayer(in_features - 7)  # 减去7个中心性特征
        
        # GCN和GAT层
        self.convs = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = 64 if i == 0 else hidden_features  # 融合后特征的维度是64
            self.convs.append(GCNConv(in_dim, hidden_features))
            self.attentions.append(GATConv(hidden_features, hidden_features // heads, heads=heads))
            self.bns.append(BatchNorm(hidden_features))
            
        # 特征融合
        self.jk = JumpingKnowledge(mode='max')  # 采用max方式
        self.lin1 = nn.Linear(hidden_features, hidden_features)
        
        # 输出层 - 简化版
        self.mlp = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_features, 1),
            nn.Sigmoid()  # 确保输出在0-1之间
        )
        
        # 对比强化层 - 新增
        self.contrast_enhancement = ContrastEnhancementLayer(
            temperature=contrast_temp, 
            power=contrast_power
        )
        
    def forward(self, x, edge_index, batch=None, apply_contrast=True):
        # 特征预处理
        x = self.feature_fusion(x)
        
        # 多层GNN处理
        xs = []
        for i in range(self.num_layers):
            # GCN层
            x = self.convs[i](x, edge_index)
            # GAT层
            x = self.attentions[i](x, edge_index)
            # 归一化+激活
            x = self.bns[i](x)
            x = F.leaky_relu(x, negative_slope=0.2)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
            xs.append(x)
            
        # 多尺度特征融合
        x = self.jk(xs)
        x = self.lin1(x)
        x = F.leaky_relu(x)
        
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
        
        # 应用对比强化（在评估阶段可选）
        if apply_contrast and not self.training:
            return self.contrast_enhancement(raw_scores)
        return raw_scores
