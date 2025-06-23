import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from models.enhanced_gcn import EnhancedGCN
from utils.data_processing import (
    preprocess_network,
    extract_features,
    load_graph_from_csv,
    remap_node_ids
)
from utils.evaluation import sir_experiment, kendall_tau_validation
from comparison_methods.d import d_method
from comparison_methods.bc import bc_method
from comparison_methods.k_shell import k_shell_method
from comparison_methods.cc import cc_method
from comparison_methods.h_index import h_index_method
import networkx as nx
import numpy as np
import cudf
import cugraph as cg
import pandas as pd
import time
import os
from collections import defaultdict
from scipy.stats import kendalltau, spearmanr
from matplotlib.font_manager import FontManager
import matplotlib.pyplot as plt

# 获取所有支持中文的字体
chinese_fonts = [f.name for f in FontManager().ttflist if 'CJK' in f.name or 'Hei' in f.name or 'Song' in f.name]

if chinese_fonts:
    plt.rcParams['font.family'] = chinese_fonts[0]  # 使用第一个找到的中文字体
    plt.rcParams['axes.unicode_minus'] = False


# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")



# 新的加权 MAE 损失函数
class WeightedMAELoss(nn.Module):
    def __init__(self, alpha=2.0, threshold=0.7):
        super().__init__()
        self.alpha = alpha
        self.threshold = threshold

    def forward(self, pred, target):
        weights = torch.where(target > self.threshold,
                              self.alpha * torch.ones_like(target),
                              torch.ones_like(target))
        return torch.mean(weights * torch.abs(pred - target))


class EnhancedRankingLoss(nn.Module):
    """增强版排序损失函数，包含多种策略使模型更好地区分关键节点"""
    def __init__(self, alpha=3.0, margin=0.2, variance_weight=0.15, cluster_weight=0.2):
        super().__init__()
        self.alpha = alpha  # 重要节点加权系数
        self.margin = margin  # 排序差异边界
        self.variance_weight = variance_weight  # 方差损失权重
        self.cluster_weight = cluster_weight  # 聚类损失权重
        
    def forward(self, pred, target):
        """
        pred: 模型输出的节点重要性评分 [batch_size]
        target: 目标评分 [batch_size]
        """
        # 1. 基础损失 - 加权MAE
        weights = torch.where(target > 0.7, 
                             self.alpha * torch.ones_like(target),
                             torch.ones_like(target))
        base_loss = torch.mean(weights * torch.abs(pred - target))
        
        # 2. 排序损失 - 加强节点间评分差异
        n = pred.size(0)
        if n <= 1:
            return base_loss
            
        # 计算相对排名一致性
        # 对预测和目标进行排序
        pred_ranks = torch.argsort(torch.argsort(pred, descending=True))
        target_ranks = torch.argsort(torch.argsort(target, descending=True))
        
        # 计算排序差异
        rank_diff = pred_ranks.float() - target_ranks.float()
        
        # 加权排序损失 - 根据目标重要性加权
        rank_weights = 1.0 + target * 4.0  # 重要节点排序错误代价更高
        ranking_loss = torch.mean(rank_weights * torch.abs(rank_diff)) / n
        
        # 3. 方差损失 - 鼓励预测分布更分散
        pred_std = pred.std()
        variance_loss = torch.exp(-5.0 * pred_std)  # 指数惩罚低方差
        
        # 4. 聚类损失 - 鼓励重要节点得分明显区分于不重要节点
        # 将预测分成两部分：高于和低于目标均值
        target_mean = target.mean()
        important_mask = (target > target_mean).float()
        
        # 计算两类节点预测均值
        important_pred = (pred * important_mask).sum() / (important_mask.sum() + 1e-8)
        unimportant_pred = (pred * (1 - important_mask)).sum() / ((1 - important_mask).sum() + 1e-8)
        
        # 鼓励两类节点预测均值差距大
        clustering_loss = torch.exp(-3.0 * (important_pred - unimportant_pred))
        
        # 5. 组合损失
        total_loss = base_loss + ranking_loss + self.variance_weight * variance_loss + self.cluster_weight * clustering_loss
        
        return total_loss, {
            'base_loss': base_loss.item(),
            'ranking_loss': ranking_loss.item(),
            'variance_loss': variance_loss.item(),
            'clustering_loss': clustering_loss.item(),
            'pred_std': pred_std.item(),
            'pred_min': pred.min().item(),
            'pred_max': pred.max().item()
        }


def load_and_preprocess_data(file_path):
    """加载数据并自动适配列名"""
    # 加载原始数据（无表头，两列）
    raw_df = pd.read_csv(file_path, header=None, names=["source", "target"])
    raw_cudf = cudf.from_pandas(raw_df)

    # 创建 cuGraph 图（显式指定列名）
    raw_G = cg.Graph()
    raw_G.from_cudf_edgelist(raw_cudf, source='source', destination='target')

    # 获取边列表时动态识别列名
    edge_list = raw_G.view_edge_list().to_pandas()
    source_col = edge_list.columns[0]  # 实际列名（可能是'src'/'source'等）
    target_col = edge_list.columns[1]  # 实际列名（可能是'dst'/'target'等）

    # 创建 NetworkX 图（使用实际列名）
    raw_G_nx = nx.from_pandas_edgelist(edge_list, source=source_col, target=target_col)

    # 预处理图（移除孤立节点）
    G, vertex_to_new_id, new_id_to_vertex = preprocess_network(raw_G)
    processed_edges = G.view_edge_list().to_pandas()
    G_nx = nx.from_pandas_edgelist(processed_edges, source=source_col, target=target_col)

    print(f"原始图节点数: {raw_G.number_of_vertices()}, 边数: {raw_G.number_of_edges()}")
    print(f"处理后图节点数: {G.number_of_vertices()}, 边数: {G.number_of_edges()}")

    return {
        'raw_G': raw_G,
        'raw_G_nx': raw_G_nx,
        'G': G,
        'G_nx': G_nx,
        'vertex_mapping': (vertex_to_new_id, new_id_to_vertex)
    }


# 增强版伪标签生成函数
def generate_enhanced_pseudo_labels(features, edge_index):
    """基于特征和图结构生成优化的伪标签"""
    # 1. 提取中心性特征 - 确保包含所有8个
    centrality_features = features[:, :8]  # 包含所有中心性特征
    
    # 2. 自适应加权 - 基于各指标的变异系数
    # 变异系数 = 标准差/均值，可以衡量数据的离散程度
    means = torch.mean(centrality_features, dim=0)
    stds = torch.std(centrality_features, dim=0) 
    feature_cv = stds / (means + 1e-8)
    feature_weights = F.softmax(feature_cv, dim=0)
    
    # 3. 加权组合中心性
    weighted_centrality = torch.sum(centrality_features * feature_weights, dim=1)
    
    # 4. 基于图结构的自适应分界点
    # 创建临时NetworkX图用于结构分析
    edge_list = edge_index.cpu().numpy().T
    G_nx = nx.Graph()
    G_nx.add_edges_from(edge_list)
    
    # 计算关键图指标
    avg_degree = np.mean([d for _, d in G_nx.degree()])
    try:
        # 对于大图，我们随机采样一些节点计算聚类系数
        if len(G_nx) > 10000:
            sampled_nodes = np.random.choice(list(G_nx.nodes()), 1000, replace=False)
            subgraph = G_nx.subgraph(sampled_nodes)
            cluster_coef = nx.average_clustering(subgraph)
        else:
            cluster_coef = nx.average_clustering(G_nx)
    except:
        cluster_coef = 0.1  # 默认值
    
    # 自适应阈值: 稀疏网络需要较少的关键节点，密集网络需要较多
    adaptive_threshold = max(0.05, min(0.4, 0.2 - 0.05 * avg_degree / max(cluster_coef, 0.01)))
    
    # 5. 计算分数
    sorted_indices = torch.argsort(weighted_centrality, descending=True)
    n = len(sorted_indices)
    rank_scores = torch.zeros_like(weighted_centrality)
    
    for i, idx in enumerate(sorted_indices):
        normalized_rank = i / (n - 1)
        if normalized_rank < adaptive_threshold:
            # 使用非线性曲线提高前部分差异
            score = 0.8 + 0.2 * (1 - (normalized_rank/adaptive_threshold) ** 0.6)
        else:
            # 使用非线性衰减，更陡峭的下降
            relative_pos = (normalized_rank - adaptive_threshold)/(1 - adaptive_threshold)
            score = 0.8 * (1 - relative_pos ** 1.3)
            
        rank_scores[idx] = score
    
    # 返回生成的伪标签和使用的权重
    return rank_scores, {
        'feature_weights': feature_weights.cpu().numpy(),
        'adaptive_threshold': adaptive_threshold,
        'avg_degree': avg_degree,
        'cluster_coef': cluster_coef
    }


def train_gcn_model(features, edge_index, num_epochs=1000, patience=50):
    """训练增强版EnhancedGCN模型，使用改进的损失函数，支持早停"""
    model = EnhancedGCN(
        in_features=features.shape[1],
        hidden_features=512,
        heads=16,  
        dropout_rate=0.4,
        num_layers=8,
        contrast_temp=0.01,
        contrast_power=4.5,
        top_k_ratio=0.1
    ).to(device)

    criterion = EnhancedRankingLoss(alpha=4.0, margin=0.2, variance_weight=0.15, cluster_weight=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=0.002, div_factor=10, 
                           final_div_factor=100, total_steps=num_epochs)

    print("\n开始训练EnhancedGCN模型...")
    best_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    # 生成增强版伪标签
    rank_scores, label_info = generate_enhanced_pseudo_labels(features, edge_index)
    
    # 输出伪标签生成信息
    print("\n伪标签生成信息:")
    print(f"图平均度: {label_info['avg_degree']:.4f}")
    print(f"图聚类系数: {label_info['cluster_coef']:.4f}")
    print(f"自适应阈值: {label_info['adaptive_threshold']:.4f}")
    print("中心性特征权重:")
    for i, weight in enumerate(label_info['feature_weights']):
        print(f"  特征 {i}: {weight:.4f}")
    
    # 分析生成的伪标签
    scores_np = rank_scores.cpu().numpy()
    print(f"伪标签分数统计 - 最小值: {scores_np.min():.4f}, 最大值: {scores_np.max():.4f}, 平均值: {scores_np.mean():.4f}, 标准差: {scores_np.std():.4f}")

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # 使用预先生成的增强版伪标签
        output = model(features, edge_index, apply_contrast=False)
        loss, metrics = criterion(output, rank_scores.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # 早停机制
        if loss.item() < best_loss - 1e-5:
            best_loss = loss.item()
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, " +
                  f"Base: {metrics['base_loss']:.4f}, Ranking: {metrics['ranking_loss']:.4f}, " +
                  f"Var: {metrics.get('variance_loss', 0):.4f}, Clust: {metrics.get('clustering_loss', 0):.4f}, " +
                  f"Pred Std: {metrics['pred_std']:.4f}, Min: {metrics['pred_min']:.4f}, Max: {metrics['pred_max']:.4f}")

        if patience_counter >= patience:
            print(f"早停触发，最佳Loss: {best_loss:.4f}，提前终止于第{epoch+1}轮")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print("训练完成!")
    return model


def get_key_nodes(model, features, edge_index, new_id_to_vertex, top_ratio=0.1):
    """获取关键节点"""
    model.eval()
    with torch.no_grad():
        # 获取原始和增强分数
        raw_scores = model(features, edge_index, apply_contrast=False)
        enhanced_scores = model(features, edge_index, apply_contrast=True)
        
    # 输出统计信息
    print(f"原始分数 - 最小值: {raw_scores.min().item():.4f}, 最大值: {raw_scores.max().item():.4f}, 标准差: {raw_scores.std().item():.4f}")
    print(f"增强分数 - 最小值: {enhanced_scores.min().item():.4f}, 最大值: {enhanced_scores.max().item():.4f}, 标准差: {enhanced_scores.std().item():.4f}")
    
    # 使用top-k策略而不是阈值
    k = int(len(enhanced_scores) * top_ratio)
    _, indices = torch.topk(enhanced_scores, k)
    key_nodes_idx = indices.tolist()
    
    key_nodes = []
    for idx in key_nodes_idx:
        try:
            original_id = new_id_to_vertex[idx]
            key_nodes.append(original_id)
        except KeyError:
            continue
    
    print(f"选取了 {len(key_nodes)} 个关键节点 (前{top_ratio*100}%)")
    return key_nodes, enhanced_scores


def run_comparison_methods(raw_G_nx, num_nodes):
    """运行所有对比方法"""
    methods = {
        'Degree': d_method,
        'Betweenness': bc_method,
        'KShell': k_shell_method,
        'Closeness': cc_method,
        'HIndex': h_index_method
    }

    results = {}
    for name, method in methods.items():
        start = time.time()
        nodes = method(raw_G_nx, top_k=num_nodes)
        results[name] = {
            'nodes': nodes,
            'time': time.time() - start
        }
        print(f"{name} 完成, 耗时: {results[name]['time']:.2f}s")

    return results


def evaluate_sir(methods_nodes, raw_G_nx, repetitions=10):
    """执行SIR评估，返回每个时间步长的感染节点数量"""
    # 计算网络的平均度数和平均平方度数
    degrees = dict(raw_G_nx.degree())
    avg_k = sum(degrees.values()) / len(degrees)
    avg_k_squared = sum(d**2 for d in degrees.values()) / len(degrees)
    
    # 计算流行病学临界阈值 βc
    beta_critical = avg_k / (avg_k_squared - avg_k)
    
    # 使用稍高于临界阈值的β以确保传播但不太快
    beta = 1.5 * beta_critical
    # beta = beta_critical
    
    # 输出参数信息
    print(f"\n网络特性:")
    print(f"平均度数(〈k〉): {avg_k:.4f}")
    print(f"平均平方度数(〈k²〉): {avg_k_squared:.4f}")
    print(f"流行病学临界阈值(βc): {beta_critical:.4f}")
    print(f"实际使用的传染率(β): {beta:.4f}")
    
    gamma = 0.3 # 恢复率保持不变
    steps = 50   # 时间步数保持不变
    
    # 存储每个方法在每次重复中每个时间步的结果
    results = {}
    
    print("\n开始SIR评估...")
    for name, data in methods_nodes.items():
        nodes = data['nodes']
        print(f"评估 {name} (节点数={len(nodes)})...")
        
        # 存储该方法的所有重复实验结果
        method_results = []
        
        for rep in range(repetitions):
            # 每次实验获取每个时间步的感染节点数
            infected_over_time = sir_experiment(
                raw_G_nx,
                nodes,
                beta=beta,
                gamma=gamma,
                steps=steps
            )
            method_results.append(infected_over_time)
            
        # 计算每个时间步的平均值和标准差
        avg_infected = np.mean(method_results, axis=0)
        std_infected = np.std(method_results, axis=0)
        
        results[name] = {
            'avg_over_time': avg_infected.tolist(),
            'std_over_time': std_infected.tolist(),
            'all_runs': method_results,
            'mean': np.mean([sum(run) for run in method_results]),
            'std': np.std([sum(run) for run in method_results]),
            'min': np.min([max(run) for run in method_results]),
            'max': np.max([max(run) for run in method_results])
        }
    
    return results


def compute_h_index(citations):
    """计算H-index值"""
    if not citations:
        return 0
    citations = sorted(citations, reverse=True)
    h = 0
    for i, c in enumerate(citations):
        if c >= i + 1:
            h = i + 1
        else:
            break
    return h


def main(dataset_path):
    # === 数据集名称提取并构造输出目录 ===
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    output_dir = os.path.join("outputs", dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # === 数据加载 ===
    data = load_and_preprocess_data(dataset_path)
    vertex_to_new_id, new_id_to_vertex = data['vertex_mapping']
    features = extract_features(data['G'], vertex_to_new_id, new_id_to_vertex)
    features = torch.tensor(features, dtype=torch.float).to(device)

    edge_index = torch.tensor(np.array(data['G_nx'].edges()).T, dtype=torch.long).to(device)
    valid_indices = (edge_index[0] < features.shape[0]) & (edge_index[1] < features.shape[0])
    edge_index = edge_index[:, valid_indices]

    model = train_gcn_model(features, edge_index)

    # === 获取预测分数并保存 ===
    model.eval()
    with torch.no_grad():
        raw_scores = model(features, edge_index, apply_contrast=False)
        enhanced_scores = model(features, edge_index, apply_contrast=True)
    
    raw_scores_np = raw_scores.cpu().numpy()
    enhanced_scores_np = enhanced_scores.cpu().numpy()
    
    node_scores = []
    for i in range(len(raw_scores_np)):
        if i < len(new_id_to_vertex):
            original_id = new_id_to_vertex[i]
            node_scores.append({
                'node_id': original_id,
                'raw_score': raw_scores_np[i],
                'enhanced_score': enhanced_scores_np[i]
            })

    scores_df = pd.DataFrame(node_scores).sort_values('enhanced_score', ascending=False)
    scores_df.to_csv(os.path.join(output_dir, "node_scores.csv"), index=False)

    # === 选取关键节点并分析 ===
    sorted_scores, indices = torch.sort(enhanced_scores, descending=True)
    top_k = int(len(sorted_scores) * 0.1)
    key_nodes_idx = indices[:top_k].tolist()
    gcn_nodes = [new_id_to_vertex[idx] for idx in key_nodes_idx if idx in new_id_to_vertex]

    top_nodes_df = scores_df.head(top_k)
    node_analytics = {
        'top_nodes_mean_raw': top_nodes_df['raw_score'].mean(),
        'top_nodes_mean_enhanced': top_nodes_df['enhanced_score'].mean(),
        'non_top_nodes_mean_raw': scores_df.iloc[top_k:]['raw_score'].mean(),
        'non_top_nodes_mean_enhanced': scores_df.iloc[top_k:]['enhanced_score'].mean(),
        'enhancement_ratio': top_nodes_df['enhanced_score'].mean() / scores_df.iloc[top_k:]['enhanced_score'].mean() 
    }
    pd.DataFrame([node_analytics]).to_csv(os.path.join(output_dir, "node_analytics.csv"), index=False)

    # === 评估比较方法 ===
    comparison_results = run_comparison_methods(data['raw_G_nx'], len(gcn_nodes))
    all_methods = {'EnhancedGCN': {'nodes': gcn_nodes, 'time': None}, **comparison_results}
    sir_stats = evaluate_sir(all_methods, data['raw_G_nx'], repetitions=500)

    # === SIR 可视化 ===
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        time_steps = list(range(len(sir_stats['EnhancedGCN']['avg_over_time'])))
        for name, data in sir_stats.items():
            avg, std = data['avg_over_time'], data['std_over_time']
            plt.plot(time_steps, avg, label=name, marker='o', markersize=4)
            plt.fill_between(time_steps, 
                             [a - s for a, s in zip(avg, std)],
                             [a + s for a, s in zip(avg, std)],
                             alpha=0.2)
        plt.xlabel('时间步长')
        plt.ylabel('感染节点数量')
        plt.title(f'SIR传播过程 - {dataset_name}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sir_diffusion.png"))
        plt.close()
    except Exception as e:
        print(f"绘图失败: {e}")

    # === 保存SIR数据 ===
    time_step_data = {}
    for name in sir_stats:
        for step in range(len(sir_stats[name]['avg_over_time'])):
            step_key = f'step_{step}'
            if step_key not in time_step_data:
                time_step_data[step_key] = {}
            time_step_data[step_key][f'{name}_avg'] = sir_stats[name]['avg_over_time'][step]
            time_step_data[step_key][f'{name}_std'] = sir_stats[name]['std_over_time'][step]
    
    pd.DataFrame.from_dict(time_step_data, orient='index').to_csv(
        os.path.join(output_dir, "sir_diffusion_data.csv")
    )

    pd.DataFrame({
        name: {'mean': stat['mean'], 'std': stat['std'], 'min': stat['min'], 'max': stat['max']}
        for name, stat in sir_stats.items()
    }).T.to_csv(os.path.join(output_dir, "results_summary.csv"))

    print(f"✅ 完成 {dataset_name} 数据集，结果保存在 {output_dir}")

if __name__ == "__main__":
    # 定义多个数据集路径
    dataset_paths = [
        # "/root/gnn/data/USairport.csv",
        # "/root/gnn/data/eo.csv",
        # "/root/gnn/data/zebra.csv"
        # "/root/gnn/data/email-univ.csv"
        # "/root/gnn/data/NS-CG.csv"
        # "/root/gnn/data/airctl.csv"
         "/root/gnn/data/C-elegans.csv"
        # "/root/gnn/data/ba_graph_1000_avg_degree_20.csv"
         # "/root/gnn/data/cora.csv"
    ]

    # 遍历每个数据集，运行主流程
    for path in dataset_paths:
        print(f"\n===== 处理数据集：{path} =====")
        try:
            main(path)
        except Exception as e:
            print(f"❌ 处理 {path} 时发生错误: {e}")
