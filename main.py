import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from models.enhanced_gcn import EnhancedGCN
from utils.data_processing import (
    preprocess_network,
    extract_features,
    load_graph_from_csv,
    remap_node_ids
)
from utils.evaluation import sir_experiment
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
from collections import defaultdict

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


def train_gcn_model(features, edge_index, num_epochs=100):
    """训练EnhancedGCN模型"""
    model = EnhancedGCN(
        in_features=features.shape[1],
        hidden_features=256,
        heads=4,
        dropout_rate=0.3,
        num_layers=4,
        contrast_temp=0.05,  # 更低的温度参数，增强对比效果
        contrast_power=2.5   # 更高的幂指数，进一步强化差异
    ).to(device)

    criterion = WeightedMAELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=0.001, total_steps=num_epochs)

    print("\n开始训练EnhancedGCN模型...")
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # 训练时不应用对比强化
        output = model(features, edge_index, apply_contrast=False)
        loss = criterion(output, torch.ones(features.shape[0]).to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

    print("训练完成!")
    return model


def get_key_nodes(model, features, edge_index, new_id_to_vertex, threshold=0.5):
    """获取关键节点并映射回原始ID"""
    model.eval()
    with torch.no_grad():
        # 评估时应用对比强化
        raw_scores = model(features, edge_index, apply_contrast=False)
        enhanced_scores = model(features, edge_index, apply_contrast=True)
        
        # 使用增强后的分数进行节点筛选
        scores = enhanced_scores

    # 计算动态阈值：选取前20%的节点
    # 或使用固定阈值，但在对比增强后的分数上
    sorted_scores, _ = torch.sort(scores, descending=True)
    dynamic_threshold = sorted_scores[int(len(sorted_scores) * 0.2)]
    threshold = max(threshold, dynamic_threshold)
    
    key_nodes_idx = (scores > threshold).nonzero().squeeze().tolist()
    if isinstance(key_nodes_idx, int):
        key_nodes_idx = [key_nodes_idx]

    key_nodes = []
    for idx in key_nodes_idx:
        try:
            original_id = new_id_to_vertex[idx]
            key_nodes.append(original_id)
        except KeyError:
            continue

    print(f"选取了 {len(key_nodes)} 个关键节点 (分数阈值: {threshold:.4f})")
    
    # 打印原始分数和增强分数的差异统计
    print(f"原始分数范围: {raw_scores.min().item():.4f}-{raw_scores.max().item():.4f}, 标准差: {raw_scores.std().item():.4f}")
    print(f"增强分数范围: {scores.min().item():.4f}-{scores.max().item():.4f}, 标准差: {scores.std().item():.4f}")
    
    return key_nodes, scores


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
    beta = 0.3
    gamma = 0.1
    steps = 30
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


def main():
    # 数据加载
    data = load_and_preprocess_data("/root/Key_Node_Identification/data/USairport.csv")
    vertex_to_new_id, new_id_to_vertex = data['vertex_mapping']

    # 特征提取
    features = extract_features(data['G'], vertex_to_new_id, new_id_to_vertex)
    features = torch.tensor(features, dtype=torch.float).to(device)

    # 边索引处理
    edge_index = torch.tensor(np.array(data['G_nx'].edges()).T, dtype=torch.long).to(device)
    valid_indices = (edge_index[0] < features.shape[0]) & (edge_index[1] < features.shape[0])
    edge_index = edge_index[:, valid_indices]

    # 训练GCN模型
    model = train_gcn_model(features, edge_index)

    # 获取关键节点 - 使用对比增强
    model.eval()
    with torch.no_grad():
        # 获取原始分数
        raw_scores = model(features, edge_index, apply_contrast=False)
        # 获取增强分数
        enhanced_scores = model(features, edge_index, apply_contrast=True)
        
    # 使用增强后的分数来选择节点
    sorted_scores, indices = torch.sort(enhanced_scores, descending=True)
    # 动态阈值：选取前20%的节点
    top_k = int(len(sorted_scores) * 0.2)
    threshold = sorted_scores[top_k].item()
    
    key_nodes_idx = indices[:top_k].tolist()
    gcn_nodes = []
    for idx in key_nodes_idx:
        if idx < len(new_id_to_vertex):
            original_id = new_id_to_vertex[idx]
            gcn_nodes.append(original_id)
    
    print(f"选取了 {len(gcn_nodes)} 个关键节点")
    print(f"原始分数范围: {raw_scores.min().item():.4f}-{raw_scores.max().item():.4f}, 标准差: {raw_scores.std().item():.4f}")
    print(f"增强分数范围: {enhanced_scores.min().item():.4f}-{enhanced_scores.max().item():.4f}, 标准差: {enhanced_scores.std().item():.4f}")

    # 运行对比方法
    comparison_results = run_comparison_methods(data['raw_G_nx'], len(gcn_nodes))

    # 合并所有方法结果
    all_methods = {
        'EnhancedGCN': {
            'nodes': gcn_nodes,
            'time': None
        },
        **comparison_results
    }

    # 评估
    sir_stats = evaluate_sir(all_methods, data['raw_G_nx'], repetitions=10)

    # 打印结果
    print("\n=== 最终评估结果 ===")
    print("{:<15} {:<10} {:<10} {:<10} {:<10}".format(
        "Method", "Mean", "Std", "Min", "Max"))
    for name, stat in sir_stats.items():
        print("{:<15} {:<10.1f} {:<10.1f} {:<10.1f} {:<10.1f}".format(
            name, stat['mean'], stat['std'], stat['min'], stat['max']))

    # 可视化SIR扩散过程
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        
        steps = len(sir_stats['EnhancedGCN']['avg_over_time'])
        time_steps = list(range(steps))
        
        for name, data in sir_stats.items():
            avg = data['avg_over_time']
            std = data['std_over_time']
            
            plt.plot(time_steps, avg, label=name, marker='o', markersize=4)
            plt.fill_between(time_steps, 
                            [a - s for a, s in zip(avg, std)],
                            [a + s for a, s in zip(avg, std)],
                            alpha=0.2)
        
        plt.xlabel('时间步长')
        plt.ylabel('感染节点数量')
        plt.title('各方法SIR扩散过程对比')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('sir_diffusion.png')
        print("SIR扩散过程图已保存到 sir_diffusion.png")
    except Exception as e:
        print(f"绘图失败: {e}")

    # 保存详细结果
    # 将每个时间步长的数据转换为DataFrame并保存
    time_step_data = {}
    for name in sir_stats:
        for step in range(len(sir_stats[name]['avg_over_time'])):
            step_key = f'step_{step}'
            if step_key not in time_step_data:
                time_step_data[step_key] = {}
            time_step_data[step_key][f'{name}_avg'] = sir_stats[name]['avg_over_time'][step]
            time_step_data[step_key][f'{name}_std'] = sir_stats[name]['std_over_time'][step]
    
    time_step_df = pd.DataFrame.from_dict(time_step_data, orient='index')
    time_step_df.to_csv("sir_diffusion_data.csv")
    
    # 保存总体结果
    result_df = pd.DataFrame({
        name: {'mean': stat['mean'], 'std': stat['std'], 'min': stat['min'], 'max': stat['max']}
        for name, stat in sir_stats.items()
    }).T
    result_df.to_csv("results_summary.csv")
    
    print("\n结果已保存到 results_summary.csv 和 sir_diffusion_data.csv")


if __name__ == "__main__":
    main()
