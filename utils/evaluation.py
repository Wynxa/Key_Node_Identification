import numpy as np
import networkx as nx
from scipy.stats import kendalltau, spearmanr


def sir_experiment(graph, initial_infected, beta=0.5, gamma=0.1, steps=30):
    valid_initial_infected = [node for node in initial_infected if node in graph.nodes]
    if not valid_initial_infected:
        print("没有有效的初始感染节点，无法进行 SIR 实验")
        return []

    infected = set(valid_initial_infected)
    susceptible = set(graph.nodes) - infected
    recovered = set()
    results = []

    for _ in range(steps):
        new_infected = set()
        for node in list(infected):
            neighbors = set(graph.neighbors(node)) & susceptible
            for neighbor in neighbors:
                if np.random.random() < beta:
                    new_infected.add(neighbor)
            if np.random.random() < gamma:
                infected.remove(node)
                recovered.add(node)
        infected.update(new_infected)
        susceptible -= new_infected
        # 输出累计感染节点数（已感染+已恢复）
        results.append(len(infected) + len(recovered))

    return results


# 牵制控制实验
# def pinning_control_experiment(G, controlled_nodes):
#     L = nx.laplacian_matrix(G).toarray()
#     Q_max = len(controlled_nodes)
#     P = 0
#     for Q in range(1, Q_max + 1):
#         L_Q = np.delete(np.delete(L, controlled_nodes[:Q], axis=0), controlled_nodes[:Q], axis=1)
#         eigenvalues = np.linalg.eigvalsh(L_Q)
#         non_zero_eigenvalues = eigenvalues[eigenvalues > 0]
#         if non_zero_eigenvalues.size > 0:
#             P += 1 / non_zero_eigenvalues.min()
#     P /= Q_max
#     return P


# 新增：肯德尔系数相关函数
def evaluate_ranking_correlation(method_scores, sir_impacts):
    """
    计算节点排名与SIR传播能力之间的肯德尔相关系数
    使用严格单调递减序列作为比较标准
    
    参数:
    method_scores - 字典，包含 {node_id: score} 形式的方法评分
    sir_impacts - 字典，包含 {node_id: impact} 形式的SIR影响力
    
    返回:
    包含肯德尔tau值、p值和其他相关统计量的字典
    """
    # 获取共同节点
    common_nodes = set(method_scores.keys()) & set(sir_impacts.keys())
    if not common_nodes:
        return {"kendall_tau": 0, "p_value": 1, "common_nodes": 0}
    
    # 基于分数创建严格单调递减序列（排名）
    # 先将节点按分数降序排序
    method_ranked_nodes = sorted(common_nodes, key=lambda node: method_scores[node], reverse=True)
    sir_ranked_nodes = sorted(common_nodes, key=lambda node: sir_impacts[node], reverse=True)
    
    # 为每个节点分配排名（排名从0开始）
    method_ranks = {node: rank for rank, node in enumerate(method_ranked_nodes)}
    sir_ranks = {node: rank for rank, node in enumerate(sir_ranked_nodes)}
    
    # 提取排名序列（保持节点顺序一致）
    method_rank_values = [method_ranks[node] for node in common_nodes]
    sir_rank_values = [sir_ranks[node] for node in common_nodes]
    
    # 计算肯德尔tau系数（比较排名而非原始分数）
    tau, p_value = kendalltau(method_rank_values, sir_rank_values)
    
    # 计算Spearman等级相关系数作为补充
    rho, rho_p = spearmanr(method_rank_values, sir_rank_values)
    
    return {
        "kendall_tau": tau, 
        "p_value": p_value,
        "spearman_rho": rho,
        "spearman_p": rho_p,
        "common_nodes": len(common_nodes),
        "method_top10": method_ranked_nodes[:10],  # 记录方法识别的前10个节点
        "sir_top10": sir_ranked_nodes[:10]         # 记录SIR影响力最大的前10个节点
    }


def run_individual_sir_tests(graph, nodes_list, beta=0.5, gamma=0.1, steps=15, repetitions=20):
    """
    为每个节点单独运行SIR实验，评估其传播影响力
    
    参数:
    graph - NetworkX图对象
    nodes_list - 要评估的节点列表
    beta, gamma, steps - SIR模型参数
    repetitions - 每个节点重复实验的次数
    
    返回:
    字典，{node: avg_impact} 形式的节点传播影响力
    """
    result_dict = {}
    total_nodes = len(nodes_list)
    
    print(f"开始对{total_nodes}个节点进行单独SIR测试...")
    
    for i, node in enumerate(nodes_list):
        if i % 10 == 0:  # 每评估10个节点打印一次进度
            print(f"进度: {i}/{total_nodes} 节点")
            
        if node not in graph.nodes:
            continue
            
        total_infected = 0
        
        for _ in range(repetitions):
            # 以单个节点为初始感染点运行SIR模型
            infected_over_time = sir_experiment(graph, [node], beta=beta, gamma=gamma, steps=steps)
            # 使用感染总量作为影响力指标
            total_infected += sum(infected_over_time)
        
        # 计算平均传播影响力
        result_dict[node] = total_infected / repetitions
        
    print(f"单独SIR测试完成，评估了{len(result_dict)}个节点")
    return result_dict


def kendall_tau_validation(methods_scores, raw_G_nx, beta=None, sample_size=None):
    """
    使用肯德尔系数验证不同方法识别的关键节点与实际传播能力的相关性
    基于严格单调递减序列的比较
    
    参数:
    methods_scores - 字典，{method_name: {node_id: score}} 形式的各方法节点分数
    raw_G_nx - NetworkX图对象
    beta - SIR模型的传播率，若为None则自动计算
    sample_size - 随机采样的节点数量，若为None则使用所有节点
    
    返回:
    各方法的肯德尔系数验证结果和节点SIR影响力结果
    """
    # 获取所有节点的并集
    all_methods_nodes = set()
    for method_scores in methods_scores.values():
        all_methods_nodes.update(method_scores.keys())
    
    # 如果指定了样本大小且节点数超过样本大小，则进行随机采样
    test_nodes = list(all_methods_nodes)
    if sample_size and len(test_nodes) > sample_size:
        np.random.seed(42)  # 固定随机种子以确保结果可复现
        test_nodes = np.random.choice(test_nodes, size=sample_size, replace=False).tolist()
    
    # 计算网络参数
    if beta is None:
        degrees = dict(raw_G_nx.degree())
        avg_k = sum(degrees.values()) / len(degrees)
        avg_k_squared = sum(d**2 for d in degrees.values()) / len(degrees)
        beta_critical = avg_k / (avg_k_squared - avg_k)
        beta = 1.5 * beta_critical
    
    gamma = 0.1  # 恢复率
    steps = 15   # 时间步数(减少以提高效率)
    
    print(f"\n开始肯德尔系数验证，测试节点数: {len(test_nodes)}")
    print(f"SIR参数 - beta: {beta:.4f}, gamma: {gamma}, steps: {steps}")
    
    # 运行单独SIR测试获取节点影响力
    node_impacts = run_individual_sir_tests(
        raw_G_nx, test_nodes, beta=beta, gamma=gamma, steps=steps, repetitions=20
    )
    
    # 将SIR影响力结果按降序排列，创建标准参考序列
    sir_ranked_nodes = sorted(node_impacts.items(), key=lambda x: x[1], reverse=True)
    print(f"\nSIR影响力最高的前10个节点:")
    for i, (node, impact) in enumerate(sir_ranked_nodes[:10]):
        print(f"  {i+1}. 节点{node}: 影响力 {impact:.2f}")
    
    # 计算各方法的肯德尔系数
    correlation_results = {}
    print("\n各方法的排序相关性:")
    for method, scores in methods_scores.items():
        correlation = evaluate_ranking_correlation(scores, node_impacts)
        correlation_results[method] = correlation
        
        # 显示每个方法识别的前10个节点
        method_top = sorted(((n, scores[n]) for n in scores if n in node_impacts), 
                           key=lambda x: x[1], reverse=True)[:10]
        
        print(f"\n方法: {method}")
        print(f"  Kendall's tau: {correlation['kendall_tau']:.4f}, p-value: {correlation['p_value']:.4e}")
        print(f"  Spearman's rho: {correlation['spearman_rho']:.4f}, 共同节点: {correlation['common_nodes']}")
        
        print(f"  该方法识别的前10个节点:")
        for i, (node, score) in enumerate(method_top):
            # 查找该节点在SIR排名中的位置
            sir_rank = next((i for i, (n, _) in enumerate(sir_ranked_nodes) if n == node), None)
            sir_value = node_impacts[node]
            print(f"    {i+1}. 节点{node}: 分数 {score:.4f}, SIR影响力 {sir_value:.2f}, SIR排名 {sir_rank+1 if sir_rank is not None else 'N/A'}")
    
    return correlation_results, node_impacts, sir_ranked_nodes