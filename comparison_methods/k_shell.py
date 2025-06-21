import networkx as nx


# K 壳分解算法
def k_shell_method(G_nx, top_k=30):
    G_nx = G_nx.copy()  # 避免修改原图
    G_nx.remove_edges_from(nx.selfloop_edges(G_nx))  # 移除自环
    ks = nx.core_number(G_nx)
    # 获取前 top_k 个核心数最大的节点
    sorted_nodes = sorted(ks.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in sorted_nodes[:top_k]]
