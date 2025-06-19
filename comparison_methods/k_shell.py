import networkx as nx


# K 壳分解算法
def k_shell_method(G_nx, top_k=30):
    ks = nx.core_number(G_nx)
    return sorted(ks, key=lambda x: -ks[x])[:top_k]
