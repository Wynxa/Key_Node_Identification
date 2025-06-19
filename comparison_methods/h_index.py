import networkx as nx


def h_index_method(G, top_k=30):
    """改进的 H-index 计算方法"""
    h_scores = {}
    degrees = dict(G.degree())

    # 迭代计算直到收敛
    prev_scores = degrees.copy()
    while True:
        current_scores = {}
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            neighbor_scores = sorted([prev_scores[n] for n in neighbors], reverse=True)
            h = 0
            while h < len(neighbor_scores) and neighbor_scores[h] > h:
                h += 1
            current_scores[node] = h
        if current_scores == prev_scores:
            break
        prev_scores = current_scores.copy()

    # 筛选 top_k 节点
    sorted_nodes = sorted(prev_scores.items(), key=lambda x: (-x[1], -degrees[x[0]]))
    return [n[0] for n in sorted_nodes[:top_k]]