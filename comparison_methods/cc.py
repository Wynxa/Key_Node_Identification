import networkx as nx


# 接近中心性算法，Closeness Centrality
def cc_method(G_nx, top_k=30):
    cc = nx.closeness_centrality(G_nx)
    return sorted(cc, key=lambda x: -cc[x])[:top_k]
