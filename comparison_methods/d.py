import networkx as nx


# 度中心性算法，Degree Centrality
def d_method(G_nx, top_k=30):
    dc = nx.degree_centrality(G_nx)
    return sorted(dc, key=lambda x: -dc[x])[:top_k]
