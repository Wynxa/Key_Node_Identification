import networkx as nx


# 介数中心性算法，Betweenness Centrality
def bc_method(G_nx, top_k=30):
    bc = nx.betweenness_centrality(G_nx)
    return sorted(bc, key=lambda x: -bc[x])[:top_k]
