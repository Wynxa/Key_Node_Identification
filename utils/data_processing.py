import cudf
import cugraph as cg
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import pandas as pd
import scipy.sparse as sp
import concurrent.futures
from functools import lru_cache
from scipy.linalg import expm
from scipy.stats import rankdata


# ==========================
# 1. SDNE 模型定义
# ==========================
class SDNE(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], alpha=1e-5, beta=5, nu1=1e-4, nu2=1e-3):
        super(SDNE, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.nu1 = nu1
        self.nu2 = nu2

        # 编码器
        encoder_layers = []
        current_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.append(nn.Linear(current_dim, dim))
            encoder_layers.append(nn.ReLU())
            current_dim = dim
        self.encoder = nn.Sequential(*encoder_layers)

        # 解码器
        decoder_layers = []
        hidden_dims_reversed = hidden_dims[::-1]
        current_dim = hidden_dims[-1]
        for dim in hidden_dims_reversed[1:]:
            decoder_layers.append(nn.Linear(current_dim, dim))
            decoder_layers.append(nn.ReLU())
            current_dim = dim
        decoder_layers.append(nn.Linear(current_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def get_embeddings(self, x):
        with torch.no_grad():
            return self.encoder(x)


# ==========================
# 2. 图数据处理
# ==========================
def remap_node_ids(G):
    """将节点ID重映射为连续整数"""
    vertices = G.nodes().to_pandas().values  # 关键修改点
    vertex_to_new_id = {v: i for i, v in enumerate(vertices)}
    new_id_to_vertex = {i: v for i, v in enumerate(vertices)}

    edge_list = G.view_edge_list().to_pandas()
    src_col, dst_col = edge_list.columns[0], edge_list.columns[1]

    new_edges = []
    for _, row in edge_list.iterrows():
        src_new = vertex_to_new_id[row[src_col]]
        dst_new = vertex_to_new_id[row[dst_col]]
        new_edges.append((src_new, dst_new))

    new_gdf = cudf.DataFrame({
        'source': [e[0] for e in new_edges],
        'target': [e[1] for e in new_edges],
        'weight': 1.0
    })

    new_G = cg.Graph()
    new_G.from_cudf_edgelist(new_gdf, source='source', destination='target', edge_attr='weight')
    return new_G, vertex_to_new_id, new_id_to_vertex


def preprocess_network(G):
    """移除孤立节点并标准化ID"""
    edge_list = G.view_edge_list().to_pandas()
    src_col, dst_col = edge_list.columns[0], edge_list.columns[1]

    degree_df = G.degree()
    non_isolated = degree_df[degree_df['degree'] > 0]['vertex'].to_pandas().values

    filtered_edges = edge_list[
        edge_list[src_col].isin(non_isolated) &
        edge_list[dst_col].isin(non_isolated)
        ]

    new_gdf = cudf.DataFrame({
        src_col: filtered_edges[src_col],
        dst_col: filtered_edges[dst_col],
        'weight': 1.0
    })

    new_G = cg.Graph()
    new_G.from_cudf_edgelist(
        new_gdf,
        source=src_col,
        destination=dst_col,
        edge_attr='weight'
    )
    return remap_node_ids(new_G)


# ==========================
# 3. 节点特征计算
# ==========================
def lraspn_centrality_cugraph(G, vertex_to_new_id, new_id_to_vertex, L=3):
    """计算LRASPN中心性"""
    vertices_df = G.degree()
    vertices = vertices_df["vertex"].to_pandas().values
    degrees_df = vertices_df.set_index("vertex")["degree"].to_pandas()
    num_vertices = len(vertices)

    edge_list = G.view_edge_list().to_pandas()
    src_col, dst_col = edge_list.columns[0], edge_list.columns[1]
    G_nx = nx.from_pandas_edgelist(edge_list, source=src_col, target=dst_col)

    lraspn_scores = {}
    for vertex in vertices:
        if vertex not in new_id_to_vertex:
            continue

        original_vertex = new_id_to_vertex[vertex]
        local_influence = degrees_df.get(vertex, 0) / num_vertices

        # 计算L-hop邻居
        neighbors = []
        current_level = {vertex}
        for _ in range(L):
            next_level = set()
            for n in current_level:
                next_level.update(G.neighbors(n).to_pandas())
            neighbors.extend(current_level)
            current_level = next_level - set(neighbors)
            if not current_level:
                break
        neighbors = list(set(neighbors) - {vertex})

        # 半局部影响力
        semi_local = 0.0
        if neighbors:
            for neighbor in neighbors:
                try:
                    path = nx.shortest_path(G_nx, vertex, neighbor)
                    path_len = len(path) - 1
                    weighted = (0.5 ** path_len)
                    semi_local += np.sqrt(
                        (weighted * degrees_df[vertex]) /
                        (path_len * (degrees_df[vertex] + degrees_df.get(neighbor, 0)))
                    )
                except nx.NetworkXNoPath:
                    continue
            semi_local /= len(neighbors)

        # 子图影响力
        subgraph_nodes = neighbors + [vertex]
        valid_nodes = [n for n in subgraph_nodes if n in G_nx]
        if len(valid_nodes) >= 2:
            subgraph = G_nx.subgraph(valid_nodes)
            try:
                asp_orig = nx.average_shortest_path_length(subgraph)
                subgraph_removed = subgraph.copy()
                subgraph_removed.remove_node(vertex)
                if len(subgraph_removed.nodes()) >= 2:
                    asp_removed = nx.average_shortest_path_length(subgraph_removed)
                else:
                    asp_removed = asp_orig

                delta_asp = abs(asp_orig - asp_removed) / asp_orig
                delta_sz = abs(len(subgraph.edges()) - len(subgraph_removed.edges())) / len(subgraph.edges())
                psi = len(subgraph.edges()) / len(G_nx.edges())
                lrasp_influence = delta_sz * np.exp(delta_asp * (psi ** (2 * L)))
            except nx.NetworkXError:
                lrasp_influence = 0.0
        else:
            lrasp_influence = 0.0

        lraspn_scores[original_vertex] = (
                0.33 * local_influence +
                0.33 * semi_local +
                0.34 * lrasp_influence
        )

    return lraspn_scores


def node_embedding_cugraph(G, vertex_to_new_id, new_id_to_vertex, hidden_dims=[128, 64], epochs=100):
    """改进的SDNE节点嵌入生成"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vertices = G.nodes().to_pandas().values
    num_vertices = len(vertices)
    
    # 构建邻接矩阵
    edge_list = G.view_edge_list().to_pandas()
    src_col, dst_col = edge_list.columns[0], edge_list.columns[1]
    rows = edge_list[src_col].values
    cols = edge_list[dst_col].values
    
    adj = sp.coo_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(num_vertices, num_vertices)
    )
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)
    adj = adj.tocsr()
    
    # 添加早停策略
    model = SDNE(adj.shape[0], hidden_dims).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    adj_tensor = torch.tensor(adj.todense(), dtype=torch.float32).to(device)
    actual_epochs = min(epochs, max(10, 500000 // num_vertices))
    
    best_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(actual_epochs):
        model.train()
        optimizer.zero_grad()
        x_hat, z = model(adj_tensor)
        
        # 损失计算
        L_1st = (adj_tensor * (x_hat - adj_tensor) ** 2).sum()
        L_2nd = nn.MSELoss()(x_hat, adj_tensor)
        reg_loss = sum(p.pow(2).sum() for p in model.parameters())
        loss = L_1st + 5.0 * L_2nd + 1e-4 * reg_loss
        
        loss.backward()
        optimizer.step()
        
        # 早停检查
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'提前停止训练: Epoch {epoch+1}')
            break
            
        scheduler.step(loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{actual_epochs}, Loss: {loss.item():.4f}')
    
    # 获取嵌入
    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings(adj_tensor).cpu().numpy()
    
    return embeddings


# ==========================
# 4. 中心性指标计算
# ==========================
def degree_centrality(G, degrees_df, num_vertices):
    return {v: degrees_df.get(v, 0) / num_vertices for v in degrees_df.index}


def betweenness_centrality(G):
    try:
        bc = cg.betweenness_centrality(G)
        return dict(zip(bc["vertex"].to_pandas(), bc["betweenness_centrality"].to_pandas()))
    except Exception as e:
        print(f"介数中心性错误: {e}")
        return {}


def k_shell(G):
    try:
        ks = cg.core_number(G)
        return dict(zip(ks["vertex"].to_pandas(), ks["core_number"].to_pandas()))
    except Exception as e:
        print(f"K-shell错误: {e}")
        return {}


def closeness_centrality(G_nx):
    try:
        return nx.closeness_centrality(G_nx)
    except Exception as e:
        print(f"紧密中心性错误: {e}")
        return {}


@lru_cache(maxsize=None)
def cached_shortest_path(subgraph, source, target):
    try:
        return nx.shortest_path_length(subgraph, source, target)
    except nx.NetworkXNoPath:
        return float('inf')


def local_efficiency(G_nx):
    """计算局部效率"""
    eff_dict = {}
    for node in G_nx.nodes():
        neighbors = list(nx.neighbors(G_nx, node))
        if len(neighbors) < 2:
            eff_dict[node] = 0.0
            continue

        subgraph = G_nx.subgraph(neighbors)
        try:
            eff = nx.global_efficiency(subgraph)
        except:
            eff = 0.0
        eff_dict[node] = eff
    return eff_dict


def communicability_centrality(G, vertex_to_new_id):
    """通信中心性（返回带索引的字典）"""
    edge_list = G.view_edge_list().to_pandas()
    src_col, dst_col = edge_list.columns[0], edge_list.columns[1]
    num_vertices = G.number_of_vertices()

    A = np.zeros((num_vertices, num_vertices))
    for _, row in edge_list.iterrows():
        src = row[src_col]
        dst = row[dst_col]
        if src in vertex_to_new_id and dst in vertex_to_new_id:
            i = vertex_to_new_id[src]
            j = vertex_to_new_id[dst]
            if i < num_vertices and j < num_vertices:
                A[i, j] = 1
                A[j, i] = 1

    C = expm(A)
    return {i: v for i, v in enumerate(C.sum(axis=1))}  # 修改返回格式


# ==========================
# 5. 特征整合与归一化
# ==========================
def quantile_normalize(scores_dict):
    """分位数归一化"""
    values = np.array(list(scores_dict.values()))
    ranks = rankdata(values, method='average')
    return {k: ranks[i] / len(ranks) for i, k in enumerate(scores_dict.keys())}


def extract_features(G, vertex_to_new_id, new_id_to_vertex):
    """改进的多特征提取函数"""
    print("开始特征提取...")
    
    # 获取边列表列名
    edge_list = G.view_edge_list().to_pandas()
    src_col, dst_col = edge_list.columns[0], edge_list.columns[1]
    G_nx = nx.from_pandas_edgelist(edge_list, source=src_col, target=dst_col)
    
    # 基础数据 - 严格过滤节点
    valid_vertices = [v for v in G.nodes().to_pandas().values
                      if (v in vertex_to_new_id) and (v in new_id_to_vertex)]  # 双重验证
    degrees_df = G.degree().set_index("vertex")["degree"].to_pandas()
    num_vertices = len(valid_vertices)
    
    # 计算各中心性特征
    print("计算各类中心性特征...")
    lraspn = quantile_normalize(lraspn_centrality_cugraph(G, vertex_to_new_id, new_id_to_vertex))
    degree = quantile_normalize(degree_centrality(G, degrees_df, num_vertices))
    betweenness = quantile_normalize(betweenness_centrality(G))
    kshell = quantile_normalize(k_shell(G))
    closeness = quantile_normalize(closeness_centrality(G_nx))
    local_eff = quantile_normalize(local_efficiency(G_nx))
    communicability = quantile_normalize(communicability_centrality(G, vertex_to_new_id))
    
    # 生成节点嵌入
    print("计算节点嵌入...")
    embedding = node_embedding_cugraph(G, vertex_to_new_id, new_id_to_vertex)
    if embedding.size == 0:
        embedding = np.zeros((num_vertices, 64))
    
    # 整合所有特征
    combined_features = []
    for vertex in valid_vertices:
        try:
            original_vertex = new_id_to_vertex[vertex]
            emb_idx = vertex_to_new_id[vertex]
            
            # 为每个节点创建完整特征向量
            node_features = [
                lraspn.get(original_vertex, 0),
                degree.get(vertex, 0),
                betweenness.get(vertex, 0),
                kshell.get(vertex, 0),
                closeness.get(vertex, 0),
                local_eff.get(vertex, 0),
                communicability.get(emb_idx, 0)
            ]
            
            # 获取嵌入向量
            emb = embedding[emb_idx] if emb_idx < len(embedding) else np.zeros(64)
            
            # 整合嵌入与特征向量
            full_features = np.concatenate([node_features, emb])
            combined_features.append(full_features)
        except (KeyError, IndexError) as e:
            print(f"警告：跳过节点 {vertex}，特征数据不完整 - {str(e)}")
            continue
    
    print(f"特征提取完成，有效节点数: {len(combined_features)}/{len(valid_vertices)}")
    return np.array(combined_features)


# ==========================
# 6. 数据加载接口
# ==========================
def load_graph_from_csv(file_path):
    """从CSV加载图数据"""
    print(f"加载图数据: {file_path}")
    gdf = cudf.read_csv(file_path, header=None, names=['source', 'target'])
    gdf['weight'] = 1.0

    G = cg.Graph()
    G.from_cudf_edgelist(gdf, source='source', destination='target', edge_attr='weight')

    # 预处理
    print("预处理网络...")
    G_processed, vmap, ivmap = preprocess_network(G)
    print(f"节点数: {G_processed.number_of_vertices()}, 边数: {G_processed.number_of_edges()}")

    return G_processed, vmap, ivmap