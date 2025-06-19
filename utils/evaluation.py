import numpy as np
import networkx as nx


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
        # 遍历当前感染节点的副本，避免在迭代中修改原集合
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
        results.append(len(infected))

    return results


# 牵制控制实验
def pinning_control_experiment(G, controlled_nodes):
    L = nx.laplacian_matrix(G).toarray()
    Q_max = len(controlled_nodes)
    P = 0
    for Q in range(1, Q_max + 1):
        L_Q = np.delete(np.delete(L, controlled_nodes[:Q], axis=0), controlled_nodes[:Q], axis=1)
        eigenvalues = np.linalg.eigvalsh(L_Q)
        non_zero_eigenvalues = eigenvalues[eigenvalues > 0]
        if non_zero_eigenvalues.size > 0:
            P += 1 / non_zero_eigenvalues.min()
    P /= Q_max
    return P