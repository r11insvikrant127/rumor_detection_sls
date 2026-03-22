import numpy as np


def build_adjacency(graph, node_list):
    n = len(node_list)
    idx_map = {node: i for i, node in enumerate(node_list)}

    adj = np.zeros((n, n))

    for u, v in graph.edges():
        if u in idx_map and v in idx_map:
            adj[idx_map[u], idx_map[v]] = 1

    # normalize
    row_sum = adj.sum(axis=1, keepdims=True) + 1e-6
    adj = adj / row_sum

    return adj


def build_node_features(graph, node_list):
    """
    SIMPLE but effective features:
    - degree
    - depth
    """

    feats = []

    for node in node_list:
        data = graph.nodes[node]

        degree = graph.out_degree(node)
        depth = data.get("depth", 0)

        feats.append([degree, depth])

    return np.array(feats, dtype=np.float32)