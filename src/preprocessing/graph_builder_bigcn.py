import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# =====================================================
# GLOBAL TF-IDF VECTORIZER
# =====================================================
vectorizer = TfidfVectorizer(
    max_features=3000,
    stop_words="english"
)


# =====================================================
# FIT TF-IDF (CALL ONCE BEFORE TRAINING)
# =====================================================
def fit_tfidf(graphs):
    texts = []

    for graph in graphs:
        for _, data in graph.nodes(data=True):
            text = data.get("text", "")
            texts.append(text)

    print(f"[TF-IDF] Fitting on {len(texts)} tweets...")
    vectorizer.fit(texts)
    print("[TF-IDF] Done.")


# =====================================================
# BUILD NODE FEATURES
# =====================================================
def build_node_features(graph, node_list):
    texts = []

    for node in node_list:
        data = graph.nodes[node]
        text = data.get("text", "")
        texts.append(text)

    features = vectorizer.transform(texts)
    return features.toarray().astype(np.float32)


# =====================================================
# BUILD ADJACENCY MATRIX (Bi-GCN CORRECT VERSION)
# =====================================================
def build_adjacency(graph, node_list):
    n = len(node_list)
    idx_map = {node: i for i, node in enumerate(node_list)}

    adj = np.zeros((n, n), dtype=np.float32)

    # Directed edges (parent -> child)
    for u, v in graph.edges():
        if u in idx_map and v in idx_map:
            adj[idx_map[u], idx_map[v]] = 1.0

    # -------------------------------
    # ADD SELF-LOOPS
    # -------------------------------
    adj = adj + np.eye(n, dtype=np.float32)

    # -------------------------------
    # SYMMETRIC NORMALIZATION
    # A_hat = D^(-1/2) A D^(-1/2)
    # -------------------------------
    degree = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(degree + 1e-6, -0.5)
    d_mat = np.diag(d_inv_sqrt)

    adj = d_mat @ adj @ d_mat

    return adj