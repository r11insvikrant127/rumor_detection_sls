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
# ADJACENCY MATRIX (USED BY GNN MODELS)
# =====================================================
def build_adjacency(graph, node_list):
    n = len(node_list)
    idx_map = {node: i for i, node in enumerate(node_list)}

    adj = np.zeros((n, n))

    for u, v in graph.edges():
        if u in idx_map and v in idx_map:
            adj[idx_map[u], idx_map[v]] = 1

    # normalize row-wise
    row_sum = adj.sum(axis=1, keepdims=True) + 1e-6
    adj = adj / row_sum

    return adj


# =====================================================
# FIT TF-IDF ON TRAINING DATA (CALL ONCE)
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
# BUILD NODE FEATURES (TF-IDF)
# =====================================================
def build_node_features(graph, node_list):
    texts = []

    for node in node_list:
        data = graph.nodes[node]
        text = data.get("text", "")
        texts.append(text)

    # transform using pre-fitted vectorizer
    features = vectorizer.transform(texts)

    return features.toarray().astype(np.float32)