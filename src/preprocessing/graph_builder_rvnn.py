import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# =====================================================
# GLOBAL TF-IDF VECTORIZER
# =====================================================
# 🔥 Paper uses vocab size = 5000
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)


# =====================================================
# ADJACENCY MATRIX (USED BY GNN MODELS)
# =====================================================
def build_adjacency(graph, node_list):
    n = len(node_list)
    idx_map = {node: i for i, node in enumerate(node_list)}

    adj = np.zeros((n, n), dtype=np.float32)

    for u, v in graph.edges():
        if u in idx_map and v in idx_map:
            adj[idx_map[u], idx_map[v]] = 1.0

    # Row-normalize
    row_sum = adj.sum(axis=1, keepdims=True) + 1e-6
    adj = adj / row_sum

    return adj


# =====================================================
# CLEAN TEXT (FIXED & SAFE)
# =====================================================
def _clean_text(text):
    if text is None:
        return None

    text = str(text).strip()

    # Remove completely empty text
    if len(text) == 0:
        return None

    # 🔥 Keep short texts like "fake", "true", etc.
    return text


# =====================================================
# FIT TF-IDF ON TRAINING DATA
# =====================================================
def fit_tfidf(graphs):
    texts = []

    for graph in graphs:
        for _, data in graph.nodes(data=True):
            text = _clean_text(data.get("text", ""))

            if text is not None:
                texts.append(text)

    print(f"[TF-IDF] Valid texts: {len(texts)}")

    if len(texts) == 0:
        raise ValueError("No valid text found for TF-IDF!")

    vectorizer.fit(texts)
    print("[TF-IDF] Done.")


# =====================================================
# BUILD NODE FEATURES (ORDER SAFE)
# =====================================================
def build_node_features(graph, node_list):

    # 🔥 Ensure TF-IDF is fitted
    if not hasattr(vectorizer, "vocabulary_") or vectorizer.vocabulary_ is None:
        raise ValueError("TF-IDF vectorizer is not fitted!")

    texts = []

    # 🔥 MUST follow node_list order (topological order)
    for node in node_list:
        data = graph.nodes[node]
        text = _clean_text(data.get("text", ""))

        if text is None:
            text = "empty tweet"

        texts.append(text)

    features = vectorizer.transform(texts)

    return features.toarray().astype(np.float32)