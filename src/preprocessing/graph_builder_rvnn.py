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
# CLEAN TEXT (CRITICAL FIX)
# =====================================================
def _clean_text(text):
    if text is None:
        return None

    text = str(text).strip()

    # remove empty
    if len(text) == 0:
        return None

    # remove very short / useless text
    if len(text.split()) < 2:
        return None

    return text


# =====================================================
# FIT TF-IDF ON TRAINING DATA (FIXED)
# =====================================================
def fit_tfidf(graphs):
    texts = []

    for graph in graphs:
        for _, data in graph.nodes(data=True):
            text = _clean_text(data.get("text", ""))

            if text is not None:
                texts.append(text)

    print(f"[TF-IDF] Valid texts: {len(texts)}")

    # safety check
    if len(texts) == 0:
        raise ValueError("No valid text found for TF-IDF!")

    vectorizer.fit(texts)
    print("[TF-IDF] Done.")


# =====================================================
# BUILD NODE FEATURES (SAFE VERSION)
# =====================================================
def build_node_features(graph, node_list):
    texts = []

    for node in node_list:
        data = graph.nodes[node]
        text = _clean_text(data.get("text", ""))

        # fallback for bad text
        if text is None:
            text = "empty tweet"

        texts.append(text)

    features = vectorizer.transform(texts)

    return features.toarray().astype(np.float32)