import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# =====================================================
# TF-IDF (PAPER SETTING)
# =====================================================
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)


# =====================================================
# FIT TF-IDF (TRAIN ONLY)
# =====================================================
def fit_tfidf(root_nodes):

    texts = []

    def collect(node):
        if node.text:
            texts.append(node.text)
        for child in node.children:
            collect(child)

    for root in root_nodes:
        collect(root)

    print(f"[TF-IDF] Valid texts: {len(texts)}")

    if len(texts) == 0:
        raise ValueError("No valid text found!")

    vectorizer.fit(texts)
    print("[TF-IDF] Done.")


# =====================================================
# ASSIGN SPARSE TF-IDF (CRITICAL)
# =====================================================
def assign_tfidf_to_nodes(root_node):

    def dfs(node):
        vec = vectorizer.transform([node.text])

        # HANDLE EMPTY VECTOR
        if len(vec.indices) == 0:
            node.word = [0.0]
            node.index = [0]
        else:
            node.word = vec.data.tolist()
            node.index = vec.indices.tolist()

        for child in node.children:
            dfs(child)

    dfs(root_node)


# =====================================================
# AUTHOR TREE LOGIC
# =====================================================
def _clear_indices(root_node):
    root_node.idx = None
    for child in root_node.children:
        if child:
            _clear_indices(child)


def _get_leaf_vals(root_node):
    all_leaves = []
    layer = [root_node]

    while layer:
        next_layer = []
        for node in layer:
            if not node.children:
                all_leaves.append(node)
            else:
                next_layer.extend([child for child in node.children[::-1]])
        layer = next_layer

    X_word, X_index = [], []

    for idx, leaf in enumerate(reversed(all_leaves)):
        leaf.idx = idx
        X_word.append(leaf.word)
        X_index.append(leaf.index)

    return X_word, X_index


def _get_tree_traversal(root_node, start_idx):

    layers = []
    layer = [root_node]

    while layer:
        layers.append(layer[:])
        next_layer = []
        for node in layer:
            next_layer.extend(node.children)
        layer = next_layer

    tree = []
    internal_word = []
    internal_index = []

    idx = start_idx

    for layer in reversed(layers):
        for node in layer:

            if node.idx is not None:
                continue

            # NO PADDING (IMPORTANT)
            child_idxs = [child.idx for child in node.children]

            node.idx = idx
            tree.append(child_idxs + [node.idx])

            internal_word.append(node.word)
            internal_index.append(node.index)

            idx += 1

    return tree, internal_word, internal_index


def gen_nn_inputs(root_node):
    _clear_indices(root_node)

    X_word, X_index = _get_leaf_vals(root_node)
    tree, internal_word, internal_index = _get_tree_traversal(root_node, len(X_word))

    X_word.extend(internal_word)
    X_index.extend(internal_index)

    return (
        np.array(X_word, dtype=object),
        np.array(X_index, dtype=object),
        np.array(tree, dtype=object)
    )


# =====================================================
# FINAL WRAPPER
# =====================================================
def build_rvnn_inputs(root_node):

    X_word, X_index, tree = gen_nn_inputs(root_node)

    return {
        "X_word": X_word,
        "X_index": X_index,
        "tree": tree
    }