import os
import json
import torch
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from torch_geometric.data import Data

BASE_PATH = "data/processed/pheme_dataset"

EVENTS = [
    "charliehebdo",
    "ferguson",
    "germanwings-crash",
    "ottawashooting",
    "sydneysiege"
]

# ===== COLLECT TEXTS =====
all_texts = []

def collect_all_texts():
    for event in EVENTS:
        event_path = os.path.join(BASE_PATH, event)
        for file in os.listdir(event_path):
            if file.endswith(".json"):
                with open(os.path.join(event_path, file), 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    tweets = [data["source"]] + data["replies"] + data.get("retweets", [])
                    for t in tweets:
                        all_texts.append(t["text"])

collect_all_texts()

# ===== TF-IDF =====
vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit(all_texts)


# ===== BUILD GRAPH =====
def process_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tweets = [data["source"]] + data["replies"] + data.get("retweets", [])

    # minimal filtering (keep most graphs)
    if len(tweets) < 2:
        return None

    id2idx = {}
    texts = []

    for i, tweet in enumerate(tweets):
        id2idx[tweet["id"]] = i
        texts.append(tweet["text"])

    x = vectorizer.transform(texts).toarray()
    x = torch.tensor(x, dtype=torch.float)

    root_id = data["source"]["id"]
    root_index = id2idx[root_id]

    edges = []

    for tweet in tweets:
        parent = tweet["in_reply_to_status_id"]
        child = tweet["id"]

        if parent is not None and parent in id2idx:
            edges.append([id2idx[parent], id2idx[child]])
        else:
            # connect orphan nodes to root
            if child != root_id:
                edges.append([root_index, id2idx[child]])

    if len(edges) == 0:
        # allow single node graph
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    BU_edge_index = edge_index.flip(0)

    label = data["label"]
    if isinstance(label, str):
        label = {"non-rumor": 0, "rumor": 1}[label]

    y = torch.tensor([label], dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        BU_edge_index=BU_edge_index,
        y=y,
        rootindex=torch.tensor([root_index])
    )


# ===== BUILD DATASET =====
dataset = []

for event in EVENTS:
    event_path = os.path.join(BASE_PATH, event)

    for file in tqdm(os.listdir(event_path), desc=f"{event}"):
        if file.endswith(".json"):
            graph = process_json(os.path.join(event_path, file))
            if graph is not None:
                dataset.append(graph)

print(f"Total graphs: {len(dataset)}")

torch.save(dataset, "pheme_bigcn_dataset.pt")