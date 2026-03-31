import os
import json
import torch
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from torch_geometric.data import Data
from datetime import datetime

# ===== PATH =====
BASE_PATH = "data/processed/pheme_dataset"

EVENTS = [
    "charliehebdo",
    "ferguson",
    "germanwings-crash",
    "ottawashooting",
    "sydneysiege"
]

# ===== STEP 1: LOAD ALL TEXTS =====
all_texts = []

def collect_all_texts():
    for event in EVENTS:
        event_path = os.path.join(BASE_PATH, event)
        for file in os.listdir(event_path):
            if file.endswith(".json"):
                with open(os.path.join(event_path, file), 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    texts = [data["source"]["text"]] + [r["text"] for r in data["replies"]]
                    all_texts.extend(texts)

collect_all_texts()

# ===== STEP 2: TF-IDF (5000 like paper) =====
vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit(all_texts)

# ===== TIME PARSER (only for stats) =====
def parse_time(time_str):
    return datetime.strptime(time_str, "%a %b %d %H:%M:%S %z %Y")

# ===== STEP 3: BUILD GRAPH =====
def process_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # ===== BUILD TWEETS =====
    tweets = [data["source"]] + data["replies"]

    # ===== FILTER SMALL TREES =====
    if len(tweets) < 3:
        return None

    id2idx = {}
    texts = []

    for i, tweet in enumerate(tweets):
        id2idx[tweet["id"]] = i
        texts.append(tweet["text"])

    # ===== NODE FEATURES =====
    x = vectorizer.transform(texts).toarray()
    x = torch.tensor(x, dtype=torch.float)

    # ===== TD EDGES =====
    edges = []
    for tweet in tweets:
        parent = tweet["in_reply_to_status_id"]
        child = tweet["id"]

        if parent is not None and parent in id2idx:
            edges.append([id2idx[parent], id2idx[child]])

    if len(edges) == 0:
        return None

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # ===== BU EDGES =====
    BU_edge_index = edge_index.flip(0)

    # ===== ROOT =====
    root_id = data["source"]["id"]
    if root_id not in id2idx:
        return None

    root_index = id2idx[root_id]

    # ===== ROOT FEATURE =====
    root_feat = x[root_index]   # shape: [5000]

    # ===== LABEL =====
    label = data["label"]
    if isinstance(label, str):
        label_map = {"non-rumor": 0, "rumor": 1}
        label = label_map[label]

    y = torch.tensor([label], dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        BU_edge_index=BU_edge_index,
        y=y,
        root=root_feat,
        rootindex=torch.tensor([root_index])
    ), data


# ===== STEP 4: CREATE DATASET =====
dataset = []
raw_data_info = []

for event in EVENTS:
    event_path = os.path.join(BASE_PATH, event)

    for file in tqdm(os.listdir(event_path), desc=f"Processing {event}"):
        if file.endswith(".json"):
            path = os.path.join(event_path, file)
            result = process_json(path)

            if result is not None:
                graph, raw = result
                dataset.append(graph)
                raw_data_info.append(raw)

print(f"\nTotal graphs: {len(dataset)}")


# ===== STEP 5: PRINT STATS =====
def print_stats(raw_data_info):
    num_events = len(raw_data_info)
    num_posts = 0
    users = set()

    rumor_count = 0
    non_rumor_count = 0

    posts_per_event = []
    time_lengths = []

    for data in raw_data_info:
        tweets = [data["source"]] + data["replies"]

        num_posts += len(tweets)
        posts_per_event.append(len(tweets))

        for t in tweets:
            users.add(t["user"]["id"])

        label = data["label"]
        if isinstance(label, str):
            label_map = {"non-rumor": 0, "rumor": 1}
            label = label_map[label]

        if label == 1:
            rumor_count += 1
        else:
            non_rumor_count += 1

        try:
            times = [parse_time(t["created_at"]) for t in tweets]
            diff = (max(times) - min(times)).total_seconds() / 3600.0
            time_lengths.append(diff)
        except:
            continue

    print("\n===== DATASET STATISTICS =====")
    print(f"# of posts: {num_posts}")
    print(f"# of users: {len(users)}")
    print(f"# of events: {num_events}")
    print(f"# of Rumors: {rumor_count}")
    print(f"# of Non-rumors: {non_rumor_count}")
    print(f"Avg # of posts / event: {sum(posts_per_event)/len(posts_per_event):.2f}")
    print(f"Max # of posts / event: {max(posts_per_event)}")
    print(f"Min # of posts / event: {min(posts_per_event)}")

    if len(time_lengths) > 0:
        print(f"Avg time length / event (hours): {sum(time_lengths)/len(time_lengths):.2f}")


print_stats(raw_data_info)

# ===== SAVE =====
torch.save(dataset, "pheme_bigcn_dataset.pt")