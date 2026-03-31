import json
import networkx as nx
from datetime import datetime
import re


class Node_tweet:
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None
        self.text = ""


class TreeBuilderRvNN:

    def __init__(self):
        self.graph = nx.DiGraph()

    def _normalize_id(self, tid):
        return str(tid).strip() if tid else ""

    def _parse_time(self, t):
        if t is None:
            return None
        try:
            return datetime.fromisoformat(str(t).replace("Z", "+00:00"))
        except:
            return None

    def _clean_text(self, text):
        if text is None:
            return ""

        text = str(text).lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#\w+", "", text)
        return text.strip()

    def build_from_json(self, json_path):

        self.graph.clear()

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        all_tweets = {}

        source = data["source"]
        root_id = self._normalize_id(source.get("id"))
        all_tweets[root_id] = source

        for t in data.get("replies", []):
            tid = self._normalize_id(t.get("id"))
            if tid:
                all_tweets[tid] = t

        for t in data.get("tweets", []):
            tid = self._normalize_id(t.get("id"))
            if tid:
                all_tweets[tid] = t

        # ADD NODES
        for tid, tweet in all_tweets.items():

            if tweet.get("retweeted", False):
                continue

            text = self._clean_text(tweet.get("text", ""))

            self.graph.add_node(
                tid,
                text=text,
                created_at=self._parse_time(tweet.get("created_at"))
            )

        # ROOT SAFETY
        if root_id not in self.graph:
            raise ValueError("Root removed during preprocessing!")

        # ADD EDGES
        for tid, tweet in all_tweets.items():

            child = self._normalize_id(tweet.get("id"))
            parent = self._normalize_id(tweet.get("in_reply_to_status_id"))

            if child not in self.graph:
                continue

            if parent and parent in self.graph and parent != child:
                self.graph.add_edge(parent, child)
            else:
                if child != root_id:
                    self.graph.add_edge(root_id, child)

        # FIX ORPHANS
        for node in list(self.graph.nodes()):
            if node == root_id:
                continue
            if self.graph.in_degree(node) == 0:
                self.graph.add_edge(root_id, node)

        # ENFORCE TREE
        for node in list(self.graph.nodes()):
            if node == root_id:
                continue
            parents = list(self.graph.predecessors(node))
            if len(parents) > 1:
                for p in parents[1:]:
                    self.graph.remove_edge(p, node)

        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Graph is not a valid tree!")

        # CONVERT TO Node_tweet
        node_map = {}

        for node in self.graph.nodes():
            node_map[node] = Node_tweet(idx=node)
            node_map[node].text = self.graph.nodes[node]["text"]

        for parent, child in self.graph.edges():
            node_map[parent].children.append(node_map[child])
            node_map[child].parent = node_map[parent]

        return node_map[root_id]