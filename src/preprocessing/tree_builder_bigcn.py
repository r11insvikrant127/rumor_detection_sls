import networkx as nx
from typing import List, Dict
from collections import deque
from datetime import datetime


class TreeBuilderBiGCN:
    """
    Robust Tree Builder for BiGCN

    Fixes:
    - Handles text / full_text
    - Handles missing IDs
    - Works with PHEME structure (source + reactions)
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    def _normalize_id(self, tid):
        if tid is None:
            return ""
        return str(tid).strip()

    def _parse_time(self, t):
        if t is None:
            return None
        try:
            return datetime.fromisoformat(str(t).replace("Z", "+00:00"))
        except:
            return None

    def _get_text(self, tweet):
        # 🔥 robust text extraction
        return (
            tweet.get("text")
            or tweet.get("full_text")
            or ""
        )

    def build_from_tweets(self, tweets: List[Dict], source_id: str):

        self.graph.clear()
        seen_ids = set()

        source_id = self._normalize_id(source_id)

        # ---------- ADD NODES ----------
        for tweet in tweets:

            tweet_id = self._normalize_id(
                tweet.get("id") or tweet.get("id_str") or tweet.get("tweet_id")
            )

            if not tweet_id or tweet_id in seen_ids:
                continue

            seen_ids.add(tweet_id)

            text = self._get_text(tweet)

            self.graph.add_node(
                tweet_id,
                text=text,
                depth=None,
                tweet_data=tweet
            )

        if source_id not in self.graph.nodes():
            raise ValueError("Source tweet not found.")

        # ---------- ADD EDGES ----------
        for tweet in tweets:

            child_id = self._normalize_id(
                tweet.get("id") or tweet.get("id_str") or tweet.get("tweet_id")
            )

            parent_id = self._normalize_id(
                tweet.get("in_reply_to_status_id")
            )

            if (
                parent_id
                and parent_id in self.graph.nodes()
                and child_id in self.graph.nodes()
                and parent_id != child_id
            ):
                self.graph.add_edge(parent_id, child_id)

        # ---------- DEPTH ----------
        self._compute_depths(source_id)

        return self.graph

    def _compute_depths(self, source_id):

        for node in self.graph.nodes():
            self.graph.nodes[node]["depth"] = None

        self.graph.nodes[source_id]["depth"] = 0

        queue = deque([source_id])

        while queue:
            parent = queue.popleft()
            parent_depth = self.graph.nodes[parent]["depth"]

            for child in self.graph.successors(parent):
                if self.graph.nodes[child]["depth"] is None:
                    self.graph.nodes[child]["depth"] = parent_depth + 1
                    queue.append(child)

        # fallback
        for node in self.graph.nodes():
            if self.graph.nodes[node]["depth"] is None:
                self.graph.nodes[node]["depth"] = -1