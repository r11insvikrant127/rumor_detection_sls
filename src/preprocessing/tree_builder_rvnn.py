import networkx as nx
from typing import List, Dict
from datetime import datetime


class TreeBuilderRvNN:
    """
    Clean tree builder ONLY for RvNN.
    Guarantees:
    - Text is preserved
    - Single root
    - Valid tree structure
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    def _normalize_id(self, tid):
        return str(tid).strip() if tid else ""

    def _parse_time(self, t):
        if t is None:
            return None
        if isinstance(t, datetime):
            return t
        try:
            return datetime.fromisoformat(str(t).replace("Z", "+00:00"))
        except:
            return None

    def build(self, tweets: List[Dict], source_id: str):

        self.graph.clear()
        source_id = self._normalize_id(source_id)

        # -------- ADD NODES --------
        for tweet in tweets:
            tid = self._normalize_id(tweet.get("id"))

            if not tid:
                continue

            self.graph.add_node(
                tid,
                text=tweet.get("text", ""),   # 🔥 CRITICAL
                created_at=self._parse_time(tweet.get("created_at"))
            )

        if source_id not in self.graph:
            raise ValueError("Source tweet missing")

        # -------- ADD EDGES --------
        for tweet in tweets:
            child = self._normalize_id(tweet.get("id"))
            parent = self._normalize_id(tweet.get("in_reply_to_status_id"))

            if parent and parent in self.graph and child != parent:
                self.graph.add_edge(parent, child)

        # -------- FIX ORPHANS (VERY IMPORTANT) --------
        for node in self.graph.nodes():
            if node == source_id:
                continue

            if self.graph.in_degree(node) == 0:
                self.graph.add_edge(source_id, node)

        return self.graph