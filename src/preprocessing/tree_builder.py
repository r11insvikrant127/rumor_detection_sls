import networkx as nx
from typing import List, Dict
from collections import deque
from datetime import datetime


class TreeBuilder:
    """
    PAPER-FAITHFUL Propagation Tree Builder.

    Paper Definition:
    -----------------
    Event = source tweet + ALL responsive tweets.
    (Connectivity reconstruction is NOT required.)

    Properties:
    - Directed edges: parent → reply
    - Graph may contain disconnected components
      due to missing/deleted tweets (allowed).
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    # --------------------------------------------------
    # UTILITIES
    # --------------------------------------------------
    def _normalize_id(self, tid) -> str:
        if tid is None:
            return ""
        return str(tid).strip()

    def _parse_time(self, t):
        if t is None:
            return None
        if isinstance(t, datetime):
            return t
        try:
            return datetime.fromisoformat(str(t).replace("Z", "+00:00"))
        except Exception:
            return None

    # --------------------------------------------------
    # MAIN BUILDER
    # --------------------------------------------------
    def build_from_tweets(
        self,
        tweets: List[Dict],
        source_id: str,
    ) -> nx.DiGraph:

        if source_id is None:
            raise ValueError(
                "Paper requires a source tweet for each event."
            )

        self.graph.clear()
        seen_ids = set()
        source_id = self._normalize_id(source_id)

        # ---------- PASS 1: ADD NODES ----------
        for tweet in tweets:

            tweet_id = self._normalize_id(
                tweet.get("id", tweet.get("tweet_id"))
            )

            if not tweet_id or tweet_id in seen_ids:
                continue

            seen_ids.add(tweet_id)

            self.graph.add_node(
                tweet_id,
                text=tweet.get("text", ""),
                user=tweet.get("user", {}),
                created_at=self._parse_time(tweet.get("created_at")),
                tweet_data=tweet,
                depth=None,
            )

        if source_id not in self.graph.nodes():
            raise ValueError("Source tweet not found in event.")

        # ---------- PASS 2: ADD EDGES ----------
        # deterministic parent → reply construction
        for tweet in tweets:

            child_id = self._normalize_id(
                tweet.get("id", tweet.get("tweet_id"))
            )

            if child_id not in self.graph.nodes():
                continue

            parent_id = self._normalize_id(
                tweet.get("in_reply_to_status_id")
            )

            if (
                parent_id
                and parent_id in self.graph.nodes()
                and parent_id != child_id
            ):
                self.graph.add_edge(parent_id, child_id)

        # NOTE:
        # We DO NOT remove unreachable nodes.
        # Paper keeps all responsive tweets even if
        # parents are missing.

        # ---------- PASS 3: COMPUTE DEPTH ----------
        self._compute_depths(source_id)

        return self.graph

    # --------------------------------------------------
    # DEPTH COMPUTATION
    # --------------------------------------------------
    def _compute_depths(self, source_id: str):
        """
        Compute depth from source tweet.

        Nodes unreachable due to missing parents
        remain with depth = None (allowed).
        """

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

      
        # fallback depth for disconnected tweets (paper-faithful)
        # -1 means: depth unknown (parent missing)
        for node in self.graph.nodes():
            if self.graph.nodes[node]["depth"] is None:
                self.graph.nodes[node]["depth"] = -1

    # --------------------------------------------------
    # METRICS (Propagation Features)
    # --------------------------------------------------
    def get_tree_metrics(self, graph: nx.DiGraph) -> Dict:

        metrics = {
            "total_nodes": graph.number_of_nodes(),
            "leaf_nodes": 0,
            "max_depth": 0,
            "average_depth": 0.0,
        }

        if graph.number_of_nodes() == 0:
            return metrics

        # leaf = no replies
        metrics["leaf_nodes"] = sum(
            1 for _, d in graph.out_degree() if d == 0
        )

        # use only valid propagation depths
        depths = [
            data["depth"]
            for _, data in graph.nodes(data=True)
            if data.get("depth", -1) >= 0
        ]
        if depths:
            metrics["max_depth"] = max(depths)
            metrics["average_depth"] = sum(depths) / len(depths)
        else:
            metrics["max_depth"] = 0
            metrics["average_depth"] = 0.0


        return metrics