import networkx as nx
from typing import List, Dict, Optional
from collections import deque
from datetime import datetime


class TreeBuilder:
    """Build propagation tree from tweet data (Paper-aligned version)."""

    def __init__(self):
        self.graph = nx.DiGraph()

    # --------------------------------------------------
    # Utility
    # --------------------------------------------------
    def _normalize_id(self, tid) -> str:
        if tid is None:
            return ''
        return str(tid).strip()

    def _parse_time(self, t):
        """Safely parse timestamps."""
        if t is None:
            return None
        if isinstance(t, datetime):
            return t
        try:
            return datetime.fromisoformat(str(t).replace('Z', '+00:00'))
        except Exception:
            return None

    # --------------------------------------------------
    # MAIN BUILDER
    # --------------------------------------------------
    def build_from_tweets(
        self,
        tweets: List[Dict],
        source_id: Optional[str] = None,
        max_depth: int = None,
    ) -> nx.DiGraph:

        self.graph.clear()
        seen_ids = set()

        # ---------- PASS 1: ADD NODES ----------
        for tweet in tweets:
            tweet_id = self._normalize_id(
                tweet.get('id', tweet.get('tweet_id', ''))
            )

            if not tweet_id or tweet_id in seen_ids:
                continue

            seen_ids.add(tweet_id)

            created_at = self._parse_time(tweet.get('created_at'))

            self.graph.add_node(
                tweet_id,
                text=tweet.get('text', ''),
                user=tweet.get('user', {}),
                created_at=created_at,
                tweet_data=tweet,
                depth=None,
            )

        # ---------- PASS 2: ADD EDGES ----------
        for tweet in tweets:
            tweet_id = self._normalize_id(
                tweet.get('id', tweet.get('tweet_id', ''))
            )

            if tweet_id not in self.graph.nodes():
                continue

            child_time = self.graph.nodes[tweet_id]['created_at']

            parent_candidates = []

            response_to = tweet.get('response_to')
            if response_to:
                parent_candidates.append(response_to)

            in_reply_to = tweet.get('in_reply_to_status_id')
            if in_reply_to:
                parent_candidates.append(in_reply_to)

            for candidate in parent_candidates:
                parent_id = self._normalize_id(candidate)

                if (
                    parent_id
                    and parent_id in self.graph.nodes()
                    and parent_id != tweet_id
                ):
                    parent_time = self.graph.nodes[parent_id]['created_at']

                    # ✅ FIX 2 — Temporal consistency
                    if parent_time and child_time:
                        if parent_time >= child_time:
                            continue

                    self.graph.add_edge(parent_id, tweet_id)
                    break

        # ---------- PASS 3: DEPTH ----------
        self._calculate_node_depths(source_id)

        # ---------- PASS 4: REMOVE ORPHANS ----------
        if source_id is not None:
            self._remove_orphan_branches(source_id)

        # ---------- PASS 5: OPTIONAL PRUNING ----------
        if max_depth is not None:
            self.graph = self._prune_by_depth(max_depth)

        return self.graph

    # --------------------------------------------------
    # DEPTH COMPUTATION
    # --------------------------------------------------
    def _calculate_node_depths(self, source_id: Optional[str]):

        for node in self.graph.nodes():
            self.graph.nodes[node]['depth'] = None

        if source_id and source_id in self.graph.nodes():
            start_nodes = [source_id]
        else:
            start_nodes = [
                n for n in self.graph.nodes()
                if self.graph.in_degree(n) == 0
            ]

        for root in start_nodes:
            self.graph.nodes[root]['depth'] = 0

        visited = set(start_nodes)
        queue = deque(start_nodes)

        while queue:
            current = queue.popleft()
            current_depth = self.graph.nodes[current]['depth']

            for child in self.graph.successors(current):
                if child not in visited:
                    self.graph.nodes[child]['depth'] = current_depth + 1
                    visited.add(child)
                    queue.append(child)

    # --------------------------------------------------
    # ✅ FIX 1 — REMOVE ORPHAN SUBTREES
    # --------------------------------------------------
    def _remove_orphan_branches(self, source_id: str):

        source_id = self._normalize_id(source_id)

        if source_id not in self.graph.nodes():
            return

        reachable = nx.descendants(self.graph, source_id)
        reachable.add(source_id)

        nodes_to_remove = [
            n for n in self.graph.nodes() if n not in reachable
        ]

        self.graph.remove_nodes_from(nodes_to_remove)

    # --------------------------------------------------
    def _prune_by_depth(self, max_depth: int) -> nx.DiGraph:
        nodes_to_keep = [
            n for n, data in self.graph.nodes(data=True)
            if data.get('depth', 0) <= max_depth
        ]
        return self.graph.subgraph(nodes_to_keep).copy()

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------
    def get_tree_metrics(self, graph: nx.DiGraph) -> Dict:

        metrics = {
            'total_nodes': graph.number_of_nodes(),
            'leaf_nodes': 0,
            'max_depth': 0,
            'max_width': 0,
            'kernel_ratio': 0.0,
            'average_depth': 0.0,
            'depth_distribution': {},
        }

        if graph.number_of_nodes() == 0:
            return metrics

        # leaf nodes
        metrics['leaf_nodes'] = sum(
            1 for _, d in graph.out_degree() if d == 0
        )

        depth_counts = {}
        depths = []

        for _, data in graph.nodes(data=True):
            depth = data.get('depth', 0)
            depths.append(depth)
            depth_counts[depth] = depth_counts.get(depth, 0) + 1

        if depths:
            metrics['max_depth'] = max(depths)
            metrics['average_depth'] = sum(depths) / len(depths)

        # ✅ FIX 3 — TRUE TREE WIDTH
        metrics['max_width'] = max(depth_counts.values())
        metrics['depth_distribution'] = depth_counts

        return metrics