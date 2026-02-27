import networkx as nx
from typing import Dict, List, Tuple, Optional


class KernelSubtreeExtractor:
    """
    PAPER-FAITHFUL Kernel Subtree Extractor.

    Paper Definition (Section IV-B):
    --------------------------------
    Kernel subtree = node with maximum number of replies
                     + its immediate children.

    Assumptions from paper:
    - Event is modeled as a propagation TREE.
    - Edge direction: parent tweet → reply tweet.
    - Influential node = tweet receiving MOST replies.
    """

    def __init__(self):
        pass

    # --------------------------------------------------
    # MAIN API
    # --------------------------------------------------
    def extract_kernel_subtree(
        self, graph: nx.DiGraph
    ) -> Tuple[Optional[str], List[str]]:
        """
        Returns:
            influential_node_id,
            list of kernel subtree nodes
        """

        if graph is None or graph.number_of_nodes() == 0:
            return None, []

        self._validate_graph(graph)

        max_node, kernel_nodes = self._paper_kernel_extraction(graph)
        return max_node, kernel_nodes

    # --------------------------------------------------
    # GRAPH VALIDATION (Paper assumes tree structure)
    # --------------------------------------------------
    def _validate_graph(self, graph: nx.DiGraph) -> None:
        """
        Paper models events as propagation trees.
        We enforce minimal structural correctness.
        """

        if not isinstance(graph, nx.DiGraph):
            raise TypeError("Graph must be a networkx.DiGraph")

        # propagation must not contain cycles
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError(
                "Propagation graph must be a DAG (tree-like structure)."
            )

    # --------------------------------------------------
    # PAPER-EXACT IMPLEMENTATION
    # --------------------------------------------------
    def _paper_kernel_extraction(
        self, graph: nx.DiGraph
    ) -> Tuple[str, List[str]]:
        """
        Paper (Section IV-B):
        influential node = tweet with MOST responses (replies).
        """

        # replies received = number of outgoing edges
        # (parent -> reply convention)
        def reply_degree(node: str) -> Tuple[int, str]:
            # deterministic tie-breaking using node id
            return (graph.out_degree(node), str(node))

        # node with maximum replies
        max_node = max(graph.nodes(), key=reply_degree)

        # children = DIRECT replies ONLY
        children = list(graph.successors(max_node))

        # kernel subtree = influential node + its children
        kernel_nodes = [max_node] + children

        return max_node, kernel_nodes

    # --------------------------------------------------
    # METRICS (Used for Feature #2)
    # --------------------------------------------------
    def get_kernel_metrics(
        self,
        graph: nx.DiGraph,
        kernel_nodes: List[str]
    ) -> Dict[str, Optional[float]]:
        """
        Computes kernel subtree statistics.

        Feature #2 (Paper Table I):
        #tweets in kernel subtree / #tweets in total
        """

        total_nodes = graph.number_of_nodes()
        kernel_size = len(kernel_nodes)

        kernel_ratio = (
            kernel_size / total_nodes if total_nodes > 0 else 0.0
        )

        return {
            "kernel_ratio": kernel_ratio,
            "kernel_size": kernel_size,
            "total_nodes": total_nodes,
            "kernel_node": kernel_nodes[0] if kernel_nodes else None,
        }