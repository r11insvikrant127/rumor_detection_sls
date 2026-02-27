import networkx as nx
from typing import Dict, List, Tuple


class KernelSubtreeExtractor:
    """
    PAPER-FAITHFUL kernel subtree extractor.

    Paper definition:
    Kernel subtree = node with maximum number of replies
                     + its immediate children.
    """

    def __init__(self):
        # Paper does NOT prune tree
        pass

    # --------------------------------------------------
    # MAIN API
    # --------------------------------------------------
    def extract_kernel_subtree(
        self, graph: nx.DiGraph
    ) -> Tuple[str, List[str]]:

        if graph.number_of_nodes() == 0:
            return None, []

        max_node, kernel_nodes = self._paper_kernel_extraction(graph)
        return max_node, kernel_nodes

    # --------------------------------------------------
    # PAPER-EXACT IMPLEMENTATION
    # --------------------------------------------------
    def _paper_kernel_extraction(
        self, graph: nx.DiGraph
    ) -> Tuple[str, List[str]]:

        """
        Paper (Section IV-B):
        influential node = tweet with MOST responses.
        """

        # reply degree independent of edge direction
        def reply_degree(node):
            return max(
                graph.out_degree(node),
                graph.in_degree(node)
            )

        max_node = max(graph.nodes(), key=reply_degree)

        # detect reply direction automatically
        if graph.out_degree(max_node) >= graph.in_degree(max_node):
            children = list(graph.successors(max_node))
        else:
            children = list(graph.predecessors(max_node))

        kernel_nodes = [max_node] + children

        return max_node, kernel_nodes

    # --------------------------------------------------
    # METRICS (Feature 2)
    # --------------------------------------------------
    def get_kernel_metrics(
        self,
        graph: nx.DiGraph,
        kernel_nodes: List[str]
    ) -> Dict:

        total_nodes = graph.number_of_nodes()
        kernel_size = len(kernel_nodes)

        return {
            "kernel_ratio":
                kernel_size / total_nodes if total_nodes else 0.0,
            "kernel_node": kernel_nodes[0] if kernel_nodes else None,
        }