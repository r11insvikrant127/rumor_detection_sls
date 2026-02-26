import networkx as nx
from typing import Dict, List, Tuple


class KernelSubtreeExtractor:
    """
    Extract kernel subtree from propagation tree for rumor detection.

    Strict implementation of SLS paper definition:
    - Kernel = node with maximum TOTAL degree + its immediate children
    - No multi-centrality combination
    - No parent inclusion
    - Only kernel_ratio feature (as per paper's Table I)
    """

    def __init__(self, max_depth: int = None):
        """
        Initialize kernel extractor with optional depth constraint.

        Args:
            max_depth: Maximum depth to consider for kernel extraction.
                       None means no pruning (use full tree).
                       Set an integer only to apply temporal constraint.
        """
        self.max_depth = max_depth

    def extract_kernel_subtree(
        self, graph: nx.DiGraph, use_pruned: bool = True
    ) -> Tuple[str, List[str]]:
        """
        Extract kernel subtree strictly following paper definition.

        Paper definition: Node with maximum degree + its children.

        Args:
            graph: Propagation tree graph
            use_pruned: If True and max_depth is set, uses only nodes up to max_depth.

        Returns:
            Tuple of (max_degree_node, kernel_nodes_list)
        """
        if graph.number_of_nodes() == 0:
            return None, []

        # Apply depth constraint only if max_depth is explicitly set
        if use_pruned and self.max_depth is not None:
            working_graph = self._prune_to_max_depth(graph)
        else:
            working_graph = graph

        if working_graph.number_of_nodes() == 0:
            return None, []

        max_node, kernel_nodes = self._paper_kernel_extraction(working_graph)
        return max_node, kernel_nodes

    def _prune_to_max_depth(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Prune graph to include only nodes up to max_depth.

        FIX: Now safely handles max_depth=None (returns full graph).
        """
        # Safety guard — should not be called with None, but defensive check
        if self.max_depth is None:
            return graph

        nodes_to_keep = [
            n for n, data in graph.nodes(data=True)
            if data.get('depth', 0) <= self.max_depth
        ]
        return graph.subgraph(nodes_to_keep).copy()

    def _paper_kernel_extraction(self, graph: nx.DiGraph) -> Tuple[str, List[str]]:
        """
        Exact kernel extraction as defined in the paper.

        Paper: "Kernel subtree consists of the node with maximum degree
               and its immediate children."

        FIX: Uses total degree (in + out) instead of out_degree only,
             which matches the paper's use of "degree" (not "out-degree").
             Total degree better captures influence: a deeply-nested node
             with many replies has both a parent edge (in) and reply edges (out).

        Returns:
            Tuple of (max_degree_node, kernel_nodes_list)
        """
        if graph.number_of_nodes() == 0:
            return None, []
        degrees = dict(graph.out_degree())
        if not degrees:
            return None, []

        max_node = max(degrees.items(), key=lambda x: x[1])[0]

        # Kernel = max node + its children (successors in directed graph)
        kernel_nodes = {max_node}
        kernel_nodes.update(graph.successors(max_node))

        return max_node, list(kernel_nodes)

    def get_kernel_metrics(self, graph: nx.DiGraph, kernel_nodes: List[str]) -> Dict:
        """
        Calculate ONLY the metrics defined in Table I of the paper.

        Returns:
            Dict with exactly the paper's features.
        """
        metrics = {}

        # Use pruned graph only if max_depth is set
        if self.max_depth is not None:
            working_graph = self._prune_to_max_depth(graph)
        else:
            working_graph = graph

        total_nodes = working_graph.number_of_nodes()
        kernel_nodes_in_graph = [n for n in kernel_nodes if n in working_graph]
        kernel_size = len(kernel_nodes_in_graph)

        # Feature 2 from Table I: kernel_ratio
        metrics['kernel_ratio'] = kernel_size / total_nodes if total_nodes > 0 else 0.0

        # Metadata (not features)
        metrics['max_depth_constraint'] = self.max_depth
        metrics['kernel_node'] = kernel_nodes[0] if kernel_nodes else None

        return metrics