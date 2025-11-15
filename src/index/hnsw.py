"""
HNSW (Hierarchical Navigable Small World) index implementation.
Custom ANN index for approximate nearest neighbor search.
"""

import heapq
import pickle
import random
from pathlib import Path

import numpy as np

from src.config import HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH, HNSW_M
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class HNSWIndex:
    """
    HNSW index for approximate nearest neighbor search.
    Uses cosine similarity (dot product on normalized vectors).
    """

    def __init__(
        self, dim: int, m: int = HNSW_M, ef_construction: int = HNSW_EF_CONSTRUCTION
    ):
        """
        Initialize HNSW index.

        Args:
            dim: Embedding dimension
            m: Maximum number of connections per node per layer
            ef_construction: Size of candidate pool during construction
        """
        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction

        self.node_levels: dict[int, int] = {}  # node_id -> level (0 = bottom layer)
        self.links: dict[
            int, dict[int, list[int]]
        ] = {}  # node_id -> {level -> [neighbor_node_ids]}
        self.entry_point: int | None = None

        # Embeddings reference (not owned, set externally)
        self.embeddings: np.ndarray | None = None

        # Statistics
        self.num_nodes = 0
        self.max_level = -1

    def set_embeddings(self, embeddings: np.ndarray) -> None:
        """
        Set embeddings array reference.

        Args:
            embeddings: Embeddings array of shape [N, dim]
        """
        self.embeddings = embeddings
        if embeddings is not None:
            assert embeddings.shape[1] == self.dim, (
                f"Embedding dim mismatch: {embeddings.shape[1]} != {self.dim}"
            )

    def _random_level(self) -> int:
        """
        Generate random level using geometric distribution.
        Probability of level L is proportional to 1/m^L.

        Returns:
            Random level (0 = bottom layer)
        """
        level = 0
        while random.random() < 1.0 / self.m and level < 16:  # Cap at 16 levels
            level += 1
        return level

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        Assumes vectors are already normalized.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (dot product)
        """
        return float(np.dot(vec1, vec2))

    def _get_vector(self, node_id: int) -> np.ndarray:
        """
        Get embedding vector for a node.

        Args:
            node_id: Node ID (row index in embeddings array)

        Returns:
            Embedding vector
        """
        if self.embeddings is None:
            raise RuntimeError("Embeddings not set")
        return self.embeddings[node_id]

    def _greedy_search_layer(
        self, query_vec: np.ndarray, entry_node: int, level: int
    ) -> int:
        """
        Greedy search at a specific layer to find closest node.

        Args:
            query_vec: Query vector (normalized)
            entry_node: Starting node ID
            level: Layer level to search

        Returns:
            Closest node ID at this layer
        """
        current = entry_node
        current_dist = self._cosine_similarity(query_vec, self._get_vector(current))

        improved = True
        while improved:
            improved = False
            neighbors = self.links.get(current, {}).get(level, [])

            for neighbor in neighbors:
                neighbor_dist = self._cosine_similarity(
                    query_vec, self._get_vector(neighbor)
                )
                if neighbor_dist > current_dist:  # Higher similarity is better
                    current = neighbor
                    current_dist = neighbor_dist
                    improved = True

        return current

    def _select_neighbors_heuristic(
        self, candidates: list[tuple[float, int]], m: int
    ) -> list[int]:
        """
        Select M neighbors from candidates using heuristic (keep closest, remove redundant).

        Args:
            candidates: List of (similarity, node_id) tuples
            m: Maximum number of neighbors to select

        Returns:
            List of selected neighbor node IDs
        """
        if len(candidates) <= m:
            return [node_id for _, node_id in candidates]

        # Sort by similarity (descending)
        candidates.sort(reverse=True)

        selected = []
        for similarity, node_id in candidates:
            if len(selected) >= m:
                break

            # Check if this node is too close to already selected nodes
            too_close = False
            for selected_id in selected:
                selected_vec = self._get_vector(selected_id)
                candidate_vec = self._get_vector(node_id)
                dist = self._cosine_similarity(selected_vec, candidate_vec)
                if dist > 0.9:  # Very similar, skip
                    too_close = True
                    break

            if not too_close:
                selected.append(node_id)

        # If we don't have enough, just take top M
        if len(selected) < m:
            selected = [node_id for _, node_id in candidates[:m]]

        return selected

    def build(self, embeddings: np.ndarray) -> None:
        """
        Build HNSW index from embeddings.

        Args:
            embeddings: Embeddings array of shape [N, dim], already normalized
        """
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dimension mismatch: {embeddings.shape[1]} != {self.dim}"
            )

        self.set_embeddings(embeddings)
        num_nodes = embeddings.shape[0]

        logger.info(
            f"Building HNSW index for {num_nodes} nodes (M={self.m}, ef_construction={self.ef_construction})"
        )

        # Initialize data structures
        self.node_levels = {}
        self.links = {}
        self.entry_point = None
        self.num_nodes = 0
        self.max_level = -1

        # Build index by inserting each node
        for node_id in range(num_nodes):
            self._insert_node(node_id, embeddings[node_id])

            if (node_id + 1) % 1000 == 0:
                logger.info(f"Inserted {node_id + 1}/{num_nodes} nodes...")

        logger.info(f"HNSW index built: {num_nodes} nodes, max_level={self.max_level}")

    def _insert_node(self, node_id: int, node_vec: np.ndarray) -> None:
        """
        Insert a node into the HNSW index.

        Args:
            node_id: Node ID to insert
            node_vec: Normalized embedding vector for this node
        """
        # Assign random level
        level = self._random_level()
        self.node_levels[node_id] = level
        self.max_level = max(self.max_level, level)

        # Initialize links for this node
        if node_id not in self.links:
            self.links[node_id] = {}

        # If this is the first node, it becomes the entry point
        if self.entry_point is None:
            self.entry_point = node_id
            self.num_nodes = 1
            return

        # Search from top layer down to node's level
        current = self.entry_point
        for layer in range(self.max_level, level, -1):
            current = self._greedy_search_layer(node_vec, current, layer)

        # At each level from node's level down to 0, find candidates and connect
        for layer in range(min(level, self.max_level), -1, -1):
            # Find candidates using greedy search with ef_construction
            candidates = self._search_layer_candidates(
                node_vec, current, layer, self.ef_construction
            )

            # Select up to M neighbors
            neighbors = self._select_neighbors_heuristic(candidates, self.m)

            # Connect node to neighbors
            if layer not in self.links[node_id]:
                self.links[node_id][layer] = []

            for neighbor_id in neighbors:
                # Add bidirectional link
                if neighbor_id not in self.links:
                    self.links[neighbor_id] = {}
                if layer not in self.links[neighbor_id]:
                    self.links[neighbor_id][layer] = []

                # Add links (avoid duplicates)
                if node_id not in self.links[neighbor_id][layer]:
                    self.links[neighbor_id][layer].append(node_id)
                if neighbor_id not in self.links[node_id][layer]:
                    self.links[node_id][layer].append(neighbor_id)

            # Prune neighbors if we have more than M (keep closest M)
            if len(self.links[node_id][layer]) > self.m:
                # Re-sort by distance to node
                neighbor_sims = [
                    (self._cosine_similarity(node_vec, self._get_vector(nid)), nid)
                    for nid in self.links[node_id][layer]
                ]
                neighbor_sims.sort(reverse=True)
                self.links[node_id][layer] = [nid for _, nid in neighbor_sims[: self.m]]

            # Update current for next level (closest neighbor)
            if neighbors:
                # Find closest neighbor
                best_neighbor = max(
                    neighbors,
                    key=lambda nid: self._cosine_similarity(
                        node_vec, self._get_vector(nid)
                    ),
                )
                current = best_neighbor

        self.num_nodes += 1

    def _search_layer_candidates(
        self, query_vec: np.ndarray, entry_node: int, level: int, ef: int
    ) -> list[tuple[float, int]]:
        """
        Search layer to find ef candidates using beam search.

        Args:
            query_vec: Query vector (normalized)
            entry_node: Starting node ID
            level: Layer level
            ef: Number of candidates to find

        Returns:
            List of (similarity, node_id) tuples, sorted by similarity descending
        """
        # Min-heap for candidates (keep worst for replacement)
        # Use negative similarity for min-heap (we want to keep worst at top)
        candidates = []  # (-similarity, node_id) for min-heap
        visited = {entry_node}

        # Start with entry node
        entry_sim = self._cosine_similarity(query_vec, self._get_vector(entry_node))
        heapq.heappush(candidates, (-entry_sim, entry_node))

        # Explore candidates
        candidates_to_explore = [entry_node]

        while candidates_to_explore:
            current = candidates_to_explore.pop(0)

            # Check neighbors
            neighbors = self.links.get(current, {}).get(level, [])
            for neighbor in neighbors:
                if neighbor in visited:
                    continue

                visited.add(neighbor)
                neighbor_sim = self._cosine_similarity(
                    query_vec, self._get_vector(neighbor)
                )

                # If we have fewer than ef candidates, add it
                if len(candidates) < ef:
                    heapq.heappush(candidates, (-neighbor_sim, neighbor))
                    candidates_to_explore.append(neighbor)
                # Otherwise, replace worst if this is better
                else:
                    worst_neg_sim, _ = candidates[0]  # Peek at worst
                    if neighbor_sim > -worst_neg_sim:
                        heapq.heapreplace(candidates, (-neighbor_sim, neighbor))
                        candidates_to_explore.append(neighbor)

        # Convert back to positive similarities and sort descending
        result = [(-sim, node_id) for sim, node_id in candidates]
        result.sort(reverse=True)
        return result

    def search(
        self, query_vec: np.ndarray, k: int, ef: int = HNSW_EF_SEARCH
    ) -> list[tuple[int, float]]:
        """
        Search for k nearest neighbors.

        Args:
            query_vec: Query vector (normalized), shape [dim]
            k: Number of results to return
            ef: Size of candidate pool during search (should be >= k)

        Returns:
            List of (node_id, similarity) tuples, sorted by similarity descending
        """
        if self.entry_point is None:
            return []

        if self.embeddings is None:
            raise RuntimeError("Embeddings not set")

        # Normalize query vector
        query_norm = query_vec / np.linalg.norm(query_vec)

        # Greedy descent from top layer
        current = self.entry_point
        for level in range(self.max_level, 0, -1):
            current = self._greedy_search_layer(query_norm, current, level)

        # At bottom layer (level 0), do beam search
        candidates = self._search_layer_candidates(query_norm, current, 0, max(ef, k))

        # Return top-k
        results = [(node_id, float(sim)) for sim, node_id in candidates[:k]]
        return results

    def save(self, path: Path) -> None:
        """
        Save HNSW index to disk using pickle.

        Args:
            path: Path to save index
        """
        logger.info(f"Saving HNSW index to {path}")

        # Prepare data for serialization (don't save embeddings, just metadata)
        data = {
            "dim": self.dim,
            "m": self.m,
            "ef_construction": self.ef_construction,
            "node_levels": self.node_levels,
            "links": self.links,
            "entry_point": self.entry_point,
            "num_nodes": self.num_nodes,
            "max_level": self.max_level,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info("HNSW index saved")

    def load(self, path: Path, embeddings: np.ndarray) -> None:
        """
        Load HNSW index from disk.

        Args:
            path: Path to load index from
            embeddings: Embeddings array (must match the index)
        """
        logger.info(f"Loading HNSW index from {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.dim = data["dim"]
        self.m = data["m"]
        self.ef_construction = data["ef_construction"]
        self.node_levels = data["node_levels"]
        self.links = data["links"]
        self.entry_point = data["entry_point"]
        self.num_nodes = data["num_nodes"]
        self.max_level = data["max_level"]

        # Set embeddings reference
        self.set_embeddings(embeddings)

        logger.info(
            f"HNSW index loaded: {self.num_nodes} nodes, max_level={self.max_level}"
        )
