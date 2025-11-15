"""
Brute-force search baseline for semantic media search.
Used for Phase 0 validation and as ground truth for HNSW recall tests.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalize vectors along the last dimension.

    Args:
        vectors: Array of shape [..., D]

    Returns:
        Normalized vectors with same shape.
    """
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    # Avoid division by zero
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return vectors / norms


def brute_force_search(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    frame_ids: Sequence[int] | None = None,
    k: int = 10,
) -> list[tuple[int, float]]:
    """
    Compute cosine similarity between the query and all embeddings.

    Args:
        query_embedding: Array of shape [D]
        embeddings: Array of shape [N, D]
        frame_ids: Optional sequence mapping embedding row -> frame_id. If None, use row indices.
        k: Number of top results to return

    Returns:
        List of (frame_id, similarity) tuples sorted by similarity descending.
    """
    if embeddings.size == 0:
        return []

    if frame_ids is None:
        frame_ids = range(len(embeddings))

    query = _normalize_vectors(query_embedding.astype(np.float32))
    emb_norm = _normalize_vectors(embeddings.astype(np.float32))

    similarities = emb_norm @ query  # [N]

    k = min(k, len(embeddings))
    top_indices = np.argpartition(similarities, -k)[-k:]
    top_pairs = [(frame_ids[idx], float(similarities[idx])) for idx in top_indices]

    # Sort by similarity descending
    top_pairs.sort(key=lambda x: x[1], reverse=True)
    return top_pairs


def brute_force_search_batch(
    query_embeddings: np.ndarray,
    embeddings: np.ndarray,
    frame_ids: Sequence[int] | None = None,
    k: int = 10,
) -> list[list[tuple[int, float]]]:
    """
    Batch version of brute-force search.

    Args:
        query_embeddings: Array [B, D]
        embeddings: Array [N, D]
        frame_ids: Optional frame_id mapping
        k: Number of results per query

    Returns:
        List (len B) of result lists like brute_force_search
    """
    results = []
    for query in query_embeddings:
        results.append(brute_force_search(query, embeddings, frame_ids, k))
    return results
