"""
ImageRetriever: text-to-image search using CLIP + HNSW index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


from src.config import (
    EMBEDDING_DIM,
    EMBEDDINGS_DAT_PATH,
    EMBEDDINGS_IDX_PATH,
    HNSW_EF_SEARCH,
    HNSW_INDEX_PATH,
    METADATA_PATH,
)
from src.embeddings.clip_encoder import CLIPEncoder
from src.embeddings.embed_store import load_embedding_store
from src.index.hnsw import HNSWIndex
from src.ingest.metadata_store import load_metadata
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    frame_id: int
    video_id: str | None
    timestamp: float | None
    frame_path: str
    width: int
    height: int
    similarity: float


class ImageRetriever:
    """
    Image retriever using CLIP encoder + HNSW index.
    """

    def __init__(
        self,
        metadata_path: Path = METADATA_PATH,
        embeddings_path: Path = EMBEDDINGS_DAT_PATH,
        embeddings_idx_path: Path = EMBEDDINGS_IDX_PATH,
        hnsw_index_path: Path = HNSW_INDEX_PATH,
        ef_search: int = HNSW_EF_SEARCH,
    ):
        self.metadata_path = metadata_path
        self.embeddings_path = embeddings_path
        self.embeddings_idx_path = embeddings_idx_path
        self.hnsw_index_path = hnsw_index_path
        self.ef_search = ef_search

        logger.info("Initializing ImageRetriever...")

        # Load metadata
        self.metadata = load_metadata(metadata_path)
        if not self.metadata:
            raise ValueError(f"No metadata found at {metadata_path}")

        # Load embeddings
        self.embedding_store = load_embedding_store(
            embeddings_path=embeddings_path,
            index_path=embeddings_idx_path,
        )
        self.embeddings = self.embedding_store.get_all_embeddings()

        # Load frame_id -> row_idx mapping
        self.frame_id_to_row_idx = self.embedding_store.frame_id_to_row_idx
        self.row_idx_to_frame_id = self.embedding_store.row_idx_to_frame_id

        # Load HNSW index
        self.hnsw_index = HNSWIndex(dim=EMBEDDING_DIM)
        self.hnsw_index.load(hnsw_index_path, embeddings=self.embeddings)

        # Initialize CLIP encoder
        self.encoder = CLIPEncoder()

        logger.info(
            "ImageRetriever initialized: "
            f"{len(self.metadata)} frames, ef_search={self.ef_search}"
        )

    def search_text(self, query: str, k: int = 20) -> list[SearchResult]:
        """
        Search for images matching a text query.

        Args:
            query: Text query
            k: Number of results to return

        Returns:
            List of SearchResult objects
        """
        if not query.strip():
            return []

        # Encode text query
        try:
            query_embedding = self.encoder.encode_texts([query])[0].cpu().numpy()
        except Exception as exc:
            logger.exception("Failed to encode query: %s", query)
            raise RuntimeError("Failed to encode query") from exc

        # Search HNSW index
        try:
            hnsw_results = self.hnsw_index.search(
                query_embedding, k=k, ef=self.ef_search
            )
        except Exception as exc:
            logger.exception("HNSW search failed for query: %s", query)
            raise RuntimeError("Search backend error") from exc

        # Build result objects
        results = []
        for node_id, similarity in hnsw_results:
            frame_id = self.row_idx_to_frame_id[node_id]
            metadata = self.metadata.get(frame_id)
            if not metadata:
                continue

            result = SearchResult(
                frame_id=frame_id,
                video_id=metadata.get("video_id"),
                timestamp=metadata.get("timestamp"),
                frame_path=metadata.get("frame_path"),
                width=metadata.get("width"),
                height=metadata.get("height"),
                similarity=similarity,
            )
            results.append(result)

        return results
