"""
Embedding store for managing frame embeddings.
Uses numpy memmap for efficient storage and access.
"""

import pickle
from pathlib import Path

import numpy as np

from src.config import (
    EMBEDDING_DIM,
    EMBEDDINGS_DAT_PATH,
    EMBEDDINGS_IDX_PATH,
    METADATA_PATH,
    PROJECT_ROOT,
)
from src.embeddings.clip_encoder import CLIPEncoder
from src.ingest.metadata_store import load_metadata
from src.utils.io import ensure_dir
from src.utils.logging_utils import get_logger
from src.utils.timing import timer

logger = get_logger(__name__)


class EmbeddingStore:
    """
    Manages storage and retrieval of frame embeddings.
    """

    def __init__(self, embeddings_path: Path, index_path: Path):
        """
        Initialize embedding store.

        Args:
            embeddings_path: Path to memmap file for embeddings
            index_path: Path to index file mapping frame_id -> row_idx
        """
        self.embeddings_path = embeddings_path
        self.index_path = index_path
        self.embeddings: np.memmap | None = None
        self.frame_id_to_row_idx: dict[int, int] = {}
        self.row_idx_to_frame_id: dict[int, int] = {}

    def create(
        self, metadata: dict[int, dict], encoder: CLIPEncoder, batch_size: int = 32
    ) -> None:
        """
        Create embedding store from metadata.

        Args:
            metadata: Dictionary mapping frame_id to metadata dict
            encoder: CLIPEncoder instance
            batch_size: Batch size for encoding
        """
        num_frames = len(metadata)
        if num_frames == 0:
            raise ValueError("No frames to embed")

        logger.info(f"Creating embedding store for {num_frames} frames")
        ensure_dir(self.embeddings_path.parent)

        # Create memmap file
        self.embeddings = np.memmap(
            self.embeddings_path,
            dtype="float32",
            mode="w+",
            shape=(num_frames, EMBEDDING_DIM),
        )

        # Build frame_id -> row_idx mapping (sequential for now)
        frame_ids = sorted(metadata.keys())
        self.frame_id_to_row_idx = {
            frame_id: idx for idx, frame_id in enumerate(frame_ids)
        }
        self.row_idx_to_frame_id = {
            idx: frame_id for frame_id, idx in self.frame_id_to_row_idx.items()
        }

        # Process frames in batches
        frame_paths = []
        row_indices = []

        for frame_id in frame_ids:
            frame_metadata = metadata[frame_id]
            frame_path = (PROJECT_ROOT / frame_metadata["frame_path"]).resolve()
            if not frame_path.exists():
                raise FileNotFoundError(
                    f"Frame file missing for frame_id {frame_id}: {frame_path}"
                )
            frame_paths.append(str(frame_path))
            row_indices.append(self.frame_id_to_row_idx[frame_id])

        # Encode in batches
        with timer("Embedding generation"):
            for i in range(0, len(frame_paths), batch_size):
                batch_paths = frame_paths[i : i + batch_size]
                batch_indices = row_indices[i : i + batch_size]

                logger.info(
                    f"Encoding batch {i // batch_size + 1}/{(len(frame_paths) + batch_size - 1) // batch_size}"
                )

                # Encode images
                embeddings = encoder.encode_images(
                    batch_paths, batch_size=len(batch_paths)
                )

                # Convert to numpy and write to memmap
                embeddings_np = embeddings.cpu().numpy().astype("float32")

                for j, row_idx in enumerate(batch_indices):
                    self.embeddings[row_idx] = embeddings_np[j]

                # Flush to disk periodically
                if (i // batch_size + 1) % 10 == 0:
                    self.embeddings.flush()

        # Final flush
        self.embeddings.flush()

        # Save index mapping
        self._save_index()

        logger.info(f"Embedding store created: {self.embeddings_path}")
        logger.info(f"Index saved: {self.index_path}")

    def load(self) -> None:
        """
        Load existing embedding store from disk.
        """
        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                f"Embeddings file not found: {self.embeddings_path}"
            )
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")

        # Load index first to get number of frames
        self._load_index()
        num_frames = len(self.frame_id_to_row_idx)

        # Load memmap
        self.embeddings = np.memmap(
            self.embeddings_path,
            dtype="float32",
            mode="r",
            shape=(num_frames, EMBEDDING_DIM),
        )

        logger.info(
            f"Loaded embedding store: {num_frames} frames, {EMBEDDING_DIM} dimensions"
        )

    def get_embedding(self, frame_id: int) -> np.ndarray:
        """
        Get embedding for a specific frame_id.

        Args:
            frame_id: Frame ID

        Returns:
            Embedding vector
        """
        if self.embeddings is None:
            raise RuntimeError("Embedding store not loaded")

        row_idx = self.frame_id_to_row_idx[frame_id]
        return self.embeddings[row_idx].copy()

    def get_all_embeddings(self) -> np.ndarray:
        """
        Get all embeddings as numpy array.

        Returns:
            Array of shape [N, EMBEDDING_DIM]
        """
        if self.embeddings is None:
            raise RuntimeError("Embedding store not loaded")

        return np.array(self.embeddings)

    def _save_index(self) -> None:
        """Save frame_id -> row_idx mapping to disk."""
        with open(self.index_path, "wb") as f:
            pickle.dump(self.frame_id_to_row_idx, f)

    def _load_index(self) -> None:
        """Load frame_id -> row_idx mapping from disk."""
        with open(self.index_path, "rb") as f:
            self.frame_id_to_row_idx = pickle.load(f)

        # Build reverse mapping
        self.row_idx_to_frame_id = {
            idx: frame_id for frame_id, idx in self.frame_id_to_row_idx.items()
        }


def create_embedding_store(
    metadata_path: Path = METADATA_PATH,
    embeddings_path: Path = EMBEDDINGS_DAT_PATH,
    index_path: Path = EMBEDDINGS_IDX_PATH,
    batch_size: int = 32,
) -> EmbeddingStore:
    """
    Create embedding store from metadata.

    Args:
        metadata_path: Path to metadata JSONL file
        embeddings_path: Path to save embeddings memmap
        index_path: Path to save index mapping
        batch_size: Batch size for encoding

    Returns:
        EmbeddingStore instance
    """
    # Load metadata
    metadata = load_metadata(metadata_path)
    if not metadata:
        raise ValueError(f"No metadata found at {metadata_path}")

    # Initialize encoder
    encoder = CLIPEncoder()

    # Create store
    store = EmbeddingStore(embeddings_path, index_path)
    store.create(metadata, encoder, batch_size)

    return store


def load_embedding_store(
    embeddings_path: Path = EMBEDDINGS_DAT_PATH, index_path: Path = EMBEDDINGS_IDX_PATH
) -> EmbeddingStore:
    """
    Load existing embedding store.

    Args:
        embeddings_path: Path to embeddings memmap
        index_path: Path to index mapping

    Returns:
        EmbeddingStore instance
    """
    store = EmbeddingStore(embeddings_path, index_path)
    store.load()
    return store
