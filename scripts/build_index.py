"""
Build index script.
Generates embeddings and builds HNSW index.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import (  # noqa: E402
    EMBEDDING_DIM,
    EMBEDDINGS_DAT_PATH,
    EMBEDDINGS_IDX_PATH,
    HNSW_INDEX_PATH,
    METADATA_PATH,
)
from src.embeddings.embed_store import create_embedding_store, load_embedding_store  # noqa: E402
from src.index.hnsw import HNSWIndex  # noqa: E402
from src.ingest.metadata_store import get_frame_count  # noqa: E402
from src.utils.logging_utils import setup_logging  # noqa: E402
from src.utils.timing import timer  # noqa: E402


def main():
    """Main function to build embeddings and HNSW index."""
    logger = setup_logging()
    logger.info("Starting index building process...")

    try:
        if not METADATA_PATH.exists():
            logger.error(f"Metadata file not found: {METADATA_PATH}")
            logger.error("Please run scripts/ingest_all.py first")
            return

        if not EMBEDDINGS_DAT_PATH.exists() or not EMBEDDINGS_IDX_PATH.exists():
            logger.info("Embeddings not found, generating...")
            with timer("Embedding generation"):
                store = create_embedding_store(
                    metadata_path=METADATA_PATH,
                    embeddings_path=EMBEDDINGS_DAT_PATH,
                    index_path=EMBEDDINGS_IDX_PATH,
                )
            logger.info("Embedding generation complete!")
        else:
            logger.info("Embeddings already exist, loading...")
            store = load_embedding_store(
                embeddings_path=EMBEDDINGS_DAT_PATH, index_path=EMBEDDINGS_IDX_PATH
            )

            # Validate embedding count matches metadata count
            metadata_count = get_frame_count(METADATA_PATH)
            embedding_count = len(store.frame_id_to_row_idx)

            if embedding_count != metadata_count:
                logger.error(
                    f"Embedding count mismatch! "
                    f"Metadata has {metadata_count} frames, but embeddings have {embedding_count} frames. "
                    f"This usually means new frames were ingested but embeddings weren't regenerated."
                )
                logger.error("To fix this, delete the embedding files and rebuild:")
                logger.error(
                    f"  rm {EMBEDDINGS_DAT_PATH} {EMBEDDINGS_IDX_PATH} {HNSW_INDEX_PATH}"
                )
                logger.error("  python scripts/build_index.py")

                response = (
                    input(
                        "Regenerate embeddings now? This will delete old embeddings. (y/n): "
                    )
                    .strip()
                    .lower()
                )
                if response == "y":
                    logger.info("Deleting old embeddings and regenerating...")
                    EMBEDDINGS_DAT_PATH.unlink(missing_ok=True)
                    EMBEDDINGS_IDX_PATH.unlink(missing_ok=True)
                    HNSW_INDEX_PATH.unlink(missing_ok=True)

                    with timer("Embedding generation"):
                        store = create_embedding_store(
                            metadata_path=METADATA_PATH,
                            embeddings_path=EMBEDDINGS_DAT_PATH,
                            index_path=EMBEDDINGS_IDX_PATH,
                        )
                    logger.info("Embedding generation complete!")
                else:
                    logger.warning(
                        "Continuing with mismatched embeddings. Search results may be incomplete."
                    )
            else:
                logger.info(
                    f"Validation passed: {embedding_count} embeddings match {metadata_count} metadata frames"
                )

        if HNSW_INDEX_PATH.exists():
            logger.warning(f"HNSW index already exists at {HNSW_INDEX_PATH}")
            response = input("Rebuild HNSW index? (y/n): ").strip().lower()
            if response != "y":
                logger.info("Skipping HNSW index building")
                return

        logger.info("Building HNSW index...")
        embeddings_array = store.get_all_embeddings()

        with timer("HNSW index building"):
            index = HNSWIndex(dim=EMBEDDING_DIM)
            index.build(embeddings_array)
            index.save(HNSW_INDEX_PATH)

        logger.info("Index building complete!")
        logger.info(f"Embeddings: {EMBEDDINGS_DAT_PATH}")
        logger.info(f"Embedding index: {EMBEDDINGS_IDX_PATH}")
        logger.info(f"HNSW index: {HNSW_INDEX_PATH}")
    except Exception as exc:
        logger.exception("Index building failed: %s", exc)


if __name__ == "__main__":
    main()
