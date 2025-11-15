"""
Main ingestion script.
Orchestrates image and video ingestion, writes metadata.jsonl.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import (  # noqa: E402
    FRAMES_DIR,
    METADATA_PATH,
    RAW_IMAGES_DIR,
    RAW_VIDEOS_DIR,
    VIDEO_FPS,
)
from src.ingest.image_ingest import ingest_images  # noqa: E402
from src.ingest.metadata_store import MetadataWriter  # noqa: E402
from src.ingest.video_ingest import ingest_videos  # noqa: E402
from src.utils.logging_utils import setup_logging  # noqa: E402
from src.utils.timing import timer  # noqa: E402


def main():
    """Main ingestion function."""
    logger = setup_logging()
    logger.info("Starting ingestion process...")

    RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RAW_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    if METADATA_PATH.exists():
        logger.warning(
            f"Existing metadata file found at {METADATA_PATH}, will be overwritten"
        )
        METADATA_PATH.unlink()

    with MetadataWriter(METADATA_PATH) as metadata_writer:
        total_frames = 0

        with timer("Image ingestion"):
            image_count = ingest_images(
                root_dir=RAW_IMAGES_DIR,
                frames_dir=FRAMES_DIR,
                metadata_writer=metadata_writer,
            )
            total_frames += image_count

        with timer("Video ingestion"):
            video_frame_count = ingest_videos(
                root_dir=RAW_VIDEOS_DIR,
                frames_dir=FRAMES_DIR,
                metadata_writer=metadata_writer,
                fps=VIDEO_FPS,
            )
            total_frames += video_frame_count

    logger.info(f"Ingestion complete! Total frames: {total_frames}")
    logger.info(f"Metadata written to: {METADATA_PATH}")
    logger.info(f"Frames saved to: {FRAMES_DIR}")


if __name__ == "__main__":
    main()
