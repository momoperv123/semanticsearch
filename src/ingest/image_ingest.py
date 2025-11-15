"""
Image ingestion module.
Walks filesystem, loads images, normalizes, and saves frames.
"""

import cv2
from pathlib import Path
from typing import Any

from src.config import MAX_FRAME_SIZE, PROJECT_ROOT
from src.utils.io import ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def resize_image(
    image: cv2.typing.MatLike, max_size: int = MAX_FRAME_SIZE
) -> cv2.typing.MatLike:
    """
    Resize image maintaining aspect ratio, with longest side = max_size.

    Args:
        image: OpenCV image (numpy array)
        max_size: Maximum size for longest side

    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    longest_side = max(height, width)

    if longest_side <= max_size:
        return image

    scale = max_size / longest_side
    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def ingest_images(
    root_dir: Path,
    frames_dir: Path,
    metadata_writer: Any,
    max_size: int = MAX_FRAME_SIZE,
) -> int:
    """
    Ingest images from root directory.

    Args:
        root_dir: Root directory containing images
        frames_dir: Directory to save processed frames
        metadata_writer: MetadataWriter instance
        max_size: Maximum size (longest side) for frame normalization

    Returns:
        Number of images ingested
    """
    ensure_dir(frames_dir)

    if not root_dir.exists():
        logger.warning(f"Image directory does not exist: {root_dir}")
        return 0

    image_files = set()
    for ext in IMAGE_EXTENSIONS:
        image_files.update(root_dir.rglob(f"*{ext}"))
        image_files.update(root_dir.rglob(f"*{ext.upper()}"))

    image_files = sorted(image_files)
    if not image_files:
        logger.info(f"No images found in {root_dir}")
        return 0

    logger.info(f"Found {len(image_files)} images to ingest")

    ingested_count = 0
    for image_path in sorted(image_files):
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Failed to load image: {image_path}")
                continue

            height, width = image.shape[:2]

            image = resize_image(image, max_size)

            frame_id = metadata_writer.frame_id_counter
            frame_path = frames_dir / f"{frame_id}.jpg"
            cv2.imwrite(str(frame_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])

            metadata_writer.add_record(
                {
                    "video_id": None,
                    "frame_path": str(
                        frame_path.relative_to(PROJECT_ROOT)
                    ),  # Relative to project root
                    "timestamp": None,
                    "width": width,
                    "height": height,
                }
            )

            ingested_count += 1

            if ingested_count % 100 == 0:
                logger.info(f"Ingested {ingested_count} images...")

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}", exc_info=True)
            continue

    logger.info(f"Successfully ingested {ingested_count} images")
    return ingested_count
