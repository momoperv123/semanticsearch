"""
Video ingestion module.
Extracts frames from videos at configurable FPS.
"""

import cv2
from pathlib import Path
from typing import Any

from src.config import MAX_FRAME_SIZE, PROJECT_ROOT, VIDEO_FPS
from src.ingest.image_ingest import resize_image
from src.utils.io import ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


def ingest_videos(
    root_dir: Path,
    frames_dir: Path,
    metadata_writer: Any,
    fps: float = VIDEO_FPS,
    max_size: int = MAX_FRAME_SIZE,
) -> int:
    """
    Ingest videos from root directory, extracting frames at specified FPS.

    Args:
        root_dir: Root directory containing videos
        frames_dir: Directory to save processed frames
        metadata_writer: MetadataWriter instance
        fps: Frames per second to extract (e.g., 1.0 = 1 frame per second)
        max_size: Maximum size (longest side) for frame normalization

    Returns:
        Number of frames extracted
    """
    ensure_dir(frames_dir)

    if not root_dir.exists():
        logger.warning(f"Video directory does not exist: {root_dir}")
        return 0

    video_files = set()
    for ext in VIDEO_EXTENSIONS:
        video_files.update(root_dir.rglob(f"*{ext}"))
        video_files.update(root_dir.rglob(f"*{ext.upper()}"))

    video_files = sorted(video_files)
    if not video_files:
        logger.info(f"No videos found in {root_dir}")
        return 0

    logger.info(f"Found {len(video_files)} videos to ingest")

    total_frames = 0

    for video_path in sorted(video_files):
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.warning(f"Failed to open video: {video_path}")
                continue

            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if video_fps <= 0:
                logger.warning(f"Invalid FPS for video {video_path}, skipping")
                cap.release()
                continue

            frame_interval = max(1, int(video_fps / fps))

            logger.info(
                f"Processing {video_path.name}: "
                f"{total_video_frames} frames @ {video_fps:.2f} FPS, "
                f"extracting every {frame_interval} frames"
            )

            video_id = video_path.stem
            frames_extracted = 0
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    frame_id = metadata_writer.frame_id_counter

                    frame = resize_image(frame, max_size)

                    frame_path = frames_dir / f"{frame_id}.jpg"
                    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                    timestamp = frame_idx / video_fps

                    metadata_writer.add_record(
                        {
                            "video_id": video_id,
                            "frame_path": str(
                                frame_path.relative_to(PROJECT_ROOT)
                            ),  # Relative to project root
                            "timestamp": round(timestamp, 3),
                            "width": video_width,
                            "height": video_height,
                        }
                    )

                    frames_extracted += 1
                    total_frames += 1

                frame_idx += 1

            cap.release()
            logger.info(f"Extracted {frames_extracted} frames from {video_path.name}")

        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}", exc_info=True)
            continue

    logger.info(
        f"Successfully extracted {total_frames} frames from {len(video_files)} videos"
    )
    return total_frames
