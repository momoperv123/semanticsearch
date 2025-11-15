"""
Metadata store for frame metadata.
Handles JSONL writing during ingestion and loading for search.
"""

import json
from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir, read_jsonl


class MetadataWriter:
    """
    Writes metadata records to JSONL file during ingestion.
    """

    def __init__(self, path: Path):
        """
        Initialize metadata writer.

        Args:
            path: Path to JSONL file for metadata storage
        """
        self.path = path
        ensure_dir(path.parent)
        self.file = open(path, "a", encoding="utf-8")
        self.frame_id_counter = 0

    def add_record(self, record: dict[str, Any]) -> int:
        """
        Add a metadata record and assign frame_id.

        Args:
            record: Dictionary with frame metadata (video_id, frame_path, timestamp, width, height)

        Returns:
            Assigned frame_id
        """
        frame_id = self.frame_id_counter
        record["frame_id"] = frame_id

        json_line = json.dumps(record) + "\n"
        self.file.write(json_line)
        self.file.flush()

        self.frame_id_counter += 1
        return frame_id

    def close(self) -> None:
        """Close the metadata file."""
        if self.file:
            self.file.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def load_metadata(path: Path) -> dict[int, dict[str, Any]]:
    """
    Load metadata from JSONL file into a dictionary keyed by frame_id.

    Args:
        path: Path to metadata JSONL file

    Returns:
        Dictionary mapping frame_id to metadata dict
    """
    records = read_jsonl(path)
    return {record["frame_id"]: record for record in records}


def get_frame_count(path: Path) -> int:
    """
    Get the number of frames from metadata file.

    Args:
        path: Path to metadata JSONL file

    Returns:
        Number of frames
    """
    if not path.exists():
        return 0
    return len(load_metadata(path))
