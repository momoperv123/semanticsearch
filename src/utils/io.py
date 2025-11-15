"""
File I/O utilities for semantic media search.
"""

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to directory

    Returns:
        Path object
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(
    path: Path, records: list[dict[str, Any]], append: bool = False
) -> None:
    """
    Write records to a JSONL file.

    Args:
        path: Path to JSONL file
        records: List of dictionaries to write
        append: If True, append to existing file; otherwise overwrite
    """
    mode = "a" if append else "w"
    ensure_dir(path.parent)

    with open(path, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Read records from a JSONL file.

    Args:
        path: Path to JSONL file

    Returns:
        List of dictionaries
    """
    records = []
    if not path.exists():
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records
