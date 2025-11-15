"""
Run FastAPI server for semantic media search.
"""

import sys
from pathlib import Path

import uvicorn

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.server import app  # noqa: E402
from src.config import API_HOST, API_PORT  # noqa: E402


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
