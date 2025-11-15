"""
Configuration file for semantic media search system.
All paths are relative to project root.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# CLIP Model Configuration
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512
BATCH_SIZE = 32  # Batch size for encoding images/text

# HNSW Index Parameters
HNSW_M = 16  # Maximum number of connections per node per layer
HNSW_EF_CONSTRUCTION = 200  # Size of candidate pool during construction
HNSW_EF_SEARCH = 50  # Size of candidate pool during search

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_IMAGES_DIR = DATA_DIR / "raw" / "images"
RAW_VIDEOS_DIR = DATA_DIR / "raw" / "videos"
PROCESSED_DIR = DATA_DIR / "processed"
FRAMES_DIR = PROCESSED_DIR / "frames"
METADATA_PATH = PROCESSED_DIR / "metadata.jsonl"
EMBEDDINGS_DAT_PATH = PROCESSED_DIR / "embeddings.dat"
EMBEDDINGS_IDX_PATH = PROCESSED_DIR / "embeddings.idx"
HNSW_INDEX_PATH = PROCESSED_DIR / "hnsw_index.bin"

# Video Ingestion
VIDEO_FPS = 1.0  # Frames per second to extract from videos
MAX_FRAME_SIZE = 512  # Maximum size (longest side) for frame normalization

# API Server
API_HOST = "0.0.0.0"
API_PORT = 8000

# Logging
LOG_LEVEL = "INFO"

# Device Configuration
# Will be set dynamically based on torch availability
try:
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"
