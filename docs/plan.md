# Semantic Media Search System

This document serves as both the specification and implementation plan for a local semantic image/video search system.

---

## V1 Scope (Locked)

**Core features for initial release:**

1. **Ingestion**
   - Image ingestion: walk filesystem, extract, normalize, save frames
   - Video ingestion: simple frame sampling at configurable FPS
   - Metadata store: JSONL format with frame_id, video_id, timestamp, dimensions

2. **Embedding Pipeline**
   - CLIP encoder: fixed model `openai/clip-vit-base-patch32` (512-dim)
   - Embedding store: memmap array with frame_id → row_idx mapping
   - Batch processing with GPU support if available
   - **Distance metric: Cosine similarity** (embeddings normalized, use dot product)

3. **HNSW Index**
   - Custom HNSW implementation (multi-layer graph, adjacency lists)
   - Build algorithm: random level assignment, greedy insertion, neighbor selection
   - Search algorithm: greedy descent + beam search at bottom layer
   - **Serialization: Pickle format** (simple, sufficient for v1)
   - **Node IDs: Contiguous [0, N-1]** mapping directly to embeddings array indices

4. **API Server**
   - FastAPI with `/search` and `/frame/:id` endpoints
   - `/health` endpoint for status
   - **Response format: Use `similarity` field (higher is better, range 0-1)**

5. **Minimal Frontend**
   - Basic HTML/JS search interface
   - Grid display of results with timestamps and similarity scores
   - Frame preview on click
   - **No video playback or timeline heatmaps**

6. **Testing & Validation**
   - Brute-force vs HNSW recall test script
   - End-to-end pipeline test
   - Simple benchmark: N random queries, print avg + p95 latency

---

## Phase 2+ (Future Enhancements)

**Deferred to post-v1:**

- `/video/:id` endpoint with seek-based playback
- Reranker using BLIP/cross-encoder models
- Timeline heatmaps and multi-query logic
- Tagging pipeline with captioning models
- Advanced serialization (custom binary format)
- Full test suite with granular per-module tests
- Comprehensive benchmarking suite
- Duplicate frame suppression
- Incremental index updates

---

## 0. Architecture Overview

Goal: local semantic image/video search.

Data flow:

1. **Ingest**: walk filesystem → extract frames → store metadata.
2. **Embed**: CLIP encodes frames → embeddings stored in mmap'd array.
3. **Index**: build HNSW over embeddings.
4. **Query**: text → embedding → ANN search → result set.
5. **Serve**: FastAPI exposes `/search`, `/frame/:id` (V1). `/video/:id` (Phase 2+).
6. **UI** (optional): thin frontend displaying grid (V1). Timeline heatmaps (Phase 2+).

Core constraints:

* Runs on a single machine.
* Handles at least tens of thousands of frames.
* No Faiss/Pinecone; ANN is your code.
* No cloud dependencies.

### Key Implementation Details

**CLIP Model (V1 - Fixed):**
- **Model:** `openai/clip-vit-base-patch32`
- **Embedding dimension:** 512
- **Config example:**
  ```python
  # src/config.py
  CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
  EMBEDDING_DIM = 512
  ```
- Support GPU if available, fallback to CPU

**Distance Metric:**
- **Use cosine similarity** (not Euclidean distance)
- Store normalized embeddings (L2-normalized)
- Compute similarity via dot product: `similarity = query_vec @ embedding_vec`
- Similarity range: [0, 1] where 1 = identical, 0 = orthogonal

**HNSW Parameters:**
- Default: `M=16`, `efConstruction=200`, `efSearch=50`
- Configurable via `config.py`

**Node ID Convention:**
- **Node IDs are contiguous integers [0, N-1]**
- Node ID directly indexes into embeddings array: `embeddings[node_id]`
- Graph stores only node IDs, not vectors (embeddings array is source of truth)
- Use `embeddings.idx` to map between frame_id and node_id

**Error Handling:**
- Graceful handling of corrupted media files
- Memory management for large datasets
- Partial failure recovery during ingestion

**Incremental Updates:**
- Deferred to Phase 2+
- V1 requires full rebuild for new media

---

## Quick Start

### Prerequisites

- Python 3.11 or higher
- CUDA-capable GPU (optional, but recommended for faster embedding generation)
- At least 8GB RAM (16GB+ recommended for large datasets)

### Installation

1. Clone the repository and navigate to the project directory:
```bash
cd semantic_media_search
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Workflow

1. **Prepare your media files:**
   - Place images in `data/raw/images/`
   - Place videos in `data/raw/videos/`

2. **Ingest media and extract frames:**
```bash
python scripts/ingest_all.py
```

3. **Generate embeddings and build index:**
```bash
python scripts/build_index.py
```

4. **Start the API server + frontend:**
```bash
python scripts/run_server.py
```

**Note:** `scripts/run_server.py` imports `src.api.server:app` and runs Uvicorn server.
The same FastAPI process also serves the static frontend at `http://localhost:8000/`.

5. **Search via API or frontend:**
   - Browser: open http://localhost:8000 and use the search UI
   - Curl:
```bash
curl "http://localhost:8000/search?q=red%20hoodie%20dog&k=10"
```

6. **(Optional) Run benchmarks:**
```bash
python scripts/benchmark.py --queries "a cat" "a skyline at night" --ef-search 32 64 128 --k 5 --output data/benchmarks/latest.json
```

---

### API Reference (V1)

| Endpoint | Description | Notes |
| --- | --- | --- |
| `GET /health` | Returns status, frame count, embedding dim, HNSW nodes | Use for readiness checks |
| `GET /search?q=...&k=...` | Runs text search via CLIP + HNSW | Returns `{query, results[]}`; each result has `frame_id`, `timestamp`, `similarity`, `frame_url` |
| `GET /frame/{frame_id}` | Streams the stored JPEG frame | Used by the frontend thumbnails and modal preview |
| `GET /` | Serves `frontend/index.html` | Minimal HTML/JS grid UI hitting `/search` |

All responses use cosine similarity (`0–1`, higher is more similar).

## 1. Tech Stack and Project Layout

**Language**: Python 3.11+
**Core libs**: `torch`, `transformers` or `open_clip`, `numpy`, `opencv-python`, `fastapi`, `uvicorn`
**Storage**: filesystem + `numpy.memmap`
**Index**: custom HNSW
**API**: FastAPI
**Frontend**: minimal React/Next.js or plain HTML/JS

**Directory structure**:

```text
semantic_media_search/
  src/
    config.py
    ingest/
      __init__.py
      video_ingest.py
      image_ingest.py
      metadata_store.py
    embeddings/
      __init__.py
      clip_encoder.py
      embed_store.py
    index/
      __init__.py
      hnsw.py
    search/
      __init__.py
      retriever.py
      brute_force.py  # For validation/testing only
    api/
      __init__.py
      server.py
    utils/
      logging_utils.py
      timing.py
      io.py
  data/
    raw/
      images/
      videos/
    processed/
      frames/
      metadata.jsonl
      embeddings.dat
      embeddings.idx
      hnsw_index.bin
  frontend/  (optional)
  scripts/
    ingest_all.py
    build_index.py
    run_server.py
    benchmark.py
  requirements.txt
  README.md
```

---

## 2. Phase 0 – Project Setup & MVP Validation

### 2.1 Project Setup

Before implementing the full HNSW index, set up the project structure and validate the pipeline with a simpler approach.

**Tasks:**
- Create directory structure as specified
- Set up `requirements.txt` with all dependencies
- Create `src/config.py` with configuration parameters
- Implement basic logging and utility functions

### 2.2 MVP Validation with Brute-Force Search

To validate the pipeline before implementing complex HNSW, create a brute-force search baseline:

**Module:** `src/search/brute_force.py` (temporary, for validation)

```python
def brute_force_search(query_embedding: np.ndarray, embeddings: np.ndarray, k: int) -> list[tuple[int, float]]:
    """
    Compute cosine similarity between query and all embeddings.
    Return top-k results with (frame_id, similarity_score).
    """
    # Normalize query embedding
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    # Compute cosine similarities
    similarities = embeddings @ query_norm
    # Get top-k indices
    top_k_indices = np.argsort(similarities)[::-1][:k]
    return [(idx, float(similarities[idx])) for idx in top_k_indices]
```

**Validation Steps:**
1. Ingest small test dataset (10-100 frames)
2. Generate embeddings
3. Test search with brute-force approach
4. Verify results are reasonable
5. Measure baseline performance (latency, accuracy)
6. Use this as ground truth for HNSW validation

**Benefits:**
- Validates CLIP encoding works correctly
- Confirms metadata and embedding storage
- Provides ground truth for HNSW recall testing
- Identifies issues early before complex index implementation

---

## 3. Data Model and Files

### 3.1 Metadata

Each **frame** has:

```json
{
  "frame_id": int,
  "video_id": str | null,
  "frame_path": "data/processed/frames/<frame_id>.jpg",
  "timestamp": float | null,
  "width": int,
  "height": int
}
```

**Frame ID Ordering:**
- `frame_id` increments sequentially starting from 0 in ingestion order
- First ingested frame gets `frame_id=0`, second gets `frame_id=1`, etc.
- This ensures `frame_id` is contiguous [0, N-1] and prevents off-by-one errors
- Order: images first (if any), then videos (if any), both processed in filesystem walk order

All stored as `metadata.jsonl` in `data/processed/`.

You can also mirror metadata in an in-memory `pandas.DataFrame` or dict keyed by `frame_id`.

### 3.2 Embeddings

* Shape: `[N, D]` (N = frames, D = 512 or 768).
* Stored as `numpy.memmap` file `embeddings.dat`.
* `embeddings.idx` is a small sidecar file mapping `frame_id → row_index`.

Example:

```python
# embeddings.idx content (Python pickle or simple text)
{
  0: 0,
  1: 1,
  ...
}
```

### 3.3 HNSW Index

Store in `hnsw_index.bin` with:

* global header: `N, D, M, efConstruction`
* entry per node: level, neighbors per level, vector index

Serialization: your own binary format or `pickle` if you accept portability tradeoffs.

---

## 4. Phase 1 – Ingestion

### 4.1 Image Ingestion

Module: `src/ingest/image_ingest.py`

Steps:

1. Walk `data/raw/images`.
2. For each image:

   * Load with OpenCV.
   * Optionally resize to max size (e.g., 512px longest side).
   * Save standardized frame as JPEG under `data/processed/frames/{next_frame_id}.jpg`.
3. Append metadata row.

Functions:

```python
def ingest_images(root_dir: str, frames_dir: str, metadata_writer) -> None:
    ...

class MetadataWriter:
    def __init__(self, path: str): ...
    def add_record(self, record: dict) -> None: ...
    def close(self) -> None: ...
```

### 4.2 Video Ingestion

Module: `src/ingest/video_ingest.py`

Steps:

1. Walk `data/raw/videos`.
2. For each video:

   * Use `cv2.VideoCapture`.
   * Determine FPS, total frames.
   * Sample frames at interval (e.g., 1 fps or `every_n_frames`).
   * Save each sampled frame to `frames_dir`.
   * Record `timestamp = frame_idx / fps`.
3. Append metadata.

Functions:

```python
def ingest_videos(root_dir: str, frames_dir: str, metadata_writer, fps: float = 1.0) -> None:
    ...
```

### 4.3 Metadata Store

Module: `src/ingest/metadata_store.py`

Responsibilities:

* Write JSONL during ingestion.
* Later, load into Python for search.

```python
def load_metadata(path: str) -> dict[int, dict]:
    # returns {frame_id: metadata_dict}
```

Script: `scripts/ingest_all.py`
Runs image and video ingestion, writes `metadata.jsonl`.

---

## 5. Phase 2 – Embedding Pipeline

### 5.1 CLIP Encoder

Module: `src/embeddings/clip_encoder.py`

Responsibilities:

* Load image encoder and text encoder (one-time).
* Offer two functions:

```python
def encode_images(image_paths: list[str]) -> np.ndarray:  # [B, D]
    ...

def encode_texts(queries: list[str]) -> np.ndarray:  # [B, D]
    ...
```

Implementation details:

* Use `torch.no_grad()` and `model.eval()`.
* Support GPU if available (`cuda`).
* Batch size configurable via `config.py`.

### 5.2 Embedding Store

Module: `src/embeddings/embed_store.py`

Responsibilities:

* Allocate a memmap file of shape `[N, D]`.
* Fill rows as you embed.
* Write index `frame_id → row_idx`.

Steps:

1. Count frames from metadata.
2. Open memmap:

```python
emb = np.memmap("embeddings.dat", dtype="float32", mode="w+", shape=(N, D))
```

3. Iterate over `frame_id` in chunks:

   * Load frame images.
   * Call `encode_images`.
   * Normalize embeddings (L2).
   * Write into memmap rows.
4. Build `frame_id → row_idx` dict, write as `embeddings.idx`.

Script: `scripts/build_index.py` will call this first, then ANN.

---

## 6. Phase 3 – ANN Index (HNSW)

Module: `src/index/hnsw.py`

HNSW basics:

* Graph with multiple layers.
* Each node has up to `M` neighbors per layer.
* Upper layers sparser.
* Search: start at entry point, greedy descend, then local BFS/beam.

### 6.1 Data Structures

**Node ID Convention:**
- **Node IDs are contiguous [0, N-1]** where N = number of embeddings
- Node ID directly indexes embeddings array: `embeddings[node_id]` is the vector
- **Graph stores only node IDs, not vectors** (embeddings array is source of truth)
- Frame ID mapping: Use `embeddings.idx` to convert between `frame_id` and `node_id`

**Data Structure:**

```python
class HNSWIndex:
    def __init__(self, dim: int, m: int, ef_construction: int):
        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self.node_levels: dict[int, int]  # node_id -> level (0 = bottom layer)
        self.links: dict[int, dict[int, list[int]]]  # node_id -> {level -> [neighbor_node_ids]}
        self.entry_point: int | None  # node_id of entry point
        self.embeddings: np.memmap  # reference, not owned
        # embeddings[node_id] gives the vector for node_id
        # Graph structure stores only integer node IDs
```

### 6.2 Build

Algorithm:

* Level for each node: random geometric distribution.
* If empty: first node becomes entry point.
* For each new node:

  * From top layer to its level:

    * Greedy search to find closest node at that layer.
    * Use `ef_construction` as candidate pool size.
  * Insert links at each layer:

    * Connect node to up to `M` nearest nodes in candidate set.
    * Update neighbors symmetrically.

### 6.3 Search

**Distance Metric:**
- Use cosine similarity (dot product on normalized vectors)
- Higher similarity = closer/more similar
- Return similarity scores, not distances

**Node ID to Frame ID Conversion:**

After HNSW search returns `(node_id, similarity)` tuples, convert to frame_ids using the reverse mapping from `embeddings.idx`:

```python
# embeddings.idx maps: frame_id -> row_idx (node_id)
# Reverse mapping: row_idx (node_id) -> frame_id
frame_id_lookup = {row_idx: frame_id for frame_id, row_idx in embeddings_idx.items()}
```

Function:

```python
def search(self, query_vec: np.ndarray, k: int, ef: int) -> list[tuple[int, float]]:
    """
    Returns: list of (node_id, similarity) tuples, sorted by similarity descending.
    similarity is in range [0, 1] where 1 = identical, 0 = orthogonal.
    """
    ...
```

Search algorithm:

1. Start at `entry_point`.
2. Greedy descent from top layer down:

   * Move to neighbor with increasing similarity until no improvement.
3. At bottom layer:

   * Maintain a min-heap of candidates size `ef` (keep worst candidates for replacement).
   * Explore neighbors, update heap if better similarity found.
4. After exploration, return top-k by similarity (highest first).

### 6.4 Serialization

**V1: Use Pickle**
- Simple and sufficient for initial release
- `save(path: str)` → pickle dump of index metadata and adjacency lists
- `load(path: str, embeddings: np.memmap)` → load index and attach embeddings reference
- Custom binary format can be added in Phase 2+ for optimization

---

## 7. Phase 4 – Retrieval and Ranking

Module: `src/search/retriever.py` (V1). `src/search/reranker.py` (Phase 2+).

### 7.1 Retriever

`retriever.py`:

```python
class ImageRetriever:
    def __init__(self, encoder, index, emb_store, metadata):
        ...

    def search_text(self, query: str, k: int = 20) -> list[dict]:
        # 1. text → embedding
        # 2. HNSW search with ef > k (e.g., 50 or 100)
        # 3. decode node ids → frame_ids
        # 4. attach metadata and similarity
        ...
```

### 7.2 Reranker (Phase 2+)

**V1: No reranker**
- Return HNSW results directly with CLIP similarity scores
- Reranker using BLIP/cross-encoder models deferred to Phase 2+

**Future implementation:**
```python
def rerank(query: str, candidates: list[dict], top_n: int = 20) -> list[dict]:
    # Phase 2+: Load cross-encoder model
    # Rerun scoring on (query, image) pairs
    # Return top_n reranked results
```

---

## 8. Phase 5 – API Server

Module: `src/api/server.py`

Use FastAPI.

### 8.1 Dependency Setup

On startup:

1. Load metadata.
2. Open `embeddings.dat` memmap.
3. Load HNSW index.
4. Construct `ImageRetriever` instance.

Use FastAPI `startup` event.

**Server Entrypoint:**

`scripts/run_server.py` imports `src.api.server:app` and runs Uvicorn:

```python
# scripts/run_server.py
from src.api.server import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 8.2 Endpoints

**V1 Endpoints:**

1. `GET /search`

Params: `q` (string), `k` (int, default=20)

Response:

```json
{
  "query": "red hoodie dog",
  "results": [
    {
      "frame_id": 123,
      "video_id": "vid_01",
      "timestamp": 10.5,
      "similarity": 0.82,
      "frame_url": "/frame/123"
    },
    ...
  ]
}
```

**Note:** Use `similarity` field (higher is better, range 0-1), not `score` or `distance`.

2. `GET /frame/{frame_id}`

* Load image from disk.
* Return `image/jpeg` response.

3. `GET /health`

* Return basic status, index size, frame count, etc.

**Phase 2+ Endpoint:**

4. `GET /video/{video_id}`

* Serve video file with seek-based playback.
* Deferred to Phase 2+

---

## 9. Phase 6 – Frontend (V1: Minimal)

Folder: `frontend/`

**V1 Requirements (Minimal):**

* Search bar.
* Grid display of returned frames with timestamp + similarity score.
* Click on result: show larger version in modal.
* Plain HTML + JS calling `/search` and `/frame/:id`.

**Phase 2+ Features:**

* Video playback with timestamp seeking.
* Timeline heatmaps.
* Next.js/React implementation (optional).

---

## 10. Phase 7 – Benchmarking & Validation (V1: Simplified)

### 10.1 Benchmark Script (V1)

Create `scripts/benchmark.py`:

**V1 Scope:**
* Run N random queries (e.g., N=100)
* Measure latency for each query
* Print: average latency, p95 latency
* Compare HNSW recall vs brute-force on small test set

**Example output:**
```
Average search latency: 45ms
P95 search latency: 120ms
HNSW recall@10 vs brute-force: 0.96
```

**Phase 2+ Enhancements:**
* Comprehensive parameter tuning (M, efConstruction, efSearch)
* P99 latency tracking
* Build time analysis
* Memory profiling
* Quantization experiments

### 10.2 Running Benchmarks

```bash
python scripts/benchmark.py \
  --queries "a cat" "a skyline at night" \
  --ef-search 32 64 128 \
  --k 5 \
  --repeat 3 \
  --output data/benchmarks/latest.json
```

- `--queries` / `--queries-file`: provide sample prompts (defaults included)
- `--ef-search`: test one or more ef values (must be ≥ k)
- `--repeat`: measure multiple times to average latency
- `--measure-build`: optionally regenerate embeddings/index in a temp dir to time the pipeline (slow, only run intentionally)
- `--output`: writes JSON summary for plotting/comparisons

JSON schema (`data/benchmarks/latest.json`):
```json
{
  "encode_ms": 210.05,
  "per_query": [
    {
      "query": "a cat",
      "ef_search": 64,
      "recall": 1.0,
      "hnsw_ms": 0.026,
      "brute_ms": 1.73,
      "speedup": 42.3,
      "hnsw_results": [0],
      "brute_results": [0]
    }
  ],
  "aggregates": {
    "64": {
      "queries": 2,
      "avg_recall": 1.0,
      "avg_hnsw_ms": 0.026,
      "avg_brute_ms": 1.009,
      "avg_speedup": 42.28
    }
  }
}
```

**Sample Results (1 frame dataset, CPU, ef_search in {32, 64, 128}):**

| ef_search | avg recall@k | avg HNSW latency (ms) | avg brute-force latency (ms) | avg speedup |
| --- | --- | --- | --- | --- |
| 32 | 1.00 | 0.0368 | 1.0089 | 28.3x |
| 64 | 1.00 | 0.0260 | 1.0089 | 42.3x |
| 128 | 1.00 | 0.0282 | 1.0089 | 34.0x |

Absolute numbers will increase with more frames, but the script validates accuracy (recall) and provides a repeatable way to tune ef_search/M for your dataset.

### 10.3 Documentation

In `README.md`:

* System overview diagram.
* Data pipeline description.
* Index internals explanation.
* API description and usage.
* Performance numbers with dataset size.

---

## 11. Testing Strategy (V1: Key Tests Only)

### 11.1 V1 Test Suite (Minimal)

Create `tests/` directory with essential tests:

**V1 Test Coverage:**
- `test_hnsw_search_recall.py`: HNSW recall vs brute-force on small set
- `test_bruteforce_pipeline.py`: End-to-end pipeline validation

**Example test structure:**
```python
# tests/test_hnsw_search_recall.py
def test_hnsw_recall_vs_bruteforce():
    # Create small test embeddings (50-100 vectors)
    # Build HNSW index
    # Run brute-force search for comparison
    # Test HNSW search on same queries
    # Verify recall@10 > 0.95

# tests/test_bruteforce_pipeline.py
def test_end_to_end_pipeline():
    # Ingest small test dataset
    # Generate embeddings
    # Build index
    # Search with known queries
    # Verify results are reasonable
```

### 11.2 Validation Dataset

**V1: Small curated test set**
- 50-100 frames with known semantic content
- Diverse queries: objects, actions, scenes, colors
- Use for HNSW recall validation

**Phase 2+ Testing:**
- Granular per-module tests (ingest, encoder, embed_store, etc.)
- Full integration test suite
- API endpoint tests
- Error handling tests
- Memory management tests
- CI/CD integration

---

## 12. Phase 2+ Features (Future Enhancements)

**Deferred to post-v1:**

1. **Video Serving & Playback**
   * `/video/:id` endpoint with seek-based playback
   * Video timeline navigation

2. **Advanced Reranking**
   * BLIP/cross-encoder reranker
   * Batch server-side reranking (ANN → top 100, rerank → top 20)

3. **Timeline Heatmaps**
   * Aggregate scores by video and timestamp bucket
   * Visual heatmap showing where in each video the query matches best

4. **Multi-query Logic**
   * Parse query into multiple tokens or phrases
   * Combine multiple embeddings (AND/OR semantics)

5. **Tagging Pipeline**
   * Run captioning model on frames
   * Store auto tags + captions in metadata
   * Support filtering by tags

6. **Duplicate Frame Suppression**
   * Use perceptual hashing on frames
   * Only keep keyframes with sufficient difference from previous

7. **Advanced Serialization**
   * Custom binary format for HNSW index (optimized for size/speed)
   * Replace pickle with more efficient format

8. **Incremental Index Updates**
   * Add new media without full rebuild
   * Incremental HNSW insertion

9. **Comprehensive Testing**
   * Full test suite with granular per-module tests
   * CI/CD integration
   * Performance regression testing

---

This is the full plan.