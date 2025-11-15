# Semantic Media Search

Local-first semantic image/video search pipeline that ingests media, encodes frames with CLIP, indexes them using a custom HNSW graph, and serves results via FastAPI plus a minimal browser UI.

- **Embeddings:** `openai/clip-vit-base-patch32` (512-dim, cosine similarity)
- **Storage:** metadata in JSONL, embeddings in `numpy.memmap`, frame images on disk
- **ANN:** fully custom HNSW implementation (build/search/serialize)
- **Serving:** FastAPI (`/search`, `/frame/{id}`, `/health`) and a static HTML grid at `/`
- **Benchmarks:** brute-force baseline + `scripts/benchmark.py` to track recall/latency

For the full specification (phase breakdown, module details, future work), see [`docs/plan.md`](docs/plan.md).

---

## Quick Start

```bash
git clone <repo> && cd semantic_media_search
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1) Drop media under data/raw/{images,videos}
python scripts/ingest_all.py

# 2) Encode + build HNSW (re-run whenever media changes)
python scripts/build_index.py

# 3) Serve API + frontend
python scripts/run_server.py
```

- Browser UI: http://localhost:8000
- REST query: `curl "http://localhost:8000/search?q=red%20hoodie%20dog&k=10"`
- Benchmark (optional): `python scripts/benchmark.py --queries "a cat" "a skyline at night" --ef-search 32 64 128 --k 5`

---

## API Surface

| Endpoint | Description | Notes |
| --- | --- | --- |
| `GET /` | Serves `frontend/index.html` | Simple search UI hitting `/search` |
| `GET /search?q=...&k=...` | Text → CLIP → HNSW results | Returns `{query, results[]}` with `frame_id`, `similarity`, `frame_url`, metadata |
| `GET /frame/{frame_id}` | Streams JPEG frame from disk | Used by UI hover/preview |
| `GET /health` | Frame count, embedding dim, HNSW nodes | Health/readiness |

All similarities are cosine scores in `[0, 1]` (higher is better). Validation errors return 400; runtime errors are logged and surfaced as 500 with a generic payload.

---

## Architecture Snapshot

1. **Ingest** (`scripts/ingest_all.py`): walks `data/raw/`, saves normalized frames under `data/processed/frames/`, appends metadata to `metadata.jsonl`.
2. **Embed** (`src/embeddings/`): CLIP encodes frames in batches, writes normalized vectors to `data/processed/embeddings.dat` (`numpy.memmap`) plus `embeddings.idx` for frame_id ↔ row index mapping.
3. **Index** (`src/index/hnsw.py` + `scripts/build_index.py`): builds a multi-layer HNSW graph, serializes to `data/processed/hnsw_index.bin`.
4. **Retrieve** (`src/search/retriever.py`): loads metadata, embedding store, HNSW index, and CLIP text encoder to answer queries.
5. **Serve** (`src/api/server.py` + `frontend/index.html`): FastAPI exposes HTTP endpoints and static UI.
6. **Benchmark** (`scripts/benchmark.py`): compares HNSW vs brute-force recall/latency and optionally measures rebuild time.

---

## Project Layout

```
semantic_media_search/
  frontend/                # HTML/JS UI
  scripts/
    ingest_all.py          # image + video ingestion
    build_index.py         # embeddings + HNSW
    run_server.py          # uvicorn entry point
    benchmark.py           # recall/latency benchmarks
  src/
    api/server.py          # FastAPI app
    embeddings/            # CLIP encoder + memmap store
    ingest/                # metadata writer, image/video ingest
    index/hnsw.py          # custom HNSW graph
    search/retriever.py    # glue layer (text -> ANN -> metadata)
    utils/                 # logging, IO, timing helpers
  data/
    raw/{images,videos}/   # user-provided inputs
    processed/             # frames, metadata.jsonl, embeddings.dat, hnsw_index.bin
  docs/plan.md             # full specification + roadmap
  README.md
  requirements.txt
```

---

## Benchmarks & Validation

- `scripts/benchmark.py` encodes a list of queries once, times both HNSW and brute-force search, computes recall@k, and summarizes results per `ef_search`.
- Sample (single-frame dataset, CPU):

| ef_search | avg recall@k | avg HNSW latency (ms) | avg brute-force latency (ms) | avg speedup |
| --- | --- | --- | --- | --- |
| 32 | 1.00 | 0.0368 | 1.0089 | 28.3x |
| 64 | 1.00 | 0.0260 | 1.0089 | 42.3x |
| 128 | 1.00 | 0.0282 | 1.0089 | 34.0x |

Absolute timings scale with dataset size, but the script provides a repeatable harness for tuning `M`, `ef_construction`, and `ef_search`.

---

## Testing

- Brute-force search module doubles as a ground-truth implementation for recall checks.
- Add lightweight integration tests under `tests/` (e.g., ingest → embed → search pipeline, HNSW recall vs brute-force on synthetic data).
- For exhaustive coverage, refer to the roadmap in [`docs/plan.md`](docs/plan.md) (Phase 5+ testing strategy).

---

## Further Reading

- [`docs/plan.md`](docs/plan.md): original end-to-end plan, phased implementation notes, and future enhancements (video playback, rerankers, incremental indexing, etc.).