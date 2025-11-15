"""
Benchmark script for semantic media search.
Measures HNSW search latency / recall against brute-force and optional build timing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np  # noqa: E402

from src.config import (  # noqa: E402
    HNSW_EF_SEARCH,
    METADATA_PATH,
)
from src.embeddings.clip_encoder import CLIPEncoder  # noqa: E402
from src.embeddings.embed_store import EmbeddingStore  # noqa: E402
from src.index.hnsw import HNSWIndex  # noqa: E402
from src.ingest.metadata_store import load_metadata  # noqa: E402
from src.search.brute_force import brute_force_search  # noqa: E402
from src.search.retriever import ImageRetriever  # noqa: E402
from src.utils.logging_utils import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark HNSW vs brute-force search."
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        help="Text queries to benchmark (space separated)",
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        help="Path to newline-delimited text queries",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of results to retrieve",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        nargs="+",
        default=[HNSW_EF_SEARCH],
        help="ef_search values to test (>= k). Specify multiple for grid search.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of times to run each search for latency averaging",
    )
    parser.add_argument(
        "--measure-build",
        action="store_true",
        help="Measure embedding generation + HNSW build times using a temporary index",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for optional build measurement",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write benchmark results as JSON",
    )
    return parser.parse_args()


def load_queries(args: argparse.Namespace) -> list[str]:
    queries: list[str] = []
    if args.queries:
        queries.extend(args.queries)
    if args.queries_file and args.queries_file.exists():
        with args.queries_file.open("r", encoding="utf-8") as f:
            queries.extend([line.strip() for line in f if line.strip()])
    if not queries:
        queries = [
            "a portrait photo",
            "a person outdoors",
            "close-up texture",
            "a city skyline at night",
            "an animal in nature",
        ]
    return queries


def measure_latency(func, repeat: int) -> tuple[float, any]:
    durations: list[float] = []
    result = None
    for _ in range(max(repeat, 1)):
        start = time.perf_counter()
        result = func()
        durations.append((time.perf_counter() - start) * 1000.0)
    avg_ms = float(mean(durations))
    return avg_ms, result


def compute_recall(
    approx_ids: Iterable[int],
    exact_ids: Iterable[int],
) -> float:
    approx_set = set(approx_ids)
    exact_set = set(exact_ids)
    if not exact_set:
        return 0.0
    return len(approx_set & exact_set) / len(exact_set)


def benchmark_searches(
    retriever: ImageRetriever,
    queries: list[str],
    k: int,
    ef_values: list[int],
    repeat: int,
    logger,
) -> dict:
    embeddings = retriever.embeddings
    frame_ids = [retriever.row_idx_to_frame_id[idx] for idx in range(len(embeddings))]

    logger.info("Encoding %d queries", len(queries))
    encode_start = time.perf_counter()
    query_embeddings = (
        retriever.encoder.encode_texts(queries).cpu().numpy().astype(np.float32)
    )
    encode_ms = (time.perf_counter() - encode_start) * 1000.0

    per_query_stats = []
    grouped_stats: dict[int, list[dict]] = defaultdict(list)

    for query, query_vec in zip(queries, query_embeddings):
        brute_ms, brute_results = measure_latency(
            lambda: brute_force_search(query_vec, embeddings, frame_ids, k),
            repeat=1,
        )
        brute_ids = [frame_id for frame_id, _ in brute_results]

        for ef in ef_values:
            if ef < k:
                logger.warning("ef_search (%d) is less than k (%d); adjusting", ef, k)
                ef = k

            hnsw_ms, hnsw_results = measure_latency(
                lambda: retriever.hnsw_index.search(query_vec, k=k, ef=ef),
                repeat=repeat,
            )
            mapped = [
                (retriever.row_idx_to_frame_id[node_id], sim)
                for node_id, sim in hnsw_results
            ]
            hnsw_ids = [frame_id for frame_id, _ in mapped]

            recall = compute_recall(hnsw_ids, brute_ids)
            speedup = brute_ms / hnsw_ms if hnsw_ms else float("inf")

            stat = {
                "query": query,
                "ef_search": ef,
                "recall": recall,
                "hnsw_ms": hnsw_ms,
                "brute_ms": brute_ms,
                "speedup": speedup,
                "hnsw_results": hnsw_ids,
                "brute_results": brute_ids,
            }
            per_query_stats.append(stat)
            grouped_stats[ef].append(stat)

            logger.info(
                "[ef=%d] query='%s' recall=%.3f hnsw=%.2fms brute=%.2fms speedup=%.2fx",
                ef,
                query,
                recall,
                hnsw_ms,
                brute_ms,
                speedup,
            )

    aggregates = {}
    for ef, stats in grouped_stats.items():
        aggregates[ef] = {
            "queries": len(stats),
            "avg_recall": mean(s["recall"] for s in stats),
            "avg_hnsw_ms": mean(s["hnsw_ms"] for s in stats),
            "avg_brute_ms": mean(s["brute_ms"] for s in stats),
            "avg_speedup": mean(s["speedup"] for s in stats),
        }

    logger.info(
        "Encoding time: %.2f ms (%.2f ms/query)", encode_ms, encode_ms / len(queries)
    )

    return {
        "encode_ms": encode_ms,
        "per_query": per_query_stats,
        "aggregates": aggregates,
    }


def measure_build_times(batch_size: int, logger) -> dict:
    logger.warning(
        "Build measurement will regenerate embeddings and HNSW index in a temp directory."
    )

    metadata = load_metadata(METADATA_PATH)
    if not metadata:
        raise RuntimeError("No metadata available. Run ingestion first.")

    temp_dir = project_root / "data" / "benchmarks"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_embeddings = temp_dir / "embeddings.dat"
    temp_index = temp_dir / "embeddings.idx"

    # Ensure previous temp files are removed
    for path in (temp_embeddings, temp_index):
        if path.exists():
            path.unlink()

    store = EmbeddingStore(temp_embeddings, temp_index)
    encoder = CLIPEncoder()

    logger.info("Measuring embedding generation (batch=%d)...", batch_size)
    start = time.perf_counter()
    store.create(metadata, encoder, batch_size=batch_size)
    embed_ms = (time.perf_counter() - start) * 1000.0

    logger.info("Measuring HNSW build...")
    embeddings_np = store.get_all_embeddings()
    hnsw = HNSWIndex(dim=embeddings_np.shape[1])
    start = time.perf_counter()
    hnsw.build(embeddings_np)
    build_ms = (time.perf_counter() - start) * 1000.0

    # Clean up temp files to avoid confusion
    for path in (temp_embeddings, temp_index):
        if path.exists():
            path.unlink()

    return {
        "embedding_ms": embed_ms,
        "hnsw_build_ms": build_ms,
    }


def main():
    args = parse_args()
    logger = setup_logging()

    queries = load_queries(args)
    logger.info("Benchmarking %d queries (k=%d)", len(queries), args.k)

    retriever = ImageRetriever()
    benchmarks = benchmark_searches(
        retriever=retriever,
        queries=queries,
        k=args.k,
        ef_values=args.ef_search,
        repeat=args.repeat,
        logger=logger,
    )

    if args.measure_build:
        try:
            build_stats = measure_build_times(batch_size=args.batch_size, logger=logger)
            benchmarks["build_times"] = build_stats
            logger.info(
                "Build timing -> embeddings: %.2fs, hnsw: %.2fs",
                build_stats["embedding_ms"] / 1000.0,
                build_stats["hnsw_build_ms"] / 1000.0,
            )
        except Exception as exc:
            logger.exception("Build measurement failed: %s", exc)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(benchmarks, f, indent=2)
        logger.info("Benchmark results written to %s", args.output)
    else:
        logger.info(
            "Benchmark summary: %s", json.dumps(benchmarks["aggregates"], indent=2)
        )


if __name__ == "__main__":
    main()
