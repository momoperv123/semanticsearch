"""
FastAPI server for semantic media search.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.config import PROJECT_ROOT
from src.search.retriever import ImageRetriever, SearchResult
from src.utils.logging_utils import setup_logging


logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup/shutdown."""
    # Startup
    logger.info("Loading retriever for API server...")
    app.state.retriever = ImageRetriever()
    logger.info("Retriever ready")
    yield
    # Shutdown (if needed in future)
    # Cleanup code here


app = FastAPI(title="Semantic Media Search", lifespan=lifespan)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchResultModel(BaseModel):
    frame_id: int
    video_id: str | None
    timestamp: float | None
    frame_path: str
    width: int
    height: int
    similarity: float
    frame_url: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultModel]


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    """Serve the frontend HTML page."""
    frontend_path = Path(PROJECT_ROOT, "frontend", "index.html")
    if not frontend_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return frontend_path.read_text(encoding="utf-8")


@app.get("/health")
async def health() -> dict[str, int | str]:
    """Simple health check."""
    retriever: ImageRetriever = app.state.retriever
    return {
        "status": "ok",
        "frames": len(retriever.metadata),
        "embeddings_dim": retriever.embeddings.shape[1],
        "hnsw_nodes": retriever.hnsw_index.num_nodes,
    }


def _result_to_model(result: SearchResult) -> SearchResultModel:
    return SearchResultModel(
        frame_id=result.frame_id,
        video_id=result.video_id,
        timestamp=result.timestamp,
        frame_path=result.frame_path,
        width=result.width,
        height=result.height,
        similarity=result.similarity,
        frame_url=f"/frame/{result.frame_id}",
    )


@app.get("/search", response_model=SearchResponse)
async def search(q: str, k: int = 20) -> SearchResponse:
    """Search images by text query."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")
    if k <= 0:
        raise HTTPException(status_code=400, detail="Parameter 'k' must be positive")

    retriever: ImageRetriever = app.state.retriever
    try:
        results = retriever.search_text(q, k)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    models = [_result_to_model(res) for res in results]
    return SearchResponse(query=q, results=models)


@app.get("/frame/{frame_id}")
async def get_frame(frame_id: int) -> FileResponse:
    """Serve the frame image."""
    retriever: ImageRetriever = app.state.retriever
    try:
        metadata = retriever.metadata[frame_id]
    except KeyError:
        raise HTTPException(status_code=404, detail="Frame not found") from None

    frame_path = Path(PROJECT_ROOT, metadata["frame_path"]).resolve()
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame file missing")

    return FileResponse(path=str(frame_path), media_type="image/jpeg")


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled server error")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
