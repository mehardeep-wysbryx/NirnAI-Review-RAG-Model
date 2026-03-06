"""
NirnAI Review API — Production FastAPI service.

Exposes the RAG-powered two-stage review pipeline as HTTP endpoints
that any web application can call.

Usage (local):
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Usage (production):
    uvicorn api:app --host 0.0.0.0 --port 8000 --workers 2
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger("nirnai")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Globals — initialized once at startup via the lifespan hook
# ---------------------------------------------------------------------------
pipeline = None
store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize heavy resources once when the server starts."""
    global pipeline, store

    logger.info("Initializing Pinecone store …")
    from src.pinecone_store import PineconeStore
    store = PineconeStore()
    stats = store.get_stats()
    logger.info("Pinecone connected — %d vectors in index '%s'",
                stats["total_vectors"], stats["index_name"])

    logger.info("Initializing ReviewPipeline …")
    from src.review import ReviewPipeline
    pipeline = ReviewPipeline(precedent_store=store, output_dir="./outputs")
    logger.info("Pipeline ready.")

    yield

    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NirnAI Review API",
    description="RAG-powered two-stage legal document review for loan-against-property title verification.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class ReviewRequest(BaseModel):
    """The merged case JSON sent by the client."""
    attachments: Optional[List[Any]] = None
    encumbranceDetails: Optional[List[Any]] = None
    reportJson: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


class ReviewResponse(BaseModel):
    overall_summary: Optional[str] = None
    overall_risk_level: Optional[str] = None
    sections: Optional[Dict[str, Any]] = None
    duration_seconds: Optional[float] = None

    class Config:
        extra = "allow"


class IngestRequest(BaseModel):
    """One or more raw case JSONs to ingest into Pinecone."""
    cases: List[Dict[str, Any]] = Field(..., min_length=1)


class IngestResponse(BaseModel):
    files_processed: int
    total_chunks: int
    errors: List[str]


class StatsResponse(BaseModel):
    index_name: str
    total_vectors: int
    estimated_precedents: int


class HealthResponse(BaseModel):
    status: str
    pinecone: str
    openai: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/review", response_model=ReviewResponse)
async def review_case(request: Request):
    """
    Run the two-stage RAG review on a merged case JSON.

    Accepts the full merged_case_json body (attachments, encumbranceDetails, reportJson).
    Returns the REVIEW_OBJECT with issues, severity, and recommendation.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized yet.")

    body = await request.json()

    start = time.time()
    try:
        result = pipeline.review(body, save_output=False, verbose=False)
    except Exception as exc:
        logger.exception("Review failed")
        raise HTTPException(status_code=500, detail=str(exc))
    elapsed = round(time.time() - start, 2)

    result["duration_seconds"] = elapsed
    logger.info("Review completed in %.2fs — risk=%s", elapsed, result.get("overall_risk_level"))
    return result


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_cases(req: IngestRequest):
    """
    Ingest one or more raw case JSONs into Pinecone as precedents.

    Use this to grow the precedent database programmatically
    (e.g. a nightly batch from your main app).
    """
    if store is None:
        raise HTTPException(status_code=503, detail="Store not initialized yet.")

    total_chunks = 0
    errors: List[str] = []

    for i, case in enumerate(req.cases):
        try:
            chunks = store.ingest_precedent(case, filename=f"api_case_{i}")
            total_chunks += chunks
        except Exception as exc:
            errors.append(f"case[{i}]: {exc}")

    return IngestResponse(
        files_processed=len(req.cases) - len(errors),
        total_chunks=total_chunks,
        errors=errors,
    )


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Return Pinecone index statistics."""
    if store is None:
        raise HTTPException(status_code=503, detail="Store not initialized yet.")

    stats = store.get_stats()
    total_vectors = stats.get("total_vectors", 0)

    return StatsResponse(
        index_name=stats.get("index_name", ""),
        total_vectors=total_vectors,
        estimated_precedents=round(total_vectors / 3.47) if total_vectors else 0,
    )


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Verify connectivity to Pinecone and OpenAI."""
    pinecone_ok = "error"
    openai_ok = "error"

    # Pinecone
    try:
        if store is not None:
            store.get_stats()
            pinecone_ok = "ok"
    except Exception as exc:
        pinecone_ok = f"error: {exc}"

    # OpenAI
    try:
        import openai
        client = openai.OpenAI()
        client.models.list()
        openai_ok = "ok"
    except Exception as exc:
        openai_ok = f"error: {exc}"

    status = "healthy" if pinecone_ok == "ok" and openai_ok == "ok" else "degraded"

    return HealthResponse(status=status, pinecone=pinecone_ok, openai=openai_ok)
