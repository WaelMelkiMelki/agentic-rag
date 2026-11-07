"""FastAPI application exposing the B1 baseline and B4 agent."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agentic_rag.api.deps import get_agent, get_baseline, get_store

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    system: str = Field(default="B4", pattern="^(B1|B4)$")


class QueryResponse(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    status: str = ""
    stop_reason: str = ""
    system: str
    latency_s: float


class HealthResponse(BaseModel):
    status: str
    qdrant_docs: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Construct and validate shared dependencies once when serving starts."""
    logger.info("Warming up dependencies")
    get_store()
    get_agent()
    get_baseline()
    logger.info("API ready")
    yield


app = FastAPI(title="Agentic RAG API", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        return HealthResponse(status="ok", qdrant_docs=get_store().count())
    except Exception:
        logger.exception("Health check failed")
        return HealthResponse(status="degraded", qdrant_docs=-1)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    started = time.perf_counter()
    try:
        if req.system == "B4":
            result = get_agent().invoke(
                {"question": req.question, "router_calls": 0, "steps": []}
            )
            answer = result.get("final_answer", "")
            citations = result.get("citations", [])
            steps = result.get("steps", [])
            status = result.get("status", "unknown")
            stop_reason = result.get("stop_reason", "unknown")
        else:
            result = get_baseline().query(req.question)
            answer = result["answer"]
            citations = result["citations"]
            steps = result["steps"]
            status = result["status"]
            stop_reason = result["stop_reason"]

        return QueryResponse(
            answer=answer,
            citations=citations,
            steps=steps,
            status=status,
            stop_reason=stop_reason,
            system=req.system,
            latency_s=round(time.perf_counter() - started, 2),
        )
    except Exception as error:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Query processing failed") from error
