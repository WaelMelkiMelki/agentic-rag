"""FastAPI application exposing the B1 baseline and B4 agent."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agentic_rag.agent.memory import standalone_question
from agentic_rag.api.deps import (
    get_agent,
    get_baseline,
    get_closed_book,
    get_router,
    get_sessions,
    get_store,
)
from agentic_rag.tracing import flush, trace_agent

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    system: str = Field(default="B4", pattern="^(B0|B1|B4)$")
    # Optional: send the same session_id on follow-up questions to get multi-turn
    # behaviour ("and the second one?"). Omit it for a stateless request.
    session_id: str | None = Field(default=None, min_length=1, max_length=64)


class QueryResponse(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    status: str = ""
    stop_reason: str = ""
    system: str
    latency_s: float
    # The question actually retrieved against (equal to `question` when no
    # session history was used).
    standalone_question: str = ""


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
    get_closed_book()
    logger.info("API ready")
    yield
    flush()
    logger.info("Flushed Langfuse traces")


app = FastAPI(title="Agentic RAG API", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse | JSONResponse:
    try:
        return HealthResponse(status="ok", qdrant_docs=get_store().count())
    except Exception:
        logger.exception("Health check failed")
        return JSONResponse(
            status_code=503,
            content=HealthResponse(status="degraded", qdrant_docs=-1).model_dump(),
        )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    started = time.perf_counter()
    try:
        question = req.question
        if req.session_id:
            transcript = get_sessions().transcript(req.session_id)
            if transcript:
                question = standalone_question(req.question, transcript, get_router())

        if req.system == "B4":
            result = get_agent().invoke(
                {"question": question, "router_calls": 0, "steps": []}
            )
            answer = result.get("final_answer", "")
            citations = result.get("citations", [])
            steps = result.get("steps", [])
            status = result.get("status", "unknown")
            stop_reason = result.get("stop_reason", "unknown")
        elif req.system == "B1":
            result = get_baseline().query(question)
            answer = result["answer"]
            citations = result["citations"]
            steps = result["steps"]
            status = result["status"]
            stop_reason = result["stop_reason"]
        else:
            result = get_closed_book().query(question)
            answer = result["answer"]
            citations = result["citations"]
            steps = result["steps"]
            status = result["status"]
            stop_reason = result["stop_reason"]

        if req.session_id:
            get_sessions().remember(req.session_id, req.question, answer)

        trace_id = trace_agent(
            req.question,
            {
                "answer": answer,
                "citations": citations,
                "steps": steps,
                "status": status,
                "stop_reason": stop_reason,
                "system": req.system,
            },
        )
        if trace_id:
            logger.info("Langfuse trace: %s", trace_id)

        return QueryResponse(
            answer=answer,
            citations=citations,
            steps=steps,
            status=status,
            stop_reason=stop_reason,
            system=req.system,
            latency_s=round(time.perf_counter() - started, 2),
            standalone_question=question,
        )
    except Exception as error:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Query processing failed") from error
