"""Application-wide singletons for API routes."""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_rag.agent.closed_book import ClosedBook
    from agentic_rag.agent.rag import SimpleRAG
    from agentic_rag.llm.router import LLMRouter
    from agentic_rag.retrieval.fast_embedder import FastEmbedder
    from agentic_rag.retrieval.store import VectorStore


@lru_cache(maxsize=1)
def get_router() -> LLMRouter:
    from agentic_rag.llm.router import LLMRouter
    from agentic_rag.llm.tracker import UsageTracker

    return LLMRouter(tracker=UsageTracker())


@lru_cache(maxsize=1)
def get_embedder() -> FastEmbedder:
    from agentic_rag.retrieval.fast_embedder import FastEmbedder

    # ONNX/FastEmbed keeps the API process small while matching the indexed model.
    return FastEmbedder()


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    from agentic_rag.retrieval.store import VectorStore

    return VectorStore()


@lru_cache(maxsize=1)
def get_agent():
    from agentic_rag.agent.graph import build_agent

    return build_agent(
        router=get_router(),
        embedder=get_embedder(),
        store=get_store(),
        top_k=5,
        max_router_calls=10,
        max_rounds=2,
        grounded_threshold=0.75,
    )


@lru_cache(maxsize=1)
def get_baseline() -> SimpleRAG:
    from agentic_rag.agent.rag import SimpleRAG

    return SimpleRAG(
        router=get_router(),
        embedder=get_embedder(),
        store=get_store(),
        top_k=5,
    )


@lru_cache(maxsize=1)
def get_closed_book() -> "ClosedBook":
    from agentic_rag.agent.closed_book import ClosedBook

    return ClosedBook(router=get_router())
