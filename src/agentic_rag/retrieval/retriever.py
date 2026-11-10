"""LangChain retriever over the Qdrant chunk collection.

``ChunkRetriever`` implements LangChain's ``BaseRetriever`` protocol, so
retrieval becomes ``retriever.invoke(query) -> list[Document]`` and automatically
participates in LangChain callbacks (that is how the Langfuse handler sees it).

Why not ``langchain_qdrant.QdrantVectorStore.as_retriever()``? That class expects
the *LangChain payload layout* — page content under ``page_content``/``text`` and a
**nested dict** under ``metadata_payload_key``. This collection stores flat
payloads (``chunk_id``, ``doc_id``, ``chunk_index``, ``text``, ``source_url``) that
were indexed before LangChain was involved. Going through ``QdrantVectorStore``
would drop ``chunk_id``, and ``chunk_id`` is what the citation and grounding
checks verify. ``ChunkRetriever`` keeps the flat payload intact while still
speaking the LangChain ``Document`` protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

#: Payload keys that travel from Qdrant into ``Document.metadata``.
METADATA_KEYS = ("chunk_id", "doc_id", "chunk_index", "source_url")


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk in the shape the agent and the baseline already use."""

    payload: dict[str, Any]
    score: float


class ChunkRetriever(BaseRetriever):
    """Embed the query, search Qdrant, and return LangChain ``Document``s."""

    store: Any
    embedder: Any
    k: int = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        if self.k <= 0:
            raise ValueError("k must be greater than zero")
        vector = self.embedder.embed_query(query)
        points = self.store.search(vector, top_k=self.k)
        return [
            Document(
                page_content=point.payload.get("text", ""),
                metadata={
                    **{key: point.payload.get(key) for key in METADATA_KEYS},
                    "score": float(point.score),
                },
            )
            for point in points
        ]


def build_retriever(store, embedder, k: int = 5) -> ChunkRetriever:
    """Construct the project's default LangChain retriever."""
    return ChunkRetriever(store=store, embedder=embedder, k=k)


def documents_to_chunks(documents: list[Document]) -> list[RetrievedChunk]:
    """Convert retriever output back into the agent's chunk shape."""
    chunks = []
    for document in documents:
        metadata = document.metadata or {}
        chunks.append(
            RetrievedChunk(
                payload={
                    "chunk_id": metadata.get("chunk_id"),
                    "doc_id": metadata.get("doc_id"),
                    "chunk_index": metadata.get("chunk_index"),
                    "source_url": metadata.get("source_url"),
                    "text": document.page_content,
                },
                score=float(metadata.get("score", 0.0)),
            )
        )
    return chunks
