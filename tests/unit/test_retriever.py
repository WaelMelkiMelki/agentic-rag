"""LangChain retriever over Qdrant: Documents in, agent chunks out."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentic_rag.retrieval.retriever import (
    ChunkRetriever,
    build_retriever,
    documents_to_chunks,
)


class FakeEmbedder:
    def __init__(self):
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return [0.1, 0.2, 0.3]


class FakeStore:
    def __init__(self, points=None):
        self.points = points if points is not None else [_point("c1", 0.9), _point("c2", 0.4)]
        self.calls = []

    def search(self, vector, top_k):
        self.calls.append((vector, top_k))
        return self.points[:top_k]


def _point(chunk_id: str, score: float):
    return SimpleNamespace(
        score=score,
        payload={
            "chunk_id": chunk_id,
            "doc_id": "attention_is_all_you_need",
            "chunk_index": 3,
            "text": f"text for {chunk_id}",
            "source_url": "https://arxiv.org/pdf/1706.03762",
        },
    )


def test_retriever_returns_documents_with_payload_metadata():
    embedder, store = FakeEmbedder(), FakeStore()
    documents = build_retriever(store, embedder, k=2).invoke("What is attention?")

    assert len(documents) == 2
    assert documents[0].page_content == "text for c1"
    assert documents[0].metadata["chunk_id"] == "c1"
    assert documents[0].metadata["doc_id"] == "attention_is_all_you_need"
    assert documents[0].metadata["chunk_index"] == 3
    assert documents[0].metadata["score"] == pytest.approx(0.9)
    assert embedder.queries == ["What is attention?"]


def test_retriever_honours_top_k():
    store = FakeStore()
    build_retriever(store, FakeEmbedder(), k=1).invoke("q")
    assert store.calls[0][1] == 1


def test_retriever_rejects_invalid_k():
    with pytest.raises(ValueError, match="k must be greater than zero"):
        ChunkRetriever(store=FakeStore(), embedder=FakeEmbedder(), k=0).invoke("q")


def test_retriever_returns_empty_list_for_no_hits():
    assert build_retriever(FakeStore(points=[]), FakeEmbedder(), k=5).invoke("q") == []


def test_documents_convert_back_to_agent_chunks():
    documents = build_retriever(FakeStore(), FakeEmbedder(), k=2).invoke("q")
    chunks = documents_to_chunks(documents)

    assert [chunk.payload["chunk_id"] for chunk in chunks] == ["c1", "c2"]
    assert chunks[0].payload["text"] == "text for c1"
    assert chunks[1].score == pytest.approx(0.4)
    # The agent formats context from these keys.
    assert set(chunks[0].payload) == {
        "chunk_id", "doc_id", "chunk_index", "source_url", "text",
    }


def test_documents_without_score_default_to_zero():
    from langchain_core.documents import Document

    chunks = documents_to_chunks([Document(page_content="x", metadata={"chunk_id": "c9"})])
    assert chunks[0].score == 0.0
    assert chunks[0].payload["text"] == "x"
