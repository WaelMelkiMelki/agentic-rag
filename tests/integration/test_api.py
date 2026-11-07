from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agentic_rag.api import main


class FakeAgent:
    def invoke(self, state: dict) -> dict:
        return {
            "final_answer": "Fake answer [chunk_1].",
            "citations": ["chunk_1"],
            "steps": ["plan", "retrieve", "generate", "finalize"],
            "status": "answered",
            "stop_reason": "completed",
        }


class FakeBaseline:
    def query(self, question: str) -> dict:
        return {
            "answer": "Fake B1 answer [chunk_2].",
            "citations": ["chunk_2"],
            "steps": ["retrieve", "generate"],
            "status": "answered",
            "stop_reason": "completed",
        }


class FakeStore:
    def count(self) -> int:
        return 42


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "get_agent", lambda: FakeAgent())
    monkeypatch.setattr(main, "get_baseline", lambda: FakeBaseline())
    monkeypatch.setattr(main, "get_store", lambda: FakeStore())
    # Do not enter TestClient as a context manager: that would run the real
    # startup lifespan and construct Qdrant, embeddings, and provider clients.
    return TestClient(main.app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "qdrant_docs": 42}


def test_query_b4(client):
    response = client.post(
        "/query", json={"question": "What is attention?", "system": "B4"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "Fake answer" in data["answer"]
    assert data["citations"] == ["chunk_1"]
    assert data["system"] == "B4"
    assert data["status"] == "answered"
    assert data["latency_s"] >= 0


def test_query_b1(client):
    response = client.post("/query", json={"question": "What is RAG?", "system": "B1"})
    assert response.status_code == 200
    data = response.json()
    assert "Fake B1" in data["answer"]
    assert data["citations"] == ["chunk_2"]
    assert data["system"] == "B1"


def test_query_invalid_system(client):
    response = client.post("/query", json={"question": "Hi", "system": "B99"})
    assert response.status_code == 422


def test_query_empty_question(client):
    response = client.post("/query", json={"question": "", "system": "B4"})
    assert response.status_code == 422


def test_simple_rag_accepts_injected_dependencies():
    from types import SimpleNamespace

    from agentic_rag.agent.rag import SimpleRAG

    class FakeEmbedder:
        def embed_query(self, question):
            return [0.0, 1.0]

    class FakeStore:
        def search(self, vector, top_k):
            return [
                SimpleNamespace(
                    score=0.9,
                    payload={
                        "chunk_id": "chunk_a",
                        "doc_id": "doc_a",
                        "chunk_index": 0,
                        "text": "Evidence",
                    },
                )
            ]

    class FakeRouter:
        def invoke_with_fallback(self, prompt):
            return "A fake answer [chunk_a]."

    baseline = SimpleRAG(
        router=FakeRouter(), embedder=FakeEmbedder(), store=FakeStore(), top_k=3
    )
    result = baseline.query("Question")

    assert result["answer"] == "A fake answer [chunk_a]."
    assert result["citations"] == ["chunk_a"]
    assert result["status"] == "answered"
    assert result["stop_reason"] == "completed"
