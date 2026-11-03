from types import SimpleNamespace

from agentic_rag.agent.graph import build_agent


class FakeRouter:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.prompts = []

    def invoke_with_fallback(self, prompt):
        self.prompts.append(prompt)
        return next(self.replies)


class FakeEmbedder:
    def embed_query(self, query):
        return [0.1, 0.2]


class FakeStore:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, vector, top_k=5):
        return self.chunks[:top_k]


def make_chunk():
    return SimpleNamespace(
        score=0.75,
        payload={
            "chunk_id": "paper__0",
            "doc_id": "paper",
            "chunk_index": 0,
            "source_url": "https://example.com/paper",
            "text": "Multi-head attention attends to different subspaces.",
        },
    )


def test_answerable_question():
    router = FakeRouter(
        [
            '{"action": "retrieve", "query": "multi-head attention"}',
            "It attends to different subspaces [paper__0].",
        ]
    )
    app = build_agent(router, FakeEmbedder(), FakeStore([make_chunk()]))

    result = app.invoke(
        {
            "question": "Why use multi-head attention?",
            "router_calls": 0,
            "steps": [],
        }
    )

    assert result["status"] == "answered_unverified"
    assert result["citations"] == ["paper__0"]
    assert result["router_calls"] == 2
    assert len(router.prompts) == 2
