from types import SimpleNamespace

from agentic_rag.agent.grade import grade_chunks


class FakeRouter:
    def __init__(self, replies):
        self.replies = iter(replies)

    def invoke_with_fallback(self, prompt):
        return next(self.replies)


def make_chunk(chunk_id, text):
    return SimpleNamespace(payload={"chunk_id": chunk_id, "text": text})


def test_grade_filters_irrelevant():
    chunks = [make_chunk("c1", "Attention has multiple heads"), make_chunk("c2", "Pizza is cheesy")]
    relevant, calls = grade_chunks("What is attention?", chunks, FakeRouter(["yes", "no"]))
    assert [c.payload["chunk_id"] for c in relevant] == ["c1"]
    assert calls == 2


def test_grade_keeps_chunk_on_router_failure():
    chunk = make_chunk("c1", "Relevant content")

    class BrokenRouter:
        def invoke_with_fallback(self, prompt):
            raise RuntimeError("offline")

    relevant, calls = grade_chunks("Question", [chunk], BrokenRouter())
    assert relevant == [chunk]
    assert calls == 1
