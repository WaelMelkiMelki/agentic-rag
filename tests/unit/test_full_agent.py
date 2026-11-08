from types import SimpleNamespace

from agentic_rag.agent.graph import build_agent


class FakeRouter:
    def __init__(self, replies):
        self.replies = list(replies)
        self.call_count = 0

    def invoke_with_fallback(self, prompt):
        reply = self.replies[self.call_count] if self.call_count < len(self.replies) else "fallback"
        self.call_count += 1
        return reply


class FakeEmbedder:
    def embed_query(self, query):
        return [0.1] * 768


class FakeStore:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, vector, top_k=5):
        return self.chunks[:top_k]


def chunk(cid="c1", text="Multi-head attention enables parallel processing."):
    return SimpleNamespace(score=0.8, payload={"chunk_id": cid, "doc_id": "paper",
        "chunk_index": 0, "source_url": "url", "text": text})


def initial_state(question="What is attention?"):
    return {"question": question, "router_calls": 0, "steps": []}


def test_happy_path():
    router = FakeRouter(['{"action":"retrieve","query":"attention"}', "yes", "Answer [c1].",
                         "CLAIM: Attention enables parallel processing\nVERDICT: supported\nCHUNKS: c1"])
    result = build_agent(router, FakeEmbedder(), FakeStore([chunk()])).invoke(initial_state())
    assert result["status"] == "answered"
    assert result["is_grounded"] is True
    assert result["citations"] == ["c1"]
    assert result["router_calls"] == 4


def test_abstain_out_of_scope():
    router = FakeRouter(['{"action":"abstain","query":""}'])
    result = build_agent(router, FakeEmbedder(), FakeStore([])).invoke(initial_state("Pizza recipe?"))
    assert result["status"] == "abstained"
    assert result["stop_reason"] == "out_of_scope"
    assert router.call_count == 1


def test_invoke_with_question_only_initializes_runtime_state():
    router = FakeRouter(['{"action":"abstain","query":""}'])
    agent = build_agent(router, FakeEmbedder(), FakeStore([]))

    result = agent.invoke({"question": "Pizza recipe?"})

    assert result["status"] == "abstained"
    assert result["stop_reason"] == "out_of_scope"
    assert result["router_calls"] == 1
    assert "Plan: abstain" in result["steps"]


def test_retry_after_bad_grading():
    router = FakeRouter(['{"action":"retrieve","query":"attention"}', "no", "better transformer query",
                         "yes", "Answer [c1].", "CLAIM: A\nVERDICT: supported\nCHUNKS: c1"])
    result = build_agent(router, FakeEmbedder(), FakeStore([chunk("c1", "Text about transformers.")]),
                         max_rounds=1).invoke(initial_state())
    assert result["round"] == 1
    assert result["router_calls"] == 6
    assert any("Rewrote query" in step for step in result["steps"])
