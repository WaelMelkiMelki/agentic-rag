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
            "yes",
            "It attends to different subspaces [paper__0].",
            "CLAIM: It attends to different subspaces\nVERDICT: supported\nCHUNKS: paper__0",
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
    assert result["status"] == "answered"
    assert result["citations"] == ["paper__0"]
    assert result["router_calls"] == 4
    assert len(router.prompts) == 4


# --- citation repair -------------------------------------------------------
# Local models (phi3) copy the example id from the generation prompt, or cite
# nothing at all. Both used to trip the old all-or-nothing citation gate and
# turn a good answer into an abstention.


def make_chunk_at(index: int):
    chunk = make_chunk()
    chunk.payload = dict(chunk.payload, chunk_id=f"paper__{index}", chunk_index=index)
    return chunk


GROUNDED = "CLAIM: It attends to different subspaces\nVERDICT: supported\nCHUNKS: paper__0"
UNGROUNDED = "CLAIM: It attends to different subspaces\nVERDICT: unsupported\nCHUNKS: none"


def run_agent(replies, chunks, **kwargs):
    router = FakeRouter(replies)
    app = build_agent(router, FakeEmbedder(), FakeStore(chunks), **kwargs)
    return app.invoke({"question": "Why use multi-head attention?", "router_calls": 0, "steps": []})


def test_citation_outside_the_retrieved_chunks_is_dropped_not_fatal():
    replies = [
        '{"action": "retrieve", "query": "multi-head attention"}',
        "yes",
        "yes",
        "It attends to different subspaces [paper__0] and more [attention_is_all_you_need__12].",
        GROUNDED,
    ]
    result = run_agent(replies, [make_chunk_at(0), make_chunk_at(1)])

    assert result["status"] == "answered"
    assert result["stop_reason"] == "grounded"
    assert result["citations"] == ["paper__0"]
    assert "attention_is_all_you_need__12" not in result["final_answer"]
    assert "It attends to different subspaces [paper__0]" in result["final_answer"]
    assert any("Dropped 1 citation" in step for step in result["steps"])


def test_answer_keeps_every_valid_citation_in_order():
    replies = [
        '{"action": "retrieve", "query": "multi-head attention"}',
        "yes",
        "yes",
        "Uses [paper__1] and [paper__0] together.",
        GROUNDED,
    ]
    result = run_agent(replies, [make_chunk_at(0), make_chunk_at(1)])

    assert result["citations"] == ["paper__1", "paper__0"]
    assert result["final_answer"] == "Uses [paper__1] and [paper__0] together."


def test_markdown_links_are_not_treated_as_citations():
    replies = [
        '{"action": "retrieve", "query": "multi-head attention"}',
        "yes",
        "Uses subspaces [paper__0], see [the paper](https://arxiv.org/abs/1706.03762).",
        GROUNDED,
    ]
    result = run_agent(replies, [make_chunk_at(0)])

    assert result["citations"] == ["paper__0"]
    assert "[the paper](https://arxiv.org/abs/1706.03762)" in result["final_answer"]


def test_uncited_but_grounded_answer_borrows_the_fact_checker_citations():
    replies = [
        '{"action": "retrieve", "query": "multi-head attention"}',
        "yes",
        "It attends to different subspaces.",
        GROUNDED,
    ]
    result = run_agent(replies, [make_chunk_at(0)])

    assert result["status"] == "answered"
    assert result["citations"] == ["paper__0"]
    assert result["final_answer"].endswith("[paper__0]")
    assert any("grounding report" in step for step in result["steps"])


def test_uncited_and_ungrounded_answer_still_abstains():
    replies = [
        '{"action": "retrieve", "query": "multi-head attention"}',
        "yes",
        "It attends to different subspaces.",
        UNGROUNDED,
    ]
    result = run_agent(replies, [make_chunk_at(0)], max_rounds=0)

    assert result["status"] == "abstained"
    assert result["stop_reason"] == "invalid_citations"
    assert result["citations"] == []
    # The draft is preserved so the caller can see what was rejected.
    assert result["draft_answer"] == "It attends to different subspaces."


def test_uncited_ungrounded_answer_is_rescued_by_a_citation_request():
    """The probe case: a correct draft, no markers, and a fact-checker that timed out."""
    replies = [
        '{"action": "retrieve", "query": "multi-head attention"}',
        "yes",
        "The Transformer uses multi-head attention to build residual streams.",
        UNGROUNDED,
        "paper__0",
    ]
    result = run_agent(replies, [make_chunk_at(0)], max_rounds=0)

    assert result["status"] == "answered"
    assert result["stop_reason"] == "grounded"
    assert result["citations"] == ["paper__0"]
    assert result["final_answer"].endswith("[paper__0]")
    assert any("citation request" in step for step in result["steps"])


def test_citation_request_that_returns_nothing_still_abstains():
    replies = [
        '{"action": "retrieve", "query": "multi-head attention"}',
        "yes",
        "An answer with no evidence trail.",
        UNGROUNDED,
        "none",
    ]
    result = run_agent(replies, [make_chunk_at(0)], max_rounds=0)

    assert result["status"] == "abstained"
    assert result["stop_reason"] == "invalid_citations"
    assert result["citations"] == []
