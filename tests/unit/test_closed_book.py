"""B0 closed-book baseline: one LLM call, no retrieval, no citations."""
from __future__ import annotations

from agentic_rag.agent.closed_book import CLOSED_BOOK_CHAT_PROMPT, ClosedBook
from agentic_rag.agent.prompts import ABSTENTION


class FakeRouter:
    def __init__(self, answer: str = "Transformers use self-attention.", error=None):
        self.answer = answer
        self.error = error
        self.calls = 0
        self.received = []

    def invoke_with_fallback(self, prompt):
        self.calls += 1
        self.received.append(prompt)
        if self.error is not None:
            raise self.error
        return self.answer


def test_answers_from_memory_without_citations():
    router = FakeRouter()
    result = ClosedBook(router=router).query("What is self-attention?")

    assert result["answer"] == "Transformers use self-attention."
    assert result["citations"] == []
    assert result["status"] == "answered"
    assert result["stop_reason"] == "completed"
    assert result["steps"] == ["generate (closed-book)"]
    assert router.calls == 1


def test_prompt_asks_for_no_context_and_no_citations():
    rendered = CLOSED_BOOK_CHAT_PROMPT.format_messages(question="q?")[0].content
    assert "without any documents" in rendered
    assert ABSTENTION in rendered
    assert "Do not invent citations" in rendered


def test_model_abstention_is_reported_as_abstained():
    result = ClosedBook(router=FakeRouter(ABSTENTION)).query("What is the capital of France?")
    assert result["status"] == "abstained"
    assert result["stop_reason"] == "no_answer"
    assert result["answer"] == ABSTENTION


def test_abstention_match_is_case_insensitive():
    result = ClosedBook(router=FakeRouter(ABSTENTION.lower())).query("q")
    assert result["status"] == "abstained"


def test_empty_answer_becomes_an_abstention():
    result = ClosedBook(router=FakeRouter("   ")).query("q")
    assert result["status"] == "abstained"
    assert result["answer"] == ABSTENTION


def test_provider_failure_degrades_gracefully():
    result = ClosedBook(router=FakeRouter(error=RuntimeError("All providers failed."))).query("q")
    assert result["status"] == "abstained"
    assert result["stop_reason"] == "llm_unavailable"
    assert result["answer"] == ABSTENTION
