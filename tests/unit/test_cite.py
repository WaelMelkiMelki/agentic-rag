"""The citation request that rescues an answer with no ``[chunk_id]`` markers."""
from __future__ import annotations

from types import SimpleNamespace

from agentic_rag.agent.cite import request_citations


class FakeRouter:
    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list = []

    def invoke_with_fallback(self, prompt):
        self.prompts.append(prompt)
        return self.reply


class BoomRouter:
    def invoke_with_fallback(self, prompt):
        raise TimeoutError("Request timed out.")


def chunk(chunk_id: str, text: str = "Multi-head attention attends to subspaces."):
    return SimpleNamespace(payload={"chunk_id": chunk_id, "text": text})


CHUNKS = [chunk("paper__0"), chunk("paper__1")]


def test_invented_ids_are_dropped():
    cited, calls = request_citations("An answer.", CHUNKS, FakeRouter("paper__0, invented__9"))

    assert cited == ["paper__0"]
    assert calls == 1


def test_bracketed_replies_are_understood_in_order():
    cited, _ = request_citations("An answer.", CHUNKS, FakeRouter("[paper__1] and [paper__0]"))

    assert cited == ["paper__1", "paper__0"]


def test_none_keeps_the_refusal():
    cited, calls = request_citations("An answer.", CHUNKS, FakeRouter("none"))

    assert cited == []
    assert calls == 1


def test_reply_without_any_known_id_is_empty():
    cited, _ = request_citations("An answer.", CHUNKS, FakeRouter("I think the first one."))

    assert cited == []


def test_citations_are_capped():
    chunks = [chunk(f"paper__{index}") for index in range(5)]
    reply = ", ".join(f"paper__{index}" for index in range(5))

    cited, _ = request_citations("An answer.", chunks, FakeRouter(reply))

    assert cited == ["paper__0", "paper__1", "paper__2"]


def test_provider_failure_reports_one_call_and_no_citations():
    cited, calls = request_citations("An answer.", CHUNKS, BoomRouter())

    assert cited == []
    assert calls == 1


def test_no_chunks_means_no_call_at_all():
    router = FakeRouter("paper__0")

    cited, calls = request_citations("An answer.", [], router)

    assert (cited, calls) == ([], 0)
    assert router.prompts == []


def test_prompt_shows_every_chunk_and_demands_ids_only():
    router = FakeRouter("paper__0")

    request_citations("An answer.", CHUNKS, router)

    # The router receives a ChatPromptValue (LCEL chain), so render it first.
    prompt = router.prompts[0].to_string()
    assert "[paper__0]" in prompt and "[paper__1]" in prompt
    assert "An answer." in prompt
    assert "ONLY the chunk ids" in prompt
