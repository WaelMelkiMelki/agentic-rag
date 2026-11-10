"""Multi-turn memory: session histories and standalone-question rewriting."""
from __future__ import annotations

import pytest

from agentic_rag.agent.memory import (
    CONDENSE_CHAT_PROMPT,
    SessionMemory,
    standalone_question,
)


class FakeRouter:
    def __init__(self, reply: str = "What is multi-head attention?", error=None):
        self.reply = reply
        self.error = error
        self.prompts = []

    def invoke_with_fallback(self, prompt):
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return self.reply


def test_remember_and_transcript():
    memory = SessionMemory()
    memory.remember("s1", "What is RAG?", "RAG combines retrieval with generation.")
    memory.remember("s1", "And who proposed it?", "Lewis et al.")

    transcript = memory.transcript("s1")
    assert "User: What is RAG?" in transcript
    assert "Assistant: RAG combines retrieval with generation." in transcript
    assert transcript.index("What is RAG?") < transcript.index("And who proposed it?")


def test_sessions_are_isolated():
    memory = SessionMemory()
    memory.remember("s1", "question one", "answer one")

    assert memory.transcript("s2") == ""
    assert "question one" not in memory.transcript("s2")


def test_history_is_trimmed_to_max_turns():
    memory = SessionMemory(max_turns=2)
    for index in range(5):
        memory.remember("s1", f"q{index}", f"a{index}")

    transcript = memory.transcript("s1", max_turns=10)
    assert "q0" not in transcript
    assert "q4" in transcript
    assert "a1" not in transcript


def test_transcript_limits_turns_rendered():
    memory = SessionMemory()
    for index in range(4):
        memory.remember("s1", f"q{index}", f"a{index}")

    transcript = memory.transcript("s1", max_turns=1)
    assert "q3" in transcript and "q2" not in transcript


def test_oldest_session_is_evicted_when_full():
    memory = SessionMemory(max_sessions=2)
    memory.remember("s1", "q1", "a1")
    memory.remember("s2", "q2", "a2")
    memory.remember("s3", "q3", "a3")

    assert memory.session_count() == 2
    assert memory.transcript("s1") == ""
    assert memory.transcript("s3") != ""


def test_clear_removes_one_or_all_sessions():
    memory = SessionMemory()
    memory.remember("s1", "q", "a")
    memory.remember("s2", "q", "a")

    memory.clear("s1")
    assert memory.transcript("s1") == ""
    assert memory.transcript("s2") != ""

    memory.clear()
    assert memory.session_count() == 0


def test_invalid_limits_are_rejected():
    with pytest.raises(ValueError):
        SessionMemory(max_sessions=0)
    with pytest.raises(ValueError):
        SessionMemory(max_turns=0)


def test_standalone_question_uses_the_conversation():
    router = FakeRouter("What is multi-head attention used for?")
    result = standalone_question("and what is it used for?", "User: What is attention?", router)

    assert result == "What is multi-head attention used for?"
    rendered = router.prompts[0].to_messages()[0].content
    assert "User: What is attention?" in rendered
    assert "and what is it used for?" in rendered


def test_standalone_question_skips_the_llm_without_history():
    router = FakeRouter()
    assert standalone_question("plain question", "", router) == "plain question"
    assert router.prompts == []


def test_standalone_question_falls_back_on_router_failure():
    router = FakeRouter(error=RuntimeError("All providers failed."))
    assert standalone_question("follow-up", "User: earlier", router) == "follow-up"


@pytest.mark.parametrize("reply", ["", "no", "x" * 500])
def test_standalone_question_rejects_unusable_rewrites(reply):
    router = FakeRouter(reply)
    assert standalone_question("follow-up", "User: earlier", router) == "follow-up"


def test_condense_prompt_forbids_answering():
    rendered = CONDENSE_CHAT_PROMPT.format_messages(history="User: hi", question="and then?")[0].content
    assert "Do not answer the question" in rendered
    assert "Respond with ONLY the rewritten question" in rendered
