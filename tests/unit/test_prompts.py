"""The prompts are LangChain ``ChatPromptTemplate`` objects.

These tests protect the one property that matters for the B1-vs-B4 comparison:
the rendered text is byte-identical to the old ``PROMPT.format(...)`` output, so
swapping string prompts for chat messages cannot change what the model sees.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from agentic_rag.agent import grade, ground, plan, prompts, rewrite
from agentic_rag.agent.prompts import format_context
from agentic_rag.llm.router import LLMRouter


def _text(messages) -> str:
    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
    return messages[0].content


def test_generate_template_matches_original_string():
    context = "[c1] (doc: d, section: 2, relevance: 0.900)\nsome chunk text"
    question = "Why multi-head attention?"
    rendered = _text(prompts.GENERATE_CHAT_PROMPT.format_messages(context=context, question=question))
    assert rendered == prompts.GENERATE_PROMPT.format(context=context, question=question)


def test_generate_template_keeps_abstention_sentence():
    rendered = _text(prompts.GENERATE_CHAT_PROMPT.format_messages(context="x", question="y"))
    assert "I cannot answer this from the provided sources." in rendered
    assert "[chunk_id]" in rendered


def test_plan_template_matches_original_string():
    question = "What is RAG?"
    rendered = _text(plan.PLAN_CHAT_PROMPT.format_messages(question=question))
    assert rendered == plan.PLAN_PROMPT.format(question=question)
    # Doubled braces in the template must still render as valid JSON braces.
    assert '{"action": "retrieve", "query": "a concise search query"}' in rendered


def test_grade_template_matches_original_string():
    values = {"question": "What is RAG?", "chunk_id": "rag_original__3", "text": "RAG combines ..."}
    rendered = _text(grade.GRADE_CHAT_PROMPT.format_messages(**values))
    assert rendered == grade.GRADE_PROMPT.format(**values)
    assert "Respond with ONLY \"yes\" or \"no\":" in rendered


def test_ground_template_matches_original_string():
    values = {"answer": "RAG uses a retriever [c1].", "chunks_text": "[c1]: RAG combines ..."}
    rendered = _text(ground.GROUND_CHAT_PROMPT.format_messages(**values))
    assert rendered == ground.GROUND_PROMPT.format(**values)
    assert "VERDICT: supported | partial | unsupported" in rendered


def test_rewrite_template_matches_original_string():
    values = {"question": "How does self-attention work?", "previous_query": "self attention",
              "feedback": "No relevant results", "round": 1}
    rendered = _text(rewrite.REWRITE_CHAT_PROMPT.format_messages(**values))
    assert rendered == rewrite.REWRITE_PROMPT.format(**values)
    assert "Respond with ONLY the new query" in rendered


def test_brace_characters_in_context_are_not_treated_as_placeholders():
    """Retrieved PDF text can contain braces; they must survive verbatim."""
    context = "def f(x): return {x: 1}  {not_a_placeholder}"
    rendered = _text(prompts.GENERATE_CHAT_PROMPT.format_messages(context=context, question="q"))
    assert "{not_a_placeholder}" in rendered


def test_format_context_is_unchanged():
    class Chunk:
        payload = {"chunk_id": "c1", "doc_id": "d1", "chunk_index": 2, "text": "hello"}
        score = 0.9

    assert format_context([Chunk()]) == "[c1] (doc: d1, section: 2, relevance: 0.900)\nhello"


def test_router_accepts_chat_messages_and_returns_a_string():
    seen = {}

    class FakeLLM:
        model_name = "fake"

        def invoke(self, prompt):
            seen["prompt"] = prompt

            class Response:
                content = "answer text"

            return Response()

    router = LLMRouter(
        ollama_phi=FakeLLM(),
        ollama_qwen=FakeLLM(),
        gemini=FakeLLM(),
        openrouter=FakeLLM(),
    )
    messages = prompts.GENERATE_CHAT_PROMPT.format_messages(context="ctx", question="q?")
    result = router.invoke_with_fallback(messages)

    assert result == "answer text"
    assert seen["prompt"] == messages
    assert isinstance(result, str)


def test_router_still_accepts_plain_strings():
    class FakeLLM:
        model_name = "fake"

        def invoke(self, prompt):
            class Response:
                content = "ok"

            assert isinstance(prompt, str)
            return Response()

    router = LLMRouter(ollama_phi=FakeLLM(), ollama_qwen=FakeLLM(),
                       gemini=FakeLLM(), openrouter=FakeLLM())
    assert router.invoke_with_fallback("plain prompt") == "ok"
