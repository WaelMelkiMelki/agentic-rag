"""LCEL plumbing: prompt -> fallback router -> StrOutputParser."""
from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate

from agentic_rag.llm.chains import text_chain


class FakeRouter:
    """Stand-in for LLMRouter: records what the chain handed it."""

    def __init__(self, answer: str = "answer text"):
        self.answer = answer
        self.received = []

    def invoke_with_fallback(self, prompt):
        self.received.append(prompt)
        return self.answer


PROMPT = ChatPromptTemplate.from_template("Context: {context}\nQuestion: {question}")


def test_chain_returns_a_plain_string():
    chain = text_chain(PROMPT, FakeRouter("hello"))
    result = chain.invoke({"context": "ctx", "question": "q?"})
    assert result == "hello"
    assert isinstance(result, str)


def test_chain_sends_rendered_messages_to_the_router():
    """LCEL hands the router a ChatPromptValue, which every provider accepts."""
    router = FakeRouter()
    text_chain(PROMPT, router).invoke({"context": "the context", "question": "why?"})

    sent = router.received[0]
    assert isinstance(sent, ChatPromptValue)
    messages = sent.to_messages()
    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
    assert "Context: the context" in messages[0].content
    assert "Question: why?" in messages[0].content


def test_chain_forwards_router_failures():
    class Broken:
        def invoke_with_fallback(self, prompt):
            raise RuntimeError("All providers failed. Check your API keys.")

    chain = text_chain(PROMPT, Broken())
    try:
        chain.invoke({"context": "c", "question": "q"})
    except RuntimeError as error:
        assert "All providers failed" in str(error)
    else:  # pragma: no cover
        raise AssertionError("the router error should propagate to the caller")


def test_chain_is_reusable_across_calls():
    router = FakeRouter()
    chain = text_chain(PROMPT, router)
    for question in ("one", "two", "three"):
        chain.invoke({"context": "c", "question": question})
    assert len(router.received) == 3


def test_router_accepts_a_chat_prompt_value():
    """The router must forward LCEL's ChatPromptValue unchanged."""
    from agentic_rag.llm.router import LLMRouter

    seen = {}

    class FakeLLM:
        model_name = "fake"

        def invoke(self, prompt):
            seen["prompt"] = prompt

            class Response:
                content = "ok"

            return Response()

    router = LLMRouter(ollama_phi=FakeLLM(), ollama_qwen=FakeLLM(),
                       gemini=FakeLLM(), openrouter=FakeLLM())
    value = PROMPT.format_prompt(context="c", question="q")

    assert router.invoke_with_fallback(value) == "ok"
    assert seen["prompt"] is value
