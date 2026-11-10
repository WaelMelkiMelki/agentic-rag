"""Tracing is optional: it must never break a query and must support the
Langfuse v2 / v3 / v4+ SDK surfaces."""
from __future__ import annotations

import pytest

from agentic_rag import tracing


class _Span:
    def __init__(self) -> None:
        self.entered = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeV5Client:
    """Mimics langfuse>=4 ``start_as_current_observation``."""

    def __init__(self) -> None:
        self.kwargs: dict = {}
        self.span = _Span()

    def start_as_current_observation(self, **kwargs):
        self.kwargs = kwargs
        return self.span

    def get_current_trace_id(self) -> str:
        return "trace-v5"

    def flush(self) -> None:
        self.flushed = True


class _FakeV3Client:
    """Mimics langfuse v3 ``start_as_current_span``."""

    def __init__(self) -> None:
        self.kwargs: dict = {}

    def start_as_current_span(self, **kwargs):
        self.kwargs = kwargs
        return _Span()

    def get_current_trace_id(self) -> str:
        return "trace-v3"


class _FakeV2Trace:
    id = "trace-v2"

    def __init__(self) -> None:
        self.updated: dict = {}

    def update(self, **kwargs):
        self.updated = kwargs


class _FakeV2Client:
    """Mimics langfuse v2 ``client.trace(...)``."""

    def __init__(self) -> None:
        self.kwargs: dict = {}
        self.trace_obj = _FakeV2Trace()

    def trace(self, **kwargs):
        self.kwargs = kwargs
        return self.trace_obj


class _BrokenClient:
    def start_as_current_observation(self, **kwargs):
        raise RuntimeError("langfuse exploded")


B4_RESULT = {
    "final_answer": "Attention maps queries to keys [c1].",
    "citations": ["c1"],
    "steps": ["plan", "retrieve", "generate", "finalize"],
    "status": "answered",
    "stop_reason": "completed",
    "system": "B4",
}


@pytest.fixture(autouse=True)
def _no_cached_client(monkeypatch):
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        monkeypatch.delenv(key, raising=False)
    tracing.reset_client()
    yield
    tracing.reset_client()


def test_tracing_is_disabled_without_keys():
    assert tracing.get_client() is None
    assert tracing.trace_agent("What is RAG?", B4_RESULT) == ""
    tracing.flush()  # must not raise


def test_does_not_create_client_with_only_one_key(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    tracing.reset_client()
    assert tracing.get_client() is None


def test_trace_supports_v5_api():
    client = _FakeV5Client()
    trace_id = tracing.trace_agent("What is RAG?", B4_RESULT, client=client)

    assert trace_id == "trace-v5"
    assert client.kwargs["name"] == tracing.TRACE_NAME
    assert client.kwargs["input"] == {"question": "What is RAG?"}
    assert client.kwargs["output"]["answer"] == B4_RESULT["final_answer"]
    assert client.kwargs["output"]["stop_reason"] == "completed"
    assert client.kwargs["metadata"]["system"] == "B4"
    assert client.kwargs["metadata"]["citations"] == ["c1"]
    assert client.kwargs["level"] == "DEFAULT"
    assert client.span.entered is True


def test_trace_supports_v3_api():
    client = _FakeV3Client()
    assert tracing.trace_agent("q", B4_RESULT, client=client) == "trace-v3"
    assert client.kwargs["name"] == tracing.TRACE_NAME


def test_trace_supports_v2_api():
    client = _FakeV2Client()
    assert tracing.trace_agent("q", B4_RESULT, client=client) == "trace-v2"
    assert client.kwargs["metadata"]["steps"] == B4_RESULT["steps"]
    assert client.trace_obj.updated["status_message"] == "completed"


def test_b1_result_shape_is_normalized():
    client = _FakeV5Client()
    tracing.trace_agent("q", {"answer": "B1 answer", "status": "answered"}, client=client)
    assert client.kwargs["output"]["answer"] == "B1 answer"


def test_sdk_failure_never_raises():
    assert tracing.trace_agent("q", B4_RESULT, client=_BrokenClient()) == ""


def test_flush_swallows_client_errors():
    class Broken:
        def flush(self):
            raise RuntimeError("nope")

    tracing.flush(client=Broken())  # must not raise


def test_flush_calls_client():
    client = _FakeV5Client()
    tracing.flush(client=client)
    assert client.flushed is True


# ------------------------------------------------- LangChain callback wiring ---
def test_callbacks_are_empty_without_keys():
    assert tracing.callbacks() == []


def test_callbacks_are_cached_once(monkeypatch):
    class FakeClient:
        pass

    sentinel = object()
    monkeypatch.setattr(tracing, "get_client", lambda: FakeClient())
    monkeypatch.setattr(tracing, "_callback_handler", sentinel, raising=False)

    assert tracing.callbacks() == [sentinel]
    assert tracing.callbacks() == [sentinel]


def test_callbacks_returns_empty_when_the_handler_cannot_be_built(monkeypatch):
    class FakeClient:
        pass

    monkeypatch.setattr(tracing, "get_client", lambda: FakeClient())
    monkeypatch.setattr(tracing, "_callback_handler", tracing._UNINITIALIZED, raising=False)

    # Whether or not the installed SDK can build a handler without a live client,
    # this must never raise and must always return a list.
    result = tracing.callbacks()
    assert isinstance(result, list)


def test_chains_bind_langfuse_callbacks_when_available(monkeypatch):
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.prompts import ChatPromptTemplate

    from agentic_rag.llm.chains import text_chain

    class RecordingHandler(BaseCallbackHandler):
        def __init__(self):
            self.starts = 0

        def on_chain_start(self, *args, **kwargs):
            self.starts += 1

    handler = RecordingHandler()
    monkeypatch.setattr(tracing, "callbacks", lambda: [handler])

    class FakeRouter:
        def invoke_with_fallback(self, prompt):
            return "answer"

    chain = text_chain(ChatPromptTemplate.from_template("Q: {question}"), FakeRouter())
    assert chain.invoke({"question": "why?"}) == "answer"
    assert handler.starts >= 1
