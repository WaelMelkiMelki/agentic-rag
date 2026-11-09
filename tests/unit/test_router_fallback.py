"""Unit tests for Ollama-first LLMRouter fallback order (no network)."""

from types import SimpleNamespace

import pytest

from agentic_rag.llm.router import LLMRouter, _normalize_ollama_base_url


def _response(content: str):
    return SimpleNamespace(content=content, usage_metadata=None, response_metadata={})


class FakeLLM:
    def __init__(self, content: str = "ok", error: Exception | None = None):
        self.content = content
        self.error = error
        self.calls = 0
        self.model = "fake-model"
        self.model_name = "fake-model"

    def invoke(self, prompt: str):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return _response(self.content)


def _router(ollama=None, gemini=None, openrouter=None, **kwargs):
    return LLMRouter(
        ollama=ollama if ollama is not None else FakeLLM("ollama"),
        gemini=gemini if gemini is not None else FakeLLM("gemini"),
        openrouter=openrouter if openrouter is not None else FakeLLM("openrouter"),
        **kwargs,
    )


def test_ollama_is_tried_first():
    ollama, gemini, openrouter = FakeLLM("ollama"), FakeLLM("gemini"), FakeLLM("openrouter")
    router = _router(ollama=ollama, gemini=gemini, openrouter=openrouter)
    assert router.invoke_with_fallback("hello") == "ollama"
    assert ollama.calls == 1
    assert gemini.calls == 0
    assert openrouter.calls == 0


def test_falls_back_to_gemini_when_ollama_fails():
    ollama = FakeLLM(error=ConnectionError("ollama down"))
    gemini = FakeLLM("gemini")
    router = _router(ollama=ollama, gemini=gemini)
    assert router.invoke_with_fallback("hello") == "gemini"
    assert ollama.calls == 1
    assert gemini.calls == 1


def test_falls_back_to_openrouter_when_ollama_and_gemini_fail():
    ollama = FakeLLM(error=ConnectionError("ollama down"))
    gemini = FakeLLM(error=RuntimeError("gemini down"))
    openrouter = FakeLLM("openrouter")
    router = _router(ollama=ollama, gemini=gemini, openrouter=openrouter)
    assert router.invoke_with_fallback("hello") == "openrouter"
    assert (ollama.calls, gemini.calls, openrouter.calls) == (1, 1, 1)


def test_ollama_failure_is_cached_for_rest_of_process():
    ollama = FakeLLM(error=ConnectionError("ollama down"))
    gemini = FakeLLM("gemini")
    router = _router(ollama=ollama, gemini=gemini)
    assert router.invoke_with_fallback("one") == "gemini"
    assert router.invoke_with_fallback("two") == "gemini"
    # Second call must not retry Ollama (avoids repeated timeouts in the agent loop).
    assert ollama.calls == 1
    assert gemini.calls == 2


def test_gemini_quota_error_skips_gemini_afterwards():
    ollama = FakeLLM(error=ConnectionError("ollama down"))
    quota = RuntimeError("quota exceeded")
    quota.status_code = 429
    gemini = FakeLLM(error=quota)
    openrouter = FakeLLM("openrouter")
    router = _router(ollama=ollama, gemini=gemini, openrouter=openrouter)
    assert router.invoke_with_fallback("one") == "openrouter"
    assert router.invoke_with_fallback("two") == "openrouter"
    assert gemini.calls == 1
    assert openrouter.calls == 2


def test_all_providers_failing_raises():
    router = _router(
        ollama=FakeLLM(error=ConnectionError("down")),
        gemini=FakeLLM(error=RuntimeError("down")),
        openrouter=FakeLLM(error=RuntimeError("down")),
    )
    with pytest.raises(RuntimeError, match="All providers failed"):
        router.invoke_with_fallback("hello")


def test_get_llm_supports_ollama_and_defaults_to_ollama():
    ollama, gemini, openrouter = FakeLLM(), FakeLLM(), FakeLLM()
    router = _router(ollama=ollama, gemini=gemini, openrouter=openrouter)
    assert router.get_llm("ollama") is ollama
    assert router.get_llm("gemini") is gemini
    assert router.get_llm("openrouter") is openrouter
    assert router.get_llm() is ollama
    with pytest.raises(ValueError, match="Unknown provider"):
        router.get_llm("bogus")


def test_normalize_ollama_base_url():
    assert _normalize_ollama_base_url("http://localhost:11434") == "http://localhost:11434/v1"
    assert _normalize_ollama_base_url("http://localhost:11434/v1") == "http://localhost:11434/v1"
    assert _normalize_ollama_base_url("http://localhost:11434/v1/") == "http://localhost:11434/v1"
    assert _normalize_ollama_base_url("") == "http://localhost:11434/v1"
