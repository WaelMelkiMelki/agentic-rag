"""Unit tests for Ollama-first LLMRouter fallback order (no network).

Fallback order: Ollama phi3 -> Ollama qwen2.5 -> Gemini -> OpenRouter.
"""

from types import SimpleNamespace

import pytest

from agentic_rag.llm.router import LLMRouter


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


def _router(ollama_phi=None, ollama_qwen=None, gemini=None, openrouter=None, **kwargs):
    return LLMRouter(
        ollama_phi=ollama_phi if ollama_phi is not None else FakeLLM("phi"),
        ollama_qwen=ollama_qwen if ollama_qwen is not None else FakeLLM("qwen"),
        gemini=gemini if gemini is not None else FakeLLM("gemini"),
        openrouter=openrouter if openrouter is not None else FakeLLM("openrouter"),
        **kwargs,
    )


def test_ollama_phi_is_tried_first():
    phi, qwen, gemini, openrouter = (
        FakeLLM("phi"),
        FakeLLM("qwen"),
        FakeLLM("gemini"),
        FakeLLM("openrouter"),
    )
    router = _router(ollama_phi=phi, ollama_qwen=qwen, gemini=gemini, openrouter=openrouter)
    assert router.invoke_with_fallback("hello") == "phi"
    assert phi.calls == 1
    assert qwen.calls == 0
    assert gemini.calls == 0
    assert openrouter.calls == 0


def test_falls_back_to_qwen_when_phi_fails():
    phi = FakeLLM(error=ConnectionError("ollama down"))
    qwen = FakeLLM("qwen")
    gemini = FakeLLM("gemini")
    router = _router(ollama_phi=phi, ollama_qwen=qwen, gemini=gemini)
    assert router.invoke_with_fallback("hello") == "qwen"
    assert phi.calls == 1
    assert qwen.calls == 1
    assert gemini.calls == 0


def test_falls_back_to_gemini_when_both_ollama_models_fail():
    phi = FakeLLM(error=ConnectionError("ollama down"))
    qwen = FakeLLM(error=ConnectionError("ollama down"))
    gemini = FakeLLM("gemini")
    router = _router(ollama_phi=phi, ollama_qwen=qwen, gemini=gemini)
    assert router.invoke_with_fallback("hello") == "gemini"
    assert (phi.calls, qwen.calls, gemini.calls) == (1, 1, 1)


def test_ollama_failure_is_cached_for_rest_of_process():
    phi = FakeLLM(error=ConnectionError("ollama down"))
    qwen = FakeLLM(error=ConnectionError("ollama down"))
    gemini = FakeLLM("gemini")
    router = _router(ollama_phi=phi, ollama_qwen=qwen, gemini=gemini)
    assert router.invoke_with_fallback("one") == "gemini"
    assert router.invoke_with_fallback("two") == "gemini"
    # Second call must not retry either Ollama model (avoids repeated
    # timeouts across the agent's router calls).
    assert phi.calls == 1
    assert qwen.calls == 1
    assert gemini.calls == 2


def test_gemini_quota_error_skips_gemini_afterwards():
    phi = FakeLLM(error=ConnectionError("ollama down"))
    qwen = FakeLLM(error=ConnectionError("ollama down"))
    quota = RuntimeError("quota exceeded")
    quota.status_code = 429
    gemini = FakeLLM(error=quota)
    openrouter = FakeLLM("openrouter")
    router = _router(ollama_phi=phi, ollama_qwen=qwen, gemini=gemini, openrouter=openrouter)
    assert router.invoke_with_fallback("one") == "openrouter"
    assert router.invoke_with_fallback("two") == "openrouter"
    assert gemini.calls == 1
    assert openrouter.calls == 2


def test_full_chain_ollama_fail_gemini_fail_openrouter_succeeds():
    phi = FakeLLM(error=ConnectionError("ollama down"))
    qwen = FakeLLM(error=ConnectionError("ollama down"))
    gemini = FakeLLM(error=RuntimeError("gemini down"))
    openrouter = FakeLLM("openrouter")
    router = _router(ollama_phi=phi, ollama_qwen=qwen, gemini=gemini, openrouter=openrouter)
    assert router.invoke_with_fallback("hello") == "openrouter"
    assert (phi.calls, qwen.calls, gemini.calls, openrouter.calls) == (1, 1, 1, 1)


def test_all_providers_failing_raises():
    router = _router(
        ollama_phi=FakeLLM(error=ConnectionError("down")),
        ollama_qwen=FakeLLM(error=ConnectionError("down")),
        gemini=FakeLLM(error=RuntimeError("down")),
        openrouter=FakeLLM(error=RuntimeError("down")),
    )
    with pytest.raises(RuntimeError, match="All providers failed"):
        router.invoke_with_fallback("hello")


def test_ollama_disabled_skips_straight_to_gemini(monkeypatch):
    monkeypatch.setenv("OLLAMA_ENABLED", "false")
    phi, qwen, gemini = FakeLLM("phi"), FakeLLM("qwen"), FakeLLM("gemini")
    router = _router(ollama_phi=phi, ollama_qwen=qwen, gemini=gemini)
    assert router.invoke_with_fallback("hello") == "gemini"
    assert phi.calls == 0
    assert qwen.calls == 0
    assert gemini.calls == 1


def test_get_llm_supports_all_providers():
    phi, qwen, gemini, openrouter = FakeLLM(), FakeLLM(), FakeLLM(), FakeLLM()
    router = _router(ollama_phi=phi, ollama_qwen=qwen, gemini=gemini, openrouter=openrouter)
    assert router.get_llm("ollama_phi") is phi
    assert router.get_llm("ollama_qwen") is qwen
    assert router.get_llm("gemini") is gemini
    assert router.get_llm("openrouter") is openrouter
    assert router.get_llm() is phi
    with pytest.raises(ValueError, match="Unknown provider"):
        router.get_llm("bogus")


class APITimeoutError(Exception):
    """Mimics openai.APITimeoutError (detected by class name)."""


def test_timeout_does_not_disable_ollama_for_the_process(monkeypatch):
    """A slow/cold phi3 must not push every later query to the cloud."""
    phi = FakeLLM(error=APITimeoutError("Request timed out."))
    qwen = FakeLLM(error=TimeoutError("socket timed out"))
    gemini = FakeLLM("gemini")
    router = _router(ollama_phi=phi, ollama_qwen=qwen, gemini=gemini)

    assert router.invoke_with_fallback("first") == "gemini"
    assert phi.calls == 1 and qwen.calls == 1
    assert router._ollama_unavailable is False

    # Next query tries the local models again, in order.
    assert router.invoke_with_fallback("second") == "gemini"
    assert phi.calls == 2 and qwen.calls == 2


def test_connection_error_still_disables_ollama(monkeypatch):
    """Ollama not running (connection refused) must be cached, not retried."""
    phi = FakeLLM(error=ConnectionError("Connection refused"))
    qwen = FakeLLM("qwen")
    gemini = FakeLLM("gemini")
    router = _router(ollama_phi=phi, ollama_qwen=qwen, gemini=gemini)

    assert router.invoke_with_fallback("first") == "qwen"
    assert router._ollama_unavailable is True

    assert router.invoke_with_fallback("second") == "gemini"
    assert phi.calls == 1          # not retried
    assert qwen.calls == 1


def test_ollama_timeout_is_configurable(monkeypatch):
    monkeypatch.setenv("OLLAMA_TIMEOUT", "240")
    router = LLMRouter()
    assert router._ollama_timeout == 240.0


def test_ollama_timeout_defaults_to_120(monkeypatch):
    monkeypatch.delenv("OLLAMA_TIMEOUT", raising=False)
    router = LLMRouter()
    assert router._ollama_timeout == 120.0
