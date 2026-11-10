import os
from collections.abc import Sequence

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()

# Prompts arrive either as plain text or as LangChain chat messages
# (``ChatPromptTemplate.format_messages(...)``), so both are accepted.
PromptInput = str | Sequence[BaseMessage] | PromptValue


class LLMRouter:
    """Ollama-first LLM router with Gemini and OpenRouter fallbacks.

    Priority order for :meth:`invoke_with_fallback`:

    1. **Ollama phi3** (``OLLAMA_PHI_MODEL``, default ``phi3:latest``) —
       primary local model via the Ollama OpenAI-compatible endpoint.
    2. **Ollama qwen2.5** (``OLLAMA_QWEN_MODEL``, default ``qwen2.5:latest``) —
       secondary local model on the same endpoint.
    3. **Gemini** (``gemini-3.8-flash``) — cloud fallback; skipped for the
       rest of the process once a quota/rate-limit error is seen.
    4. **OpenRouter** (``openrouter/free``) — last resort.

    If either Ollama model fails with a connection error (e.g. Ollama is not
    running, or the Cloudflare Tunnel is down), Ollama is marked unavailable
    for the rest of the process so the agent's repeated router calls don't
    each pay a connection cost. A *timeout* is treated differently: it usually
    means the model is still loading or generating, so Ollama stays enabled and
    the next query retries it. ``OLLAMA_TIMEOUT`` (default 120s) bounds each
    local call.

    Set ``OLLAMA_ENABLED=false`` to skip Ollama entirely and use only
    Gemini + OpenRouter (e.g. on Render, where Ollama is local-only).
    """

    def __init__(
        self,
        tracker=None,
        ollama_phi=None,
        ollama_qwen=None,
        gemini=None,
        openrouter=None,
    ):
        self.tracker = tracker
        self._gemini_quota_exhausted = False
        # OLLAMA_ENABLED=false skips Ollama (local-only use; Render uses
        # Gemini + OpenRouter only).
        self._ollama_unavailable = os.getenv(
            "OLLAMA_ENABLED", "true"
        ).strip().lower() in ("0", "false", "no", "off")

        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        ollama_api_key = os.getenv("OLLAMA_API_KEY", "ollama")
        phi_model = os.getenv("OLLAMA_PHI_MODEL", "phi3:latest")
        qwen_model = os.getenv("OLLAMA_QWEN_MODEL", "qwen2.5:latest")
        # phi3 needs time to load into memory on a laptop on the first call.
        ollama_timeout = float(os.getenv("OLLAMA_TIMEOUT", "120"))
        self._phi_model = phi_model
        self._qwen_model = qwen_model
        self._ollama_timeout = ollama_timeout

        self.ollama_phi = (
            ollama_phi
            if ollama_phi is not None
            else ChatOpenAI(
                model=phi_model,
                base_url=ollama_base_url,
                api_key=ollama_api_key,
                temperature=0,
                timeout=ollama_timeout,
                max_retries=0,
            )
        )

        self.ollama_qwen = (
            ollama_qwen
            if ollama_qwen is not None
            else ChatOpenAI(
                model=qwen_model,
                base_url=ollama_base_url,
                api_key=ollama_api_key,
                temperature=0,
                timeout=ollama_timeout,
                max_retries=0,
            )
        )

        self.gemini = (
            gemini
            if gemini is not None
            else ChatGoogleGenerativeAI(
                model="gemini-3.8-flash",
                api_key=os.getenv("GEMINI_API_KEY") or "missing-key",
                temperature=0,
                # Bound the primary request so invoke_with_fallback can reach OpenRouter.
                timeout=120,
                max_retries=0,
            )
        )

        self.openrouter = (
            openrouter
            if openrouter is not None
            else ChatOpenAI(
                model="openrouter/free",
                base_url=os.getenv(
                    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
                ),
                # Placeholder keeps router construction working when only
                # Ollama is configured; the call will fail and surface as
                # "All providers failed" instead of crashing at startup.
                api_key=os.getenv("OPENROUTER_API_KEY") or "missing-key",
                temperature=0,
                timeout=180,
                max_retries=1,
            )
        )

    def get_llm(self, provider: str = "ollama_phi"):
        """Get an LLM by provider name.

        Accepts ``"ollama_phi"``, ``"ollama_qwen"``, ``"gemini"``,
        and ``"openrouter"``.
        """
        if provider == "ollama_phi":
            return self.ollama_phi
        if provider == "ollama_qwen":
            return self.ollama_qwen
        if provider == "gemini":
            return self.gemini
        if provider == "openrouter":
            return self.openrouter
        raise ValueError(f"Unknown provider: {provider}")

    def invoke_with_fallback(self, prompt: PromptInput) -> str:
        """Use Ollama phi3, then Ollama qwen2.5, then Gemini, then OpenRouter.

        ``prompt`` may be a string or a list of chat messages produced by a
        :class:`~langchain_core.prompts.ChatPromptTemplate`; every provider
        accepts both. The return value is always the answer text.
        """
        # Snapshot so a phi failure in this call still allows qwen to be
        # tried once; subsequent calls skip Ollama entirely.
        ollama_was_unavailable = self._ollama_unavailable

        if self.ollama_phi is not None and not ollama_was_unavailable:
            print(f"🔄 Using Ollama ({self._phi_model})...")
            try:
                response = self.ollama_phi.invoke(prompt)
                self._record_usage(response, "ollama_phi", self.ollama_phi)
                return response.content
            except Exception as error:
                print(f"⚠️ Ollama failed: {error}")
                self._note_ollama_failure(error)

        if self.ollama_qwen is not None and not ollama_was_unavailable:
            print(f"🔄 Using Ollama ({self._qwen_model})...")
            try:
                response = self.ollama_qwen.invoke(prompt)
                self._record_usage(response, "ollama_qwen", self.ollama_qwen)
                return response.content
            except Exception as error:
                print(f"⚠️ Ollama failed: {error}")
                self._note_ollama_failure(error)

        if self.gemini is not None and not self._gemini_quota_exhausted:
            print("🔄 Using Gemini...")
            try:
                response = self.gemini.invoke(prompt)
                self._record_usage(response, "gemini", self.gemini)
                return response.content
            except Exception as error:
                print(f"⚠️ Gemini failed: {error}")
                if _is_quota_error(error):
                    self._gemini_quota_exhausted = True
                    print("Gemini quota appears exhausted; skipping it for remaining requests in this run.")

        print("🔄 Using OpenRouter...")

        try:
            response = self.openrouter.invoke(prompt)
            self._record_usage(response, "openrouter", self.openrouter)
            return response.content
        except Exception as error:
            print(f"⚠️ OpenRouter also failed: {error}")

        raise RuntimeError("All providers failed. Check your API keys.")

    def _note_ollama_failure(self, error: Exception) -> None:
        """Decide whether this failure should disable Ollama for the process.

        A connection error means Ollama is not running (or the tunnel is down),
        so skip both local models from now on. A timeout usually means the model
        is still loading/generating, so the next query tries Ollama again.
        """
        if _is_timeout_error(error):
            print(
                "   Ollama timed out; keeping it enabled for the next query "
                "(raise OLLAMA_TIMEOUT if phi3 is still loading)."
            )
            return
        self._ollama_unavailable = True

    def _record_usage(self, response, provider: str, model) -> None:
        if self.tracker is not None:
            model_name = getattr(model, "model_name", None) or getattr(
                model, "model", "unknown"
            )
            self.tracker.record(response, provider, model_name)


def _is_timeout_error(error: Exception) -> bool:
    """Recognize request timeouts (model busy/cold) as opposed to unreachable servers."""
    if isinstance(error, TimeoutError):
        return True
    for klass in type(error).__mro__:
        if "timeout" in klass.__name__.casefold():
            return True
    message = str(error).casefold()
    return "timed out" in message or "timeout" in message


def _is_quota_error(error: Exception) -> bool:
    """Recognize quota/rate-limit failures that won't improve on an immediate retry."""
    status = getattr(error, "status_code", None)
    if status == 429:
        return True
    message = str(error).casefold()
    return any(marker in message for marker in (
        "quota exceeded", "resource_exhausted", "too many requests", "rate limit exceeded"
    ))
