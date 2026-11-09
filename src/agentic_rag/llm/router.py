import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()

# Ollama's OpenAI-compatible endpoint. Override with OLLAMA_BASE_URL
# (e.g. http://localhost:11434 or https://my-ollama.example.com/v1).
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_TIMEOUT = 60


def _normalize_ollama_base_url(url: str) -> str:
    """Ensure the Ollama base URL points at the OpenAI-compatible /v1 API."""
    url = (url or "").strip().rstrip("/")
    if not url:
        return DEFAULT_OLLAMA_BASE_URL
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


def _ollama_timeout() -> float:
    try:
        return float(os.getenv("OLLAMA_TIMEOUT", str(DEFAULT_OLLAMA_TIMEOUT)))
    except ValueError:
        return float(DEFAULT_OLLAMA_TIMEOUT)


def _ollama_disabled() -> bool:
    return os.getenv("OLLAMA_ENABLED", "true").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    )


class LLMRouter:
    """Ollama-first LLM router with Gemini and OpenRouter fallbacks.

    Priority order for :meth:`invoke_with_fallback`:

    1. **Ollama** (local, ``OLLAMA_BASE_URL`` / ``OLLAMA_MODEL``) — tried
       first so local/self-hosted inference is preferred. If Ollama is
       unreachable or errors, it is marked unavailable for the rest of
       the process so the agent's repeated router calls don't each pay
       the connection/timeout cost.
    2. **Gemini** (``gemini-3.8-flash``) — skipped for the rest of the
       run once a quota/rate-limit error is seen.
    3. **OpenRouter** (``openrouter/free``) — last resort.

    Set ``OLLAMA_ENABLED=false`` to skip Ollama entirely (e.g. on Render
    where no Ollama endpoint is reachable and you want to go straight to
    Gemini/OpenRouter).
    """

    def __init__(self, tracker=None, ollama=None, gemini=None, openrouter=None):
        self.tracker = tracker
        self._gemini_quota_exhausted = False
        self._ollama_unavailable = _ollama_disabled()

        self.ollama = (
            ollama
            if ollama is not None
            else ChatOpenAI(
                model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
                base_url=_normalize_ollama_base_url(
                    os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
                ),
                # Ollama ignores the key but ChatOpenAI requires one.
                api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
                temperature=0,
                timeout=_ollama_timeout(),
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

    def get_llm(self, provider: str = "ollama"):
        """Get an LLM by provider name (``ollama``, ``gemini``, ``openrouter``)."""
        if provider == "ollama":
            return self.ollama
        if provider == "openrouter":
            return self.openrouter
        if provider == "gemini":
            return self.gemini
        raise ValueError(f"Unknown provider: {provider}")

    def invoke_with_fallback(self, prompt: str) -> str:
        """Use Ollama first, then Gemini, then OpenRouter."""
        if self.ollama is not None and not self._ollama_unavailable:
            try:
                response = self.ollama.invoke(prompt)
                self._record_usage(response, "ollama", self.ollama)
                return response.content
            except Exception as error:
                print(f"⚠️  Ollama failed for this request: {error}")
                self._ollama_unavailable = True
                print(
                    "Ollama unavailable; skipping it for remaining requests in this run."
                )

        if self.gemini is not None and not self._gemini_quota_exhausted:
            try:
                response = self.gemini.invoke(prompt)
                self._record_usage(response, "gemini", self.gemini)
                return response.content
            except Exception as error:
                print(f"⚠️  Gemini failed for this request: {error}")
                if _is_quota_error(error):
                    self._gemini_quota_exhausted = True
                    print("Gemini quota appears exhausted; skipping it for remaining requests in this run.")

        print("🔄 Using OpenRouter...")

        try:
            response = self.openrouter.invoke(prompt)
            self._record_usage(response, "openrouter", self.openrouter)
            return response.content
        except Exception as error:
            print(f"⚠️  OpenRouter also failed: {error}")

        raise RuntimeError("All providers failed. Check your API keys.")

    def _record_usage(self, response, provider: str, model) -> None:
        if self.tracker is not None:
            model_name = getattr(model, "model_name", None) or getattr(
                model, "model", "unknown"
            )
            self.tracker.record(response, provider, model_name)


def _is_quota_error(error: Exception) -> bool:
    """Recognize quota/rate-limit failures that won't improve on an immediate retry."""
    status = getattr(error, "status_code", None)
    if status == 429:
        return True
    message = str(error).casefold()
    return any(marker in message for marker in (
        "quota exceeded", "resource_exhausted", "too many requests", "rate limit exceeded"
    ))
