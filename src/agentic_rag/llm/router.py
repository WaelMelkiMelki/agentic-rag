import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()

#nice

class LLMRouter:
    def __init__(self, tracker=None):
        self.tracker = tracker
        self._gemini_quota_exhausted = False
        self.openrouter = ChatOpenAI(
            model="openrouter/free",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0,
            timeout=180,
            max_retries=1,
        )

        self.gemini = ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0,
            # Bound the primary request so invoke_with_fallback can reach OpenRouter.
            timeout=120,
            max_retries=0,
        )

    def get_llm(self, provider: str = "gemini"):
        """Get an LLM by provider name."""
        if provider == "openrouter":
            return self.openrouter
        if provider == "gemini":
            return self.gemini
        raise ValueError(f"Unknown provider: {provider}")

    def invoke_with_fallback(self, prompt: str) -> str:
        """Use Gemini first unless its daily quota is exhausted, then use OpenRouter."""
        if not self._gemini_quota_exhausted:
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
            self.tracker.record(response, provider, getattr(model, "model", "unknown"))


def _is_quota_error(error: Exception) -> bool:
    """Recognize quota/rate-limit failures that won't improve on an immediate retry."""
    status = getattr(error, "status_code", None)
    if status == 429:
        return True
    message = str(error).casefold()
    return any(marker in message for marker in (
        "quota exceeded", "resource_exhausted", "too many requests", "rate limit exceeded"
    ))
