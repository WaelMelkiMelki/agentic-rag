import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()


class LLMRouter:
    def __init__(self):
        self.openrouter = ChatOpenAI(
            model="openrouter/free",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0,
        )

        self.gemini = ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0,
        )

    def get_llm(self, provider: str = "gemini"):
        """Get an LLM by provider name."""
        if provider == "openrouter":
            return self.openrouter
        if provider == "gemini":
            return self.gemini
        raise ValueError(f"Unknown provider: {provider}")

    def invoke_with_fallback(self, prompt: str) -> str:
        """Try Gemini first. If it fails, fall back to OpenRouter."""
        try:
            response = self.gemini.invoke(prompt)
            return response.content
        except Exception as error:
            print(f"⚠️  Gemini failed: {error}")
            print("🔄 Falling back to OpenRouter...")

        try:
            response = self.openrouter.invoke(prompt)
            return response.content
        except Exception as error:
            print(f"⚠️  OpenRouter also failed: {error}")

        raise RuntimeError("All providers failed. Check your API keys.")
