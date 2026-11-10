"""LCEL chains built around the fallback router.

``text_chain`` composes the three pieces the agent uses everywhere:

    ChatPromptTemplate | LLMRouter.invoke_with_fallback | StrOutputParser

so every node is a single ``chain.invoke({...}) -> str`` call instead of manual
``format_messages`` + ``invoke`` + ``.content`` plumbing. The router keeps the
Ollama -> Gemini -> OpenRouter fallback and still returns a plain string, so the
chain has no idea which provider answered.
"""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda


def text_chain(prompt: ChatPromptTemplate, router) -> Runnable:
    """Build ``prompt | router | StrOutputParser()`` for one prompt template."""
    return prompt | RunnableLambda(router.invoke_with_fallback) | StrOutputParser()
