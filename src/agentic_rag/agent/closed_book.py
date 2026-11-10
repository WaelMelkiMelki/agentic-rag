"""B0 — closed-book baseline: answer from parametric knowledge, no retrieval.

This is the ablation floor. It uses one LLM call with no context, no citations
and no grounding check, so the delta measured against B1 (plain RAG) and B4 (the
self-correcting agent) isolates what retrieval and self-correction actually add.

It is deliberately kept out of the shared ``GENERATE_PROMPT``: that prompt tells
the model to answer *only* from context, which would make a closed-book run
meaningless.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from agentic_rag.agent.prompts import ABSTENTION
from agentic_rag.llm.chains import text_chain
from agentic_rag.llm.router import LLMRouter

CLOSED_BOOK_PROMPT = """You are answering a question from memory, without any documents.

Rules:
1. Answer using your own knowledge. There is no context and there are no sources.
2. If you do not know the answer, respond with EXACTLY this sentence: "I cannot answer this from the provided sources."
3. Be concise. 2-5 sentences maximum.
4. Do not invent citations.

Question: {question}

Answer:"""

CLOSED_BOOK_CHAT_PROMPT = ChatPromptTemplate.from_template(CLOSED_BOOK_PROMPT)


class ClosedBook:
    """B0 baseline: a single retrieval-free generation call."""

    def __init__(self, router=None):
        self.router = router if router is not None else LLMRouter()
        self.chain = text_chain(CLOSED_BOOK_CHAT_PROMPT, self.router)

    def query(self, question: str) -> dict:
        """Answer from parametric knowledge; return the API's result shape."""
        try:
            answer = self.chain.invoke({"question": question}).strip()
        except Exception:
            return {
                "answer": ABSTENTION,
                "citations": [],
                "steps": ["generate (closed-book)", "Generation failed"],
                "status": "abstained",
                "stop_reason": "llm_unavailable",
            }

        answered = bool(answer) and answer.casefold() != ABSTENTION.casefold()
        return {
            "answer": answer if answered else ABSTENTION,
            "citations": [],
            "steps": ["generate (closed-book)"],
            "status": "answered" if answered else "abstained",
            "stop_reason": "completed" if answered else "no_answer",
        }
