"""Ask the model for citations when a draft answer arrived without any.

Local models (phi3) often answer correctly and simply ignore the ``[chunk_id]``
rule, and the grounding pass that could attribute the claims is the longest LLM
call in the graph — on a CPU-only laptop it can time out, leaving the answer
with no evidence trail at all. ``finalize`` then has to refuse it.

This module is the cheap way out: one call that returns **only ids**
(a handful of tokens, so seconds even on CPU), validated against the chunks
that were actually retrieved. Nothing is invented here — an id that is not in
the retrieved set is dropped, and ``none`` keeps the refusal.
"""
from __future__ import annotations

import re

from langchain_core.prompts import ChatPromptTemplate

from agentic_rag.llm.chains import text_chain

CITE_PROMPT = """You are a citation assistant. The answer below was written from the chunks listed.

Answer:
{answer}

Chunks:
{chunks_text}

Which chunk ids support the answer? Reply with ONLY the chunk ids, comma-separated, copied exactly from the brackets above. If none of them support it, reply with only the word: none"""

CITE_CHAT_PROMPT = ChatPromptTemplate.from_template(CITE_PROMPT)

#: Ids are short; more than a few means the model is guessing.
MAX_CITATION_IDS = 3


def request_citations(answer: str, chunks: list, router, max_ids: int = MAX_CITATION_IDS) -> tuple[list[str], int]:
    """Return ``(validated chunk ids, router calls)`` for ``answer``.

    Ids that the model invents are dropped by intersecting with the retrieved
    set, so a hallucinated citation can never reach the caller.
    """
    valid = {chunk.payload["chunk_id"] for chunk in chunks}
    if not valid:
        return [], 0
    chunks_text = "\n\n".join(
        f"[{chunk.payload['chunk_id']}] {chunk.payload['text'][:200]}" for chunk in chunks
    )
    chain = text_chain(CITE_CHAT_PROMPT, router)
    try:
        raw = chain.invoke({"answer": answer, "chunks_text": chunks_text})
    except Exception:
        return [], 1

    cited: list[str] = []
    for token in re.findall(r"[A-Za-z0-9_.:-]+", raw):
        if token in valid and token not in cited:
            cited.append(token)
    return cited[:max_ids], 1
