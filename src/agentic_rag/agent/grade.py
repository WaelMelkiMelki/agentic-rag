from langchain_core.prompts import ChatPromptTemplate

GRADE_PROMPT = """You are a relevance grader. For each chunk, decide if it contains information that helps answer the question.

Question: {question}

Chunk [{chunk_id}]:
{text}

Is this chunk relevant to answering the question?
Respond with ONLY \"yes\" or \"no\":"""

GRADE_CHAT_PROMPT = ChatPromptTemplate.from_template(GRADE_PROMPT)


def grade_chunks(question: str, chunks: list, router) -> tuple[list, int]:
    """Return relevant chunks and the number of router calls used."""
    relevant = []
    calls = 0
    for chunk in chunks:
        messages = GRADE_CHAT_PROMPT.format_messages(
            question=question,
            chunk_id=chunk.payload["chunk_id"],
            text=chunk.payload["text"][:500],
        )
        keep = False
        try:
            verdict = router.invoke_with_fallback(messages).strip().lower()
            keep = verdict == "yes"
        except Exception:
            # Preserve evidence when the optional filter is unavailable.
            keep = True
        calls += 1
        if keep:
            relevant.append(chunk)
    return relevant, calls
