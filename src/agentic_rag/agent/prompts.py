GENERATE_PROMPT = """You are a research assistant. Answer the question using ONLY the provided context.

Rules:
1. Cite sources using [chunk_id] notation after each claim. Example: "The Transformer uses attention [attention_is_all_you_need__12]."
2. If the context does NOT contain the answer, respond with EXACTLY this sentence: "I cannot answer this from the provided sources."
3. Do NOT use your own knowledge. Only use what is in the context.
4. Be concise. 2-5 sentences maximum.
5. If the answer spans multiple chunks, cite all relevant ones.

Context:
{context}

Question: {question}

Answer:"""


def format_context(chunks) -> str:
    """Format retrieved chunks into a readable context block."""
    parts = []
    for chunk in chunks:
        parts.append(
            f"[{chunk.payload['chunk_id']}] "
            f"(doc: {chunk.payload['doc_id']}, "
            f"section: {chunk.payload['chunk_index']}, "
            f"relevance: {chunk.score:.3f})\n"
            f"{chunk.payload['text']}"
        )
    return "\n\n---\n\n".join(parts)
