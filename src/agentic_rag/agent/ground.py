from langchain_core.prompts import ChatPromptTemplate

from agentic_rag.agent.parsing import ClaimsParser
from agentic_rag.llm.chains import text_chain

GROUND_PROMPT = """You are a fact-checker. Check if each claim in the answer is supported by the provided chunks.

Answer to check:
{answer}

Available chunks:
{chunks_text}

For each distinct factual claim, respond in this exact format:
CLAIM: <the claim>
VERDICT: supported | partial | unsupported
CHUNKS: <comma-separated chunk_ids, or none>

supported means clearly stated; partial means hinted at but not confirmed; unsupported means absent.
"""

GROUND_CHAT_PROMPT = ChatPromptTemplate.from_template(GROUND_PROMPT)


def check_grounding(answer: str, chunks: list, router, grounded_threshold: float = 0.75) -> dict:
    chunks_text = "\n\n".join(
        f"[{c.payload['chunk_id']}]: {c.payload['text'][:300]}" for c in chunks
    )
    chain = text_chain(GROUND_CHAT_PROMPT, router)
    try:
        raw = chain.invoke({"answer": answer, "chunks_text": chunks_text})
    except Exception:
        return {"is_grounded": False, "claims": [],
                "feedback": "Grounding check failed: LLM unavailable", "router_calls": 1}

    claims = ClaimsParser().parse(raw)
    if not claims:
        return {"is_grounded": False, "claims": [],
                "feedback": "No claims could be extracted from the answer", "router_calls": 1}
    unsupported = sum(c["verdict"] == "unsupported" for c in claims)
    supported = sum(c["verdict"] == "supported" for c in claims)
    grounded = unsupported == 0 and supported / len(claims) >= grounded_threshold
    bad = [c for c in claims if c["verdict"] in {"partial", "unsupported"}]
    feedback = "" if grounded else "The following claims are not supported by the chunks:\n" + "".join(
        f"- '{c['claim']}' ({c['verdict']})\n" for c in bad[:3]
    )
    return {"is_grounded": grounded, "claims": claims, "feedback": feedback, "router_calls": 1}
