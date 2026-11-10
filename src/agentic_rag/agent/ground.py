from langchain_core.prompts import ChatPromptTemplate

from agentic_rag.agent.state import ClaimCheck

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
    try:
        raw = router.invoke_with_fallback(
            GROUND_CHAT_PROMPT.format_messages(answer=answer, chunks_text=chunks_text)
        )
    except Exception:
        return {"is_grounded": False, "claims": [],
                "feedback": "Grounding check failed: LLM unavailable", "router_calls": 1}

    claims: list[ClaimCheck] = []
    current = {}
    for line in raw.strip().splitlines():
        line = line.strip()
        if line.startswith("CLAIM:"):
            if current:
                claims.append(_complete_claim(current))
            current = {"claim": line[6:].strip()}
        elif line.startswith("VERDICT:") and current:
            verdict = line[8:].strip().lower()
            current["verdict"] = verdict if verdict in {"supported", "partial", "unsupported"} else "unsupported"
        elif line.startswith("CHUNKS:") and current:
            values = line[7:].strip()
            current["chunk_ids"] = [] if values.lower() == "none" else [v.strip() for v in values.split(",")]
    if current:
        claims.append(_complete_claim(current))

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


def _complete_claim(claim: dict) -> ClaimCheck:
    return {
        "claim": claim.get("claim", ""),
        "verdict": claim.get("verdict", "unsupported"),
        "chunk_ids": claim.get("chunk_ids", []),
    }
