"""LangChain output parsers for the agent's structured LLM replies.

The agent asks the model for three shapes of output (a routing decision, a
yes/no grading verdict, and a claim-by-claim grounding report). Each one is
parsed by a LangChain ``BaseOutputParser`` so the nodes stop hand-rolling
``text.strip().lower() == "yes"`` and ``line[8:]`` string slicing.

Every parser is deliberately **tolerant**: free local models (phi3) do not always
follow instructions, so a parse failure falls back to the same behaviour the
agent had before instead of crashing the graph.
"""
from __future__ import annotations

from langchain_core.output_parsers import BaseOutputParser, PydanticOutputParser
from pydantic import ValidationError

from agentic_rag.agent.plan import Plan
from agentic_rag.agent.state import ClaimCheck


class VerdictParser(BaseOutputParser[bool]):
    """Parse a relevance-grading reply into ``True`` / ``False``.

    Semantics are unchanged from the pre-parser code: only an exact ``yes``
    (case/whitespace insensitive) keeps a chunk; anything else is ``False``.
    """

    def parse(self, text: str) -> bool:
        return text.strip().lower() == "yes"

    @property
    def _type(self) -> str:
        return "verdict_parser"


class ClaimsParser(BaseOutputParser[list[ClaimCheck]]):
    """Parse the ``CLAIM: / VERDICT: / CHUNKS:`` grounding report."""

    def parse(self, text: str) -> list[ClaimCheck]:
        claims: list[ClaimCheck] = []
        current: dict = {}
        for raw_line in text.strip().splitlines():
            line = raw_line.strip()
            if line.startswith("CLAIM:"):
                if current:
                    claims.append(_complete_claim(current))
                current = {"claim": line[len("CLAIM:"):].strip()}
            elif line.startswith("VERDICT:") and current:
                verdict = line[len("VERDICT:"):].strip().lower()
                current["verdict"] = (
                    verdict if verdict in {"supported", "partial", "unsupported"} else "unsupported"
                )
            elif line.startswith("CHUNKS:") and current:
                values = line[len("CHUNKS:"):].strip()
                current["chunk_ids"] = (
                    [] if values.lower() == "none" else [v.strip() for v in values.split(",")]
                )
        if current:
            claims.append(_complete_claim(current))
        return claims

    @property
    def _type(self) -> str:
        return "claims_parser"


def _complete_claim(claim: dict) -> ClaimCheck:
    return {
        "claim": claim.get("claim", ""),
        "verdict": claim.get("verdict", "unsupported"),
        "chunk_ids": claim.get("chunk_ids", []),
    }


#: Pydantic-based parser for the planner. It also strips ```json fences, which
#: phi3 and qwen2.5 emit from time to time.
plan_parser: PydanticOutputParser[Plan] = PydanticOutputParser(pydantic_object=Plan)


def parse_plan(raw: str) -> Plan | None:
    """Return the planner decision, or ``None`` when the reply is unusable."""
    try:
        return plan_parser.parse(raw)
    except (ValidationError, ValueError):
        return None
