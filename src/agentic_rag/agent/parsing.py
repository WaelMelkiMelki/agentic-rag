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

import re

from langchain_core.output_parsers import BaseOutputParser, PydanticOutputParser
from pydantic import ValidationError

from agentic_rag.agent.plan import Plan
from agentic_rag.agent.state import ClaimCheck


#: ``yes`` as the first word, allowing the emphasis/quotes local models add.
_YES_RE = re.compile(r"^[*_`\"'\s]*yes\b", re.IGNORECASE)


class VerdictParser(BaseOutputParser[bool]):
    """Parse a relevance-grading reply into ``True`` / ``False``.

    The prompt asks for a bare ``yes`` / ``no``, but local models decorate it
    (``Yes.``, ``**Yes**``, ``yes, it explains attention``). The first word
    decides, so a decorated answer no longer silently drops a usable chunk —
    phi3 was grading 5 chunks down to 2 on a question the corpus answers.

    Anything that does not start with ``yes`` is ``False``: a hedged or
    unparseable reply still drops the chunk rather than passing it on.
    """

    def parse(self, text: str) -> bool:
        return bool(_YES_RE.match(text))

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
