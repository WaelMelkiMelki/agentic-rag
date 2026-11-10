"""Output parsers behind the planner, grader and grounding check."""
from __future__ import annotations

import pytest

from agentic_rag.agent.parsing import ClaimsParser, VerdictParser, parse_plan


# ---------------------------------------------------------------- verdicts ---
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("yes", True),
        (" YES \n", True),
        # Local models decorate the verdict; the first word is the decision.
        ("Yes.", True),
        ("**Yes**", True),
        ("yes, it is relevant", True),
        ("no", False),
        ("No.", False),
        ("not relevant, no", False),
        # Never interpret silence as evidence.
        ("", False),
        ("maybe", False),
        ("I am not sure", False),
    ],
)
def test_verdict_parser_reads_the_first_word(text, expected):
    assert VerdictParser().parse(text) is expected


# ------------------------------------------------------------------ claims ---
def test_claims_parser_parses_multiple_claims():
    raw = """CLAIM: The Transformer uses multi-head attention.
VERDICT: supported
CHUNKS: attention_is_all_you_need__4, attention_is_all_you_need__5

CLAIM: Transformers were trained on 8 GPUs for 12 hours.
VERDICT: partial
CHUNKS: attention_is_all_you_need__1"""
    claims = ClaimsParser().parse(raw)

    assert len(claims) == 2
    assert claims[0]["verdict"] == "supported"
    assert claims[0]["chunk_ids"] == ["attention_is_all_you_need__4", "attention_is_all_you_need__5"]
    assert claims[1]["verdict"] == "partial"
    assert claims[1]["claim"].startswith("Transformers were trained")


def test_claims_parser_handles_none_chunks_and_missing_fields():
    raw = """CLAIM: RAG combines retrieval with generation.
VERDICT: supported
CHUNKS: none

CLAIM: DPR is the retriever used in every RAG system."""
    claims = ClaimsParser().parse(raw)

    assert claims[0]["chunk_ids"] == []
    assert claims[1]["verdict"] == "unsupported"   # no VERDICT line -> unsupported


def test_claims_parser_treats_unknown_verdict_as_unsupported():
    raw = "CLAIM: x\nVERDICT: maybe\nCHUNKS: c1"
    assert ClaimsParser().parse(raw)[0]["verdict"] == "unsupported"


def test_claims_parser_returns_empty_list_for_garbage():
    assert ClaimsParser().parse("I cannot answer this from the provided sources.") == []


# -------------------------------------------------------------------- plan ---
def test_parse_plan_reads_plain_json():
    plan = parse_plan('{"action": "retrieve", "query": "multi-head attention"}')
    assert plan is not None
    assert plan.action == "retrieve"
    assert plan.query == "multi-head attention"


def test_parse_plan_strips_markdown_code_fences():
    """qwen2.5/phi3 often wrap JSON in fences; the parser tolerates it."""
    raw = '```json\n{"action": "abstain", "query": ""}\n```'
    plan = parse_plan(raw)
    assert plan is not None
    assert plan.action == "abstain"


@pytest.mark.parametrize("raw", ["not json at all", "", '{"action": "maybe"}'])
def test_parse_plan_returns_none_when_unusable(raw):
    assert parse_plan(raw) is None
