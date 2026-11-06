from agentic_rag.agent.ground import check_grounding


class FakeRouter:
    def __init__(self, reply):
        self.reply = reply

    def invoke_with_fallback(self, prompt):
        return self.reply


def test_grounded_answer():
    reply = "CLAIM: Uses attention\nVERDICT: supported\nCHUNKS: c1"
    result = check_grounding("answer", [], FakeRouter(reply))
    assert result["is_grounded"] is True
    assert result["claims"][0]["chunk_ids"] == ["c1"]


def test_unsupported_claim_fails():
    reply = "CLAIM: Invented in 2025\nVERDICT: unsupported\nCHUNKS: none"
    result = check_grounding("answer", [], FakeRouter(reply))
    assert result["is_grounded"] is False
    assert "Invented in 2025" in result["feedback"]


def test_threshold_and_partial_claims():
    reply = "CLAIM: A\nVERDICT: supported\nCHUNKS: c1\n\nCLAIM: B\nVERDICT: partial\nCHUNKS: c2\n\nCLAIM: C\nVERDICT: supported\nCHUNKS: c3"
    result = check_grounding("answer", [], FakeRouter(reply))
    assert result["is_grounded"] is False


def test_three_of_four_supported_pass():
    reply = "\n\n".join(
        f"CLAIM: {name}\nVERDICT: {verdict}\nCHUNKS: c{i}"
        for i, (name, verdict) in enumerate([
            ("A", "supported"), ("B", "supported"), ("C", "partial"), ("D", "supported")
        ])
    )
    assert check_grounding("answer", [], FakeRouter(reply))["is_grounded"] is True
