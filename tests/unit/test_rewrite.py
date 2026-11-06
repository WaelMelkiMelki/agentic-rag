from agentic_rag.agent.rewrite import rewrite_query


def test_rewrite_query():
    class FakeRouter:
        def invoke_with_fallback(self, prompt):
            return "multi-head attention mechanism transformer"

    query, calls = rewrite_query("How does attention work?", "attention", "No relevant chunks", 1, FakeRouter())
    assert "multi-head" in query
    assert calls == 1
