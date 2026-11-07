"""Manual live provider usage check; run with ``python tests/test_tracker.py``."""


def main():
    from agentic_rag.llm.router import LLMRouter
    from agentic_rag.llm.tracker import UsageTracker

    router = LLMRouter()
    tracker = UsageTracker()

    for question in ["What is AI?", "What is RAG?", "What is LangGraph?"]:
        response = router.get_llm("gemini").invoke(question)
        tracker.record(response, provider="gemini", model="gemini-3.8-flash")
        print(f"Q: {question}")
        print(f"A: {response.content[:80]}...\n")

    print(tracker.summary())
    response = router.get_llm("openrouter").invoke("What is an agent?")
    tracker.record(response, provider="openrouter", model="openrouter/free")
    print(tracker.summary())


if __name__ == "__main__":
    main()
