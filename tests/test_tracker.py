from agentic_rag.llm.router import LLMRouter
from agentic_rag.llm.tracker import UsageTracker


router = LLMRouter()
tracker = UsageTracker()

questions = [
    "What is AI?",
    "What is RAG?",
    "What is LangGraph?",
]

for question in questions:
    llm = router.get_llm("gemini")
    response = llm.invoke(question)
    tracker.record(response, provider="gemini", model="gemini-3.8-flash")
    print(f"Q: {question}")
    print(f"A: {response.content[:80]}...\n")

print(tracker.summary())

llm = router.get_llm("openrouter")
response = llm.invoke("What is an agent?")
tracker.record(response, provider="openrouter", model="openrouter/free")

print(tracker.summary())
