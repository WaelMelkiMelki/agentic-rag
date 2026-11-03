from agentic_rag.agent.graph import build_agent
from agentic_rag.llm.router import LLMRouter
from agentic_rag.retrieval.embedder import CachedEmbedder
from agentic_rag.retrieval.store import VectorStore

agent = build_agent(
    router=LLMRouter(),
    embedder=CachedEmbedder(),
    store=VectorStore(),
    top_k=5,  # Keep this equal to B1 for comparison.
    max_router_calls=2,
)

result = agent.invoke(
    {
        "question": "Why does the Transformer use multi-head attention?",
        "router_calls": 0,
        "steps": [],
    },
    config={"recursion_limit": 10},
)

print("Answer:", result["final_answer"])
print("Status:", result["status"])
print("Citations:", result["citations"])
print("Router calls:", result["router_calls"])
print("Steps:")
for step in result["steps"]:
    print("-", step)
