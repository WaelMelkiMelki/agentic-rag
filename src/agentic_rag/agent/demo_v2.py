"""Small live demo of the Phase 6 agent (uses LLM and embedding services)."""
from agentic_rag.agent.graph import build_agent
from agentic_rag.llm.router import LLMRouter
from agentic_rag.retrieval.embedder import CachedEmbedder
from agentic_rag.retrieval.store import VectorStore


def main():
    agent = build_agent(LLMRouter(), CachedEmbedder(), VectorStore(), top_k=5,
                        max_router_calls=10, max_rounds=2)
    questions = [
        "Why does the Transformer use multi-head attention?",
        "What datasets were used to evaluate RAG?",
        "What is the best pizza in New York?",
    ]
    for question in questions:
        result = agent.invoke({"question": question, "router_calls": 0, "steps": []},
                              config={"recursion_limit": 25})
        print(f"\nQ: {question}\nStatus: {result['status']} | reason: {result['stop_reason']} "
              f"| calls: {result['router_calls']} | grounded: {result.get('is_grounded', 'N/A')}\n"
              f"Answer: {result['final_answer']}\nSteps: {' → '.join(result['steps'])}")


if __name__ == "__main__":
    main()
