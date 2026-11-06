"""Compare B1 and the Phase 6 agent on the dev set; this makes live LLM calls."""
import json

from agentic_rag.agent.graph import build_agent
from agentic_rag.agent.rag import SimpleRAG
from agentic_rag.llm.router import LLMRouter
from agentic_rag.retrieval.embedder import CachedEmbedder
from agentic_rag.retrieval.store import VectorStore
from evals.judges import judge_answer


def run_ablation():
    router = LLMRouter()
    b1 = SimpleRAG()
    b4 = build_agent(router, CachedEmbedder(), VectorStore(), top_k=5,
                     max_router_calls=10, max_rounds=2)
    judge_llm = router.get_llm("openrouter")
    with open("evals/dataset/dev_set.jsonl", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    answerable = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]
    b1_scores, b4_scores = [], []
    b1_refusals = b4_refusals = b4_calls = 0

    for q in answerable:
        r1 = b1.ask(q["question"])
        b1_scores.append(judge_answer(q["question"], q["gold_answer"], r1["answer"], judge_llm))
        r4 = b4.invoke({"question": q["question"], "router_calls": 0, "steps": []},
                       config={"recursion_limit": 25})
        b4_scores.append(judge_answer(q["question"], q["gold_answer"], r4["final_answer"], judge_llm))
        b4_calls += r4["router_calls"]
        print(f"{q['id']}: B1={b1_scores[-1]}/5 B4={b4_scores[-1]}/5 calls={r4['router_calls']}")

    for q in unanswerable:
        r1 = b1.ask(q["question"])
        r4 = b4.invoke({"question": q["question"], "router_calls": 0, "steps": []},
                       config={"recursion_limit": 25})
        b1_refusals += "cannot answer" in r1["answer"].lower()
        b4_refusals += r4["status"] == "abstained"

    avg1 = sum(b1_scores) / len(b1_scores) if b1_scores else 0
    avg4 = sum(b4_scores) / len(b4_scores) if b4_scores else 0
    ref1 = b1_refusals / len(unanswerable) if unanswerable else 0
    ref4 = b4_refusals / len(unanswerable) if unanswerable else 0
    print("\nMetric                 B1       B4")
    print(f"Avg answer score      {avg1:.2f}     {avg4:.2f}")
    print(f"Refusal rate          {ref1:.0%}     {ref4:.0%}")
    print(f"Avg B4 router calls      —      {b4_calls / len(questions) if questions else 0:.2f}")


if __name__ == "__main__":
    run_ablation()
