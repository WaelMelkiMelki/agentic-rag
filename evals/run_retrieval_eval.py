import json

from agentic_rag.retrieval.embedder import CachedEmbedder
from agentic_rag.retrieval.store import VectorStore

from metrics import avg_score, hit_at_k, mrr


def evaluate_retrieval() -> None:
    embedder = CachedEmbedder()
    store = VectorStore()

    with open("evals/dataset/dev_set.jsonl", encoding="utf-8") as file:
        questions = [json.loads(line) for line in file if line.strip()]

    answerable = [question for question in questions if question["answerable"]]
    hits = []
    mrrs = []
    scores = []

    print("=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)

    for question in answerable:
        query_vector = embedder.embed_query(question["question"])
        results = store.search(query_vector, top_k=5)

        hit = hit_at_k(results, question["gold_doc"], k=5)
        reciprocal_rank = mrr(results, question["gold_doc"])
        score = avg_score(results, question["gold_doc"], k=5)

        hits.append(hit)
        mrrs.append(reciprocal_rank)
        scores.append(score)

        status = "PASS" if hit == 1.0 else "FAIL"
        print(
            f"{status} {question['id']}: hit={hit:.1f} "
            f"mrr={reciprocal_rank:.2f} score={score:.3f} | "
            f"{question['question'][:50]}..."
        )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Questions evaluated: {len(answerable)}")
    print(f"Hit@5:  {sum(hits) / len(hits):.2f}  (target: >= 0.70)")
    print(f"MRR:    {sum(mrrs) / len(mrrs):.2f}  (target: >= 0.50)")
    print(f"Avg score of relevant chunks: {sum(scores) / len(scores):.3f}")


if __name__ == "__main__":
    evaluate_retrieval()
