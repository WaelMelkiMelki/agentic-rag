import json

from agentic_rag.agent.rag import SimpleRAG
from agentic_rag.llm.router import LLMRouter

from judges import judge_answer


def evaluate_answers() -> None:
    rag = SimpleRAG()
    router = LLMRouter()
    judge_llm = router.get_llm("openrouter")

    with open("evals/dataset/dev_set.jsonl", encoding="utf-8") as file:
        questions = [json.loads(line) for line in file if line.strip()]

    answerable = [question for question in questions if question["answerable"]]
    unanswerable = [question for question in questions if not question["answerable"]]

    print("=" * 70)
    print("ANSWER QUALITY EVALUATION")
    print("=" * 70)

    scores = []
    for question in answerable:
        print(f"\nEvaluating {question['id']}: {question['question'][:50]}...")
        result = rag.ask(question["question"], top_k=5)
        score = judge_answer(
            question=question["question"],
            gold_answer=question["gold_answer"],
            system_answer=result["answer"],
            judge_llm=judge_llm,
        )
        scores.append(score)
        print(f"   Score: {score}/5")
        print(f"   System: {result['answer'][:100]}...")

    average_score = sum(scores) / len(scores) if scores else 0
    print("\n" + "=" * 70)
    print("ANSWERABLE QUESTIONS SUMMARY")
    print("=" * 70)
    print(f"Questions: {len(answerable)}")
    print(f"Avg judge score: {average_score:.2f}/5  (target: >= 3.5)")
    print(f"Score distribution: {sorted(scores)}")

    print("\n" + "=" * 70)
    print("UNANSWERABLE QUESTIONS (REFUSAL CHECK)")
    print("=" * 70)
    correct_refusals = 0
    for question in unanswerable:
        result = rag.ask(question["question"], top_k=5)
        refused = "cannot answer" in result["answer"].lower()
        if refused:
            correct_refusals += 1
        status = "PASS" if refused else "FAIL"
        print(f"{status} {question['id']}: {result['answer'][:80]}...")

    refusal_rate = correct_refusals / len(unanswerable) if unanswerable else 0
    print(
        f"\nCorrect refusals: {correct_refusals}/{len(unanswerable)} "
        f"({refusal_rate:.0%})  (target: >= 80%)"
    )

    print("\n" + "=" * 70)
    print("B1 BASELINE SUMMARY")
    print("=" * 70)
    print(f"Avg answer score:  {average_score:.2f}/5")
    print(f"Refusal rate:      {refusal_rate:.0%}")
    print("\nRecord these numbers in docs/eval_report.md!")


if __name__ == "__main__":
    evaluate_answers()
