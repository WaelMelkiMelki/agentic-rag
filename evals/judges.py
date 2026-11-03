JUDGE_PROMPT = """You are an expert evaluator. Rate the following answer on a scale of 1-5.

Question: {question}

Gold (correct) Answer: {gold_answer}

System's Answer: {system_answer}

Scoring rubric:
1 = Completely wrong or hallucinated
2 = Mostly wrong, with minor correct elements
3 = Partially correct, but missing key details or containing errors
4 = Mostly correct, with minor omissions or imprecisions
5 = Fully correct and complete

IMPORTANT:
- Compare the system's answer against the gold answer
- Penalize hallucinated facts that are not in the gold answer
- Reward correct citations
- A refusal ("I cannot answer") on an answerable question = score 1

Respond with ONLY a single number (1-5):"""


def judge_answer(
    question: str,
    gold_answer: str,
    system_answer: str,
    judge_llm,
) -> int:
    """Use an LLM to score an answer against the gold standard."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        system_answer=system_answer,
    )
    response = judge_llm.invoke(prompt)
    text = response.content.strip()

    for char in text:
        if char in "12345":
            return int(char)
    return 0
