import json


with open("evals/dataset/dev_set.jsonl", encoding="utf-8") as file:
    questions = [json.loads(line) for line in file if line.strip()]

answerable = [question for question in questions if question["answerable"]]
unanswerable = [question for question in questions if not question["answerable"]]

print(f"Total questions: {len(questions)}")
print(f"Answerable: {len(answerable)}")
print(f"Unanswerable: {len(unanswerable)}")

for question in questions:
    assert "id" in question, f"Missing id in {question}"
    assert "question" in question, f"Missing question in {question}"
    assert "answerable" in question, f"Missing answerable in {question}"
    if question["answerable"]:
        assert question["gold_doc"] is not None, (
            f"Answerable q {question['id']} missing gold_doc"
        )
        assert question["gold_answer"] is not None, (
            f"Answerable q {question['id']} missing gold_answer"
        )
    else:
        assert question["gold_doc"] is None, (
            f"Unanswerable q {question['id']} should have null gold_doc"
        )

print("All questions valid!")
