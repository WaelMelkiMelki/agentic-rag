def hit_at_k(results, gold_doc: str, k: int = 5) -> float:
    """Return 1.0 when the gold document appears in the first k results."""
    for result in results[:k]:
        if result.payload["doc_id"] == gold_doc:
            return 1.0
    return 0.0


def mrr(results, gold_doc: str) -> float:
    """Return the reciprocal rank of the first result from the gold document."""
    for index, result in enumerate(results):
        if result.payload["doc_id"] == gold_doc:
            return 1.0 / (index + 1)
    return 0.0


def avg_score(results, gold_doc: str, k: int = 5) -> float:
    """Return the average score of gold-document results in the first k."""
    relevant = [
        result.score
        for result in results[:k]
        if result.payload["doc_id"] == gold_doc
    ]
    if not relevant:
        return 0.0
    return sum(relevant) / len(relevant)
