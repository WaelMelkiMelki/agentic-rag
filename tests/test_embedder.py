"""Manual embedding/cache smoke demo; run with ``python tests/test_embedder.py``."""


def main():
    from agentic_rag.retrieval.embedder import CachedEmbedder

    embedder = CachedEmbedder()
    print("=== First run (should embed) ===")
    texts = [
        "The transformer uses self-attention mechanisms.",
        "RAG combines retrieval with generation.",
    ]
    vectors = embedder.embed(texts)
    print(f"Vector dimension: {len(vectors[0])}")
    print(f"First 5 values: {vectors[0][:5]}")

    print("\n=== Second run (should be cached) ===")
    vectors2 = embedder.embed(texts)
    print(f"Same vectors? {vectors[0][:5] == vectors2[0][:5]}")

    print("\n=== Query embedding ===")
    print(f"Query vector dimension: {len(embedder.embed_query('What is attention?'))}")


if __name__ == "__main__":
    main()
