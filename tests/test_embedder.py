from agentic_rag.retrieval.embedder import CachedEmbedder


embedder = CachedEmbedder()

print("=== First run (should embed) ===")
vectors = embedder.embed(
    [
        "The transformer uses self-attention mechanisms.",
        "RAG combines retrieval with generation.",
    ]
)
print(f"Vector dimension: {len(vectors[0])}")
print(f"First 5 values: {vectors[0][:5]}")

print("\n=== Second run (should be cached) ===")
vectors2 = embedder.embed(
    [
        "The transformer uses self-attention mechanisms.",
        "RAG combines retrieval with generation.",
    ]
)
print(f"Same vectors? {vectors[0][:5] == vectors2[0][:5]}")

print("\n=== Query embedding ===")
q_vec = embedder.embed_query("What is attention?")
print(f"Query vector dimension: {len(q_vec)}")
