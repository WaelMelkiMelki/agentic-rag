"""Manual ingestion and Qdrant smoke demo; run with ``python tests/test_full_pipeline.py``."""


def main():
    from agentic_rag.ingestion.chunk import chunk_text
    from agentic_rag.ingestion.parse import parse_pdf
    from agentic_rag.retrieval.embedder import CachedEmbedder
    from agentic_rag.retrieval.store import VectorStore

    print("=== FULL PIPELINE TEST ===\n")
    print("1. Parsing PDF...")
    text = parse_pdf("data/raw/attention_is_all_you_need.pdf")
    print(f"   -> {len(text)} chars\n")

    print("2. Chunking...")
    chunks = chunk_text(
        text,
        doc_id="attention_is_all_you_need",
        source_url="https://arxiv.org/abs/1706.03762",
        chunk_size=500,
    )
    print(f"   -> {len(chunks)} chunks\n")

    print("3. Embedding...")
    embedder = CachedEmbedder()
    vectors = embedder.embed([chunk.text for chunk in chunks])
    print(f"   -> {len(vectors)} vectors, dim={len(vectors[0])}\n")

    print("4. Storing in Qdrant...")
    store = VectorStore()
    store.create_collection(vector_size=embedder.embedding_dimension)
    store.upsert_chunks(chunks, vectors)
    print(f"   -> Total chunks in DB: {store.count()}\n")

    print("5. Searching...\n")
    for query in [
        "What is self-attention?",
        "How does the encoder work?",
        "What is positional encoding?",
    ]:
        results = store.search(embedder.embed_query(query), top_k=2)
        print(f"Query: '{query}'")
        for result in results:
            print(f"   Score: {result.score:.3f} | {result.payload['text'][:100]}...")
        print()


if __name__ == "__main__":
    main()
