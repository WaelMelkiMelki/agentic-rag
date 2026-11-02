from agentic_rag.ingestion.parse import parse_pdf
from agentic_rag.ingestion.chunk import chunk_text
from agentic_rag.retrieval.embedder import CachedEmbedder
from agentic_rag.retrieval.store import VectorStore


print("=== FULL PIPELINE TEST ===\n")

# Step 1: Parse
print("1. Parsing PDF...")
text = parse_pdf("data/raw/attention_is_all_you_need.pdf")
print(f"   -> {len(text)} chars\n")

# Step 2: Chunk
print("2. Chunking...")
chunks = chunk_text(
    text,
    doc_id="attention_is_all_you_need",
    source_url="https://arxiv.org/abs/1706.03762",
    chunk_size=500,
)
print(f"   -> {len(chunks)} chunks\n")

# Step 3: Embed
print("3. Embedding...")
embedder = CachedEmbedder()
vectors = embedder.embed([chunk.text for chunk in chunks])
print(f"   -> {len(vectors)} vectors, dim={len(vectors[0])}\n")

# Step 4: Store
print("4. Storing in Qdrant...")
store = VectorStore()
store.create_collection(vector_size=embedder.embedding_dimension)
store.upsert_chunks(chunks, vectors)
print(f"   -> Total chunks in DB: {store.count()}\n")

# Step 5: Search
print("5. Searching...\n")
queries = [
    "What is self-attention?",
    "How does the encoder work?",
    "What is positional encoding?",
]

for query in queries:
    query_vec = embedder.embed_query(query)
    results = store.search(query_vec, top_k=2)
    print(f"Query: '{query}'")
    for result in results:
        print(f"   Score: {result.score:.3f} | {result.payload['text'][:100]}...")
    print()
