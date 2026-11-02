from pathlib import Path

from agentic_rag.ingestion.chunk import chunk_text
from agentic_rag.ingestion.parse import parse_pdf
from agentic_rag.retrieval.embedder import CachedEmbedder
from agentic_rag.retrieval.store import VectorStore


def ingest_all() -> None:
    embedder = CachedEmbedder()
    store = VectorStore()
    store.create_collection(vector_size=embedder.embedding_dimension)

    raw_dir = Path("data/raw")
    total_chunks = 0

    for pdf in sorted(raw_dir.glob("*.pdf")):
        doc_id = pdf.stem
        print(f"\n{'=' * 50}")
        print(f"Processing: {doc_id}")
        print(f"{'=' * 50}")

        text = parse_pdf(str(pdf))
        print(f"   Parsed: {len(text)} chars")

        chunks = chunk_text(
            text,
            doc_id=doc_id,
            source_url=f"https://arxiv.org/abs/{doc_id}",
            chunk_size=500,
        )
        print(f"   Chunked: {len(chunks)} chunks")

        vectors = embedder.embed([chunk.text for chunk in chunks])
        store.upsert_chunks(chunks, vectors)
        total_chunks += len(chunks)

    print(f"\nDone! Total chunks in database: {store.count()}")


if __name__ == "__main__":
    ingest_all()
