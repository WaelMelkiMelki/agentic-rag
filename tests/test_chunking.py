"""Manual chunk-size experiment; run with ``python tests/test_chunking.py``."""


def main():
    from pathlib import Path

    from agentic_rag.ingestion.chunk import chunk_text

    text = Path("data/processed/attention_is_all_you_need.md").read_text(encoding="utf-8")
    print("=== Chunk Size Experiment ===\n")
    for size in [200, 500, 1000]:
        chunks = chunk_text(text, "attention", "url", chunk_size=size)
        avg_chars = sum(len(chunk.text) for chunk in chunks) / len(chunks)
        print(f"Size {size:>4}: {len(chunks):>3} chunks | avg {avg_chars:>6.0f} chars/chunk")

    print("\n=== Sample Chunk (size=500) ===")
    chunks = chunk_text(text, "attention", "url", chunk_size=500)
    print(f"Chunk 0 ID: {chunks[0].chunk_id}")
    print(f"Chunk 0 text (first 200 chars): {chunks[0].text[:200]}...")
    print(f"\nChunk 1 text (first 200 chars): {chunks[1].text[:200]}...")
    print("\n👆 Notice how chunk 1 starts with words from the end of chunk 0? That's the overlap.")


if __name__ == "__main__":
    main()
