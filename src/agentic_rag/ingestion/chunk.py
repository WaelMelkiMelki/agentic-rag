from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    source_url: str


def chunk_text(
    text: str,
    doc_id: str,
    source_url: str,
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[Chunk]:
    """Split text into overlapping chunks by word count."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for idx, start in enumerate(range(0, len(words), step)):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}__{idx}",
                doc_id=doc_id,
                text=" ".join(chunk_words),
                chunk_index=idx,
                source_url=source_url,
            )
        )

    return chunks
