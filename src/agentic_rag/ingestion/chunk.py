from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    source_url: str


def _word_count(text: str) -> int:
    return len(text.split())


def chunk_text(
    text: str,
    doc_id: str,
    source_url: str,
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[Chunk]:
    """Split Markdown text into overlapping chunks of ``chunk_size`` words.

    Uses LangChain's ``RecursiveCharacterTextSplitter``, which prefers paragraph
    breaks, then line breaks, then spaces, so chunks follow the document's
    structure instead of being cut mid-sentence at a fixed word offset. The
    length function is a word counter, so the public contract (500 words with 80
    words of overlap) is unchanged and ``chunk_id`` keeps its stable
    ``{doc_id}__{index}`` form.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=_word_count,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=False,
    )

    chunks = []
    for idx, piece in enumerate(splitter.split_text(text)):
        piece = piece.strip()
        if not piece:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}__{idx}",
                doc_id=doc_id,
                text=piece,
                chunk_index=idx,
                source_url=source_url,
            )
        )

    return chunks
