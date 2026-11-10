"""Chunking goes through LangChain's RecursiveCharacterTextSplitter."""
from __future__ import annotations

import pytest

from agentic_rag.ingestion.chunk import Chunk, chunk_text


def _words(n: int) -> str:
    return " ".join(f"word{i}" for i in range(n))


def test_chunks_keep_the_stable_id_scheme():
    chunks = chunk_text(_words(1000), "attention_is_all_you_need", "https://arxiv.org/pdf/1706.03762")

    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert [chunk.chunk_id for chunk in chunks] == [
        f"attention_is_all_you_need__{i}" for i in range(len(chunks))
    ]
    assert all(chunk.doc_id == "attention_is_all_you_need" for chunk in chunks)
    assert all(chunk.source_url.endswith("1706.03762") for chunk in chunks)


def test_chunks_respect_the_word_budget():
    chunks = chunk_text(_words(2000), "doc", "url", chunk_size=500, overlap=80)
    assert len(chunks) == 5
    assert all(len(chunk.text.split()) <= 500 for chunk in chunks)


def test_overlap_is_applied_on_flowing_text():
    chunks = chunk_text(_words(1200), "doc", "url", chunk_size=500, overlap=80)
    first, second = chunks[0].text.split(), chunks[1].text.split()
    # The second chunk must start 80 words before the end of the first one.
    assert second[0] == first[500 - 80]


def test_paragraph_boundaries_are_preferred():
    text = "First paragraph about attention.\n\nSecond paragraph about retrieval.\n\nThird."
    chunks = chunk_text(text, "doc", "url", chunk_size=500, overlap=80)
    assert len(chunks) == 1
    assert "First paragraph" in chunks[0].text
    assert "Third." in chunks[0].text


def test_long_paragraphs_are_split_further():
    text = "\n\n".join(_words(300) for _ in range(6))
    chunks = chunk_text(text, "doc", "url", chunk_size=200, overlap=40)
    assert len(chunks) > 6
    assert all(len(chunk.text.split()) <= 200 for chunk in chunks)


def test_blank_text_returns_no_chunks():
    assert chunk_text("   \n\n  ", "doc", "url") == []


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-5, 0), (100, 100), (100, 120), (100, -1)])
def test_invalid_parameters_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text here", "doc", "url", chunk_size=size, overlap=overlap)
