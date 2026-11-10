"""``CachedEmbedder`` must work after a plain ``uv sync``.

``sentence-transformers`` lives in the optional ``ingestion`` extra, so the
eval scripts used to die with ``ModuleNotFoundError`` unless that extra was
installed. The embedder now degrades to FastEmbed/ONNX (a core dependency, and
the backend the API serves with) while keeping the same MiniLM model.
"""
from __future__ import annotations

import pytest

from agentic_rag.retrieval import embedder as embedder_module
from agentic_rag.retrieval import fast_embedder as fast_embedder_module


class FakeFastEmbedder:
    """Stands in for ``FastEmbedder`` without downloading an ONNX model."""

    model_name = embedder_module.DEFAULT_MODEL
    embedding_dimension = 384
    instances: list["FakeFastEmbedder"] = []

    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir
        FakeFastEmbedder.instances.append(self)

    def embed(self, texts):
        return [[0.1] * 384 for _ in texts]

    def embed_query(self, query):
        return [0.2] * 384


@pytest.fixture
def no_sentence_transformers(monkeypatch, tmp_path):
    monkeypatch.setattr(embedder_module, "SentenceTransformer", None)
    monkeypatch.setattr(fast_embedder_module, "FastEmbedder", FakeFastEmbedder)
    FakeFastEmbedder.instances = []
    return tmp_path


def test_embedder_falls_back_to_fastembed(no_sentence_transformers, capsys):
    embedder = embedder_module.CachedEmbedder(cache_dir=str(no_sentence_transformers))

    assert embedder.embedding_dimension == 384
    assert embedder.embed(["a", "b"]) == [[0.1] * 384] * 2
    assert embedder.embed_query("q") == [0.2] * 384

    output = capsys.readouterr().out
    assert "uv sync --extra ingestion" in output


def test_fallback_keeps_vectors_out_of_the_torch_cache(no_sentence_transformers):
    embedder = embedder_module.CachedEmbedder(cache_dir=str(no_sentence_transformers))

    embedder.embed(["a"])
    # FastEmbed has its own cache; writing JSON vectors here would mix backends.
    assert list(no_sentence_transformers.glob("*.json")) == []
    assert FakeFastEmbedder.instances[0].cache_dir.endswith("fastembed")


def test_non_default_model_still_requires_sentence_transformers(no_sentence_transformers):
    with pytest.raises(RuntimeError, match="ingestion"):
        embedder_module.CachedEmbedder(
            cache_dir=str(no_sentence_transformers),
            model_name="some/other-model",
        )
