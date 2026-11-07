"""Small ONNX embedder for API deployments with tight memory limits."""
from __future__ import annotations

import os


class FastEmbedder:
    """Embed text with the same 384-dimensional MiniLM model via ONNX Runtime."""

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension = 384

    def __init__(self, cache_dir: str | None = None):
        from fastembed import TextEmbedding

        cache_dir = cache_dir or os.getenv("FASTEMBED_CACHE_DIR")
        self.model = TextEmbedding(model_name=self.model_name, cache_dir=cache_dir)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return document vectors in input order."""
        if not texts:
            return []
        return [vector.tolist() for vector in self.model.embed(texts)]

    def embed_query(self, query: str) -> list[float]:
        """Embed one query with the same path used for document vectors."""
        return self.embed([query])[0]
