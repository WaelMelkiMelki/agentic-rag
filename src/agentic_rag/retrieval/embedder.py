import hashlib
import json
from pathlib import Path

try:  # optional 'ingestion' extra; the API uses FastEmbed instead
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - depends on the installed extras
    SentenceTransformer = None

#: Model served by both backends, so cached vectors stay interchangeable.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class CachedEmbedder:
    #ok
    def __init__(
        self,
        cache_dir: str = "data/cache",
        model_name: str = DEFAULT_MODEL,
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._delegate = False
        if SentenceTransformer is None:
            if model_name != DEFAULT_MODEL:
                raise RuntimeError(
                    "sentence-transformers is not installed; install the 'ingestion' "
                    f"extra to use the '{model_name}' embedding model"
                )
            # FastEmbed is a core dependency (it is what the API serves with) and
            # runs the same MiniLM model through ONNX, so evals and ingestion
            # still work after a plain `uv sync`. Only the JSON cache is lost.
            from agentic_rag.retrieval.fast_embedder import FastEmbedder

            print(
                "sentence-transformers is not installed (the 'ingestion' extra); "
                "falling back to FastEmbed/ONNX with the same MiniLM model. "
                "Run `uv sync --extra ingestion` for the cached torch embedder."
            )
            self.model = FastEmbedder(cache_dir=str(self.cache_dir / "fastembed"))
            self.embedding_dimension = self.model.embedding_dimension
            self._delegate = True
            return
        self.model = SentenceTransformer(model_name)
        self.embedding_dimension = self.model.get_embedding_dimension()
        if self.embedding_dimension is None:
            raise RuntimeError("Embedding model did not provide its vector dimension")

    def _cache_key(self, text: str) -> str:
        """Create a unique filename from the text content."""
        value = f"{self.model_name}\0{text}".encode()
        return hashlib.md5(value).hexdigest()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts, reusing vectors saved in the disk cache."""
        if self._delegate:
            return self.model.embed(texts)
        results: list[list[float] | None] = [None] * len(texts)
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        for i, text in enumerate(texts):
            cache_file = self.cache_dir / f"{self._cache_key(text)}.json"
            if cache_file.exists():
                results[i] = json.loads(cache_file.read_text())
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            print(
                f"Embedding {len(uncached_texts)} new texts "
                f"({len(texts) - len(uncached_texts)} cached)..."
            )
            batch_size = 32
            for start in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[start : start + batch_size]
                vectors = self.model.encode(
                    batch,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )

                for j, vector in enumerate(vectors):
                    global_index = uncached_indices[start + j]
                    vector_list = vector.tolist()
                    results[global_index] = vector_list
                    key = self._cache_key(uncached_texts[start + j])
                    (self.cache_dir / f"{key}.json").write_text(json.dumps(vector_list))
        else:
            print(f"All {len(texts)} embeddings cached!")

        if any(vector is None for vector in results):
            raise RuntimeError("Embedding provider returned fewer vectors than requested")

        return [vector for vector in results if vector is not None]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query."""
        if self._delegate:
            return self.model.embed_query(query)
        return self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()
