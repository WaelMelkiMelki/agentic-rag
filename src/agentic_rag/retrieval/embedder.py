import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()


class CachedEmbedder:
    def __init__(self, cache_dir: str = "data/cache"):
        self.embedding_dimension = 768
        self.model = GoogleGenerativeAIEmbeddings(
            model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, text: str) -> str:
        """Create a unique filename from the text content."""
        return hashlib.md5(text.encode()).hexdigest()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts, reusing vectors saved in the disk cache."""
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
            batch_size = 50
            for start in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[start : start + batch_size]
                vectors = self.model.embed_documents(
                    batch, output_dimensionality=self.embedding_dimension
                )

                for j, vector in enumerate(vectors):
                    global_index = uncached_indices[start + j]
                    results[global_index] = vector
                    key = self._cache_key(uncached_texts[start + j])
                    (self.cache_dir / f"{key}.json").write_text(json.dumps(vector))
        else:
            print(f"All {len(texts)} embeddings cached!")

        if any(vector is None for vector in results):
            raise RuntimeError("Embedding provider returned fewer vectors than requested")

        return [vector for vector in results if vector is not None]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query."""
        return self.model.embed_query(
            query, output_dimensionality=self.embedding_dimension
        )
