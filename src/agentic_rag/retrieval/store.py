import hashlib
import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class VectorStore:
    def __init__(self, url: str | None = None, api_key: str | None = None):
        load_dotenv()
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.client = QdrantClient(
            url=self.url,
            api_key=api_key or os.getenv("QDRANT_API_KEY") or None,
        )
        self.collection = "agentic_rag_docs"

    def create_collection(self, vector_size: int = 768) -> None:
        """Create the document chunk collection when it does not exist."""
        if vector_size <= 0:
            raise ValueError("vector_size must be greater than zero")

        if self.client.collection_exists(self.collection):
            print(f"Collection '{self.collection}' already exists")
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        print(f"Created collection '{self.collection}' (dim={vector_size})")

    def upsert_chunks(self, chunks, vectors) -> None:
        """Insert or update document chunks in the vector store."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        points = []
        for chunk, vector in zip(chunks, vectors):
            point_id = int(hashlib.md5(chunk.chunk_id.encode()).hexdigest(), 16) % (
                2**63
            )
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "text": chunk.text,
                        "chunk_index": chunk.chunk_index,
                        "source_url": chunk.source_url,
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection,
            points=points,
        )
        print(f"Upserted {len(points)} chunks")

    def search(self, query_vector: list[float], top_k: int = 5):
        """Find the most similar chunks to a query vector."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
        )
        return results.points

    def count(self) -> int:
        """Return the total number of stored chunks."""
        info = self.client.get_collection(self.collection)
        return int(info.points_count or 0)
