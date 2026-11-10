import hashlib
import os
import time

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


#: Qdrant's REST server closes idle keep-alive connections after 5 s, while
#: httpx keeps them pooled. A request that arrives after an idle gap (embedding
#: 15 chunks takes longer than that) can be written into a half-closed socket
#: and die with ``WriteTimeout``. Writes are idempotent, so one retry on a fresh
#: connection is both safe and sufficient.
WRITE_ATTEMPTS = 2
WRITE_RETRY_DELAY = 0.5


def _timeout() -> float:
    """Client timeout in seconds; Qdrant's own default is 5 s."""
    try:
        return float(os.getenv("QDRANT_TIMEOUT", "30"))
    except ValueError:
        return 30.0


class VectorStore:
    def __init__(self, url: str | None = None, api_key: str | None = None):
        load_dotenv()
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        configured_api_key = api_key or os.getenv("QDRANT_API_KEY")
        if configured_api_key:
            self.client = QdrantClient(url=self.url, api_key=configured_api_key, timeout=_timeout())
        else:
            self.client = QdrantClient(url=self.url, timeout=_timeout())
        self.collection = "agentic_rag_docs"

    def _write(self, description: str, operation, **kwargs):
        """Run an idempotent write, retrying once if the connection is stale."""
        for attempt in range(1, WRITE_ATTEMPTS + 1):
            try:
                return operation(**kwargs)
            except Exception as error:  # noqa: BLE001 - transport errors are opaque
                if attempt == WRITE_ATTEMPTS:
                    raise
                print(f"{description} failed ({error}); retrying on a fresh connection")
                time.sleep(WRITE_RETRY_DELAY)

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

    def recreate_collection(self, vector_size: int) -> None:
        """Drop and re-create the collection so stale points cannot survive.

        Chunk ids are derived from the splitter, so after a chunking change an
        upsert only overwrites the ids that happen to collide: every point with
        a higher index would linger and pollute retrieval (and the chunk count).
        """
        if vector_size <= 0:
            raise ValueError("vector_size must be greater than zero")

        if self.client.collection_exists(self.collection):
            self.client.delete_collection(collection_name=self.collection)
            print(f"Dropped collection '{self.collection}'")
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
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

        self._write(
            f"Upsert of {len(points)} chunks",
            self.client.upsert,
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
