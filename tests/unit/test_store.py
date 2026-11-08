from agentic_rag.retrieval import store as store_module


class FakeQdrantClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_vector_store_passes_cloud_url_and_api_key(monkeypatch):
    monkeypatch.setattr(store_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(store_module, "QdrantClient", FakeQdrantClient)
    monkeypatch.setenv("QDRANT_URL", "https://cloud.example:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "test-api-key")

    store = store_module.VectorStore()

    assert store.url == "https://cloud.example:6333"
    assert store.client.kwargs == {
        "url": "https://cloud.example:6333",
        "api_key": "test-api-key",
    }


def test_vector_store_omits_api_key_for_local_qdrant(monkeypatch):
    monkeypatch.setattr(store_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(store_module, "QdrantClient", FakeQdrantClient)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)

    store = store_module.VectorStore()

    assert store.url == "http://localhost:6333"
    assert store.client.kwargs == {"url": "http://localhost:6333"}
