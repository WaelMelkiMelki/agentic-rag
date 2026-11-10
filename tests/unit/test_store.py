import pytest

from agentic_rag.retrieval import store as store_module


class FakeQdrantClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_vector_store_passes_cloud_url_and_api_key(monkeypatch):
    monkeypatch.setattr(store_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(store_module, "QdrantClient", FakeQdrantClient)
    monkeypatch.setenv("QDRANT_URL", "https://cloud.example:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "test-api-key")
    monkeypatch.delenv("QDRANT_TIMEOUT", raising=False)

    store = store_module.VectorStore()

    assert store.url == "https://cloud.example:6333"
    assert store.client.kwargs == {
        "url": "https://cloud.example:6333",
        "api_key": "test-api-key",
        "timeout": 30.0,
    }


def test_vector_store_omits_api_key_for_local_qdrant(monkeypatch):
    monkeypatch.setattr(store_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(store_module, "QdrantClient", FakeQdrantClient)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    monkeypatch.delenv("QDRANT_TIMEOUT", raising=False)

    store = store_module.VectorStore()

    assert store.url == "http://localhost:6333"
    assert store.client.kwargs == {"url": "http://localhost:6333", "timeout": 30.0}


def test_qdrant_timeout_is_configurable(monkeypatch):
    monkeypatch.setattr(store_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(store_module, "QdrantClient", FakeQdrantClient)
    monkeypatch.setenv("QDRANT_TIMEOUT", "90")

    store = store_module.VectorStore()

    assert store.client.kwargs["timeout"] == 90.0


class RecordingClient:
    def __init__(self, exists=True):
        self.collection_exists_value = exists
        self.calls = []

    def collection_exists(self, name):
        self.calls.append(("exists", name))
        return self.collection_exists_value

    def delete_collection(self, collection_name):
        self.calls.append(("delete", collection_name))

    def create_collection(self, collection_name, vectors_config):
        self.calls.append(("create", collection_name, vectors_config.size))

    def count(self):
        return 0


def make_store(monkeypatch, exists=True):
    monkeypatch.setattr(store_module, "load_dotenv", lambda: None)
    client = RecordingClient(exists=exists)
    monkeypatch.setattr(store_module, "QdrantClient", lambda **kwargs: client)
    store_module.VectorStore.__init__.__globals__  # noqa: B018 - keep the module import obvious
    store = store_module.VectorStore()
    return store, client


def test_create_collection_is_a_no_op_when_it_exists(monkeypatch):
    store, client = make_store(monkeypatch)

    store.create_collection(vector_size=384)

    assert [call[0] for call in client.calls] == ["exists"]


def test_recreate_collection_drops_stale_points_before_creating(monkeypatch):
    store, client = make_store(monkeypatch)

    store.recreate_collection(vector_size=384)

    assert [call[0] for call in client.calls] == ["exists", "delete", "create"]
    assert client.calls[1] == ("delete", "agentic_rag_docs")
    assert client.calls[2] == ("create", "agentic_rag_docs", 384)


def test_recreate_collection_creates_when_absent(monkeypatch):
    store, client = make_store(monkeypatch, exists=False)

    store.recreate_collection(vector_size=384)

    assert [call[0] for call in client.calls] == ["exists", "create"]


# ---------------------------------------------------------------- writes ---
class FlakyClient:
    """Raises a transport error on the first upsert, then behaves."""

    def __init__(self, failures=1):
        self.failures = failures
        self.attempts = 0

    def upsert(self, **kwargs):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise RuntimeError("The write operation timed out")
        return True


def make_write_store(monkeypatch, client):
    monkeypatch.setattr(store_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(store_module, "QdrantClient", lambda **kwargs: client)
    monkeypatch.setattr(store_module.time, "sleep", lambda seconds: None)
    return store_module.VectorStore()


class Chunk:
    chunk_id = "paper__0"
    doc_id = "paper"
    text = "text"
    chunk_index = 0
    source_url = "https://example.com/paper"


def test_upsert_retries_once_after_a_stale_connection(monkeypatch, capsys):
    client = FlakyClient(failures=1)
    store = make_write_store(monkeypatch, client)

    store.upsert_chunks([Chunk()], [[0.1, 0.2]])

    assert client.attempts == 2
    assert "retrying on a fresh connection" in capsys.readouterr().out


def test_upsert_gives_up_after_the_single_retry(monkeypatch):
    client = FlakyClient(failures=2)
    store = make_write_store(monkeypatch, client)

    with pytest.raises(RuntimeError, match="timed out"):
        store.upsert_chunks([Chunk()], [[0.1, 0.2]])
    assert client.attempts == 2
