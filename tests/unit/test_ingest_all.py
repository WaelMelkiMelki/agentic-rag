"""Ingestion must be able to start from a clean collection.

Chunk ids are ``{doc_id}__{index}``, so re-ingesting after a chunking change
only overwrites colliding ids: points with a higher index survive and pollute
retrieval. ``--recreate`` drops the collection first.
"""
from __future__ import annotations

from agentic_rag.ingestion import ingest_all as ingest_module


class FakeEmbedder:
    embedding_dimension = 384

    def embed(self, texts):
        return [[0.0] * 384 for _ in texts]

    def embed_query(self, query):
        return [0.0] * 384


class FakeStore:
    instances: list["FakeStore"] = []

    def __init__(self):
        self.recreated = 0
        self.created = 0
        self.upserted = 0
        FakeStore.instances.append(self)

    def recreate_collection(self, vector_size):
        self.recreated += 1

    def create_collection(self, vector_size):
        self.created += 1

    def upsert_chunks(self, chunks, vectors):
        self.upserted += len(chunks)

    def count(self):
        return self.upserted


def run_ingest(monkeypatch, tmp_path, recreate):
    FakeStore.instances = []
    monkeypatch.setattr(ingest_module, "FastEmbedder", FakeEmbedder)
    monkeypatch.setattr(ingest_module, "VectorStore", FakeStore)
    monkeypatch.setattr(ingest_module, "Path", lambda *_: tmp_path)  # empty data/raw
    monkeypatch.chdir(tmp_path)

    ingest_module.ingest_all(recreate=recreate)
    return FakeStore.instances[0]


def test_default_ingest_appends_to_the_existing_collection(monkeypatch, tmp_path):
    store = run_ingest(monkeypatch, tmp_path, recreate=False)

    assert (store.created, store.recreated) == (1, 0)


def test_recreate_ingest_drops_the_collection_first(monkeypatch, tmp_path):
    store = run_ingest(monkeypatch, tmp_path, recreate=True)

    assert (store.created, store.recreated) == (0, 1)
