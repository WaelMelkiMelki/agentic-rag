# Phase 7 — FastAPI and Streamlit UI

## Delivered

- Added cached API dependencies for the Gemini/OpenRouter router, usage tracker,
  local Sentence Transformers embedder, Qdrant store, B4 agent, and B1 baseline.
- Added `GET /health` and validated `POST /query` routes. The query route returns
  answer, citations, steps, status, stop reason, selected system, and latency.
- Made `SimpleRAG` accept injected dependencies and added an API-shaped `query`
  method while preserving its existing `ask` interface.
- Added a Streamlit chat UI that uses HTTP calls to the API and offers B1/B4,
  citations, agent steps, status, stop reason, and latency.
- Changed the manual `tests/test_*.py` demos to run only when invoked directly.
  Pytest collection no longer triggers live provider requests, model loading, or
  Qdrant writes.

The embedder intentionally uses the project's existing local
`all-MiniLM-L6-v2` model. The existing Qdrant collection has 384-dimensional
vectors, so changing to the example's `text-embedding-004` would make queries
incompatible with the ingested collection.

## Validation

- `from agentic_rag.api.deps import get_router, get_agent, get_baseline` imports
  successfully and prints `OK`.
- `pytest -q`: **17 passed**. API tests use fakes and make no provider or Qdrant
  calls.
- Live API startup succeeded. `GET /health` returned
  `{"status":"ok","qdrant_docs":77}` and `/docs` returned HTTP 200.
- Streamlit `AppTest` rendered without exceptions and submitted one mocked B4
  request through the UI's HTTP client.

The local model printed its normal Hugging Face unauthenticated-request warning
at API startup. The model was cached and startup completed. The real chat
provider path was not used for the UI smoke check: Gemini's free-tier quota and
OpenRouter's free-model daily quota were exhausted during this session. Use
provider quota resets or an available paid/key-backed model to verify a live
answer in the browser.

## Run locally

Start Qdrant, then run these in separate terminals from the repository root:

```bash
uv run uvicorn agentic_rag.api.main:app --app-dir src --reload --port 8000
uv run streamlit run ui/app.py
```

The API docs are at `http://localhost:8000/docs`; the UI is at
`http://localhost:8501`.
