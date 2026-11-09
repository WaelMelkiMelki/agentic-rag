# Architecture

## Components

| Layer | Module | Responsibility |
|---|---|---|
| UI | `ui/app.py` | Streamlit chat, B1/B4 toggle, citations + agent-step expanders |
| API | `src/agentic_rag/api/main.py` | `GET /health`, `POST /query`, Langfuse tracing |
| Wiring | `src/agentic_rag/api/deps.py` | Singletons: router, embedder, store, agent, baseline |
| Baseline (B1) | `src/agentic_rag/agent/rag.py` | Retrieve → generate, no self-correction |
| Agent (B4) | `src/agentic_rag/agent/graph.py` | LangGraph state machine with self-correction |
| Prompts | `src/agentic_rag/agent/prompts.py` | `GENERATE_PROMPT`, shared by B1 and B4 |
| Retrieval | `src/agentic_rag/retrieval/` | FastEmbed (ONNX) embedder + Qdrant vector store |
| Ingestion | `src/agentic_rag/ingestion/` | PDF download → parse → chunk → embed → upsert |
| LLM routing | `src/agentic_rag/llm/router.py` | 4-tier fallback with per-process failure caching |
| Observability | `src/agentic_rag/tracing.py` | Optional Langfuse trace per query, fail-safe |
| Evals | `evals/` | Retrieval metrics (Hit@k, MRR), LLM judge, ablation harness |

## Request flow (B4)

1. `POST /query {question, system}` reaches FastAPI.
2. `get_agent()` returns the compiled LangGraph graph (built once via `lru_cache`).
3. The graph runs: `initialize → plan → retrieve → grade → (generate → ground) → finalize`,
   with `rewrite → retrieve` loops bounded by `max_rounds=2` and a
   `max_router_calls=10` budget. Out-of-budget or unrecoverable states route to `abstain`.
4. Every LLM call (`plan`, `grade`, `generate`, `ground`, `rewrite`) goes through
   `LLMRouter.invoke_with_fallback()`.
5. The final state exposes `final_answer`, `citations`, `steps`, `status`
   (`answered` | `answered_unverified` | `abstained`) and `stop_reason`.
6. `trace_agent()` records the invocation in Langfuse when credentials are set.

## LLM routing

| Environment | Order |
|---|---|
| Local (`OLLAMA_ENABLED=true`, default) | Ollama `phi3:latest` → Ollama `qwen2.5:latest` → Gemini `gemini-3.8-flash` → OpenRouter `openrouter/free` |
| Render (`OLLAMA_ENABLED=false`) | Gemini `gemini-3.8-flash` → OpenRouter `openrouter/free` |

Failure caching keeps the agent's ~8–10 calls per query cheap:

* the first Ollama *connection* failure sets `_ollama_unavailable`, skipping both
  local models for the rest of the process; a timeout (`OLLAMA_TIMEOUT`, default
  120s) does not, because it usually means the model is still loading;
* a Gemini 429/quota error sets `_gemini_quota_exhausted`;
* `invoke_with_fallback()` returns a plain string and raises `RuntimeError`
  only when every provider fails.

Cloud clients are constructed with a `"missing-key"` placeholder so an
Ollama-only setup (local laptop, no cloud keys) still starts.

## Storage

* **Qdrant** collection with 384-dimension cosine vectors, populated from the
  3 arXiv PDFs in `configs/corpus_manifest.yaml` (77 points at evaluation time).
* **FastEmbed** (`sentence-transformers/all-MiniLM-L6-v2`, ONNX) is used by the API;
  `CachedEmbedder` (Sentence Transformers, optional `ingestion` extra) is the legacy path.

## Deployment

* **API:** Docker image on Render. `OLLAMA_ENABLED=false` — free tunneling
  services return 403 on API traffic, so Ollama is local-only.
* **UI:** Streamlit Community Cloud, pointing at the Render API through `API_URL`.
* **Vector DB:** Qdrant Cloud (eu-west-1).
* **Observability:** Langfuse Cloud, enabled by `LANGFUSE_*` env vars.
* **Keep-alive:** `.github/workflows/keepalive.yml` pings Qdrant and Render weekly.

## Testing strategy

* **Offline** (`pytest -q`): fakes for router/LLM, fake deps for the API, in-memory
  state tests for every graph node. No network, no credentials.
* **Live** (`pytest -m integration`): prompt-injection suite in
  `tests/integration/test_injection.py`. Skips itself when credentials or Qdrant
  are unavailable.
