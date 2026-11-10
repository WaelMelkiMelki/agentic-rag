# Architecture

## Components

| Layer | Module | Responsibility |
|---|---|---|
| UI | `ui/app.py` | Streamlit chat, B1/B4 toggle, citations + agent-step expanders |
| API | `src/agentic_rag/api/main.py` | `GET /health`, `POST /query`, Langfuse tracing |
| Wiring | `src/agentic_rag/api/deps.py` | Singletons: router, embedder, store, agent, baseline |
| Closed-book (B0) | `src/agentic_rag/agent/closed_book.py` | One retrieval-free call — the ablation floor |
| Baseline (B1) | `src/agentic_rag/agent/rag.py` | Retrieve → generate, no self-correction |
| Agent (B4) | `src/agentic_rag/agent/graph.py` | LangGraph state machine with self-correction |
| Prompts | `src/agentic_rag/agent/{prompts,plan,grade,ground,rewrite}.py` | `ChatPromptTemplate` objects (`GENERATE_CHAT_PROMPT` etc.), rendered text identical to the original prompt strings; `GENERATE_CHAT_PROMPT` is shared by B1 and B4 |
| Retrieval | `src/agentic_rag/retrieval/` | FastEmbed (ONNX) embedder, Qdrant vector store, and `ChunkRetriever` (LangChain `BaseRetriever` → `Document`s) |
| Ingestion | `src/agentic_rag/ingestion/` | PDF download → PyMuPDF4LLM parse → `RecursiveCharacterTextSplitter` chunking → embed → upsert |
| Memory | `src/agentic_rag/agent/memory.py` | Bounded per-session `BaseChatMessageHistory` + follow-up condensing (`session_id`) |
| Parsing | `src/agentic_rag/agent/parsing.py` | LangChain output parsers: planner JSON, yes/no verdict, claim report |
| Chains | `src/agentic_rag/llm/chains.py` | LCEL `prompt \| router \| StrOutputParser()` used by every node |
| LLM routing | `src/agentic_rag/llm/router.py` | 4-tier fallback with per-process failure caching |
| Observability | `src/agentic_rag/tracing.py` | Optional Langfuse: one trace per query (`trace_agent`) plus the LangChain `CallbackHandler` bound to every chain, fail-safe |
| Evals | `evals/` | Retrieval metrics (Hit@k, MRR), LLM judge, ablation harness |

## Request flow (B4)

1. `POST /query {question, system}` reaches FastAPI.
2. `get_agent()` returns the compiled LangGraph graph (built once via `lru_cache`).
3. The graph runs: `initialize → plan → retrieve → grade → (generate → ground) → finalize`,
   with `rewrite → retrieve` loops bounded by `max_rounds=2` and a
   `max_router_calls=10` budget. Out-of-budget or unrecoverable states route to `abstain`.
4. Every LLM call (`plan`, `grade`, `generate`, `ground`, `rewrite`) is an LCEL chain:
   `ChatPromptTemplate | LLMRouter.invoke_with_fallback | StrOutputParser()`. The router
   accepts a string, chat messages, or a `ChatPromptValue` and always returns text.
5. Retrieval goes through `ChunkRetriever.invoke(query)` (LangChain `BaseRetriever`),
   so it participates in callbacks and returns `Document`s; the graph converts them
   back to lightweight chunks that keep `chunk_id` for citations.
6. Structured replies are read by LangChain output parsers (`agent/parsing.py`):
   `PydanticOutputParser` for the planner (fence-tolerant), `VerdictParser` for grading,
   `ClaimsParser` for the claim-by-claim grounding report.
7. `finalize` enforces the evidence trail: `[chunk_id]` markers are matched against the
   chunks retrieved for *this* question, unmatched ones are stripped from the text, and
   a draft with no valid marker borrows the fact-checker's attribution when the grounding
   check passed. Anything else abstains with `invalid_citations` — and the rejected draft
   stays visible in the response (`draft_answer`) instead of being discarded.
8. The final state exposes `final_answer`, `citations`, `steps`, `status`
   (`answered` | `answered_unverified` | `abstained`) and `stop_reason`.
9. When the request carries a `session_id`, the follow-up is first rewritten into a
   standalone question using the last turns (`agent/memory.py`), then the result is
   appended to that session's history (6 turns, LRU-bounded, in process memory).
10. `trace_agent()` records the invocation in Langfuse when credentials are set, and
   the LangChain callback handler streams the individual steps as spans.

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
* **CI:** `.github/workflows/tests.yml` runs `pytest -q -m "not integration"`, a
  whitespace check, and a no-credentials import smoke test.
* **Keep-alive:** `.github/workflows/keepalive.yml` pings Qdrant and Render weekly.

## Testing strategy

* **Baselines:** B0 (closed-book), B1 (plain RAG) and B4 (agent) share the same API
  surface; `evals/run_ablation.py` scores all three on the dev set.
* **Offline** (`pytest -q`): fakes for router/LLM, fake deps for the API, in-memory
  state tests for every graph node. No network, no credentials.
* **Live** (`pytest -m integration`): prompt-injection suite in
  `tests/integration/test_injection.py`. Skips itself when credentials or Qdrant
  are unavailable.
