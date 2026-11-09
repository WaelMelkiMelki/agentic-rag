# Agentic RAG

A self-correcting document QA agent that retrieves, generates, and **verifies its own
answers** against source documents — built on three arXiv ML papers.

The API serves two systems side by side so they can be compared directly:

* **B1 — Plain RAG:** retrieve → generate.
* **B4 — Agentic RAG:** a LangGraph state machine that plans, grades retrieved
  chunks, generates, checks every claim against the sources, and rewrites the query
  when grounding fails.

## 🚀 Live Demo

| Service | URL |
|---------|-----|
| **Chat UI** | [waelmelki-agentic-rag.streamlit.app](https://waelmelki-agentic-rag.streamlit.app/) |
| **API** | [agentic-rag-1jii.onrender.com](https://agentic-rag-1jii.onrender.com) |
| **API Docs** | [agentic-rag-1jii.onrender.com/docs](https://agentic-rag-1jii.onrender.com/docs) |
| **Demo Video** | _coming soon_ |

> ⚠️ Free-tier hosting: the first request can take 30–60 s while Render wakes the
> container. Long agent queries can take another 30–60 s.

## 📊 Results

Retrieval was measured on the 12 answerable questions in `evals/dataset/dev_set.jsonl`
against the 77-chunk collection.

| System | Hit@5 | MRR | Answer quality | Abstention | LLM calls / query |
|--------|-------|-----|----------------|------------|-------------------|
| Closed-book (B0) | — | — | __ / 5 | — | 1 |
| Plain RAG (B1) | 0.92 (11/12) | 0.81 | __ / 5 | __ % | 1 |
| Full Agent (B4) | __ | __ | __ / 5 | __ % | ~8 |

**Key finding:** _to be filled after the B0/B1/B4 ablation run — e.g. “the agent raised
abstention accuracy on unanswerable questions from __% to __% at ~8× the LLM cost.”_

Known gaps (see [Evaluation Report](docs/eval_report.md)): the B1 answer-quality run
stopped early on Gemini's daily free-tier quota, so the three recorded judge scores
(5, 3, 4) are a partial sample and refusal rates are not yet established.

## 🏗 Architecture

```mermaid
graph LR
    User([User]) --> UI[Streamlit UI]
    UI --> API[FastAPI<br/>/query]
    API --> B1[Plain RAG B1]
    API --> B4[LangGraph Agent B4]

    B4 --> Init[Initialize]
    Init --> Plan[Plan]
    Plan --> Retrieve[Retrieve]
    Retrieve --> Grade[Grade chunks]
    Grade -->|relevant| Generate[Generate]
    Grade -->|irrelevant| Rewrite[Rewrite query]
    Rewrite --> Retrieve
    Generate --> Ground[Grounding check]
    Ground -->|grounded| Finalize[Finalize]
    Ground -->|unsupported claims| Rewrite
    Plan -->|out of scope| Abstain[Abstain]

    Retrieve --> Embed[FastEmbed<br/>MiniLM 384-d]
    Embed --> Qdrant[(Qdrant Cloud)]
    B1 --> Embed

    Plan --> Router[LLM Router]
    Grade --> Router
    Generate --> Router
    Ground --> Router
    Rewrite --> Router

    Router --> Ollama[Ollama<br/>phi3 → qwen2.5<br/>local only]
    Router --> Gemini[Gemini<br/>gemini-3.8-flash]
    Router --> OpenRouter[OpenRouter<br/>openrouter/free]

    API --> Langfuse[Langfuse]
```

Full component breakdown: [docs/architecture.md](docs/architecture.md).

## 🤖 LLM provider priority

| Environment | Order |
|---|---|
| **Local** | Ollama `phi3:latest` → Ollama `qwen2.5:latest` → Gemini `gemini-3.8-flash` → OpenRouter `openrouter/free` |
| **Deployed** | Gemini `gemini-3.8-flash` → OpenRouter `openrouter/free` |

Ollama is **local-only**: free tunneling services (Cloudflare Quick Tunnel, ngrok
free tier) block API traffic with **403** responses, so Render runs with
`OLLAMA_ENABLED=false`.

Failure caching keeps the agent's ~8–10 calls per query from paying repeated
timeouts: a **connection** error skips both local models for the remaining
process lifetime, and a Gemini quota (429) error skips Gemini. A local
**timeout** only affects that call — it usually means phi3 was still loading,
so the next query retries Ollama instead of silently going cloud-only.

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_ENABLED` | `true` | Set `false` on Render to skip Ollama |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible endpoint |
| `OLLAMA_API_KEY` | `ollama` | Dummy key (Ollama ignores it, ChatOpenAI requires it) |
| `OLLAMA_PHI_MODEL` | `phi3:latest` | Primary local model |
| `OLLAMA_QWEN_MODEL` | `qwen2.5:latest` | Secondary local model |
| `OLLAMA_TIMEOUT` | `120` | Seconds per local call (phi3 loading on first use is slow) |
| `GEMINI_API_KEY` | — | Gemini key |
| `OPENROUTER_API_KEY` | — | OpenRouter key |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter endpoint |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | — | Optional tracing; disabled if unset |

## 🛠 Tech Stack

* **LLM routing:** 4-tier fallback — Ollama `phi3:latest`, Ollama `qwen2.5:latest`,
  Gemini `gemini-3.8-flash`, OpenRouter `openrouter/free`
* **Agent framework:** LangGraph (typed `AgentState`, conditional edges, retry loops)
* **Embeddings:** FastEmbed ONNX — `sentence-transformers/all-MiniLM-L6-v2`, 384-dim
* **Vector DB:** Qdrant Cloud (free tier, cosine distance)
* **API:** FastAPI + Uvicorn, Pydantic v2 request/response models
* **UI:** Streamlit
* **Observability:** Langfuse (per-query traces, optional at runtime)
* **LLM SDKs:** `langchain-openai`, `langchain-google-genai`
* **Ingestion:** PyMuPDF4LLM → chunking → embed → upsert
* **Deployment:** Docker → Render (API), Streamlit Community Cloud (UI), GitHub Actions keep-alive
* **Testing:** pytest with fake LLM/router injection; live injection suite behind `-m integration`

## 📁 Project Structure

```
agentic-rag/
├── src/agentic_rag/
│   ├── agent/          # LangGraph agent (initialize → plan → retrieve → grade → generate → ground → finalize)
│   │   ├── prompts.py  # GENERATE_PROMPT shared by B1 and B4
│   │   └── rag.py      # SimpleRAG baseline (B1)
│   ├── llm/            # LLMRouter (phi3 → qwen2.5 → Gemini → OpenRouter) + usage tracker
│   ├── retrieval/      # FastEmbed embedder + Qdrant vector store
│   ├── ingestion/      # PDF download → parse → chunk → embed pipeline
│   ├── api/            # FastAPI app + singleton deps
│   └── tracing.py      # Optional Langfuse tracing (fail-safe, version-tolerant)
├── ui/                 # Streamlit chat interface
├── evals/              # Retrieval metrics, LLM judge, ablation harness, dev set
├── tests/              # unit (offline, fakes) + integration (API, UI, live injection)
├── configs/            # Corpus manifest
└── docs/               # Architecture, eval report, learning log, phase reports
```

## 🚀 Quick Start

### Local (with Ollama)

```bash
git clone https://github.com/WaelMelkiMelki/agentic-rag.git
cd agentic-rag
cp .env.example .env          # add GEMINI_API_KEY / OPENROUTER_API_KEY / QDRANT_*

uv sync
ollama pull phi3:latest
ollama pull qwen2.5:latest

docker compose up -d qdrant
python -m agentic_rag.ingestion.ingest_all

uvicorn agentic_rag.api.main:app --reload      # http://localhost:8000/docs
uv run --extra ui streamlit run ui/app.py      # http://localhost:8501
```

Leave `OLLAMA_ENABLED` at its default (`true`) so local queries use Ollama first and
only fall back to the cloud.

### Docker Compose (API + Qdrant + optional Ollama)

```bash
docker compose --profile ollama up --build -d
docker compose exec ollama ollama pull phi3:latest
docker compose exec ollama ollama pull qwen2.5:latest
curl http://localhost:8000/health
```

### Deployed configuration (Render)

Set `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `QDRANT_URL`,
`QDRANT_API_KEY`, optional `LANGFUSE_*`, and **`OLLAMA_ENABLED=false`**.

## 🧪 Tests

```bash
pytest -q                                   # offline suite, no keys needed
pytest -m integration tests/integration/test_injection.py -v   # live LLM injection tests
```

Offline tests inject fake LLMs and fake API dependencies. The injection suite makes
real LLM calls, is marked `integration`, and skips itself when no credentials or no
reachable Qdrant collection is present.

## 📖 Documentation

* [Architecture](docs/architecture.md)
* [Evaluation Report](docs/eval_report.md)
* [Learning Log](docs/learning_log.md)
* [Phase 8 Report](docs/phase8_report.md)
* [Corpus manifest](configs/corpus_manifest.yaml)
