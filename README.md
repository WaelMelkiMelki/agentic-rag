# Agentic RAG

An agentic retrieval-augmented QA app with a FastAPI API and a Streamlit UI.
The API exposes a plain RAG baseline (B1) and the self-correcting LangGraph
agent (B4).

## 🚀 Live Demo

| Service | URL |
|---------|-----|
| **Chat UI** | [waelmelki-agentic-rag.streamlit.app](https://waelmelki-agentic-rag.streamlit.app/) |
| **API** | [agentic-rag-1jii.onrender.com](https://agentic-rag-1jii.onrender.com) |
| **API Docs** | [agentic-rag-1jii.onrender.com/docs](https://agentic-rag-1jii.onrender.com/docs) |

> ⚠️ Free-tier hosting: the first request may take 30–60 seconds to wake the server.

## Run locally with Docker Compose

Requirements: Docker Compose, `uv`, and Python 3.12 for host-side ingestion.

1. Copy `.env.example` to `.env` and add your Gemini and OpenRouter keys.
2. If the Phase 2 `qdrant` container is running, stop it first so Compose can
   bind port 6333. Its `qdrant_storage` volume is reused by Compose.
3. Start the API and Qdrant:

   ```bash
   docker compose up --build -d
   curl http://localhost:8000/health
   ```

   A new Qdrant volume reports zero documents until you ingest the corpus.
   The API docs are at `http://localhost:8000/docs`.

4. Run the Streamlit UI in another terminal:

   ```bash
   uv run --extra ui streamlit run ui/app.py
   ```

   Open `http://localhost:8501`. The UI uses `API_URL` from the environment or
   Streamlit secrets; it defaults to `http://localhost:8000`.

Ingest PDFs with the same FastEmbed ONNX model used by the API:

```bash
QDRANT_URL=http://localhost:6333 uv run python -m agentic_rag.ingestion.ingest_all
```

The older `CachedEmbedder` implementation still uses Sentence Transformers;
install the optional `ingestion` extra only if you use that legacy embedder.

Stop the containers with `docker compose down`. The named Qdrant volume remains
so the collection persists.

## LLM routing (Ollama-first locally, Gemini/OpenRouter on Render)

`LLMRouter` tries providers in order: **Ollama phi3 → Ollama qwen2.5 →
Gemini → OpenRouter**. Ollama models are **local-only**; on Render set
`OLLAMA_ENABLED=false` so production uses **Gemini (`gemini-3.8-flash`)
then OpenRouter (`openrouter/free`)** only. If Ollama is unreachable it
is skipped for the rest of the process and the router falls back to
Gemini, then OpenRouter.

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_ENABLED` | `true` | Set to `false` on Render to skip Ollama (Gemini + OpenRouter only) |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible endpoint |
| `OLLAMA_API_KEY` | `ollama` | Dummy key (Ollama ignores it, ChatOpenAI requires one) |
| `OLLAMA_PHI_MODEL` | `phi3:latest` | Primary Ollama model |
| `OLLAMA_QWEN_MODEL` | `qwen2.5:latest` | Secondary Ollama model |

For local Docker Ollama-first, start the optional Ollama service and pull
the models:

```bash
docker compose --profile ollama up -d
docker compose exec ollama ollama pull phi3:latest
docker compose exec ollama ollama pull qwen2.5:latest
```

## Cloud deployment

- **API:** Create a Render Docker web service from the repository root using
  `Dockerfile`. Set `GEMINI_API_KEY`, `OPENROUTER_API_KEY`,
  `OPENROUTER_BASE_URL`, `QDRANT_URL`, and `QDRANT_API_KEY` in Render's
  environment settings. Point Qdrant variables at your cloud cluster.
  Ollama is local-only: set `OLLAMA_ENABLED=false` in Render so
  production uses Gemini + OpenRouter only (no Ollama calls).
- **UI:** Create a Streamlit Community Cloud app with `ui/app.py` as the entry
  point and `ui/requirements.txt` as its dependency file. Set this secret in
  the app settings:

  ```toml
  API_URL = "https://<your-render-service>.onrender.com"
  ```

- **Weekly health checks:** Add `QDRANT_URL`, `QDRANT_API_KEY`, and
  `RENDER_API_URL` as GitHub Actions repository secrets.
  `.github/workflows/keepalive.yml` checks Qdrant and Render every Monday.
  The weekly Render check does not prevent the free service from sleeping.
