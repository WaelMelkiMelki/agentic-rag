# Agentic RAG

An agentic retrieval-augmented QA app with a FastAPI API and a Streamlit UI.
The API exposes a plain RAG baseline (B1) and the self-correcting LangGraph
agent (B4).

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

To ingest PDFs with the existing Sentence Transformers pipeline, install its
optional dependency and point it at local Qdrant:

```bash
uv sync --extra ingestion
QDRANT_URL=http://localhost:6333 uv run python -m agentic_rag.ingestion.ingest_all
```

Stop the containers with `docker compose down`. The named Qdrant volume remains
so the collection persists.

## Cloud deployment

- **API:** Create a Render Docker web service from the repository root using
  `Dockerfile`. Set `GEMINI_API_KEY`, `OPENROUTER_API_KEY`,
  `OPENROUTER_BASE_URL`, `QDRANT_URL`, and `QDRANT_API_KEY` in Render's
  environment settings. Point Qdrant variables at your cloud cluster.
- **UI:** Create a Streamlit Community Cloud app with `ui/app.py` as the entry
  point and `ui/requirements.txt` as its dependency file. Set this secret in
  the app settings:

  ```toml
  API_URL = "https://<your-render-service>.onrender.com"
  ```

- **Qdrant keep-alive:** Add `QDRANT_URL` and `QDRANT_API_KEY` as GitHub Actions
  repository secrets. `.github/workflows/keepalive.yml` pings the cluster
  weekly; it does not keep the Render service awake.

Live service links will be added after the cloud services are created and
verified.
