# Phase 8 — Docker and Free Deployment

## Local implementation status

The local Docker and deployment preparation is implemented. The API image
builds, the Compose stack starts, and the API health endpoint responds when
Qdrant is available.

### Implemented

- Added a multi-stage `Dockerfile` with a non-root runtime user, configurable
  `PORT`, a container health check, and a preloaded FastEmbed ONNX model.
- Added `docker-compose.yml` for the API and persistent local Qdrant storage.
- Added `.dockerignore` and the Streamlit Cloud dependency file.
- Added configurable Qdrant URL and API-key handling.
- Switched the API to FastEmbed's ONNX runtime while keeping the same
  `sentence-transformers/all-MiniLM-L6-v2` vectors used by ingestion. A local
  comparison produced 384-dimensional embeddings and cosine similarity 1.0
  against Sentence Transformers on three sample texts.
- Kept Sentence Transformers as an optional dependency for the legacy embedder
  and Streamlit as an optional UI dependency, reducing the API image's packages.
- Updated the Streamlit UI to read `API_URL` from environment variables or
  Streamlit secrets and to report a waking API separately from an unreachable
  or degraded API.
- Added a weekly Qdrant Cloud ping workflow. It does not ping Render because
  Render's free service sleeps after inactivity; a weekly request would not
  prevent that behavior.
- Updated the README with local and cloud setup instructions.

## Verification

- `uv lock --check` — passed.
- `docker compose config --quiet` — passed.
- `ty check` on the changed API, embedder, store, and Streamlit files — passed.
- `.venv/bin/pytest -q` — 25 passed after the B4 initialization, Qdrant
  configuration, and ingestion-import regression tests.
- `docker build -t agentic-rag-api .` — passed.
- Ran Compose with a dedicated temporary Qdrant volume and alternate host
  ports; `/health` returned `{"status":"ok","qdrant_docs":0}` after creating
  an empty 384-dimensional smoke-test collection.
- API container memory at idle was approximately 315 MiB in this environment.
- Removed the temporary containers, network, and test volume after the smoke
  test. Existing Qdrant data was not used or modified.

## Follow-up local B4 verification

The first direct `get_agent().invoke({"question": ...})` check exposed that the
graph expected `router_calls` and `steps` to be present in the input. The graph
now initializes those bookkeeping fields, so callers can pass only a question.
The regression test and full suite pass (25 tests total).

With the existing local 77-point Qdrant collection, a real B4 invocation using
OpenRouter returned a cited `answered_unverified` response. This check skipped
Gemini because its 120-second request timeout made the multiple B4 model calls
too slow after Gemini returned 504 errors. B1 was also exercised through the
API using the normal Gemini-to-OpenRouter fallback and returned a cited answer.

## Ingestion crash follow-up

The reported segmentation fault was reproduced while importing PyTorch from
`CachedEmbedder`; each PDF parsed successfully on its own. Ingestion now uses
the API's FastEmbed ONNX implementation, avoiding the crashing PyTorch import
while retaining the same 384-dimensional MiniLM vectors. A local parse/chunk/
embed smoke check processed all three PDFs and embedded 77 chunks successfully;
it did not write to Qdrant. The Qdrant API key pasted during troubleshooting
should be revoked and replaced before cloud ingestion.

## Cloud deployment still needs account setup

No Qdrant Cloud cluster, Render service, or Streamlit Community Cloud app was
created or verified in this environment. Those checklist items require the
account-owned services and their dashboard configuration. The code is ready
for those values, but there are no live URLs to report yet.

Before deploying, create and populate the Qdrant Cloud collection with the
existing ingestion command. Configure the API keys and Qdrant URL in Render,
set `API_URL` in Streamlit Cloud secrets, and add the Qdrant URL and API key to
GitHub Actions secrets. Then verify `/health` and `/query` through the deployed
services and add the verified URLs to the README. Do not describe the phase as
fully deployed until those checks pass.

The Render Free service currently has a 512 MiB memory limit. The measured
container idle use was about 315 MiB, leaving limited headroom for requests;
monitor peak memory during a live query before relying on that plan.

