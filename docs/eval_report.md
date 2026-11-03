# Evaluation Report

## B1 Baseline (SimpleRAG)
Date: 2026-11-03
Models: `gemini-3.8-flash` (configured generator; Gemini quota exhausted, so OpenRouter fallback generated the first four answers), `openrouter/free` (judge)
Corpus: 3 papers; Qdrant collection contained 77 points at retrieval evaluation time (below the approximately 500 chunk target)

### Retrieval Metrics
| Metric | Result | Target | Pass? |
|--------|--------|--------|-------|
| Hit@5  | 0.92 (11/12) | ≥ 0.70 | ✅ |
| MRR    | 0.81 | ≥ 0.50 | ✅ |

### Answer Quality
| Metric | Result | Target | Pass? |
|--------|--------|--------|-------|
| Avg judge score | Incomplete: 3 scores recorded (5, 3, 4); full-set average unavailable | ≥ 3.5 | — |
| Refusal rate | Not evaluated (run stopped at q05 before refusal checks) | ≥ 80% | — |

### Observations
- What worked well: Retrieval found a gold document in the top five for 11 of 12 answerable questions; MRR was 0.81.
- What failed: One retrieval miss (q09). Answer evaluation could not complete: Gemini calls hit the daily free-tier quota, and the OpenRouter judge request for q05 failed. The three completed judge scores are only a partial sample.
- Biggest weakness: Answer-quality and refusal performance are not established by this run. The evaluator's refusal check also only recognizes the exact phrase “cannot answer,” so it may undercount equivalent refusals.

### Run notes
- `uv run python evals/run_retrieval_eval.py` completed on the 12 answerable dev-set questions.
- `uv run python evals/run_answer_eval.py` started but exited with an OpenRouter request error during q05 after recording scores for q01–q04 only partially (q04 score was 4; total completed scores: q01=5, q02=3, q04=4). No refusal checks ran.
- `src/agentic_rag/llm/router.py` currently configures `gemini-3.8-flash`, not the Gemini 2.0 Flash model named in the task template. Generator output used OpenRouter fallback for the completed questions because Gemini was over quota.
