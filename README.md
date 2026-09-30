# arXiv Ingest → Store → Visualize → Ask (RAG)

A small backend pipeline: pull papers from the live arXiv API, clean them, store them in PostgreSQL (pgvector), serve aggregated stats for charting, and answer natural-language questions about the papers with a grounded RAG pipeline (Claude).

**Stack:** Python 3.12 · FastAPI · PostgreSQL 16 + pgvector · SQLAlchemy 2 (async) · Alembic · Anthropic Claude (Haiku 4.5) · fastembed (local ONNX embeddings) · BM25 + concept graph · Docker Compose

## Quick start

```bash
cp .env.example .env          # then set ANTHROPIC_API_KEY in .env
docker compose up --build     # db + api; migrations run automatically
```

The API is on <http://localhost:8000> (interactive docs at `/docs`). The image is slim (~800 MB, no PyTorch) and builds in a few minutes.

### Ingest data

Either through the API or the CLI (both run the same pipeline):

```bash
# CLI (inside the running container) - defaults come from .env
docker compose exec api python -m scripts.ingest
docker compose exec api python -m scripts.ingest --from 2026-06-01 --to 2026-06-30 --categories cs.AI,cs.CL

# API
curl -X POST localhost:8000/api/v1/ingest/run -H 'content-type: application/json' \
     -d '{"date_from":"2026-06-01","date_to":"2026-06-30","categories":["cs.CL"]}'
curl localhost:8000/api/v1/ingest/status
```

With the defaults (`cs.AI,cs.LG,cs.CL`, 2026-01-01 → 2026-09-29, `ARXIV_MAX_PAPERS=3500`) a run fetches ~3.5k papers spread evenly across the date range (the cap is split into a per-category-per-week quota) in roughly 10–15 minutes, because requests are throttled to one per 3.5 s as arXiv asks.

**Re-running is safe and is also the update mechanism.** Papers are upserted by `arxiv_id`; rows are only rewritten when arXiv reports a newer `updated` timestamp. After the fetch the pipeline re-embeds only new/changed papers (or papers embedded with a different model), then rebuilds the BM25 index and concept graph.

**Wipe everything** (papers, authors, discovered categories, concepts, embeddings, BM25 index, graph):

```bash
docker compose exec api python -m scripts.ingest --reset
# or
curl -X POST localhost:8000/api/v1/ingest/reset -H 'content-type: application/json' \
     -d '{"confirmation_token":"<RESET_CONFIRMATION_TOKEN>"}'
```

### Tests

```bash
docker compose run --rm --no-deps -v ./tests:/app/tests -v ./pytest.ini:/app/pytest.ini api python -m pytest -q
```

## Configuration (`.env`)

See `.env.example` for every variable. The important ones:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | LLM for answers, query rewriting, faithfulness check and concept extraction |
| `LLM_MODEL` | default `claude-haiku-4-5-20251001` |
| `EMBEDDING_PROVIDER` | `local` (default, no key) · `openai` · `azure` |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | default MiniLM-L6-v2 / 384. Changing provider usually changes the dimension → `docker compose down -v` to recreate the vector column |
| `AZURE_OPENAI_*` | optional: Azure embeddings (`EMBEDDING_PROVIDER=azure`, set `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` and `EMBEDDING_DIM=1536`) or chat fallback when no Anthropic key is set (`AZURE_OPENAI_CHAT_DEPLOYMENT`) |
| `RAG_SIMILARITY_THRESHOLD` | best cosine similarity below this ⇒ "no match" (default 0.30) |
| `ARXIV_*`, `INGEST_DATE_*` | categories, date range, paper cap, throttle |

Anthropic has no embeddings endpoint, so embeddings are local by default (or OpenAI/Azure if you prefer).

## Architecture

```
arXiv Atom API ──► ArxivClient (throttle, retry/backoff, weekly windows, paginated)
                      │
                      ▼
               parser.py ──► cleaner/normalizer (LaTeX/HTML stripped, UTC dates, null doi/journal_ref ok)
                      │
                      ▼  idempotent bulk upserts + per-window checkpoints
   PostgreSQL: papers · authors · paper_authors · categories · paper_categories
               (+ papers.embedding vector(384), ingestion_runs, concepts, paper_concepts)
                      │
        ┌─────────────┴───────────────┐
        ▼                             ▼
  /api/v1/stats/*            post-ingest: embeddings → BM25 → concept graph (Claude)
  (SQL aggregates)                       │
                                         ▼
  POST /api/v1/rag/ask:  dense (pgvector) + BM25 + graph-hop → RRF fusion
        → similarity gate → Claude answer with [arXiv:ID] citations → faithfulness check
```

### Schema

- `papers(arxiv_id PK, title, abstract, abstract_clean, published_date, updated_date, doi, journal_ref, primary_category_code → categories, embedding vector, embedding_model, embedding_updated_at, created_at, updated_at)`
- `authors(id, name, name_normalized UNIQUE)` ↔ `paper_authors(paper_id, author_id, position)`
- `categories(code PK, display_name)` ↔ `paper_categories(paper_id, category_code, is_primary)`
- `ingestion_runs` (status, counters, `last_checkpoint` JSONB for resume), `concepts` / `paper_concepts` (concept graph)

Migrations: `alembic/versions/0001…0005` (run on container start; reproducible from an empty DB).

## API

### Visualization — `GET /api/v1/stats/...` (response values below are illustrative)

| Endpoint | Returns |
|---|---|
| `/stats/summary` | totals, date range, top 5 categories |
| `/stats/top-categories?limit=10&from=&to=` | top N categories by paper count |
| `/stats/papers-by-category-over-time?granularity=month\|year\|week&category=&from=&to=` | category × period counts |
| `/stats/top-authors?limit=20&from=&to=` | top N authors by paper count |
| `/stats/publication-velocity?granularity=week` | papers per period |
| `/stats/category-cooccurrence?limit=20` | category pairs appearing together |

```bash
curl "localhost:8000/api/v1/stats/top-categories?limit=3"
```
```json
{"data":[{"code":"cs.AI","name":"Artificial Intelligence","count":1204},
         {"code":"cs.LG","name":"Machine Learning","count":1011},
         {"code":"cs.CL","name":"Computation and Language","count":987}],
 "meta":{"total":3,"limit":3,"generated_at":"2026-09-30T17:25:39Z"}}
```
```bash
curl "localhost:8000/api/v1/stats/papers-by-category-over-time?granularity=month&category=cs.LG"
# {"data":[{"period":"2026-08","category":"cs.LG","count":318}, ...]}
```

### RAG — `POST /api/v1/rag/ask`, `GET /api/v1/rag/search`

```bash
curl -X POST localhost:8000/api/v1/rag/ask -H 'content-type: application/json' \
     -d '{"question":"How does LoRA reduce the number of trainable parameters?","top_k":5}'
```
```json
{"answer":"LoRA freezes the pretrained weights and trains low-rank adapter matrices ... [arXiv:2609.00008]",
 "sources":[{"arxiv_id":"2609.00008","title":"LoRA: low-rank adaptation of large language models",
             "similarity_score":0.51,"retrieval_source":"dense","published_date":"2026-09-08",
             "url":"https://arxiv.org/abs/2609.00008","concept_overlap":null}],
 "confidence":"medium","iterations":1,"faithfulness_passed":true,
 "model_used":"claude-haiku-4-5-20251001","retrieval_ms":930,"llm_ms":1800}
```

**No good match** is handled without calling the LLM, so nothing can be hallucinated:

```bash
curl -X POST localhost:8000/api/v1/rag/ask -H 'content-type: application/json' \
     -d '{"question":"What is the airspeed velocity of an unladen swallow?"}'
```
```json
{"answer":"I couldn't find papers in the ingested arXiv dataset that answer this question, so I can't give a grounded answer.",
 "sources":[],"confidence":"none","iterations":1,"faithfulness_passed":null,"llm_ms":0, "...":"..."}
```

`stream: true` returns Server-Sent Events (`retrieval`, `answer`, `faithfulness`, `done`). LLM rate limit / outage map to HTTP 429 / 503.

`GET /api/v1/rag/search?q=low-rank adapters&top_k=5` is retrieval only (no LLM).

### Ingestion — `POST /ingest/run`, `GET /ingest/status`, `POST /ingest/reset`

Health: `GET /health`, `GET /ready` (DB reachable + paper/embedding counts).

## How RAG works

1. Embed the question (abstract + title were embedded at ingest); retrieve top candidates by **pgvector cosine**, **BM25**, and a **concept-graph hop** (Claude extracts concepts per paper at ingest), fuse with **Reciprocal Rank Fusion**.
2. **Similarity gate:** candidates whose cosine similarity to the question is below `RAG_SIMILARITY_THRESHOLD` are dropped. If nothing is left, optionally rewrite the query once with the LLM and retry, else return the honest "no match" answer.
3. Claude answers **only from the retrieved abstracts**, citing `[arXiv:ID]`; a second call checks the answer for unsupported claims (`faithfulness_passed`).

## Known limitations

- Only arXiv metadata + abstracts are indexed (no full text).
- Concept extraction needs `ANTHROPIC_API_KEY`; without it the graph leg is skipped (dense + BM25 still work).
- Similarity thresholds are tuned for MiniLM / `text-embedding-3-small` and were calibrated on a small sample; different embedding models need different `RAG_*_THRESHOLD` values.
- arXiv rate-limits by IP. After many rapid retries it can return HTTP 429 for several minutes; the client backs off (honouring `Retry-After`) and the run is marked `failed` with its checkpoint saved.
- Ingestion runs as an in-process background task (single API worker); no job queue.
- The faithfulness check is best-effort and does not regenerate the answer on failure.
- Reset keeps seeded category names and removes categories discovered during ingestion.

## What I'd improve with more time

- Real job queue (and resumable runs across API restarts), incremental graph updates.
- Cross-encoder reranking and LLM relevance grading of retrieved abstracts.
- Retrieval/answer evaluation set (`scripts/evaluate.py` has a lightweight F1 / citation-recall harness) and calibrated thresholds.
- Integration tests against a throwaway Postgres in CI; auth and rate limiting on the public endpoints.
- Query analyzer (multi-query / HyDE) node and conversational follow-ups via `session_id`.
