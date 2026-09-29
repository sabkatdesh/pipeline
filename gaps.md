# Phase 2 — Data Ingestion: Gaps vs Spec

## Gap 1 — `current_week` missing from `GET /ingest/status` response

**File:** `app/schemas/paper.py:21`

The spec ([docs/TECHNICAL_SPEC.md](docs/TECHNICAL_SPEC.md), section 5.2) includes `"current_week": "2026-03-09"` in the status response body. The `IngestStatus` schema has no such field, so callers polling for progress cannot tell which date window is currently being processed.

**Fix:** Add `current_week: str | None` to `IngestStatus` and populate it from `run.last_checkpoint["current"]["date_window"]` in the `get_status` handler.

---

## Gap 2 — Race condition in `POST /ingest/run`

**File:** `app/api/v1/ingestion.py:66`

`db.flush()` is called to obtain the new `run_id`, but the session is not committed until after the handler returns (the `get_db` dependency commits on cleanup). `asyncio.create_task()` schedules the background pipeline immediately — it can begin executing before that commit lands, meaning the pipeline's first `get_checkpoint` query may not find the run row, and subsequent `UPDATE ingestion_runs` calls silently update 0 rows.

**Fix:** Call `await db.commit()` explicitly after `await db.flush()`, before `asyncio.create_task(...)`.

---

## Gap 3 — `IngestRequest` date defaults are hardcoded, not env-driven

**File:** `app/schemas/paper.py:7-8`

The schema hardcodes `date_from = "2026-01-01"` and `date_to = "2026-09-29"` as Pydantic field defaults. The spec states these should default to the `INGEST_DATE_FROM` / `INGEST_DATE_TO` environment variables. Anyone who sets different dates in `.env` will find the API endpoint ignoring them.

**Fix:** Remove the hardcoded defaults from the Pydantic model and resolve them inside the `run_ingestion` handler using `cfg.ingest_date_from` / `cfg.ingest_date_to` as fallbacks when the request body omits the fields.

---

## Gap 4 — Checkpoint not saved before propagating arxiv fetch failure

**File:** `app/ingestion/arxiv_client.py` / `app/ingestion/pipeline.py`

The spec ([docs/TECHNICAL_SPEC.md](docs/TECHNICAL_SPEC.md), section 7) states: "After 5 attempts → save checkpoint → raise". The `ArxivClient` raises `RuntimeError` after exhausting retries; the pipeline's outer `except` block marks the run as `failed` but does not call `chk.save_progress()` first. The last successfully completed batch's checkpoint is intact, but the in-progress batch offset is lost.

**Fix:** In `IngestionPipeline._process_window`, wrap the `client.fetch()` call in a try/except and call `await chk.save_progress(...)` with the current offset before re-raising on unrecoverable fetch failure.
