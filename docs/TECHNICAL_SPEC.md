# arXiv Intelligence Pipeline
## Technical Specification

---

## 1. System Overview

| Property | Value |
|---|---|
| Language | Python 3.12 |
| API framework | FastAPI 0.115+ |
| Database | PostgreSQL 16 + pgvector extension |
| ORM | SQLAlchemy 2.0 (fully async) |
| Migrations | Alembic |
| Validation | Pydantic v2 |
| Embedding provider | OpenAI `text-embedding-3-small` (default) OR `sentence-transformers/all-MiniLM-L6-v2` (local) |
| LLM for RAG | Anthropic Claude (`claude-haiku-4-5-20251001` default, `claude-sonnet-5` optional) |
| RAG orchestration | LangGraph 0.2+ (state machine with typed state, conditional edges) |
| LLM framework | LangChain 0.3+ (LangChain-Anthropic, LangChain-OpenAI) |
| Sparse retrieval | rank-bm25 (in-memory BM25 index, serialized to disk) |
| Reranker | sentence-transformers cross-encoder `ms-marco-MiniLM-L-6-v2` |
| Concept graph | NetworkX (in-memory, serialized to disk as pickle/JSON) |
| RAG evaluation | RAGAS (faithfulness, context precision, context recall, answer relevancy) |
| Entity extraction | LLM-based (Claude Haiku) during ingestion post-processing |
| Containerization | Docker + docker-compose |
| Test runner | pytest + pytest-asyncio + httpx |
| Logging | structlog (JSON output) |

---

## 2. Environment Variables

```
# .env.example

# Database
DATABASE_URL=postgresql+asyncpg://pipeline:password@db:5432/pipeline
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20

# Embedding
EMBEDDING_PROVIDER=openai          # openai | local
OPENAI_API_KEY=sk-...              # required if EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536                 # 1536 for openai, 384 for local

# LLM
ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL=claude-haiku-4-5-20251001
LLM_MAX_TOKENS=1024

# RAG settings
RAG_TOP_K=5
RAG_SIMILARITY_THRESHOLD=0.65     # below this = "no relevant papers"
RAG_MAX_RETRY_ITERATIONS=3        # CRAG: max query rewrites before giving up
RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RAG_BM25_INDEX_PATH=/app/data/bm25_index.pkl
RAG_GRAPH_PATH=/app/data/concept_graph.pkl

# Ingestion
ARXIV_RATE_LIMIT_SECONDS=3.5
ARXIV_BATCH_SIZE=500              # papers per API request (max 2000)
ARXIV_CATEGORIES=cs.AI,cs.LG,cs.CL
ARXIV_MAX_PAPERS=3500             # hard cap — never fetch more than this
INGEST_DATE_FROM=2026-01-01
INGEST_DATE_TO=2026-09-29
EMBEDDING_BATCH_SIZE=100          # papers per embedding API call
CONCEPT_EXTRACTION_BATCH_SIZE=20  # papers per LLM concept-extraction call

# API
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
RESET_CONFIRMATION_TOKEN=my-secret-reset-token  # guard against accidental wipes
```

---

## 3. Database Schema

### 3.1 papers

```sql
CREATE TABLE papers (
    arxiv_id             TEXT PRIMARY KEY,
    title                TEXT NOT NULL,
    abstract             TEXT NOT NULL,
    abstract_clean       TEXT NOT NULL,
    published_date       TIMESTAMPTZ NOT NULL,
    updated_date         TIMESTAMPTZ NOT NULL,
    doi                  TEXT,
    journal_ref          TEXT,
    primary_category_code TEXT NOT NULL REFERENCES categories(code),
    embedding            vector(1536),           -- NULL until embedded; dim matches EMBEDDING_DIM
    embedding_model      TEXT,
    embedding_updated_at TIMESTAMPTZ,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_papers_published_date     ON papers(published_date);
CREATE INDEX idx_papers_primary_category   ON papers(primary_category_code);
CREATE INDEX idx_papers_updated_date       ON papers(updated_date);
-- IVFFlat approximate nearest-neighbor index (built after bulk ingest)
-- lists = sqrt(row_count), roughly 370 for 135k papers
CREATE INDEX idx_papers_embedding ON papers
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 370);
```

### 3.2 authors

```sql
CREATE TABLE authors (
    id               BIGSERIAL PRIMARY KEY,
    name             TEXT NOT NULL,
    name_normalized  TEXT NOT NULL UNIQUE    -- lowercase, unicode-normalized, whitespace-stripped
);

CREATE INDEX idx_authors_normalized ON authors(name_normalized);
```

### 3.3 paper_authors

```sql
CREATE TABLE paper_authors (
    paper_id   TEXT    NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    author_id  BIGINT  NOT NULL REFERENCES authors(id),
    position   SMALLINT NOT NULL,            -- 0 = first/corresponding author
    PRIMARY KEY (paper_id, author_id)
);

CREATE INDEX idx_paper_authors_paper_id  ON paper_authors(paper_id);
CREATE INDEX idx_paper_authors_author_id ON paper_authors(author_id);
```

### 3.4 categories

```sql
CREATE TABLE categories (
    code         TEXT PRIMARY KEY,           -- e.g. "cs.AI"
    display_name TEXT                        -- e.g. "Artificial Intelligence"
);
-- Pre-seeded with all known arXiv CS subcategories via migration
```

### 3.5 paper_categories

```sql
CREATE TABLE paper_categories (
    paper_id      TEXT    NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    category_code TEXT    NOT NULL REFERENCES categories(code),
    is_primary    BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (paper_id, category_code)
);

CREATE INDEX idx_paper_categories_code       ON paper_categories(category_code);
CREATE INDEX idx_paper_categories_paper_id   ON paper_categories(paper_id);
CREATE INDEX idx_paper_categories_primary    ON paper_categories(category_code) WHERE is_primary = TRUE;
```

### 3.6 concepts

```sql
CREATE TABLE concepts (
    id            BIGSERIAL PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,          -- e.g. "LoRA", "RLHF", "FlashAttention"
    concept_type  TEXT                           -- method | model | dataset | task | other
);

CREATE INDEX idx_concepts_name ON concepts(name);
```

### 3.7 paper_concepts

```sql
CREATE TABLE paper_concepts (
    paper_id    TEXT   NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    concept_id  BIGINT NOT NULL REFERENCES concepts(id),
    PRIMARY KEY (paper_id, concept_id)
);

CREATE INDEX idx_paper_concepts_concept_id ON paper_concepts(concept_id);
CREATE INDEX idx_paper_concepts_paper_id   ON paper_concepts(paper_id);
```

### 3.8 ingestion_runs

```sql
CREATE TABLE ingestion_runs (
    id               BIGSERIAL PRIMARY KEY,
    started_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at     TIMESTAMPTZ,
    status           TEXT NOT NULL,          -- running | completed | failed | cancelled
    date_from        DATE NOT NULL,
    date_to          DATE NOT NULL,
    categories       TEXT[] NOT NULL,
    papers_fetched   INT NOT NULL DEFAULT 0,
    papers_inserted  INT NOT NULL DEFAULT 0,
    papers_updated   INT NOT NULL DEFAULT 0,
    papers_embedded  INT NOT NULL DEFAULT 0,
    last_checkpoint  JSONB,                  -- {"week_start": "2026-03-02", "offset": 500}
    error_message    TEXT
);
```

---

## 4. Project File Map

```
pipeline/
│
├── alembic/                         # Database migration system
│   ├── versions/                    # One file per schema change
│   │   └── (generated by alembic)
│   ├── env.py                       # Alembic runtime config, connects to DATABASE_URL
│   └── script.py.mako               # Template for new migration files
│
├── app/
│   ├── main.py                      # FastAPI app factory, mounts routers, lifespan events
│   │
│   ├── api/
│   │   └── v1/
│   │       ├── ingestion.py         # POST /ingest/run, POST /ingest/reset, GET /ingest/status
│   │       ├── stats.py             # GET /stats/* (6 visualization endpoints)
│   │       └── rag.py               # POST /ask, GET /search (semantic search bonus)
│   │
│   ├── core/
│   │   ├── config.py                # pydantic-settings: reads .env, typed settings object
│   │   ├── database.py              # async engine, sessionmaker, get_db dependency
│   │   └── logging.py               # structlog setup, JSON formatter
│   │
│   ├── ingestion/
│   │   ├── arxiv_client.py          # async HTTP client (httpx), rate limiter, retry logic
│   │   ├── parser.py                # Atom/XML → Pydantic RawPaper models
│   │   ├── pipeline.py              # orchestrator: date-chunked loop, calls client→parser→db
│   │   └── checkpoint.py            # read/write last_checkpoint in ingestion_runs table
│   │
│   ├── models/                      # SQLAlchemy ORM (declarative, mapped_column style)
│   │   ├── paper.py                 # Paper model with vector column
│   │   ├── author.py                # Author model
│   │   ├── category.py              # Category model
│   │   └── ingestion_run.py         # IngestionRun model
│   │
│   ├── preprocessing/
│   │   ├── cleaner.py               # Abstract text cleanup (LaTeX, HTML entities, unicode)
│   │   └── normalizer.py            # Date normalization, author name normalization
│   │
│   ├── embeddings/
│   │   ├── provider.py              # Abstract base + OpenAI impl + LocalSentenceTransformer impl
│   │   └── indexer.py               # Fetches un-embedded papers, batches, stores vectors
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── langgraph_pipeline.py    # StateGraph: nodes wired with conditional edges
│   │   ├── retriever.py             # HybridRetriever: pgvector + BM25 + graph; RRF fusion
│   │   ├── prompt.py                # Prompt templates for each LangGraph node
│   │   ├── answerer.py              # LangChain-Anthropic answer generation
│   │   ├── nodes/
│   │   │   ├── __init__.py
│   │   │   ├── query_analyzer.py    # classify, decompose, 3 variants, HyDE document
│   │   │   ├── retriever_node.py    # invokes hybrid retriever, populates state.candidates
│   │   │   ├── reranker.py          # cross-encoder: top-20 → top-5
│   │   │   ├── doc_grader.py        # CRAG: grade each doc; set state.all_irrelevant flag
│   │   │   ├── compressor.py        # extract relevant sentences, reduces LLM context noise
│   │   │   ├── generator.py         # LLM answer with mandatory citation
│   │   │   └── faithfulness.py      # check answer claims against source docs
│   │   ├── graph/
│   │   │   ├── __init__.py
│   │   │   ├── builder.py           # LLM concept extraction + NetworkX graph construction
│   │   │   ├── traverser.py         # graph hop retrieval by shared concept lookup
│   │   │   └── models.py            # ConceptNode, ConceptEdge dataclasses
│   │   └── evaluation/
│   │       ├── __init__.py
│   │       └── ragas_eval.py        # RAGAS: load test set, score, print report
│   │
│   └── schemas/                     # Pydantic v2 request/response models (not ORM models)
│       ├── paper.py                 # PaperOut, PaperSearchResult
│       ├── stats.py                 # CategoryCount, TimeSeriesPoint, StatsResponse
│       └── rag.py                   # AskRequest, AskResponse, Source, ConfidenceLevel
│
├── tests/
│   ├── conftest.py                  # Shared fixtures: test DB, mock arxiv server, mock LLM
│   ├── unit/
│   │   ├── test_parser.py           # XML parsing edge cases
│   │   ├── test_cleaner.py          # Abstract cleanup correctness
│   │   └── test_retriever.py        # Similarity threshold logic, no-match detection
│   └── integration/
│       ├── test_ingest_pipeline.py  # Full ingest with mocked arXiv → verify DB state
│       ├── test_stats_api.py        # All 6 stat endpoints with seeded DB
│       └── test_rag_api.py          # /ask with mocked embeddings + mocked LLM
│
├── scripts/
│   └── ingest.py                    # CLI entry point: python -m scripts.ingest [options]
│
├── docs/
│   ├── PROBLEM_AND_STRATEGY.md      # Non-technical overview
│   └── TECHNICAL_SPEC.md            # This file
│
├── docker-compose.yml               # db (pgvector) + api services
├── Dockerfile                       # Multi-stage: builder → runtime
├── .env.example                     # All env vars with safe placeholder values
├── .gitignore                       # .env, __pycache__, .venv, *.pyc, vectors/
├── alembic.ini                      # Points alembic at DATABASE_URL
├── requirements.txt                 # Pinned dependencies
└── README.md                        # Setup, run, sample requests, schema overview
```

---

## 5. API Contract

### 5.1 Health & Readiness

```
GET /health
→ 200 { "status": "ok" }

GET /ready
→ 200 { "status": "ready", "papers_count": 134821, "embedded_count": 134821 }
→ 503 { "status": "not_ready", "reason": "database unreachable" }
```

### 5.2 Ingestion

```
POST /api/v1/ingest/run
Body: {
  "date_from": "2026-01-01",       // optional, defaults to INGEST_DATE_FROM env
  "date_to":   "2026-09-29",       // optional
  "categories": ["cs.*"]           // optional
}
→ 202 {
  "run_id": 1,
  "status": "running",
  "message": "Ingestion started in background"
}
→ 409 { "detail": "Ingestion already running (run_id: 1)" }

GET /api/v1/ingest/status
→ 200 {
  "run_id": 1,
  "status": "running",
  "papers_fetched": 45200,
  "papers_inserted": 44980,
  "papers_updated": 220,
  "papers_embedded": 40000,
  "current_week": "2026-03-09",
  "started_at": "2026-09-29T10:00:00Z",
  "elapsed_seconds": 3840
}

POST /api/v1/ingest/reset
Body: { "confirmation_token": "my-secret-reset-token" }
→ 200 { "message": "Database wiped. Ready for fresh import." }
→ 400 { "detail": "Invalid confirmation token" }
→ 409 { "detail": "Cannot reset while ingestion is running" }
```

### 5.3 Visualization / Stats

```
GET /api/v1/stats/summary
→ 200 {
  "total_papers": 134821,
  "total_authors": 287443,
  "total_categories": 48,
  "date_range": { "from": "2026-01-01", "to": "2026-09-29" },
  "top_categories": [
    { "code": "cs.LG", "name": "Machine Learning", "count": 18420 }
  ]
}

GET /api/v1/stats/top-categories?limit=10&from=2026-01-01&to=2026-09-29
→ 200 {
  "data": [
    { "code": "cs.LG", "name": "Machine Learning", "count": 18420 },
    { "code": "cs.CV", "name": "Computer Vision", "count": 14310 },
    ...
  ],
  "meta": { "total": 48, "limit": 10, "generated_at": "2026-09-29T12:00:00Z" }
}

GET /api/v1/stats/papers-by-category-over-time
  ?granularity=month          // month | year | week
  &category=cs.LG             // optional: filter to one category
  &from=2026-01-01
  &to=2026-09-29
→ 200 {
  "data": [
    { "period": "2026-01", "category": "cs.LG", "count": 1842 },
    { "period": "2026-02", "category": "cs.LG", "count": 1960 },
    ...
  ]
}

GET /api/v1/stats/top-authors?limit=20&from=...&to=...
→ 200 {
  "data": [
    { "author": "Yann LeCun", "paper_count": 12 },
    ...
  ]
}

GET /api/v1/stats/publication-velocity?granularity=week
→ 200 {
  "data": [
    { "period": "2026-W01", "count": 3421 },
    { "period": "2026-W02", "count": 3589 },
    ...
  ]
}

GET /api/v1/stats/category-cooccurrence?limit=20
→ 200 {
  "data": [
    { "category_a": "cs.LG", "category_b": "cs.AI", "cooccurrences": 4210 },
    ...
  ]
}
```

### 5.4 RAG

```
POST /api/v1/ask
Body: {
  "question": "What are the recent approaches to reducing memory in LLMs?",
  "top_k": 5,               // optional, 1–20, default 5
  "session_id": "abc123",   // optional — enables multi-turn context
  "stream": false           // optional — set true for SSE streaming response
}
→ 200 {
  "answer": "Based on the retrieved papers, several approaches have emerged...",
  "sources": [
    {
      "arxiv_id": "2601.12345",
      "title": "MemEfficient Transformers via ...",
      "similarity_score": 0.89,
      "reranker_score": 0.94,
      "retrieval_source": "dense",     // dense | sparse | graph
      "published_date": "2026-01-15",
      "url": "https://arxiv.org/abs/2601.12345"
    }
  ],
  "confidence": "high",               // high | medium | low | none
  "iterations": 1,                    // number of CRAG rewrites needed
  "faithfulness_passed": true,
  "model_used": "claude-haiku-4-5-20251001",
  "retrieval_ms": 180,
  "llm_ms": 2400
}
→ 200 (no match) {
  "answer": "I don't have relevant papers in the dataset to answer this question confidently.",
  "sources": [],
  "confidence": "none",
  "iterations": 3,
  "model_used": "claude-haiku-4-5-20251001",
  "retrieval_ms": 520,
  "llm_ms": 0
}
→ 422 { "detail": "question: field required" }
→ 503 { "detail": "LLM service temporarily unavailable", "retry_after": 30 }

// SSE stream (when stream: true) — each event is a partial token
GET /api/v1/ask/stream?question=...&top_k=5
→ text/event-stream
  data: {"type": "token", "content": "Based"}
  data: {"type": "token", "content": " on"}
  ...
  data: {"type": "done", "sources": [...], "confidence": "high"}

GET /api/v1/search?q=efficient+transformer+inference&limit=10
→ 200 {
  "results": [
    {
      "arxiv_id": "2601.12345",
      "title": "...",
      "abstract_clean": "...",
      "similarity_score": 0.91,
      "reranker_score": 0.95,
      "published_date": "2026-01-15",
      "primary_category": "cs.LG",
      "concepts": ["FlashAttention", "KV cache", "quantization"]
    }
  ]
}

POST /api/v1/rag/evaluate
Body: {
  "test_set": [
    { "question": "...", "ground_truth": "..." }
  ]
}
→ 200 {
  "faithfulness": 0.91,
  "answer_relevancy": 0.87,
  "context_precision": 0.83,
  "context_recall": 0.79,
  "num_samples": 20
}
```

---

## 6. Detailed Data Flow

### 6.1 Ingestion Data Flow

```
scripts/ingest.py
  │  parse CLI args (--from, --to, --categories)
  │  load Settings from .env
  ▼
app/ingestion/pipeline.py :: IngestionPipeline.run()
  │
  ├─ create ingestion_run row (status=running) → get run_id
  │
  ├─ generate weekly date windows: [(2026-01-01, 2026-01-07), (2026-01-08, 2026-01-14), ...]
  │
  └─ for each window:
       │
       ├─ checkpoint.get_checkpoint(run_id) → if window already done, skip
       │
       ├─ for offset in range(0, window_total, ARXIV_BATCH_SIZE):
       │   │
       │   ├─ arxiv_client.fetch(query, start=offset, max_results=500)
       │   │     httpx.AsyncClient.get(url, timeout=30)
       │   │     on 429/503: exponential_backoff(attempt) → sleep → retry
       │   │     on 5 failures: raise IngestionError, save checkpoint, exit
       │   │     returns: raw XML bytes
       │   │
       │   ├─ parser.parse_feed(xml_bytes)
       │   │     xml.etree.ElementTree.fromstring(xml)
       │   │     extract: arxiv_id, title, summary, authors[], categories[], dates, doi, journal_ref
       │   │     returns: List[RawPaper]  (Pydantic model, all fields optional/nullable)
       │   │
       │   ├─ for each RawPaper:
       │   │     normalizer.normalize_dates(paper)    → UTC TIMESTAMPTZ
       │   │     normalizer.normalize_authors(paper)  → strip whitespace, unicode NFC
       │   │     cleaner.clean_abstract(paper.summary)→ LaTeX stripped, HTML decoded
       │   │     → CleanPaper (Pydantic model, stricter, no nulls except doi/journal_ref)
       │   │
       │   ├─ db_session.execute(upsert_papers_bulk)
       │   │     INSERT INTO papers (...) VALUES (...) ON CONFLICT (arxiv_id) DO UPDATE
       │   │     SET ... WHERE papers.updated_date < EXCLUDED.updated_date
       │   │     (batch insert: all papers in one statement)
       │   │
       │   ├─ upsert_authors_bulk (INSERT ... ON CONFLICT (name_normalized) DO NOTHING)
       │   │
       │   ├─ upsert_paper_authors_bulk
       │   │
       │   ├─ upsert_paper_categories_bulk
       │   │
       │   ├─ checkpoint.save(run_id, week_start, offset)
       │   │
       │   └─ await asyncio.sleep(ARXIV_RATE_LIMIT_SECONDS)
       │
       └─ window complete → checkpoint.mark_week_done(run_id, week_start)
  │
  ├─ update ingestion_run: papers_fetched, papers_inserted, papers_updated
  │
  └─ embeddings/indexer.py :: EmbeddingIndexer.run()
       │
       ├─ query: SELECT arxiv_id, title, abstract_clean FROM papers WHERE embedding IS NULL
       │         (or WHERE embedding_updated_at < updated_at for re-embedding)
       │
       └─ for each batch of EMBEDDING_BATCH_SIZE:
            │
            ├─ texts = [f"{p.title}\n\n{p.abstract_clean}" for p in batch]
            │
            ├─ provider.embed(texts)
            │     OpenAIProvider: openai.embeddings.create(model=..., input=texts)
            │     LocalProvider:  sentence_transformers.SentenceTransformer.encode(texts)
            │     returns: List[List[float]]  (shape: batch_size × embedding_dim)
            │
            ├─ UPDATE papers SET embedding=?, embedding_model=?, embedding_updated_at=NOW()
            │   WHERE arxiv_id = ?   (batch update)
            │
            └─ ingestion_run.papers_embedded += len(batch)
  │
  └─ ingestion_run status: completed, completed_at: NOW()
```

### 6.2 Stats API Data Flow

```
HTTP GET /api/v1/stats/top-categories?limit=10
  │
  ▼
app/api/v1/stats.py :: get_top_categories(limit, from_date, to_date, db)
  │  Pydantic validates: limit ∈ [1, 100], dates valid ISO format
  ▼
SQL:
  SELECT c.code, c.display_name, COUNT(pc.paper_id) AS count
  FROM categories c
  JOIN paper_categories pc ON c.code = pc.category_code
  JOIN papers p ON pc.paper_id = p.arxiv_id
  WHERE p.published_date BETWEEN :from_date AND :to_date
  GROUP BY c.code, c.display_name
  ORDER BY count DESC
  LIMIT :limit
  │
  ▼
app/schemas/stats.py :: CategoryCount list → StatsResponse
  │
  ▼
HTTP 200 JSON  (~10–40ms end to end)
```

### 6.3 RAG Data Flow (LangGraph Agentic Pipeline)

**LangGraph State definition:**
```python
class RAGState(TypedDict):
    question: str
    session_id: str | None
    query_variants: list[str]       # 3 rephrased queries
    hyde_document: str              # hypothetical ideal answer
    candidates: list[RetrievedDoc]  # up to 20 after RRF fusion
    graded_docs: list[GradedDoc]    # after CRAG grading
    compressed_contexts: list[str]  # after context compressor
    answer: str
    sources: list[Source]
    confidence: ConfidenceLevel
    iterations: int                 # how many query rewrites happened
    faithfulness_passed: bool
    retrieval_ms: int
    llm_ms: int
```

**Full flow:**
```
HTTP POST /api/v1/ask  { "question": "...", "top_k": 5, "session_id": "..." }
  │  Pydantic: question non-empty, len ≤ 1000, top_k ∈ [1, 20]
  ▼
app/rag/langgraph_pipeline.py :: compiled_graph.ainvoke(initial_state)
  │
  ├─ [Node] query_analyzer.py
  │     LLM call (Haiku) — classify question: factual | comparative | exploratory
  │     Generate 3 query variants with different phrasing
  │     Generate HyDE document: "A paper that would answer this question would say: ..."
  │     → state.query_variants, state.hyde_document
  │
  ├─ [Node] retriever_node.py
  │     Dense: embed([question] + variants + [hyde_doc]) → 5 query vectors
  │            pgvector search each → merge → deduplicate → top-20 dense candidates
  │     Sparse: BM25 index search (original question + variants) → top-20 sparse
  │     Graph:  traverser.py — look up question concepts in concept graph
  │             → papers sharing ≥2 concepts with query concepts → up to 10 graph candidates
  │     Fusion: Reciprocal Rank Fusion (RRF) over all candidates → top-20 unique
  │     → state.candidates (List[RetrievedDoc] with source: dense|sparse|graph)
  │
  ├─ [Node] reranker.py
  │     cross-encoder.predict([(question, doc.text) for doc in candidates])
  │     sort by cross-encoder score descending → keep top min(top_k, 5)
  │     → state.candidates (trimmed to 5)
  │
  ├─ [Node] doc_grader.py   ← CRAG
  │     For each candidate:
  │       LLM (Haiku): "Is this paper relevant to: '{question}'? YES / PARTIAL / NO"
  │     If all NO AND state.iterations < RAG_MAX_RETRY_ITERATIONS:
  │       → [Edge: rewrite] → query_rewriter → retriever_node (loop)
  │     If all NO AND iterations == MAX:
  │       → [Edge: no_match] → return no-match response
  │     Else (at least 1 YES or PARTIAL):
  │       → [Edge: continue] → compressor
  │     → state.graded_docs
  │
  ├─ [Node] compressor.py
  │     For each relevant/partial doc:
  │       LLM: "Extract only sentences relevant to: '{question}'"
  │     → state.compressed_contexts (shorter, focused texts)
  │
  ├─ [Node] generator.py
  │     Build prompt: system (grounding rules) + user (question + compressed contexts)
  │     LangChain-Anthropic call with structured output (answer + cited arxiv_ids)
  │     → state.answer, state.sources
  │
  ├─ [Node] faithfulness.py
  │     LLM: "Does the answer make claims NOT supported by the source papers? YES/NO"
  │     If YES AND faithfulness retries < 2: → back to generator
  │     If YES AND retries == 2: strip unsupported claims, return partial answer
  │     → state.faithfulness_passed
  │
  └─ [Edge: done] → response builder
       confidence = f(max reranker_score):
         ≥ 0.80 → "high", ≥ 0.70 → "medium", ≥ 0.65 → "low", < 0.65 → "none"

HTTP 200 / SSE stream:
{
  "answer": "...",
  "sources": [{ arxiv_id, title, similarity_score, reranker_score, url, concept_overlap }],
  "confidence": "high",
  "iterations": 1,
  "retrieval_ms": 180,
  "llm_ms": 2400
}
```

---

## 7. Error Handling Matrix

| Module | Error | Behavior |
|---|---|---|
| arxiv_client | HTTP 429 Too Many Requests | Exponential backoff: 3.5s, 7s, 14s, 28s, 56s. After 5 attempts → save checkpoint → raise |
| arxiv_client | HTTP 503 Service Unavailable | Same as 429 |
| arxiv_client | Connection timeout (>30s) | Retry up to 3 times, then raise |
| arxiv_client | Malformed XML | Log full response bytes → skip batch → continue next |
| parser | Missing `<summary>` | abstract = "" → abstract_clean = "" → embed title only |
| parser | 0 authors | authors = [] → stored as empty list, paper still saved |
| parser | 500+ authors | Bulk insert in sub-batches of 100 to avoid query size limits |
| parser | Missing doi/journal_ref | None in DB — nullable columns |
| normalizer | Unparseable date string | Use published as fallback for updated; log warning |
| db upsert | DB connection lost | Rollback transaction → retry up to 3 times → fail with checkpoint saved |
| db upsert | Unique constraint violation (race) | ON CONFLICT handles it; never crashes |
| embeddings | OpenAI API rate limit | Exponential backoff same as arxiv_client |
| embeddings | OpenAI API error 500 | Retry 3 times; on failure skip batch (papers stay with embedding=NULL) |
| embeddings | Embedding dim mismatch | Raise at startup if EMBEDDING_DIM does not match pgvector column size |
| rag/retriever | No papers with embeddings | Return confidence=none, skip entire LangGraph pipeline |
| rag/retriever | BM25 index not found on disk | Fall back to dense-only retrieval, log warning |
| rag/retriever | Concept graph not found on disk | Skip graph hop retrieval, use dense+sparse only |
| rag/nodes/doc_grader | All docs irrelevant after 3 retries | Return no-match response, set confidence=none |
| rag/nodes/faithfulness | Hallucination detected × 2 | Strip unsupported sentences, return partial answer |
| rag/nodes/reranker | Cross-encoder model not loaded | Fall back to vector similarity score ordering |
| rag/answerer | Anthropic API timeout | Return HTTP 503 with Retry-After: 30 |
| rag/answerer | Anthropic API rate limit | Return HTTP 429 with Retry-After from response headers |
| rag/graph/builder | LLM concept extraction fails for a batch | Skip batch, paper gets no concepts (graph hop won't find it) |
| api/reset | Reset during active ingestion | Return HTTP 409 Conflict |
| api/ask | Empty question | Return HTTP 422 Validation Error |
| api/ask | Question > 1000 chars | Truncate silently to 1000 before embedding |

---

## 8. arXiv API Details

**Base URL:** `https://export.arxiv.org/api/query`

**Query format:**
```
GET https://export.arxiv.org/api/query
  ?search_query=cat:cs.AI+AND+submittedDate:[20260101+TO+20260107]
  &start=0
  &max_results=500
  &sortBy=submittedDate
  &sortOrder=ascending
```

**Response format:** Atom/XML feed. Key elements:
```xml
<feed>
  <opensearch:totalResults>3421</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2601.00001v1</id>          <!-- arxiv_id: "2601.00001" -->
    <title>Paper Title Here</title>
    <summary>Abstract text here...</summary>
    <published>2026-01-02T00:00:00Z</published>
    <updated>2026-01-03T00:00:00Z</updated>
    <author><name>Author Name</name></author>
    <author><name>Another Author</name></author>
    <arxiv:primary_category term="cs.AI" />
    <category term="cs.AI" />
    <category term="cs.LG" />
    <arxiv:doi>10.1000/xyz123</arxiv:doi>              <!-- may be absent -->
    <arxiv:journal_ref>ICML 2026</arxiv:journal_ref>   <!-- may be absent -->
  </entry>
</feed>
```

**Known quirks:**
- `start` offset is unreliable beyond ~10,000 for large result sets → use date-chunking
- `totalResults` can change between paginated requests (papers added during fetch) → safe to ignore minor drift
- The `<id>` field includes version suffix (v1, v2) → strip to get canonical arxiv_id
- Rate limit: 1 req/3 sec per IP (we use 3.5s buffer)

---

## 9. Preprocessing Details

### Abstract Cleanup Pipeline (`cleaner.py`)

```python
# Applied in order:
1. html.unescape(text)                   # &amp; → & etc.
2. re.sub(r'\\[a-zA-Z]+\{[^}]*\}', ...)  # \textbf{word} → word
3. re.sub(r'\$[^$]+\$', '[MATH]', ...)   # $x^2$ → [MATH]
4. re.sub(r'\\\([^)]+\\\)', '[MATH]', .) # \(...\) → [MATH]
5. re.sub(r'\\\[[^\]]+\\\]', '[MATH]', .)# \[...\] → [MATH]
6. unicodedata.normalize('NFC', text)    # unicode normalization
7. re.sub(r'\s+', ' ', text).strip()    # collapse whitespace
8. text[:2000]                           # truncate (safety for embedding token limits)
```

### Author Name Normalization (`normalizer.py`)

```python
def normalize_name(name: str) -> str:
    name = name.strip()
    name = unicodedata.normalize('NFC', name)
    name = name.lower()
    name = re.sub(r'\s+', ' ', name)
    return name
# "  Yann  LeCun  " → "yann lecun"
# "Yann LeCun" and "yann lecun" → same DB row, deduplicated
```

---

## 10. Embedding & Vector Search Details

### Text Input to Embedding
```python
text = f"{paper.title}\n\n{paper.abstract_clean}"
# Title adds topic context signal
# Max 2000 chars of abstract ensures we stay within model token limits
```

### pgvector Cosine Search
```sql
-- The <=> operator computes cosine distance (1 - cosine_similarity)
-- So ORDER BY <=> ASC = most similar first
-- 1 - distance = similarity_score

SELECT arxiv_id, title, abstract_clean, published_date,
       1 - (embedding <=> :query_vector::vector) AS similarity_score
FROM papers
WHERE embedding IS NOT NULL
ORDER BY embedding <=> :query_vector::vector
LIMIT :top_k;
```

### Index Strategy
- **IVFFlat** index: fast approximate search, good for read-heavy workloads at 100k+ scale
- Index is built **after** bulk insert (not during) — building mid-insert is slow
- `lists = 370` for ~135k rows (rule: lists ≈ sqrt(rows))
- `SET ivfflat.probes = 20` at query time for recall/speed balance

### Similarity Thresholds
```
≥ 0.80  high confidence   — strongly relevant paper
≥ 0.70  medium confidence — probably relevant
≥ 0.65  low confidence    — possibly relevant, answer with caveats
< 0.65  no match          — return honest "I don't know"
```

---

## 11. LLM Prompt Design (Per Node)

### Node: QueryAnalyzer
```
System: "You are a query analysis engine. Output JSON only."
User:   "Question: {question}
         Output:
         {
           'type': 'factual' | 'comparative' | 'exploratory',
           'variants': ['...', '...', '...'],   // 3 rephrasings
           'hyde': '...'                         // hypothetical paper abstract that would answer this
         }"
```

### Node: DocumentGrader (CRAG)
```
System: "You are a relevance grader. Output JSON only."
User:   "Question: {question}
         Paper abstract: {abstract_clean}
         Is this paper relevant to the question?
         Output: { 'score': 'relevant' | 'partial' | 'irrelevant', 'reason': '...' }"
```

### Node: ContextCompressor
```
System: "Extract only sentences directly relevant to the question. Return extracted text only."
User:   "Question: {question}
         Paper text: {abstract_clean}
         Return only relevant sentences:"
```

### Node: Generator (Answer)
```
System: "You are a research assistant with access to arXiv papers.
         Rules:
         - Use ONLY the provided papers as your source of truth
         - Cite papers using [arXiv:XXXXXXX] inline
         - If papers lack enough info, say so explicitly
         - Synthesize across papers; do not summarize each individually"

User:   "Question: {question}

         Papers:
         [1] arXiv:{arxiv_id} — {title} ({date})
         {compressed_context}

         [2] ...

         Answer:"
```

### Node: FaithfulnessChecker
```
System: "You are a fact-checker. Output JSON only."
User:   "Answer: {answer}
         Source papers: {compressed_contexts}
         Does the answer make any claims NOT supported by the source papers?
         Output: { 'hallucination_detected': true | false, 'unsupported_claims': ['...'] }"
```

---

## 12. Docker Setup

### docker-compose.yml (structure)

```yaml
version: "3.9"
services:

  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: pipeline
      POSTGRES_USER: pipeline
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U pipeline -d pipeline"]
      interval: 5s
      timeout: 5s
      retries: 10
    ports:
      - "5432:5432"

  api:
    build:
      context: .
      dockerfile: Dockerfile
    depends_on:
      db:
        condition: service_healthy
    env_file: .env
    ports:
      - "8000:8000"
    command: >
      sh -c "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"

volumes:
  pgdata:
```

### Dockerfile (multi-stage)
```
Stage 1 (builder): install all deps including dev tools
Stage 2 (runtime): copy only installed packages + app code, no build tools
```

### Running ingestion (separate container run)
```bash
docker-compose run --rm api python -m scripts.ingest \
  --from 2026-01-01 \
  --to 2026-09-29 \
  --categories "cs.*"
```

---

## 13. Testing Strategy

| File | What it tests | Approach |
|---|---|---|
| `unit/test_parser.py` | XML parsing: missing fields, v1/v2 suffixes, 500 authors, empty abstract | Parameterized with XML fixtures |
| `unit/test_cleaner.py` | LaTeX stripping, HTML entities, unicode NFC, length truncation | Property-based with hypothesis |
| `unit/test_retriever.py` | Threshold logic (high/medium/low/none), empty results handling | Mock DB results |
| `integration/test_ingest_pipeline.py` | Full ingest loop against mocked arXiv (respx mock) → verify paper/author/category rows in test DB | pytest-asyncio + test PostgreSQL |
| `integration/test_stats_api.py` | All 6 stat endpoints return correct aggregations against seeded test DB | httpx.AsyncClient against running FastAPI app |
| `integration/test_rag_api.py` | /ask: happy path, no-match, empty question, LLM timeout → mocked embeddings + mocked Anthropic | respx + httpx |

### Test DB Setup
- pytest fixture spins up a real PostgreSQL test database (in Docker or via `pytest-postgresql`)
- Runs Alembic migrations on it
- Seeds it with ~50 deterministic fake papers
- Tears down after session

---

## 14. Migrations Plan

```
0001_initial.py     — create categories, papers, authors, paper_authors, paper_categories
0002_ingestion.py   — create ingestion_runs table
0003_vectors.py     — enable pgvector extension, add embedding column, add IVFFlat index
0004_seed_cats.py   — seed categories table with cs.AI, cs.LG, cs.CL (+ other CS codes)
0005_concepts.py    — create concepts, paper_concepts tables
```

Each migration is fully reversible (has `downgrade()` implemented).

---

## 15. Implementation Order

```
Phase 1: Foundation                                  ~2h
  alembic setup → migrations 0001-0005 → Settings → database.py → /health endpoint

Phase 2: Ingestion                                   ~3h
  arxiv_client.py → parser.py → normalizer.py → cleaner.py
  → pipeline.py (category-based fetch, checkpointing, hard cap at 3500)
  → scripts/ingest.py CLI

Phase 3: Visualization API                           ~2h
  All 6 stats endpoints → schemas → SQL queries → tests

Phase 4: Embeddings + BM25 + Concept Graph           ~3h
  provider.py (OpenAI + local) → indexer.py
  → BM25 index builder → serialized to disk
  → graph/builder.py (LLM concept extraction) → NetworkX graph → disk
  → all three wired into pipeline.py post-ingest

Phase 5: LangGraph RAG Pipeline                      ~4h
  All 7 nodes (nodes/) → hybrid retriever.py → langgraph_pipeline.py (StateGraph)
  → /ask endpoint (sync + SSE stream) → /search endpoint
  → evaluation/ragas_eval.py → scripts/evaluate.py CLI

Phase 6: Tests + Docker + README                     ~2h
  conftest.py → all test files → docker-compose → Dockerfile → README
```
