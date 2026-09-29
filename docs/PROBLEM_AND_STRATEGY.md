# arXiv Intelligence Pipeline
## Problem, Strategy & Architecture

---

## 1. The Problem We Are Solving

Academic research is accelerating. In the field of computer science alone, arXiv — the world's
largest open-access preprint server — receives hundreds of new papers every single day. From
January 1 to September 29, 2026, that translates to roughly **135,000 CS papers** published
publicly, for free, with no login required.

The problem is not access. The problem is **findability and synthesis**.

- A researcher wants to know: *"What are the recent approaches to reducing memory usage in large
  language models?"* They can't read 135,000 abstracts. A keyword search returns 500 results
  with no sense of which are actually relevant.
- A team lead wants a chart: *"How many papers were published in AI vs. Systems vs. Security
  each month this year?"* There is no ready-made answer — the data exists but is not aggregated.
- A product manager wants a plain English answer: *"Are there any papers about using RAG for
  medical diagnosis?"* They don't know how to write a search query.

This project solves all three of those problems by building a **data pipeline** that:
1. Automatically downloads all relevant papers from arXiv
2. Cleans and organizes them into a structured database
3. Provides an API for charts and statistics
4. Allows anyone to ask a natural-language question and get a grounded, cited answer

---

## 2. What We Are Building (Plain English)

Think of it as building a **private, intelligent research library** in five stages:

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   arXiv.org            Our Pipeline               Users / Apps     │
│   (the source)         (what we build)             (consumers)     │
│                                                                     │
│  ┌──────────┐       ┌──────────────────────┐    ┌──────────────┐  │
│  │ 135,000  │──────▶│ 1. Fetch & Download  │    │ Dashboards   │  │
│  │ CS papers│       │ 2. Clean & Organize  │───▶│ Charts       │  │
│  │ (XML/API)│       │ 3. Store in Database │    │ Q&A Chatbot  │  │
│  └──────────┘       │ 4. Build Search Index│    └──────────────┘  │
│                     │ 5. Answer Questions  │                       │
│                     └──────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Stage-by-Stage Strategy

### Stage 1 — Ingestion (Fetching Papers)

**The challenge:** arXiv has a public API, but it has rules. You can only make 1 request every
3 seconds. Each request returns at most 2,000 papers.

**Our strategy:** Fetch **1,000–3,500 papers** from three focused categories — `cs.AI`,
`cs.LG`, and `cs.CL` — using a small number of paginated batches. At 500 papers per batch,
that's just 7 requests taking about 25 seconds. Fast, reliable, and well within safe limits.

```
cs.AI   → batch 1 (500), batch 2 (500), batch 3 (up to 500)  ✓
cs.LG   → batch 1 (500), batch 2 (500)                        ✓
cs.CL   → batch 1 (500), batch 2 (500)                        ✓
                              ~1,000–3,500 papers total
```

We chose these three categories because they have the richest abstracts for semantic search
(AI, Machine Learning, Computational Linguistics) and naturally overlap — a cross-domain
question like *"how does NLP use reinforcement learning?"* is well served by all three.

**Safety rule:** Running ingestion twice never creates duplicates. If a paper was already
downloaded and hasn't changed, we skip it. If arXiv revised the paper since last time, we update
our copy and re-embed it.

---

### Stage 2 — Preprocessing (Cleaning Papers)

**The challenge:** arXiv returns data in XML format — a computer-readable but messy structure.
Papers can have 1 author or 500 authors. Abstracts contain LaTeX math notation like `$\nabla f$`.
Some papers have no journal reference. Dates come in inconsistent formats.

**Our strategy:** Every paper passes through a cleaning pipeline before it touches the database:

```
Raw XML from arXiv
       │
       ▼
  Parse XML → extract fields (title, authors, abstract, dates, categories...)
       │
       ▼
  Normalize → convert all dates to UTC, strip extra whitespace, handle missing fields
       │
       ▼
  Clean Abstract → remove/replace LaTeX math, normalize unicode, trim to usable length
       │
       ▼
  Flatten structure → one paper → many authors (separate), many categories (separate)
       │
       ▼
  Clean records ready for storage
```

---

### Stage 3 — Storage (The Database)

**The challenge:** A paper is not a flat record. It has multiple authors, multiple categories,
and a long abstract. We need to store it in a way that lets us run fast queries like
*"how many papers did author X write?"* or *"what categories had the most papers in March?"*

**Our strategy:** Use a **relational database** (PostgreSQL) with separate tables for each
concept, linked together:

```
┌──────────────┐      ┌─────────────────┐      ┌──────────┐
│    papers    │◀────▶│  paper_authors  │◀────▶│ authors  │
│              │      └─────────────────┘      └──────────┘
│  arxiv_id    │
│  title       │      ┌───────────────────┐    ┌────────────┐
│  abstract    │◀────▶│ paper_categories  │◀──▶│ categories │
│  dates       │      └───────────────────┘    └────────────┘
│  doi         │
│  embedding ──┼──── (vector for semantic search, same table)
└──────────────┘

┌──────────────────┐
│  ingestion_runs  │  ← full audit log of every data fetch
└──────────────────┘
```

Every time we run ingestion, a record is created in `ingestion_runs` logging how many papers
were fetched, how many were new, how many were updates, and where we stopped if something failed.

---

### Stage 4 — Visualization API (The Charts Layer)

**The challenge:** Displaying data visually requires pre-computed aggregations.
You can't send 135,000 rows to a browser and ask it to draw a pie chart.

**Our strategy:** Build a small set of API endpoints that each run a specific database query
and return ready-to-chart data. These run in milliseconds because PostgreSQL is doing the
aggregation work.

```
Request:  GET /api/v1/stats/top-categories?limit=10
Response: [
            { "category": "cs.LG", "count": 18420 },
            { "category": "cs.CV", "count": 14310 },
            { "category": "cs.AI", "count": 11870 },
            ...
          ]
```

Six endpoints total — enough to power a real research analytics dashboard.

---

### Stage 5 — Advanced RAG Pipeline (The Question-Answering Layer)

**The challenge:** SQL queries can't answer *"what are the most exciting ideas in efficient AI
inference?"* — that's a meaning-based question, not a keyword match. And naive RAG (just embed
+ search) produces mediocre results — it misses papers that use different words for the same
concept, retrieves irrelevant papers, and sometimes still hallucinates.

**Our strategy:** A production-grade **Agentic RAG system** built on **LangGraph** that
self-corrects, uses multiple retrieval strategies, and verifies its own answers before returning.

**Preparation (done once during ingestion):**
```
For each paper abstract:
  1. Generate a meaning vector (embedding) → stored in pgvector
  2. Build a BM25 keyword index for sparse retrieval
  3. Extract key concepts/methods via LLM (e.g. "attention mechanism", "LoRA", "RLHF")
  4. Build a concept co-occurrence graph — papers sharing concepts are connected
```

**Query time (LangGraph state machine — the pipeline thinks for itself):**
```
User question: "What are recent approaches to efficient LLM inference?"
       │
       ▼
Node 1: Query Analyzer
  - Classify: factual / comparative / exploratory?
  - Decompose multi-part questions into sub-questions
  - Generate 3 rephrased variants of the question
  - Generate a "HyDE" document — a fake ideal answer, used as retrieval bait
       │
       ▼
Node 2: Hybrid Retriever
  - Dense search: pgvector cosine similarity on 3 query variants + HyDE (top-20)
  - Sparse search: BM25 keyword matching (top-20)
  - Graph search: look up shared concepts → retrieve related papers
  - Merge all results with Reciprocal Rank Fusion → 20 unique candidates
       │
       ▼
Node 3: Cross-Encoder Reranker
  - A dedicated re-ranking model scores all 20 candidates against the question
  - Much more accurate than vector similarity alone
  - Keeps top 5
       │
       ▼
Node 4: Document Grader (CRAG — Corrective RAG)
  - LLM reads each of the 5 papers and rates: relevant / partial / irrelevant
  - If all 5 are irrelevant AND we haven't retried 3 times yet:
      → rewrite the query → go back to Node 2
  - If all irrelevant after 3 tries:
      → return honest "I can't find relevant papers"
       │
       ▼
Node 5: Context Compressor
  - Extract only the sentences from each abstract relevant to the question
  - Removes noise, reduces hallucination
       │
       ▼
Node 6: Answer Generator (Claude LLM)
  - Strict prompt: "Use ONLY these papers. Cite by arXiv ID. No invention."
  - Generates answer with inline citations
       │
       ▼
Node 7: Faithfulness Checker
  - Verify: does the answer's claims exist in the source papers?
  - If not: regenerate (max 2 tries)
       │
       ▼
Return: answer + sources + confidence score + timing breakdown
```

**Why this is better than naive RAG:**
- HyDE + multi-query catches papers that use different vocabulary for the same idea
- Reranker dramatically improves which 5 papers the LLM actually reads
- CRAG stops the system from answering from irrelevant context
- Faithfulness check catches hallucinations before they reach the user
- The concept graph retrieves papers that are semantically related even if they don't share keywords

---

## 4. Full Data Flow (End to End)

```
╔══════════════════════════════════════════════════════════════════════╗
║  INGESTION FLOW (one-time + incremental updates)                    ║
╚══════════════════════════════════════════════════════════════════════╝

  CLI / API trigger
       │
       ▼
  IngestionPipeline starts
  Creates ingestion_run record (status: running)
       │
       ▼
  ┌─────────────────────────────────────────────┐
  │  For each category batch (cs.AI/cs.LG/cs.CL)│◀── resumes from checkpoint if interrupted
  │                                             │
  │  1. ArxivClient fetches XML                 │
  │     (rate-limited 3.5s, retry on error)     │
  │         │                                   │
  │         ▼                                   │
  │  2. Parser extracts fields                  │
  │         │                                   │
  │         ▼                                   │
  │  3. Normalizer + Cleaner process data       │
  │         │                                   │
  │         ▼                                   │
  │  4. DB upsert (no duplicates)               │
  │     papers + authors + categories           │
  │         │                                   │
  │         ▼                                   │
  │  5. Save checkpoint to DB                   │
  │  6. Sleep 3.5 seconds                       │
  └─────────────────────────────────────────────┘
       │
       ▼
  EmbeddingIndexer starts
  ┌───────────────────────────────────────────┐
  │  For each paper with no embedding yet:    │
  │  1. Combine title + clean_abstract        │
  │  2. Send batch of 100 to embedding API    │
  │  3. Store 1536-dim vectors in pgvector    │
  └───────────────────────────────────────────┘
       │
       ▼
  BM25 index built from all abstracts → saved to disk
       │
       ▼
  ConceptGraphBuilder starts
  ┌───────────────────────────────────────────┐
  │  For each paper:                          │
  │  1. LLM extracts concepts/methods         │
  │     ("LoRA", "RLHF", "FlashAttention"...) │
  │  2. Store in concepts + paper_concepts    │
  │  3. Build co-occurrence graph (NetworkX)  │
  │  4. Serialize graph to disk               │
  └───────────────────────────────────────────┘
       │
       ▼
  ingestion_run status: completed
  pgvector IVFFlat index rebuilt


╔══════════════════════════════════════════════════════════════════════╗
║  VISUALIZATION API FLOW (real-time, per request)                    ║
╚══════════════════════════════════════════════════════════════════════╝

  HTTP GET /api/v1/stats/papers-by-category-over-time?granularity=month
       │
       ▼
  FastAPI router validates query parameters
       │
       ▼
  PostgreSQL aggregation query runs (~5-50ms)
  (GROUP BY category, date_trunc('month', published_date))
       │
       ▼
  Results serialized to JSON
       │
       ▼
  HTTP 200 response with chart-ready data array


╔══════════════════════════════════════════════════════════════════════╗
║  RAG QUESTION-ANSWERING FLOW (LangGraph Agentic Pipeline)           ║
╚══════════════════════════════════════════════════════════════════════╝

  HTTP POST /api/v1/ask  { "question": "...", "top_k": 5, "session_id": "..." }
       │
       ▼
  Pydantic validates input → LangGraph pipeline invoked with initial state
       │
       ▼
  [Node 1] QueryAnalyzer
    - classify question type
    - generate 3 rephrasings
    - generate HyDE document (hypothetical ideal answer)
       │
       ▼
  [Node 2] HybridRetriever
    - pgvector search × 3 query variants + HyDE → top-20 dense
    - BM25 search → top-20 sparse
    - concept graph lookup → related papers
    - RRF fusion → top-20 unique candidates
       │
       ▼
  [Node 3] CrossEncoderReranker
    - score all 20 candidates → keep top 5
       │
       ▼
  [Node 4] DocumentGrader (CRAG)
    - grade each doc: relevant / partial / irrelevant
    - all irrelevant + iterations < 3? → rewrite query → back to Node 2
    - all irrelevant + 3 tries? → return "no relevant papers found"
       │
       ▼
  [Node 5] ContextCompressor → extract relevant sentences only
       │
       ▼
  [Node 6] AnswerGenerator (Claude) → grounded answer with citations
       │
       ▼
  [Node 7] FaithfulnessChecker → if hallucination detected → regenerate
       │
       ▼
  HTTP 200 response (or SSE stream):
  {
    "answer": "...",
    "sources": [{ arxiv_id, title, similarity_score, url }],
    "confidence": "high",
    "iterations": 1,
    "retrieval_ms": 180,
    "llm_ms": 2100
  }
```

---

## 5. What Makes This Production-Grade

### Reliability
- **Checkpointing:** a 3-hour ingestion can crash at hour 2 and resume from exactly where it
  stopped. No data is lost, no work is duplicated.
- **Idempotency:** running the same ingestion twice produces identical results. Safe to schedule
  as a daily job.
- **Graceful degradation:** if the embedding API is down, papers still land in the database.
  Embedding runs separately and can retry.

### Honesty
- The RAG system explicitly refuses to answer when it doesn't have relevant data. It will not
  invent citations or hallucinate paper titles. This is non-negotiable for a research tool.

### Observability
- Every ingestion is logged in the database with full statistics.
- Every API request logs timing, status, and paper counts in structured JSON.
- `/health` and `/ready` endpoints for infrastructure monitoring.

### Safety
- Resetting the database requires an explicit confirmation token — it cannot happen by accident.
- No real API keys ever committed to the repository.
- Re-ingestion only updates papers that have actually changed on arXiv.

### Portability
- `docker-compose up` brings up the entire stack (database + API) with a single command.
- No manual database setup, no manual migration steps, no platform-specific dependencies.

---

## 6. Known Limitations & What We'd Improve

| Limitation | Improvement with more time |
|---|---|
| Concept extraction via LLM is slow at scale | Batch with smaller model (Haiku); cache extracted concepts |
| BM25 index stored on disk (not in DB) | Move to Elasticsearch or PostgreSQL full-text search |
| LangGraph state is in-memory (not persisted) | Persist LangGraph checkpoints in Redis for distributed workers |
| Single API server | Horizontal scaling behind a load balancer |
| No user authentication | OAuth2 / API key system for multi-tenant use |
| No caching on stats endpoints | Redis cache with 5-minute TTL for expensive aggregations |
| RAGAS eval requires manual test set | Automatically generate evaluation questions from papers |
| Papers capped at 3,500 | Scale to 50k+ with async batched embedding pipeline |

---

## 7. Scope of This Build

- **Data source:** arXiv categories cs.AI, cs.LG, cs.CL — 1,000 to 3,500 papers
- **Estimated volume:** ~3,500 papers, ~8,000 unique authors, 3 primary categories
- **Storage estimate:** ~200 MB PostgreSQL (data + vectors)
- **Expected ingestion time:** ~25 seconds fetch, ~5 minutes embedding, ~10 minutes concept extraction
- **API response times:** stats endpoints <50ms, /ask endpoint 2–5 seconds (LangGraph pipeline)
