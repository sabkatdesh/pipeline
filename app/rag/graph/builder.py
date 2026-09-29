"""
Concept extraction (LLM) + concept co-occurrence graph (NetworkX).

Two steps, run after embeddings/BM25 in the ingestion pipeline:

  1. extract: papers with no paper_concepts rows are sent to Claude Haiku in batches
     of CONCEPT_EXTRACTION_BATCH_SIZE; concepts land in `concepts` + `paper_concepts`.
     Incremental and retry-safe: a failed batch just leaves those papers pending.
  2. graph: rebuilt from the DB every run and pickled to RAG_GRAPH_PATH.

Graph shape (consumed by rag/graph/traverser.py in Phase 5):
  node  = concept, id = casefolded name
          attrs: name, concept_type, papers (sorted arxiv_ids)
  edge  = two concepts appear in the same paper; attr weight = #papers sharing both
  G.graph["paper_concepts"] = {arxiv_id: [concept keys]}   # for "shares >=2 concepts"
"""

from __future__ import annotations

import asyncio
import json
import os
import pickle
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.core.config import get_settings
from app.core.database import AsyncSessionLocal
from app.core.logging import get_logger
from app.models.concept import Concept, PaperConcept
from app.rag.graph.models import ConceptEdge, ConceptNode

logger = get_logger(__name__)

_VALID_TYPES = frozenset({"method", "model", "dataset", "task", "other"})
_MAX_CONCEPTS_PER_PAPER = 8
_MAX_NAME_LEN = 80
_ABSTRACT_CHARS = 1200      # per paper in the prompt; keeps 20-paper calls small
_CONCURRENCY = 4            # parallel LLM calls (DB writes are serialized)
_LLM_ATTEMPTS = 3
_LLM_BACKOFF_SECONDS = 3.5
# 20 papers x ~8 concepts of JSON overflows the default LLM_MAX_TOKENS=1024.
_LLM_MAX_TOKENS = 6000

_SYSTEM = "You extract technical concepts from arXiv abstracts. Output JSON only, no prose."

_INSTRUCTIONS = f"""For each paper, list up to {_MAX_CONCEPTS_PER_PAPER} key technical concepts.

Rules:
- Only specific named things: methods (LoRA, RLHF, FlashAttention), models (LLaMA, BERT),
  datasets/benchmarks (GSM8K, ImageNet), tasks (machine translation, code generation).
- Use the canonical name people actually use; keep acronyms as written.
- No generic terms ("deep learning", "neural network", "results", "framework").
- type is one of: method, model, dataset, task, other.

Return exactly this JSON shape, one entry per paper, using the given arxiv_id:
{{"papers": [{{"arxiv_id": "...", "concepts": [{{"name": "...", "type": "method"}}]}}]}}
"""


class ExtractionError(Exception):
    """This batch failed; skip it and leave its papers pending."""


class ExtractionConfigError(ExtractionError):
    """Fatal (bad API key): stop the whole extraction step."""


@dataclass
class GraphBuildResult:
    papers_extracted: int = 0
    batches_failed: int = 0
    nodes: int = 0
    edges: int = 0
    extraction_skipped: str | None = None


class ConceptGraphBuilder:
    def __init__(self) -> None:
        self._cfg = get_settings()
        self._canon: dict[str, str] | None = None  # casefold -> canonical stored name
        self._fatal: ExtractionConfigError | None = None
        self._failed = 0

    async def run(self) -> GraphBuildResult:
        result = GraphBuildResult()

        if not self._cfg.anthropic_api_key:
            result.extraction_skipped = "ANTHROPIC_API_KEY not set"
            logger.warning("concept_extraction_skipped", reason=result.extraction_skipped)
        else:
            result.papers_extracted = await self._extract_all()
            result.batches_failed = self._failed

        # Always rebuild from the DB so the graph reflects every stored concept,
        # even if this run's extraction was skipped or partial.
        graph = await self._build_graph()
        result.nodes, result.edges = graph.number_of_nodes(), graph.number_of_edges()
        await asyncio.to_thread(_atomic_pickle_dump, graph, self._cfg.rag_graph_path)
        logger.info(
            "concept_graph_built",
            nodes=result.nodes,
            edges=result.edges,
            path=self._cfg.rag_graph_path,
        )
        return result

    # ── Extraction ────────────────────────────────────────────────────────────

    async def _extract_all(self) -> int:
        import anthropic

        async with AsyncSessionLocal() as session:
            pending = (
                await session.execute(
                    text(
                        """
                        SELECT p.arxiv_id, p.title, p.abstract_clean
                        FROM papers p
                        WHERE p.abstract_clean <> ''
                          AND NOT EXISTS (
                              SELECT 1 FROM paper_concepts pc WHERE pc.paper_id = p.arxiv_id
                          )
                        ORDER BY p.arxiv_id
                        """
                    )
                )
            ).all()
            self._canon = {
                n.casefold(): n for n in (await session.execute(select(Concept.name))).scalars()
            }
        if not pending:
            return 0

        size = self._cfg.concept_extraction_batch_size
        batches = [pending[i : i + size] for i in range(0, len(pending), size)]
        client = anthropic.AsyncAnthropic(api_key=self._cfg.anthropic_api_key, max_retries=0)
        sem = asyncio.Semaphore(_CONCURRENCY)
        write_lock = asyncio.Lock()
        done = 0

        async def process(batch) -> int:
            if self._fatal:
                return 0
            async with sem:
                try:
                    extracted = await self._call_llm(client, batch)
                except ExtractionConfigError as exc:
                    self._fatal = exc
                    return 0
                except ExtractionError as exc:
                    self._failed += 1
                    logger.error("concept_batch_skipped", error=str(exc), size=len(batch))
                    return 0
            async with write_lock:
                try:
                    await self._store(extracted)
                except Exception as exc:
                    self._failed += 1
                    logger.exception("concept_store_failed", error=str(exc))
                    return 0
            return len(extracted)

        for n in await asyncio.gather(*(process(b) for b in batches)):
            done += n
        if self._fatal:
            raise self._fatal
        logger.info("concept_extraction_done", papers=done, batches_failed=self._failed)
        return done

    async def _call_llm(self, client, batch) -> dict[str, list[tuple[str, str]]]:
        import anthropic

        papers_block = "\n\n".join(
            f"[arxiv_id: {r.arxiv_id}]\nTitle: {r.title}\nAbstract: {r.abstract_clean[:_ABSTRACT_CHARS]}"
            for r in batch
        )
        prompt = f"{_INSTRUCTIONS}\nPapers:\n\n{papers_block}"

        attempt = 0
        while True:
            try:
                resp = await client.messages.create(
                    model=self._cfg.llm_model,
                    max_tokens=_LLM_MAX_TOKENS,
                    temperature=0,
                    system=_SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
                raise ExtractionConfigError(f"Anthropic rejected credentials: {exc}") from exc
            except (anthropic.APIConnectionError, anthropic.APIStatusError) as exc:
                status = getattr(exc, "status_code", None)
                retryable = status is None or status == 429 or status >= 500
                attempt += 1
                if not retryable or attempt >= _LLM_ATTEMPTS:
                    raise ExtractionError(f"LLM call failed ({status}): {exc}") from exc
                await asyncio.sleep(_LLM_BACKOFF_SECONDS * 2 ** (attempt - 1))

        text_out = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return _parse_response(text_out, {r.arxiv_id for r in batch})

    async def _store(self, extracted: dict[str, list[tuple[str, str]]]) -> None:
        """Persist one batch. Reuses existing spelling when names differ only by case."""
        assert self._canon is not None
        new_names: dict[str, str] = {}
        resolved: dict[str, list[str]] = {}
        for pid, concepts in extracted.items():
            names = []
            for name, ctype in concepts:
                key = name.casefold()
                canon = self._canon.get(key)
                if canon is None:
                    self._canon[key] = canon = name
                    new_names[name] = ctype
                names.append(canon)
            resolved[pid] = names

        try:
            async with AsyncSessionLocal() as session:
                if new_names:
                    await session.execute(
                        pg_insert(Concept)
                        .values([{"name": n, "concept_type": t} for n, t in new_names.items()])
                        .on_conflict_do_nothing(index_elements=["name"])
                    )
                wanted = {n for names in resolved.values() for n in names}
                id_map: dict[str, int] = {}
                if wanted:
                    rows = await session.execute(
                        select(Concept.id, Concept.name).where(Concept.name.in_(wanted))
                    )
                    id_map = {r.name: r.id for r in rows}
                links = [
                    {"paper_id": pid, "concept_id": id_map[n]}
                    for pid, names in resolved.items()
                    for n in set(names)
                    if n in id_map
                ]
                if links:
                    await session.execute(
                        pg_insert(PaperConcept).values(links).on_conflict_do_nothing()
                    )
                await session.commit()
        except Exception:
            for n in new_names:  # roll back the in-memory canon so a retry re-inserts
                self._canon.pop(n.casefold(), None)
            raise

    # ── Graph ─────────────────────────────────────────────────────────────────

    async def _build_graph(self) -> nx.Graph:
        async with AsyncSessionLocal() as session:
            rows = (
                await session.execute(
                    text(
                        "SELECT pc.paper_id, c.name, c.concept_type "
                        "FROM paper_concepts pc JOIN concepts c ON c.id = pc.concept_id"
                    )
                )
            ).all()
        return await asyncio.to_thread(assemble_graph, [tuple(r) for r in rows])


def assemble_graph(rows: list[tuple[str, str, str | None]]) -> nx.Graph:
    """Pure function: (paper_id, concept_name, concept_type) rows -> concept graph."""
    nodes: dict[str, ConceptNode] = {}
    paper_keys: dict[str, set[str]] = {}
    for pid, name, ctype in rows:
        key = name.casefold()
        node = nodes.setdefault(key, ConceptNode(key, name, ctype or "other"))
        node.papers.append(pid)
        paper_keys.setdefault(pid, set()).add(key)

    weights: Counter[tuple[str, str]] = Counter()
    for keys in paper_keys.values():
        for a, b in combinations(sorted(keys), 2):
            weights[(a, b)] += 1
    edges = [ConceptEdge(a, b, w) for (a, b), w in weights.items()]

    graph = nx.Graph()
    for n in nodes.values():
        graph.add_node(n.key, name=n.name, concept_type=n.concept_type, papers=sorted(n.papers))
    graph.add_edges_from((e.source, e.target, {"weight": e.weight}) for e in edges)
    graph.graph.update(
        built_at=datetime.now(timezone.utc).isoformat(),
        n_papers=len(paper_keys),
        paper_concepts={pid: sorted(keys) for pid, keys in paper_keys.items()},
    )
    return graph


def load_concept_graph(path: str | None = None) -> nx.Graph | None:
    """None if missing/corrupt (spec: skip graph-hop retrieval, use dense + sparse)."""
    path = path or get_settings().rag_graph_path
    graph = _safe_pickle_load(path)
    if not isinstance(graph, nx.Graph):
        logger.warning("concept_graph_unavailable", path=path)
        return None
    return graph


def _parse_response(raw: str, expected_ids: set[str]) -> dict[str, list[tuple[str, str]]]:
    """Tolerant JSON parse: strips code fences/preamble, drops unknown ids and junk names."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ExtractionError("no JSON object in LLM response")
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"invalid JSON from LLM: {exc}") from exc

    out: dict[str, list[tuple[str, str]]] = {}
    items = data.get("papers") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ExtractionError("LLM JSON missing 'papers' list")

    for item in items:
        if not isinstance(item, dict) or item.get("arxiv_id") not in expected_ids:
            continue
        seen: set[str] = set()
        concepts: list[tuple[str, str]] = []
        for c in item.get("concepts") or []:
            name, ctype = (c.get("name"), c.get("type")) if isinstance(c, dict) else (c, None)
            if not isinstance(name, str):
                continue
            name = re.sub(r"\s+", " ", name).strip()
            key = name.casefold()
            if not name or len(name) > _MAX_NAME_LEN or key in seen:
                continue
            seen.add(key)
            ctype = ctype.lower() if isinstance(ctype, str) else "other"
            concepts.append((name, ctype if ctype in _VALID_TYPES else "other"))
            if len(concepts) >= _MAX_CONCEPTS_PER_PAPER:
                break
        out[item["arxiv_id"]] = concepts
    return out


# ── Pickle helpers (private) ──────────────────────────────────────────────────

def _atomic_pickle_dump(obj: Any, path: str | Path) -> None:
    """Write to a temp file then os.replace(), so readers never see a half-written file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _safe_pickle_load(path: str | Path) -> Any | None:
    """
    Load a pickle this app wrote earlier. Returns None if missing or unreadable
    so callers can degrade (spec: no BM25 / no graph -> skip that retrieval leg).
    Only ever point this at files produced by _atomic_pickle_dump.
    """
    target = Path(path)
    if not target.exists():
        return None
    try:
        with target.open("rb") as fh:
            return pickle.load(fh)
    except Exception as exc:  # corrupt / version-skewed pickle
        logger.warning("pickle_load_failed", path=str(target), error=str(exc))
        return None