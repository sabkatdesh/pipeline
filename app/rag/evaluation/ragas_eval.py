"""RAG evaluation utilities.

This module provides a small, self-contained evaluation harness for the
project's `SimpleRAGPipeline` implementation. It expects a JSONL dataset
where each line is a JSON object with at least a `question` field and one of
`answers` (list) or `answer` (str). Optional `sources` or `citations` may
contain ground-truth arXiv ids for citation-based scoring.

Example dataset line:
  {"id":"q1","question":"What is transformer?","answers":["A model..."],"sources":["2101.00001"]}

The evaluation computes:
 - token-level F1 (best match against gold answers)
 - exact-match (normalized)
 - citation recall (fraction of gold cited ids recovered by the model)

The implementation is deliberately lightweight and synchronous (runs the
RAG pipeline sequentially). It is intended as a reproducible baseline and
as a starting point for integrating `ragas` or other evaluation tooling.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Iterable, Dict, Any, List

from app.core.logging import get_logger
from app.rag.answerer import normalize_arxiv_id

logger = get_logger(__name__)

_WORD_RE = re.compile(r"\w+")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _WORD_RE.findall(text.lower())


def _f1_score(pred: str, gold: str) -> float:
    p = _tokenize(pred)
    g = _tokenize(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    pc = Counter(p)
    gc = Counter(g)
    common = sum((pc & gc).values())
    if common == 0:
        return 0.0
    prec = common / sum(pc.values())
    rec = common / sum(gc.values())
    return 2 * prec * rec / (prec + rec)


def _best_f1(pred: str, golds: Iterable[str]) -> float:
    return max((_f1_score(pred, g) for g in golds), default=0.0)


def _normalize_text_for_em(text: str) -> str:
    # Lowercase, keep word characters and collapse spacing
    return " ".join(_tokenize(text))


def _exact_match(pred: str, golds: Iterable[str]) -> bool:
    n = _normalize_text_for_em(pred)
    for g in golds:
        if n == _normalize_text_for_em(g):
            return True
    return False


def _normalize_ids(ids: Iterable[str]) -> List[str]:
    out: List[str] = []
    for i in (ids or []):
        if i is None:
            continue
        try:
            out.append(normalize_arxiv_id(str(i)))
        except Exception:
            out.append(str(i))
    return out


def load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file (skips empty lines)."""
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            yield json.loads(ln)


async def evaluate_pipeline(
    pipeline,
    samples: Iterable[Dict[str, Any]],
    top_k: int = 5,
    limit: int | None = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Run the pipeline over `samples` and return aggregated metrics.

    `samples` is an iterable of dicts; each dict should contain either
    `answers` (list[str]) or `answer` (str) as ground truth. Optional
    `sources`/`citations` provide ground-truth arXiv ids.

    The return dict contains overall metrics and a short per-example
    summary under the `examples` key.
    """
    total = 0
    em_count = 0
    f1_sum = 0.0
    citation_recall_sum = 0.0
    citation_counted = 0
    examples: List[Dict[str, Any]] = []

    for row in samples:
        if limit is not None and total >= limit:
            break

        qid = row.get("id") or f"q{total}"
        question = row.get("question") or row.get("query") or ""

        golds = []
        if "answers" in row and isinstance(row.get("answers"), list):
            golds = [g for g in row.get("answers") if g is not None]
        elif "answer" in row and row.get("answer") is not None:
            golds = [row.get("answer")]

        gold_citations = row.get("sources") or row.get("citations") or row.get("arxiv_ids") or []

        try:
            out = await pipeline.ask(question, top_k=top_k)
        except Exception as exc:  # pragma: no cover - runtime errors are surfaced
            logger.exception("evaluation: pipeline.ask failed", error=str(exc), question=question)
            predicted = ""
            pred_ids: List[str] = []
        else:
            # Extract predicted answer and cited ids from the pipeline output.
            predicted = ""
            pred_ids = []
            if isinstance(out, dict):
                if "answer" in out and isinstance(out["answer"], str):
                    predicted = out["answer"]
                elif "generated" in out:
                    gen = out["generated"]
                    # generated may be a dataclass-like object; be permissive
                    if hasattr(gen, "answer"):
                        predicted = str(getattr(gen, "answer") or "")
                    elif isinstance(gen, dict):
                        predicted = gen.get("answer", "") or ""

                if "cited_arxiv_ids" in out and isinstance(out["cited_arxiv_ids"], list):
                    pred_ids = [str(x) for x in out["cited_arxiv_ids"]]
                elif "generated" in out:
                    gen = out["generated"]
                    if hasattr(gen, "cited_arxiv_ids"):
                        pred_ids = list(getattr(gen, "cited_arxiv_ids") or [])
                    elif isinstance(gen, dict):
                        pred_ids = gen.get("cited_arxiv_ids") or []

        # Metrics
        this_em = 1 if (golds and _exact_match(predicted, golds)) else 0
        this_f1 = _best_f1(predicted, golds) if golds else 0.0

        pred_ids_norm = _normalize_ids(pred_ids)
        gold_ids_norm = _normalize_ids(gold_citations)

        citation_recall = None
        if gold_ids_norm:
            matched = len(set(pred_ids_norm) & set(gold_ids_norm))
            citation_recall = matched / len(set(gold_ids_norm))
            citation_recall_sum += citation_recall
            citation_counted += 1

        examples.append(
            {
                "id": qid,
                "question": question,
                "predicted": predicted,
                "gold_count": len(golds),
                "em": bool(this_em),
                "f1": this_f1,
                "predicted_citations": pred_ids_norm,
                "gold_citations": gold_ids_norm,
                "citation_recall": citation_recall,
            }
        )

        total += 1
        em_count += this_em
        f1_sum += this_f1

        if show_progress and total % 10 == 0:
            logger.info("evaluation_progress", processed=total)

    metrics: Dict[str, Any] = {
        "total": total,
        "exact_match": em_count,
        "exact_match_rate": (em_count / total) if total else 0.0,
        "avg_f1": (f1_sum / total) if total else 0.0,
        "citation_recall_avg": (citation_recall_sum / citation_counted) if citation_counted else None,
        "examples": examples,
    }
    return metrics
