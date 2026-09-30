"""CLI to run evaluation for the RAG pipeline.

Usage:
  python -m scripts.evaluate --dataset path/to/eval.jsonl [--top-k 5] [--limit 100] [--output out.json]

The input JSONL should follow the format described by `app/rag/evaluation/ragas_eval.py`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Optional

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.rag.langgraph_pipeline import SimpleRAGPipeline
from app.rag.evaluation.ragas_eval import load_jsonl, evaluate_pipeline

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    cfg = get_settings()
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline on JSONL dataset")
    parser.add_argument("--dataset", "-d", required=True, help="Path to JSONL dataset")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K documents to pass to generator")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples (for quick runs)")
    parser.add_argument("--output", "-o", default=None, help="Write JSON metrics to this path")
    parser.add_argument("--examples", type=int, default=5, help="Number of example rows to keep in output (0 to omit)")
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    cfg = get_settings()
    setup_logging(cfg.log_level)

    pipeline = SimpleRAGPipeline()
    logger.info("starting_evaluation", dataset=args.dataset, top_k=args.top_k, limit=args.limit)

    samples = load_jsonl(args.dataset)
    metrics = await evaluate_pipeline(pipeline, samples, top_k=args.top_k, limit=args.limit, show_progress=True)

    # Trim examples if requested
    if args.examples is not None and args.examples >= 0:
        metrics["examples"] = metrics.get("examples", [])[: args.examples]
    else:
        metrics.pop("examples", None)

    out = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(out)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(out)
        logger.info("wrote_metrics", path=args.output)


if __name__ == "__main__":
    asyncio.run(_main())
