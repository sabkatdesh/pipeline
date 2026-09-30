"""StateGraph-based RAG pipeline orchestrator.

This file implements a `StateGraph` orchestrator that wires together the
RAG "nodes" into a small state-machine. The historical `SimpleRAGPipeline`
remains as a thin wrapper around the `StateGraph` for backward compat.

Nodes are implemented as lightweight, local objects that call into the
existing retriever and answerer helpers. This keeps behavior identical to
the previous `SimpleRAGPipeline` while making it straightforward to
replace nodes with implementations under `app/rag/nodes/*` later.
"""
from __future__ import annotations

from time import perf_counter
from typing import List, Dict, Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.retriever import HybridRetriever, RetrievedDoc, confidence_from_score
from app.rag.answerer import (
    generate_answer,
    llm_call,
    llm_json,
    LLMUnavailableError,
    LLMRateLimitError,
)
from app.rag.prompt import (
    GENERATOR_SYSTEM,
    generator_user,
    REWRITER_SYSTEM,
    rewriter_user,
    FAITHFULNESS_SYSTEM,
    faithfulness_user,
)
from app.schemas.rag import ConfidenceLevel

logger = get_logger(__name__)

NO_MATCH_ANSWER = (
    "I couldn't find papers in the ingested arXiv dataset that answer this question, "
    "so I can't give a grounded answer."
)


class Node:
    """Base node interface. Nodes receive a mutable `state` dict and may
    update it in-place.
    """

    async def run(self, state: Dict[str, Any]) -> None:  # pragma: no cover - thin adapter
        return None


class QueryAnalyzerNode(Node):
    """Analyze the question and produce variants / hyde strings.

    Current implementation is a pass-through; a fuller implementation can
    live under `app/rag/nodes/query_analyzer.py` and be plugged in later.
    """

    async def run(self, state: Dict[str, Any]) -> None:
        state.setdefault("variants", [])
        state.setdefault("hyde", "")


class RetrieverNode(Node):
    def __init__(self) -> None:
        self._retriever = HybridRetriever()

    async def run(self, state: Dict[str, Any]) -> None:
        question = state.get("search_query") or state.get("question", "")
        variants = state.get("variants", [])
        hyde = state.get("hyde", "")
        t0 = perf_counter()
        docs = await self._retriever.retrieve(question, variants=variants, hyde=hyde)
        state["docs"] = docs
        state["retrieval_ms"] = int((perf_counter() - t0) * 1000)
        state["confidence"] = confidence_from_score(max((d.similarity_score for d in docs), default=None))
        logger.debug("retriever_node", q=question, found=len(docs), confidence=state["confidence"].name)


class RerankerNode(Node):
    """Placeholder reranker. If a cross-encoder reranker is added later, it
    can be called here to populate `d.reranker_score` on each doc.
    """

    async def run(self, state: Dict[str, Any]) -> None:
        # no-op for now; keep existing doc order
        return None


class GeneratorNode(Node):
    async def run(self, state: Dict[str, Any]) -> None:
        question = state.get("question", "")
        docs: List[RetrievedDoc] = state.get("docs", []) or []
        top_k = state.get("top_k", 5)
        top_docs = docs[:top_k]
        state["top_docs"] = top_docs
        contexts = [d.abstract_clean for d in top_docs]
        user_prompt = generator_user(question, top_docs, contexts)

        t1 = perf_counter()
        generated = await generate_answer(GENERATOR_SYSTEM, user_prompt)
        state["llm_ms"] = int((perf_counter() - t1) * 1000)
        # generated may be a dataclass or model with .answer and .cited_arxiv_ids
        if hasattr(generated, "answer"):
            state["answer"] = getattr(generated, "answer")
        else:
            state["answer"] = str(generated)
        # store the raw generated object for downstream nodes
        state["generated"] = generated
        # best-effort: model id
        state.setdefault("model_used", get_settings().llm_model)
        logger.debug("generator_node", answer_len=len(state["answer"]))


class FaithfulnessNode(Node):
    async def run(self, state: Dict[str, Any]) -> None:
        gen = state.get("generated")
        top_docs: List[RetrievedDoc] = state.get("top_docs", [])
        contexts = [d.abstract_clean for d in top_docs]
        try:
            chk = await llm_json(FAITHFULNESS_SYSTEM, faithfulness_user(state.get("answer", ""), top_docs, contexts))
            if isinstance(chk, dict):
                state["faithfulness_passed"] = not bool(chk.get("hallucination_detected", False))
            else:
                state["faithfulness_passed"] = None
        except Exception as exc:
            logger.debug("faithfulness_failed", error=str(exc))
            state["faithfulness_passed"] = None


class DocGraderNode(Node):
    """Placeholder doc grader: could score or filter docs. No-op for now."""

    async def run(self, state: Dict[str, Any]) -> None:
        return None


class CompressorNode(Node):
    """Optional answer compressor/shortener. Currently leaves the answer as-is."""

    async def run(self, state: Dict[str, Any]) -> None:
        ans = state.get("answer")
        if isinstance(ans, str):
            state["answer"] = ans.strip()


class StateGraph:
    """Orchestrator that wires nodes into a simple stateful flow.

    The flow implemented here mirrors the previous CRAG-like loop:
      - retrieve
      - if no docs -> rewrite (via LLM) and retry up to cfg.rag_max_retry_iterations
      - generate
      - faithfulness check

    Nodes are executed in sequence and can be replaced with richer
    implementations under `app/rag/nodes/` in the future.
    """

    def __init__(self) -> None:
        self._cfg = get_settings()
        self._logger = logger
        self._query_node = QueryAnalyzerNode()
        self._retriever_node = RetrieverNode()
        self._reranker_node = RerankerNode()
        self._generator_node = GeneratorNode()
        self._faith_node = FaithfulnessNode()
        self._grader_node = DocGraderNode()
        self._compressor_node = CompressorNode()

    async def run(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "question": question,
            "top_k": top_k,
            "variants": [],
            "hyde": "",
            "docs": [],
            "retrieval_ms": 0,
            "llm_ms": 0,
            "iterations": 0,
            "model_used": self._cfg.llm_model,
        }

        last_query = ""
        iterations = 0

        # CRAG-style retry loop (rewrite + retrieve)
        while iterations < self._cfg.rag_max_retry_iterations:
            # analyze (noop-able)
            await self._query_node.run(state)

            # retrieve
            await self._retriever_node.run(state)

            docs: List[RetrievedDoc] = state.get("docs", []) or []
            best_score = max((d.similarity_score for d in docs), default=None)
            confidence = confidence_from_score(best_score)
            state["confidence"] = confidence

            if confidence != ConfidenceLevel.none or iterations + 1 >= self._cfg.rag_max_retry_iterations:
                break

            # try rewriting the query via LLM
            try:
                rewritten = await llm_call(REWRITER_SYSTEM, rewriter_user(state["question"], last_query or state["question"]))
            except (LLMRateLimitError, LLMUnavailableError) as exc:
                # The rewrite is best-effort; generation surfaces real LLM outages as 429/503.
                logger.warning("query_rewrite_skipped", error=str(exc))
                break
            rewritten = (rewritten or "").strip()
            if not rewritten or rewritten == (state.get("search_query") or state["question"]):
                break

            last_query = state.get("search_query") or state["question"]
            state["search_query"] = rewritten
            iterations += 1

        state["iterations"] = iterations + 1

        # Keep only papers that are actually similar to the question. If nothing
        # clears the threshold, answer honestly without calling the LLM.
        threshold = self._cfg.rag_similarity_threshold
        docs = [d for d in (state.get("docs", []) or []) if d.similarity_score >= threshold]
        state["docs"] = docs
        if not docs:
            return {
                "answer": NO_MATCH_ANSWER,
                "docs": [],
                "confidence": ConfidenceLevel.none,
                "iterations": state.get("iterations", 0),
                "retrieval_ms": state.get("retrieval_ms", 0),
                "llm_ms": state.get("llm_ms", 0),
                "faithfulness_passed": None,
                "model_used": state.get("model_used", self._cfg.llm_model),
            }

        # Rerank (noop by default)
        await self._reranker_node.run(state)

        # Generate answer
        await self._generator_node.run(state)

        # Faithfulness
        await self._faith_node.run(state)

        # Doc grading and compression (no-op placeholders)
        await self._grader_node.run(state)
        await self._compressor_node.run(state)

        top_docs = state.get("top_docs", [])

        return {
            "answer": state.get("answer", ""),
            "docs": top_docs,
            "confidence": confidence_from_score(max((d.similarity_score for d in top_docs), default=None)),
            "iterations": state.get("iterations", 0),
            "retrieval_ms": state.get("retrieval_ms", 0),
            "llm_ms": state.get("llm_ms", 0),
            "faithfulness_passed": state.get("faithfulness_passed", None),
            "model_used": state.get("model_used", self._cfg.llm_model),
        }


class SimpleRAGPipeline:
    """Backward-compatible wrapper around `StateGraph` used by the API."""

    def __init__(self) -> None:
        self._graph = StateGraph()

    async def ask(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        return await self._graph.run(question, top_k=top_k)
