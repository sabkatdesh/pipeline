"""RAG pipeline behaviour with the retriever and LLM stubbed out (no DB, no network)."""

from datetime import date

import pytest

from app.rag import langgraph_pipeline as lp
from app.rag.answerer import GeneratedAnswer
from app.rag.retriever import RetrievedDoc
from app.schemas.rag import ConfidenceLevel


def _doc(sim: float) -> RetrievedDoc:
    return RetrievedDoc(
        arxiv_id="2601.00001", title="Attention", abstract_clean="We study attention.",
        published_date=date(2026, 1, 2), similarity_score=sim,
    )


@pytest.fixture
def stub_llm(monkeypatch):
    calls = {"generate": 0}

    async def fake_generate(system, user):
        calls["generate"] += 1
        return GeneratedAnswer(answer="Attention is studied [arXiv:2601.00001].", cited_arxiv_ids=["2601.00001"])

    async def fake_json(system, user, max_tokens=None):
        return {"hallucination_detected": False, "unsupported_claims": []}

    async def fake_call(system, user, max_tokens=None):
        return "rewritten query"

    monkeypatch.setattr(lp, "generate_answer", fake_generate)
    monkeypatch.setattr(lp, "llm_json", fake_json)
    monkeypatch.setattr(lp, "llm_call", fake_call)
    return calls


def _stub_retriever(monkeypatch, docs):
    async def fake_retrieve(self, query, variants, hyde=""):
        return docs

    monkeypatch.setattr(lp.HybridRetriever, "retrieve", fake_retrieve)


async def test_no_relevant_papers_returns_honest_answer_without_calling_llm(monkeypatch, stub_llm):
    _stub_retriever(monkeypatch, [_doc(0.05)])
    out = await lp.SimpleRAGPipeline().ask("What is the airspeed of a swallow?")
    assert out["answer"] == lp.NO_MATCH_ANSWER
    assert out["docs"] == [] and out["confidence"] == ConfidenceLevel.none
    assert stub_llm["generate"] == 0


async def test_relevant_papers_produce_grounded_answer_with_sources(monkeypatch, stub_llm):
    _stub_retriever(monkeypatch, [_doc(0.7)])
    out = await lp.SimpleRAGPipeline().ask("What is attention?")
    assert "2601.00001" in out["answer"]
    assert [d.arxiv_id for d in out["docs"]] == ["2601.00001"]
    assert out["confidence"] == ConfidenceLevel.high
    assert out["faithfulness_passed"] is True
