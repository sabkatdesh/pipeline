from datetime import date, timedelta

from app.embeddings.indexer import build_bm25_index_from_rows, tokenize
from app.ingestion.checkpoint import is_window_done, resume_offset
from app.ingestion.pipeline import _weekly_windows
from app.rag.graph.builder import assemble_graph
from app.rag.graph.traverser import traverse
from app.rag.retriever import confidence_from_score, rrf_fuse
from app.schemas.rag import ConfidenceLevel


def test_rrf_prefers_papers_ranked_by_multiple_legs():
    fused = rrf_fuse({"dense": ["a", "b", "c"], "sparse": ["b", "a"], "graph": []})
    ids = [pid for pid, _, _ in fused]
    assert set(ids[:2]) == {"a", "b"} and ids[2] == "c"


def test_confidence_bands_and_no_match():
    assert confidence_from_score(0.9) == ConfidenceLevel.high
    assert confidence_from_score(0.5) == ConfidenceLevel.medium
    assert confidence_from_score(0.35) == ConfidenceLevel.low
    assert confidence_from_score(0.1) == ConfidenceLevel.none
    assert confidence_from_score(None) == ConfidenceLevel.none


def test_bm25_ranks_keyword_match_first():
    index = build_bm25_index_from_rows(
        [
            ("1", "Protein folding", "Predicting protein structure with deep nets"),
            ("2", "Retrieval augmented generation", "RAG combines retrieval with language models"),
            ("3", "Graph networks", "Message passing on molecules"),
        ]
    )
    assert index.search("retrieval augmented generation")[0][0] == "2"
    assert index.search("zzzz") == []
    assert "the" not in tokenize("the retrieval")


def test_concept_graph_traversal_finds_papers_sharing_two_concepts():
    graph = assemble_graph(
        [
            ("p1", "LoRA", "method"), ("p1", "LLaMA", "model"),
            ("p2", "LoRA", "method"), ("p2", "LLaMA", "model"),
            ("p3", "ImageNet", "dataset"),
        ]
    )
    hits = dict(traverse(graph, ["How does LoRA fine-tune LLaMA?"]))
    assert set(hits) == {"p1", "p2"}


def test_weekly_windows_cover_range_without_gaps():
    wins = _weekly_windows(date(2026, 1, 1), date(2026, 1, 16))
    assert wins[0] == (date(2026, 1, 1), date(2026, 1, 7))
    assert wins[-1][1] == date(2026, 1, 16)
    assert all(b[0] == a[1] + timedelta(days=1) for a, b in zip(wins, wins[1:]))


def test_checkpoint_helpers():
    cp = {"done": ["cs.AI:2026-01-01"], "current": {"category": "cs.LG", "date_window": "2026-01-08", "offset": 40}}
    assert is_window_done(cp, "cs.AI", "2026-01-01") and not is_window_done(cp, "cs.LG", "2026-01-01")
    assert resume_offset(cp, "cs.LG", "2026-01-08") == 40
    assert resume_offset(cp, "cs.AI", "2026-01-08") == 0
    assert resume_offset(None, "cs.AI", "x") == 0
