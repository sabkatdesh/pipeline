"""
Graph-hop retrieval: find papers that share >= 2 concepts with the query.

1. Match concept names that literally appear in the query texts (question + variants).
2. Expand one hop to each matched concept's strongest co-occurring neighbours.
3. Papers touching >= MIN_OVERLAP concepts of that set are candidates, ranked by
   (direct matches, total overlap).
"""

from __future__ import annotations

import re
from collections import Counter

import networkx as nx

MIN_OVERLAP = 2
NEIGHBOURS_PER_CONCEPT = 3

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_index_cache: tuple[nx.Graph, dict[str, str]] | None = None


def _norm(text: str) -> str:
    return _NON_ALNUM.sub(" ", text.lower()).strip()


def _name_index(graph: nx.Graph) -> dict[str, str]:
    """normalized concept name -> node key (cached per graph object)."""
    global _index_cache
    if _index_cache is not None and _index_cache[0] is graph:
        return _index_cache[1]
    index: dict[str, str] = {}
    for key, data in graph.nodes(data=True):
        norm = _norm(data.get("name", key))
        if len(norm) >= 2:
            index[norm] = key
    _index_cache = (graph, index)
    return index


def find_query_concepts(graph: nx.Graph, texts: list[str]) -> set[str]:
    haystack = f" {' '.join(_norm(t) for t in texts)} "
    return {key for norm, key in _name_index(graph).items() if f" {norm} " in haystack}


def traverse(
    graph: nx.Graph,
    texts: list[str],
    max_results: int = 10,
    min_overlap: int = MIN_OVERLAP,
) -> list[tuple[str, int]]:
    """Return up to max_results (arxiv_id, concept_overlap), best first."""
    direct = find_query_concepts(graph, texts)
    if not direct:
        return []

    expanded = set(direct)
    for key in direct:
        neighbours = sorted(
            graph[key].items(), key=lambda kv: kv[1].get("weight", 0), reverse=True
        )
        expanded.update(n for n, _ in neighbours[:NEIGHBOURS_PER_CONCEPT])

    total: Counter[str] = Counter()
    direct_hits: Counter[str] = Counter()
    for key in expanded:
        for pid in graph.nodes[key].get("papers", []):
            total[pid] += 1
            if key in direct:
                direct_hits[pid] += 1

    ranked = sorted(
        (pid for pid, n in total.items() if n >= min_overlap),
        key=lambda pid: (-direct_hits[pid], -total[pid], pid),
    )
    return [(pid, total[pid]) for pid in ranked[:max_results]]