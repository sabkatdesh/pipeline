"""Plain dataclasses for the concept co-occurrence graph."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ConceptNode:
    key: str                # casefolded name; the NetworkX node id
    name: str               # canonical display spelling, e.g. "LoRA"
    concept_type: str       # method | model | dataset | task | other
    papers: list[str] = field(default_factory=list)  # arxiv_ids mentioning it

    @property
    def paper_count(self) -> int:
        return len(self.papers)


@dataclass(slots=True)
class ConceptEdge:
    source: str             # ConceptNode.key
    target: str             # ConceptNode.key
    weight: int             # number of papers in which both concepts appear