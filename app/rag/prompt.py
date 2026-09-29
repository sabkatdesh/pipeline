"""Prompt templates for each LangGraph node (spec section 11)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.rag.retriever import RetrievedDoc

# ── QueryAnalyzer ─────────────────────────────────────────────────────────────
QUERY_ANALYZER_SYSTEM = "You are a query analysis engine. Output JSON only."


def query_analyzer_user(question: str, history: str = "") -> str:
    convo = f"Conversation so far (resolve references like 'it' or 'that'):\n{history}\n\n" if history else ""
    return (
        f"{convo}Question: {question}\n"
        "Output:\n"
        "{\n"
        '  "type": "factual" | "comparative" | "exploratory",\n'
        '  "variants": ["...", "...", "..."],\n'
        '  "hyde": "..."\n'
        "}\n"
        "variants = 3 rephrasings of the question with different wording.\n"
        "hyde = a hypothetical paper abstract that would answer this question."
    )


# ── Query rewriter (CRAG loop) ────────────────────────────────────────────────
REWRITER_SYSTEM = "You rewrite search queries for an arXiv paper index. Return the rewritten query only."


def rewriter_user(question: str, previous_query: str) -> str:
    return (
        f"Original question: {question}\n"
        f"Previous search query (retrieved nothing relevant): {previous_query}\n"
        "Write a different search query using alternative technical terminology "
        "that might match how relevant papers describe this topic:"
    )


# ── DocumentGrader (CRAG) ─────────────────────────────────────────────────────
GRADER_SYSTEM = "You are a relevance grader. Output JSON only."


def grader_user(question: str, abstract_clean: str) -> str:
    return (
        f"Question: {question}\n"
        f"Paper abstract: {abstract_clean}\n"
        "Is this paper relevant to the question?\n"
        'Output: { "score": "relevant" | "partial" | "irrelevant", "reason": "..." }'
    )


# ── ContextCompressor ─────────────────────────────────────────────────────────
COMPRESSOR_SYSTEM = (
    "Extract only sentences directly relevant to the question. Return extracted text only."
)


def compressor_user(question: str, abstract_clean: str) -> str:
    return (
        f"Question: {question}\n"
        f"Paper text: {abstract_clean}\n"
        "Return only relevant sentences:"
    )


# ── Generator ─────────────────────────────────────────────────────────────────
GENERATOR_SYSTEM = """You are a research assistant with access to arXiv papers.
Rules:
- Use ONLY the provided papers as your source of truth
- Cite papers using [arXiv:XXXXXXX] inline
- If papers lack enough info, say so explicitly
- Synthesize across papers; do not summarize each individually
Return the answer text and the list of arXiv IDs you cited."""


def generator_user(
    question: str,
    docs: list[RetrievedDoc],
    contexts: list[str],
    unsupported_claims: list[str] | None = None,
) -> str:
    blocks = "\n\n".join(
        f"[{i}] arXiv:{d.arxiv_id} — {d.title} ({d.published_date.isoformat()})\n{ctx}"
        for i, (d, ctx) in enumerate(zip(docs, contexts, strict=True), start=1)
    )
    retry = ""
    if unsupported_claims:
        listed = "\n".join(f"- {c}" for c in unsupported_claims)
        retry = (
            "\n\nYour previous draft made claims NOT supported by the papers. "
            f"Do not repeat them:\n{listed}"
        )
    return f"Question: {question}\n\nPapers:\n{blocks}{retry}\n\nAnswer:"


# ── FaithfulnessChecker ───────────────────────────────────────────────────────
FAITHFULNESS_SYSTEM = "You are a fact-checker. Output JSON only."


def faithfulness_user(answer: str, docs: list[RetrievedDoc], contexts: list[str]) -> str:
    sources = "\n\n".join(f"[arXiv:{d.arxiv_id}] {c}" for d, c in zip(docs, contexts, strict=True))
    return (
        f"Answer: {answer}\n"
        f"Source papers: {sources}\n"
        "Does the answer make any claims NOT supported by the source papers?\n"
        'Output: { "hallucination_detected": true | false, "unsupported_claims": ["..."] }'
    )