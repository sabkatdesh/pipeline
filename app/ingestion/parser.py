import re
import xml.etree.ElementTree as ET

from pydantic import BaseModel, Field

# Atom / arXiv XML namespaces
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

# Strips version suffix: '.../abs/2601.00001v2' → '2601.00001'
_ID_RE = re.compile(r"arxiv\.org/abs/([^v\s]+)")


class RawPaper(BaseModel):
    """
    One arXiv paper entry as parsed from the feed.

    All fields are optional/have defaults — the feed can omit almost anything.
    Stricter validation happens in the pipeline after normalization.
    """

    arxiv_id: str
    title: str = ""
    abstract: str = ""
    published: str | None = None
    updated: str | None = None
    authors: list[str] = Field(default_factory=list)
    primary_category: str | None = None
    categories: list[str] = Field(default_factory=list)
    doi: str | None = None
    journal_ref: str | None = None


class FeedMeta(BaseModel):
    total_results: int = 0
    start_index: int = 0
    items_per_page: int = 0


class ParsedFeed(BaseModel):
    meta: FeedMeta
    papers: list[RawPaper]


def parse_feed(xml_bytes: bytes) -> ParsedFeed:
    """
    Parse an arXiv Atom/XML response into a structured ParsedFeed.

    Raises ValueError on malformed XML so callers can log and skip the batch.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise ValueError(f"Malformed arXiv XML: {exc}") from exc

    meta = FeedMeta(
        total_results=_int_text(root, "opensearch:totalResults"),
        start_index=_int_text(root, "opensearch:startIndex"),
        items_per_page=_int_text(root, "opensearch:itemsPerPage"),
    )

    papers = [_parse_entry(e) for e in root.findall("atom:entry", _NS)]
    return ParsedFeed(meta=meta, papers=papers)


# ── Private helpers ───────────────────────────────────────────────────────────

def _parse_entry(entry: ET.Element) -> RawPaper:
    raw_id = _text(entry, "atom:id") or ""
    arxiv_id = _strip_version(raw_id)

    authors = [
        name
        for author in entry.findall("atom:author", _NS)
        if (name := _text(author, "atom:name") or "").strip()
    ]

    primary_el = entry.find("arxiv:primary_category", _NS)
    primary = primary_el.get("term") if primary_el is not None else None

    # Deduplicated, preserving primary first if present.
    all_cats: list[str] = []
    seen: set[str] = set()
    if primary:
        all_cats.append(primary)
        seen.add(primary)
    for cat_el in entry.findall("atom:category", _NS):
        code = cat_el.get("term", "").strip()
        if code and code not in seen:
            all_cats.append(code)
            seen.add(code)

    doi_el = entry.find("arxiv:doi", _NS)
    jref_el = entry.find("arxiv:journal_ref", _NS)

    return RawPaper(
        arxiv_id=arxiv_id,
        title=(_text(entry, "atom:title") or "").replace("\n", " ").strip(),
        abstract=(_text(entry, "atom:summary") or "").strip(),
        published=_text(entry, "atom:published"),
        updated=_text(entry, "atom:updated"),
        authors=authors,
        primary_category=primary,
        categories=all_cats,
        doi=(doi_el.text or "").strip() or None if doi_el is not None else None,
        journal_ref=(jref_el.text or "").strip() or None if jref_el is not None else None,
    )


def _strip_version(raw_id: str) -> str:
    match = _ID_RE.search(raw_id)
    return match.group(1) if match else raw_id


def _text(element: ET.Element, tag: str) -> str | None:
    el = element.find(tag, _NS)
    return el.text if el is not None else None


def _int_text(element: ET.Element, tag: str) -> int:
    val = _text(element, tag)
    try:
        return int(val) if val else 0
    except ValueError:
        return 0
