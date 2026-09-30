from app.preprocessing.cleaner import clean_abstract
from app.preprocessing.normalizer import normalize_author_name, normalize_date


def test_clean_abstract_strips_latex_entities_and_whitespace():
    raw = "We study \\textbf{sparse}   attention &amp; $O(n^2)$\n cost."
    assert clean_abstract(raw) == "We study sparse attention & [MATH] cost."


def test_clean_abstract_handles_empty_and_truncates():
    assert clean_abstract("") == ""
    assert len(clean_abstract("a " * 5000)) <= 2000


def test_normalize_date_parses_z_suffix_and_falls_back():
    dt = normalize_date("2026-03-09T12:30:00Z")
    assert (dt.year, dt.month, dt.day) == (2026, 3, 9) and dt.tzinfo is not None
    assert normalize_date(None, fallback=dt) == dt
    assert normalize_date("garbage", fallback=dt) == dt


def test_normalize_author_name():
    assert normalize_author_name("  Yann   LeCun ") == "yann lecun"
