import re
import unicodedata
from datetime import datetime, timezone

_WHITESPACE = re.compile(r"\s+")


def normalize_date(
    date_str: str | None,
    fallback: datetime | None = None,
) -> datetime:
    """
    Parse an ISO 8601 / RFC 3339 date string into a timezone-aware UTC datetime.

    Falls back to `fallback` (or now()) when the string is absent or unparseable.
    """
    if not date_str:
        return fallback or _now()

    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, AttributeError):
        return fallback or _now()


def normalize_author_name(name: str) -> str:
    """
    Produce a canonical form of an author name for deduplication.

    '  Yann  LeCun  ' and 'Yann LeCun' both map to 'yann lecun'.
    """
    name = name.strip()
    name = unicodedata.normalize("NFC", name)
    name = name.lower()
    name = _WHITESPACE.sub(" ", name)
    return name


def _now() -> datetime:
    return datetime.now(timezone.utc)
