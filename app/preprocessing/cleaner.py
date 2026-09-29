import html
import re
import unicodedata

# Safety cap: keeps abstract within model token limits for embedding.
_MAX_LENGTH = 2000

# Applied in this exact order — sequence matters.
_LATEX_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\\[a-zA-Z]+\{([^}]*)\}"), r"\1"),  # \textbf{word} → word
    (re.compile(r"\$[^$]+\$"), "[MATH]"),              # $x^2$ → [MATH]
    (re.compile(r"\\\([^)]+\\\)"), "[MATH]"),          # \(...\) → [MATH]
    (re.compile(r"\\\[[^\]]+\\\]"), "[MATH]"),         # \[...\] → [MATH]
    (re.compile(r"\\[a-zA-Z]+"), ""),                  # remaining bare \commands
]

_WHITESPACE = re.compile(r"\s+")


def clean_abstract(text: str) -> str:
    """
    Normalize an arXiv abstract for storage and downstream embedding.

    Pipeline (order is significant):
      1. HTML entity decoding   — &amp; → &
      2. LaTeX stripping        — \textbf{x} → x, $x^2$ → [MATH]
      3. Unicode NFC            — canonicalize composed forms
      4. Whitespace collapse    — all runs → single space
      5. Length truncation      — cap at 2000 chars for embedding safety
    """
    if not text:
        return ""

    text = html.unescape(text)

    for pattern, replacement in _LATEX_RULES:
        text = pattern.sub(replacement, text)

    text = unicodedata.normalize("NFC", text)
    text = _WHITESPACE.sub(" ", text).strip()

    return text[:_MAX_LENGTH]
