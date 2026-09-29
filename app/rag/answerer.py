"""
LangChain-Anthropic plumbing shared by every LLM node.

  llm_call / llm_json / generate_answer   - the calls
  LLMUnavailableError / LLMRateLimitError - mapped to HTTP 503 / 429 by the API layer
"""

from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

import anthropic
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_RETRY_AFTER = 30
_CITE_RE = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})", re.IGNORECASE)


class LLMServiceError(Exception):
    retry_after: int = _DEFAULT_RETRY_AFTER


class LLMUnavailableError(LLMServiceError):
    """Anthropic timed out / is down / rejected our key -> HTTP 503 + Retry-After."""

    def __init__(self, message: str, retry_after: int = _DEFAULT_RETRY_AFTER) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMRateLimitError(LLMServiceError):
    """Anthropic rate limit -> HTTP 429 + Retry-After from the response headers."""

    def __init__(self, message: str, retry_after: int = _DEFAULT_RETRY_AFTER) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class GeneratedAnswer(BaseModel):
    answer: str = Field(description="Answer text with inline [arXiv:ID] citations")
    cited_arxiv_ids: list[str] = Field(default_factory=list, description="arXiv IDs cited")


def elapsed_ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


@lru_cache(maxsize=8)
def get_llm(max_tokens: int | None = None) -> ChatAnthropic:
    cfg = get_settings()
    if not cfg.anthropic_api_key:
        raise LLMUnavailableError("ANTHROPIC_API_KEY is not configured")
    return ChatAnthropic(
        model=cfg.llm_model,
        max_tokens=max_tokens or cfg.llm_max_tokens,
        temperature=0,
        api_key=cfg.anthropic_api_key,
        timeout=60,
        max_retries=1,
    )


@contextmanager
def _translate_errors():
    try:
        yield
    except anthropic.RateLimitError as exc:
        retry = _DEFAULT_RETRY_AFTER
        try:
            retry = int(float(exc.response.headers.get("retry-after", retry)))
        except (AttributeError, ValueError):
            pass
        raise LLMRateLimitError("LLM rate limit reached", retry) from exc
    except (anthropic.APITimeoutError, anthropic.APIConnectionError) as exc:
        raise LLMUnavailableError("LLM service temporarily unavailable") from exc
    except anthropic.APIStatusError as exc:  # 5xx, 401/403, overloaded, bad request
        logger.error("llm_api_error", status=exc.status_code, error=str(exc))
        raise LLMUnavailableError("LLM service temporarily unavailable") from exc


def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


async def llm_call(system: str, user: str, max_tokens: int | None = None) -> str:
    llm = get_llm(max_tokens)
    with _translate_errors():
        msg = await llm.ainvoke([("system", system), ("human", user)])
    return _as_text(msg.content).strip()


def parse_json(raw: str) -> dict | None:
    """Tolerant: strips code fences / chatter around the outermost {...}."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


async def llm_json(system: str, user: str, max_tokens: int | None = None) -> dict | None:
    """Returns None if the model's output isn't valid JSON (callers pick a fallback)."""
    return parse_json(await llm_call(system, user, max_tokens))


def normalize_arxiv_id(value: str) -> str:
    value = value.strip()
    if value.lower().startswith("arxiv:"):
        value = value[6:].strip()
    return re.sub(r"v\d+$", "", value)


async def generate_answer(system: str, user: str) -> GeneratedAnswer:
    """Structured output (answer + cited ids); falls back to plain text + regex."""
    llm = get_llm().with_structured_output(GeneratedAnswer)
    with _translate_errors():
        result = await llm.ainvoke([("system", system), ("human", user)])
    if isinstance(result, dict):
        result = GeneratedAnswer(**result)
    if isinstance(result, GeneratedAnswer) and result.answer.strip():
        return result

    text = await llm_call(system, user)
    return GeneratedAnswer(answer=text, cited_arxiv_ids=_CITE_RE.findall(text))


def cited_ids_in(text: str) -> list[str]:
    return _CITE_RE.findall(text)