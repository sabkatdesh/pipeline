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


class _AzureChat:
    """Minimal chat adapter over Azure OpenAI (fallback when no ANTHROPIC_API_KEY).

    Exposes the same `ainvoke` / `with_structured_output` surface as ChatAnthropic
    for the subset the app uses.
    """

    def __init__(self, max_tokens: int | None) -> None:
        import openai

        cfg = get_settings()
        self._cfg = cfg
        self._client = openai.AsyncAzureOpenAI(
            azure_endpoint=cfg.azure_openai_api_base.rstrip("/"),
            api_key=cfg.azure_openai_api_key,
            api_version=cfg.azure_openai_api_version,
            max_retries=1,
            timeout=60.0,
        )
        self._deployment = cfg.azure_openai_chat_deployment or cfg.llm_model
        self._max_tokens = max_tokens or cfg.llm_max_tokens

    async def ainvoke(self, messages: list[tuple[str, str]]):
        import openai

        payload = [
            {"role": "system" if role == "system" else "user", "content": text}
            for role, text in messages
        ]
        try:
            resp = await self._client.chat.completions.create(
                model=self._deployment, messages=payload, max_completion_tokens=self._max_tokens
            )
        except openai.RateLimitError as exc:
            raise LLMRateLimitError("LLM rate limit reached") from exc
        except openai.OpenAIError as exc:
            logger.error("azure_chat_error", error=str(exc))
            raise LLMUnavailableError("LLM service temporarily unavailable") from exc

        class _Resp:
            content = resp.choices[0].message.content or ""

        return _Resp()

    def with_structured_output(self, schema):
        parent = self

        class _Structured:
            async def ainvoke(self, messages):
                resp = await parent.ainvoke(messages)
                data = parse_json(resp.content)
                return data if data is not None else resp.content

        return _Structured()


@lru_cache(maxsize=8)
def get_llm(max_tokens: int | None = None):
    """Chat model: Anthropic when ANTHROPIC_API_KEY is set, else Azure OpenAI."""
    cfg = get_settings()
    if cfg.anthropic_api_key:
        return ChatAnthropic(
            model=cfg.llm_model,
            max_tokens=max_tokens or cfg.llm_max_tokens,
            temperature=0,
            api_key=cfg.anthropic_api_key,
            timeout=60,
            max_retries=1,
        )
    if cfg.azure_configured:
        return _AzureChat(max_tokens)
    raise LLMUnavailableError(
        "No LLM configured: set ANTHROPIC_API_KEY (or AZURE_OPENAI_API_BASE + AZURE_OPENAI_API_KEY)"
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


def normalize_arxiv_id(value: str) -> str:
    """'arXiv:2601.00001v2' / 'https://arxiv.org/abs/2601.00001' -> '2601.00001'."""
    m = re.search(r"(\d{4}\.\d{4,5})", value or "")
    return m.group(1) if m else (value or "").strip()


def cited_ids_in(text: str) -> list[str]:
    return _CITE_RE.findall(text)
