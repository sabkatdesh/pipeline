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
def get_llm(max_tokens: int | None = None):
    """Return a langchain-compatible chat model.

    Priority:
      1. Anthropic (if ANTHROPIC_API_KEY is set)
      2. Azure/OpenAI (if AZURE_OPENAI_API_BASE + key OR OPENAI_API_KEY is set)

    The returned object implements an async `ainvoke(messages)` method where
    `messages` is a list of `(role, text)` tuples where `role` is 'system' or
    'human'. It also supports `with_structured_output(schema)` returning an
    object whose `ainvoke` returns parsed JSON / dict when possible.
    """
    cfg = get_settings()

    # 1) Anthropic (preferred if configured)
    if cfg.anthropic_api_key:
        return ChatAnthropic(
            model=cfg.llm_model,
            max_tokens=max_tokens or cfg.llm_max_tokens,
            temperature=0,
            api_key=cfg.anthropic_api_key,
            timeout=60,
            max_retries=1,
        )

    # 2) Azure/OpenAI (use openai.AsyncOpenAI under the hood)
    # We implement a small adapter that provides the `ainvoke` and
    # `with_structured_output` methods the rest of the app expects.
    openai_cfg_present = bool(cfg.azure_openai_api_base and cfg.azure_openai_api_key) or bool(
        getattr(cfg, "openai_api_key", None)
    )
    if not openai_cfg_present:
        raise LLMUnavailableError("No LLM provider configured: set ANTHROPIC_API_KEY or AZURE_OPENAI_API_BASE+AZURE_OPENAI_API_KEY")

    try:
        import openai
    except Exception as exc:
        raise LLMUnavailableError(f"OpenAI python package not available: {exc}") from exc

    # Configure global openai client for Azure if requested. The AsyncOpenAI
    # client will pick these up when making requests.
    if cfg.azure_openai_api_base and cfg.azure_openai_api_key:
        openai.api_type = "azure"
        openai.api_base = cfg.azure_openai_api_base.rstrip("/")
        openai.api_key = cfg.azure_openai_api_key
        if cfg.azure_openai_api_version:
            openai.api_version = cfg.azure_openai_api_version
        client = openai.AsyncOpenAI(max_retries=0, timeout=60.0)
    else:
        # standard openai.com
        client = openai.AsyncOpenAI(api_key=cfg.openai_api_key, max_retries=0, timeout=60.0)

    class _OpenAIAdapter:
        def __init__(self, client, model, max_tokens):
            self._client = client
            self.model = model
            self.max_tokens = max_tokens or cfg.llm_max_tokens

        async def ainvoke(self, messages: list[tuple[str, str]]):
            # Convert tuples to chat messages for the OpenAI chat completion API
            msg_list = []
            for role, text in messages:
                role_name = "system" if role == "system" else "user"
                msg_list.append({"role": role_name, "content": text})

            # Call the chat completion endpoint
            try:
                resp = await self._client.chat.completions.create(
                    model=self.model, messages=msg_list, max_tokens=self.max_tokens
                )
            except Exception as exc:
                # Normalize errors to match Anthropic mapping in _translate_errors
                # We raise a generic LLMUnavailableError so the API layer returns 503.
                logger.error("openai_chat_error", error=str(exc))
                raise LLMUnavailableError("LLM service temporarily unavailable") from exc

            # Extract text from the SDK response. Different client shapes exist
            # between OpenAI releases and Azure, so be permissive.
            content = ""
            try:
                choice = None
                if hasattr(resp, "choices") and resp.choices:
                    choice = resp.choices[0]
                elif isinstance(resp, dict) and resp.get("choices"):
                    choice = resp["choices"][0]

                if choice is not None:
                    # Try common locations
                    if hasattr(choice, "message"):
                        msg = choice.message
                        if isinstance(msg, dict):
                            content = msg.get("content", "")
                        else:
                            content = getattr(msg, "content", "")
                    elif isinstance(choice, dict):
                        content = choice.get("message", {}).get("content", "") or choice.get("text", "")
                else:
                    # Fallback: try top-level text
                    content = getattr(resp, "text", "") or resp.get("text", "") if isinstance(resp, dict) else str(resp)
            except Exception:
                content = str(resp)

            class _Resp:
                def __init__(self, content):
                    self.content = content

            return _Resp(content)

        def with_structured_output(self, schema):
            parent = self

            class _Structured:
                async def ainvoke(self, messages: list[tuple[str, str]]):
                    resp = await parent.ainvoke(messages)
                    text = resp.content
                    start, end = text.find("{"), text.rfind("}")
                    if start == -1 or end <= start:
                        return text
                    try:
                        data = json.loads(text[start : end + 1])
                        return data
                    except Exception:
                        return text

            return _Structured()

    return _OpenAIAdapter(client, cfg.llm_model, max_tokens)


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


def cited_ids_in(text: str) -> list[str]:
    return _CITE_RE.findall(text)
