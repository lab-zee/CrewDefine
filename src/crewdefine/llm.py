"""Thin wrapper around the Anthropic Messages API.

Scope: enough to drive a tool-use-based interview and a single-shot persona
drafter. Retry is deliberately simple (bounded, exponential on transient
errors) — anything more elaborate belongs in `anthropic` itself.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anthropic import Anthropic, APIError, APIStatusError, RateLimitError

from crewdefine.config import Settings

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 4
_BACKOFF_BASE = 1.5


@dataclass
class LLMResponse:
    """Unified view of a Messages API response for the interviewer loop."""

    content_blocks: list[dict[str, Any]]
    stop_reason: str
    raw: Any


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self._client = Anthropic(api_key=settings.api_key)
        self._model = settings.model
        self._debug_log = Path(settings.debug_log_path) if settings.debug_log_path else None

    @property
    def model(self) -> str:
        return self._model

    def messages(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> LLMResponse:
        """Call the Messages API with retry on transient failures."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        self._log({"direction": "request", "payload": kwargs})
        response = self._call_with_retry(kwargs)
        blocks = [_block_to_dict(b) for b in response.content]
        self._log(
            {
                "direction": "response",
                "stop_reason": response.stop_reason,
                "content": blocks,
            }
        )
        return LLMResponse(
            content_blocks=blocks, stop_reason=response.stop_reason or "", raw=response
        )

    def _call_with_retry(self, kwargs: dict[str, Any]) -> Any:
        last_err: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                return self._client.messages.create(**kwargs)
            except RateLimitError as e:
                last_err = e
            except APIStatusError as e:
                if e.status_code not in _RETRYABLE_STATUS:
                    raise
                last_err = e
            except APIError as e:
                last_err = e

            if attempt == _MAX_ATTEMPTS:
                break
            sleep_for = (_BACKOFF_BASE**attempt) + random.uniform(0, 0.5)
            time.sleep(sleep_for)

        assert last_err is not None
        raise last_err

    def _log(self, payload: dict[str, Any]) -> None:
        if self._debug_log is None:
            return
        self._debug_log.parent.mkdir(parents=True, exist_ok=True)
        with self._debug_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")


def _block_to_dict(block: Any) -> dict[str, Any]:
    """Convert an SDK content block into a plain dict for logging/inspection."""
    if hasattr(block, "model_dump"):
        dumped: dict[str, Any] = block.model_dump()
        return dumped
    if isinstance(block, dict):
        return block
    return {"type": getattr(block, "type", "unknown"), "repr": repr(block)}
