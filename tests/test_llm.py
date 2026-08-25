from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import crewdefine.llm as llm_module
from crewdefine.config import Settings
from crewdefine.llm import LLMClient, _block_to_dict


class FakeMessages:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class DumpableBlock:
    def model_dump(self) -> dict[str, str]:
        return {"type": "text", "text": "hello"}


def _settings(debug_log_path: str | None = None) -> Settings:
    return Settings(
        api_key="test-key",
        model="test-model",
        max_turns=10,
        max_agents_per_crew=5,
        debug_log_path=debug_log_path,
    )


def test_client_initialization_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    api = SimpleNamespace(messages=FakeMessages([]))
    monkeypatch.setattr(llm_module, "Anthropic", lambda **_kwargs: api)
    client = LLMClient(_settings())
    assert client.model == "test-model"


def test_messages_maps_blocks_and_writes_debug_log(tmp_path: Path) -> None:
    response = SimpleNamespace(
        content=[DumpableBlock(), {"type": "text", "text": "world"}],
        stop_reason="end_turn",
    )
    messages = FakeMessages([response])
    client = object.__new__(LLMClient)
    client._client = SimpleNamespace(messages=messages)
    client._model = "test-model"
    client._debug_log = tmp_path / "nested" / "llm.jsonl"

    result = client.messages(
        system="system",
        messages=[{"role": "user", "content": "hello"}],
        tools=[{"name": "tool"}],
        tool_choice={"type": "auto"},
        max_tokens=100,
        temperature=0.2,
    )

    assert result.stop_reason == "end_turn"
    assert result.content_blocks[0]["text"] == "hello"
    assert messages.calls[0]["tools"] == [{"name": "tool"}]
    rows = [
        json.loads(line)
        for line in (tmp_path / "nested" / "llm.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["direction"] for row in rows] == ["request", "response"]


def test_retryable_api_error_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    class RetryableError(Exception):
        pass

    expected = SimpleNamespace(content=[], stop_reason=None)
    messages = FakeMessages([RetryableError("temporary"), expected])
    client = object.__new__(LLMClient)
    client._client = SimpleNamespace(messages=messages)
    monkeypatch.setattr(llm_module, "APIError", RetryableError)
    monkeypatch.setattr(llm_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(llm_module.random, "uniform", lambda _start, _end: 0)

    assert client._call_with_retry({}) is expected
    assert len(messages.calls) == 2


def test_rate_limit_exhausts_retry_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    class RetryableError(Exception):
        pass

    errors = [RetryableError("limited") for _ in range(llm_module._MAX_ATTEMPTS)]
    messages = FakeMessages(errors)
    client = object.__new__(LLMClient)
    client._client = SimpleNamespace(messages=messages)
    monkeypatch.setattr(llm_module, "RateLimitError", RetryableError)
    monkeypatch.setattr(llm_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(RetryableError, match="limited"):
        client._call_with_retry({})


def test_non_retryable_status_error_is_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    class StatusError(Exception):
        status_code = 400

    messages = FakeMessages([StatusError("bad request")])
    client = object.__new__(LLMClient)
    client._client = SimpleNamespace(messages=messages)
    monkeypatch.setattr(llm_module, "APIStatusError", StatusError)

    with pytest.raises(StatusError, match="bad request"):
        client._call_with_retry({})


def test_block_to_dict_falls_back_for_unknown_object() -> None:
    block = SimpleNamespace(type="custom")
    converted = _block_to_dict(block)
    assert converted["type"] == "custom"
    assert "namespace" in converted["repr"]
