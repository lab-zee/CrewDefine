from __future__ import annotations

import pytest

import crewdefine.config as config
from crewdefine.config import (
    DEFAULT_MAX_AGENTS_PER_CREW,
    DEFAULT_MAX_TURNS,
    DEFAULT_MODEL,
    _int_env,
    load_settings,
)


def test_load_settings_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        load_settings()


def test_load_settings_reads_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", " test-key ")
    monkeypatch.setenv("CREWDEFINE_MODEL", " test-model ")
    monkeypatch.setenv("CREWDEFINE_MAX_TURNS", "12")
    monkeypatch.setenv("CREWDEFINE_MAX_AGENTS", "7")
    monkeypatch.setenv("CREWDEFINE_DEBUG_LOG", "debug.jsonl")
    settings = load_settings()
    assert settings.api_key == "test-key"
    assert settings.model == "test-model"
    assert settings.max_turns == 12
    assert settings.max_agents_per_crew == 7
    assert settings.debug_log_path == "debug.jsonl"


def test_load_settings_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    for name in (
        "CREWDEFINE_MODEL",
        "CREWDEFINE_MAX_TURNS",
        "CREWDEFINE_MAX_AGENTS",
        "CREWDEFINE_DEBUG_LOG",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = load_settings()
    assert settings.model == DEFAULT_MODEL
    assert settings.max_turns == DEFAULT_MAX_TURNS
    assert settings.max_agents_per_crew == DEFAULT_MAX_AGENTS_PER_CREW
    assert settings.debug_log_path is None


@pytest.mark.parametrize("value", ["not-a-number", "0", "-1"])
def test_int_env_rejects_invalid_values(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COUNT", value)
    with pytest.raises(RuntimeError, match="COUNT"):
        _int_env("COUNT", 3)


def test_int_env_uses_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COUNT", raising=False)
    assert _int_env("COUNT", 3) == 3
