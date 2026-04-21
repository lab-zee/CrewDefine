"""Runtime settings loaded from env / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TURNS = 40
DEFAULT_MAX_AGENTS_PER_CREW = 10


@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str
    max_turns: int
    max_agents_per_crew: int
    debug_log_path: str | None


def load_settings() -> Settings:
    """Load from environment (with .env fallback). Raises if API key is missing."""
    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Put it in your environment or a local .env file."
        )

    model = os.environ.get("CREWDEFINE_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    max_turns = _int_env("CREWDEFINE_MAX_TURNS", DEFAULT_MAX_TURNS)
    max_agents = _int_env("CREWDEFINE_MAX_AGENTS", DEFAULT_MAX_AGENTS_PER_CREW)
    debug_log = os.environ.get("CREWDEFINE_DEBUG_LOG") or None

    return Settings(
        api_key=api_key,
        model=model,
        max_turns=max_turns,
        max_agents_per_crew=max_agents,
        debug_log_path=debug_log,
    )


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as e:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}.") from e
    if value < 1:
        raise RuntimeError(f"{name} must be >= 1, got {value}.")
    return value
