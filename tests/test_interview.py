"""Interview-loop tests using a scripted fake LLM.

We don't hit the Anthropic API here — we stand up a fake that emits a
deterministic sequence of tool_use blocks and verify the state machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from crewdefine.config import Settings
from crewdefine.interview import InterviewError, run_interview
from crewdefine.llm import LLMResponse


@dataclass
class FakeIO:
    answers: list[str]
    asked: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.asked = []

    def ask(self, question: str, options: list[str] | None, allow_skip: bool) -> str:
        self.asked.append(question)
        if not self.answers:
            return "(skipped)"
        return self.answers.pop(0)

    def info(self, message: str) -> None: ...
    def warn(self, message: str) -> None: ...


class ScriptedLLM:
    """Fake LLMClient: emits pre-scripted content block lists on each call."""

    def __init__(self, script: list[list[dict[str, Any]]]) -> None:
        self._script = list(script)
        self.call_count = 0

    @property
    def model(self) -> str:
        return "fake-model"

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
        self.call_count += 1
        if not self._script:
            raise RuntimeError("ScriptedLLM exhausted — test script is too short.")
        blocks = self._script.pop(0)
        stop = "end_turn"
        for b in blocks:
            if b.get("type") == "tool_use":
                stop = "tool_use"
        return LLMResponse(content_blocks=blocks, stop_reason=stop, raw=None)


def _settings() -> Settings:
    return Settings(
        api_key="test",
        model="fake-model",
        max_turns=20,
        max_agents_per_crew=10,
        debug_log_path=None,
    )


def _tool_use(tool_name: str, /, **inp: Any) -> dict[str, Any]:
    """Build a tool_use content block. `tool_name` is positional-only so that
    `_tool_use("record_crew_meta", name="market-intel", ...)` works — otherwise
    the kwarg `name` would collide with the positional param."""
    return {"type": "tool_use", "id": f"tu_{tool_name}", "name": tool_name, "input": inp}


def test_happy_path_finalizes_with_two_agents() -> None:
    persona_text = (
        "You are the Director. Your responsibilities are: 1. **Orchestrate**. 2. **Synthesize**. "
        "You delegate to the researcher. Always cite sources. Stay in scope."
    )
    script = [
        # Turn 1: record meta + ask about specialists
        [
            _tool_use(
                "record_crew_meta",
                name="market-intel",
                description="Small crew that gathers market intel and synthesizes it.",
            ),
        ],
        # Turn 2: define director
        [
            _tool_use(
                "record_agent",
                id="director",
                name="Director",
                role="Director who orchestrates specialists",
                tools=[],
                can_delegate_to=["researcher"],
            ),
        ],
        # Turn 3: define researcher
        [
            _tool_use(
                "record_agent",
                id="researcher",
                name="Researcher",
                role="Researcher who gathers market intel from public sources",
                tools=["web_search"],
                can_delegate_to=[],
            ),
        ],
        # Turn 4: finalize
        [
            _tool_use("finalize_crew", confirmation_note="Looks good."),
        ],
        # Drafting: one call per agent. Both return a text block.
        [{"type": "text", "text": persona_text}],
        [{"type": "text", "text": persona_text}],
    ]
    llm = ScriptedLLM(script)
    crew = run_interview(
        llm,  # type: ignore[arg-type]
        _settings(),
        FakeIO(answers=[]),
        seed_user_message="build a crew",
    )
    assert crew.name == "market-intel"
    assert {a.id for a in crew.agents} == {"director", "researcher"}


def test_empty_interview_raises() -> None:
    script = [
        [_tool_use("finalize_crew", confirmation_note="nothing to do")],
    ]
    with pytest.raises(InterviewError, match="without any agents"):
        run_interview(
            ScriptedLLM(script),  # type: ignore[arg-type]
            _settings(),
            FakeIO(answers=[]),
            seed_user_message="hi",
        )


def test_text_only_turn_gets_nudged_back() -> None:
    script = [
        # First call: text only (no tool use). Interview should nudge.
        [{"type": "text", "text": "I think we should..."}],
        # Second call: get back to work.
        [
            _tool_use(
                "record_crew_meta",
                name="tiny",
                description="tiny crew",
            ),
        ],
        [
            _tool_use(
                "record_agent",
                id="only",
                name="Only",
                role="Agent who does everything",
                tools=[],
                can_delegate_to=[],
            ),
        ],
        [_tool_use("finalize_crew", confirmation_note="done")],
        # Persona drafting.
        [
            {
                "type": "text",
                "text": "You are Only. Your responsibilities: 1. **Handle everything**. 2. **Stay focused**. 3. **Cite sources**. Always return structured findings to the caller.",
            }
        ],
    ]
    llm = ScriptedLLM(script)
    crew = run_interview(
        llm,  # type: ignore[arg-type]
        _settings(),
        FakeIO(answers=[]),
        seed_user_message="hi",
    )
    assert [a.id for a in crew.agents] == ["only"]


def test_ask_user_routes_to_io() -> None:
    script = [
        [_tool_use("ask_user", question="What are we building?")],
        [
            _tool_use(
                "record_crew_meta",
                name="q-crew",
                description="a crew",
            ),
            _tool_use(
                "record_agent",
                id="only",
                name="Only",
                role="Agent who does everything",
                tools=[],
                can_delegate_to=[],
            ),
            _tool_use("finalize_crew", confirmation_note="done"),
        ],
        [
            {
                "type": "text",
                "text": "You are Only. Your responsibilities: 1. **Do things**. 2. **Report back**. 3. **Cite sources**. Return structured findings to the caller when done.",
            }
        ],
    ]
    llm = ScriptedLLM(script)
    io = FakeIO(answers=["a thing for market stuff"])
    crew = run_interview(
        llm,  # type: ignore[arg-type]
        _settings(),
        io,
        seed_user_message="hi",
    )
    assert io.asked == ["What are we building?"]
    assert crew.name == "q-crew"


def test_unknown_tool_error_does_not_crash() -> None:
    script = [
        [_tool_use("nonexistent_tool", foo="bar")],
        [
            _tool_use(
                "record_crew_meta",
                name="recovery",
                description="recovery crew",
            ),
            _tool_use(
                "record_agent",
                id="only",
                name="Only",
                role="Agent who recovers gracefully",
                tools=[],
                can_delegate_to=[],
            ),
            _tool_use("finalize_crew", confirmation_note="ok"),
        ],
        [
            {
                "type": "text",
                "text": "You are Only. Your responsibilities: 1. **Recover gracefully**. 2. **Handle errors**. 3. **Cite sources**. Return structured findings to the caller.",
            }
        ],
    ]
    llm = ScriptedLLM(script)
    crew = run_interview(
        llm,  # type: ignore[arg-type]
        _settings(),
        FakeIO(answers=[]),
        seed_user_message="hi",
    )
    assert crew.name == "recovery"
