"""LLM-driven interview loop.

Design:
    1. The interviewer model acts ONLY through a fixed set of tools
       (`ask_user`, `record_crew_meta`, `record_agent`, `record_custom_tool`,
       `finalize_crew`). It cannot produce free-form user-facing text.
    2. Every LLM turn is dispatched by `_handle_tool_use`, which mutates a
       local `_InterviewState`, then replies with a `tool_result` content
       block. That becomes the next user message in the Messages API.
    3. Hard caps: `settings.max_turns` total turns, `settings.max_agents_per_crew`
       agents. Hitting either ends the interview with what we have.

The UX side (reading stdin, printing questions) is delegated to a `UserIO`
protocol so the CLI can swap in a rich prompt and tests can use a scripted
fake.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from crewdefine.config import Settings
from crewdefine.llm import LLMClient
from crewdefine.prompts import (
    INTERVIEWER_SYSTEM_PROMPT,
    INTERVIEWER_TOOLS,
    PERSONA_SYSTEM_PROMPT,
    persona_user_prompt,
)
from crewdefine.schema import AgentConfig, CrewConfig, ToolParameter, ToolSpec


class UserIO(Protocol):
    def ask(self, question: str, options: list[str] | None, allow_skip: bool) -> str: ...
    def info(self, message: str) -> None: ...
    def warn(self, message: str) -> None: ...


@dataclass
class _InterviewState:
    crew_name: str | None = None
    crew_description: str | None = None
    agents: dict[str, dict[str, Any]] = field(default_factory=dict)
    custom_tools: dict[str, ToolSpec] = field(default_factory=dict)
    finalized: bool = False
    finalize_note: str | None = None


class InterviewError(RuntimeError):
    """Raised when the interview cannot produce a usable crew."""


def run_interview(
    client: LLMClient,
    settings: Settings,
    io: UserIO,
    *,
    seed_user_message: str,
    existing_crew: CrewConfig | None = None,
) -> CrewConfig:
    """Conduct the interview, then draft personas for each agent. Returns a validated `CrewConfig`.

    If `existing_crew` is supplied, it pre-seeds state so `add-agent` / `update-agent`
    flows start from a coherent baseline.
    """
    state = _seed_state(existing_crew)
    messages: list[dict[str, Any]] = [{"role": "user", "content": seed_user_message}]

    for _turn in range(1, settings.max_turns + 1):
        response = client.messages(
            system=INTERVIEWER_SYSTEM_PROMPT,
            messages=messages,
            tools=INTERVIEWER_TOOLS,
            max_tokens=2048,
        )
        assistant_content = response.content_blocks
        messages.append({"role": "assistant", "content": assistant_content})

        tool_uses = [b for b in assistant_content if b.get("type") == "tool_use"]
        if not tool_uses:
            # Model produced text only — nudge it back to tool use. This shouldn't
            # happen often; the system prompt is explicit. But we guard anyway.
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Please act only through tools. Use `ask_user` to ask the next "
                        "question, or call `finalize_crew` if you're done."
                    ),
                }
            )
            continue

        tool_results: list[dict[str, Any]] = []
        stop_after = False
        for block in tool_uses:
            result = _handle_tool_use(block, state, io)
            tool_results.append(result)
            if state.finalized:
                stop_after = True

        messages.append({"role": "user", "content": tool_results})

        if stop_after:
            break
        if len(state.agents) >= settings.max_agents_per_crew:
            io.warn(f"Reached max agents per crew ({settings.max_agents_per_crew}). Wrapping up.")
            state.finalized = True
            break
    else:
        io.warn(f"Reached max turns ({settings.max_turns}). Wrapping up with what we have.")
        state.finalized = True

    if not state.agents:
        raise InterviewError(
            "Interview ended without any agents defined. "
            "Try again and give the interviewer more to work with."
        )
    if state.crew_name is None or state.crew_description is None:
        raise InterviewError(
            "Interview ended without crew name/description. "
            "Re-run and answer the opening questions."
        )

    io.info("Drafting agent personas...")
    fully_formed_agents = _draft_all_personas(client, state, io)
    crew = CrewConfig(
        name=state.crew_name,
        description=state.crew_description,
        agents=fully_formed_agents,
        custom_tools=list(state.custom_tools.values()),
    )
    return crew


def _seed_state(existing: CrewConfig | None) -> _InterviewState:
    if existing is None:
        return _InterviewState()
    state = _InterviewState(
        crew_name=existing.name,
        crew_description=existing.description,
        custom_tools={t.id: t for t in existing.custom_tools},
    )
    for a in existing.agents:
        state.agents[a.id] = {
            "id": a.id,
            "name": a.name,
            "role": a.role,
            "tools": list(a.tools),
            "can_delegate_to": list(a.can_delegate_to),
            "model": a.model,
            "data_extraction_note": a.data_extraction_note,
            "_existing_system_prompt": a.system_prompt,
        }
    return state


def _handle_tool_use(block: dict[str, Any], state: _InterviewState, io: UserIO) -> dict[str, Any]:
    name = block.get("name")
    tool_use_id = block.get("id") or str(uuid.uuid4())
    raw_input = block.get("input") or {}

    try:
        if name == "ask_user":
            answer = io.ask(
                question=str(raw_input.get("question", "")).strip(),
                options=raw_input.get("options"),
                allow_skip=bool(raw_input.get("allow_skip", True)),
            )
            return _tool_result(tool_use_id, answer)

        if name == "record_crew_meta":
            state.crew_name = str(raw_input["name"]).strip()
            state.crew_description = str(raw_input["description"]).strip()
            return _tool_result(tool_use_id, f"Recorded crew '{state.crew_name}'.")

        if name == "record_agent":
            return _record_agent(raw_input, tool_use_id, state)

        if name == "record_custom_tool":
            return _record_custom_tool(raw_input, tool_use_id, state)

        if name == "finalize_crew":
            state.finalized = True
            state.finalize_note = str(raw_input.get("confirmation_note", "")).strip()
            return _tool_result(tool_use_id, "Finalized. Drafting personas next.")

        return _tool_result(tool_use_id, f"Unknown tool: {name!r}.", is_error=True)
    except Exception as e:
        # Bubble a structured error back into the conversation so the model
        # can correct itself on the next turn.
        return _tool_result(tool_use_id, f"Tool error: {e}", is_error=True)


def _record_agent(raw: dict[str, Any], tool_use_id: str, state: _InterviewState) -> dict[str, Any]:
    agent_id = str(raw["id"]).strip()
    record = {
        "id": agent_id,
        "name": str(raw["name"]).strip(),
        "role": str(raw["role"]).strip(),
        "tools": [str(t).strip() for t in raw.get("tools", [])],
        "can_delegate_to": [str(t).strip() for t in raw.get("can_delegate_to", [])],
        "model": (str(raw["model"]).strip() if raw.get("model") else None),
        "data_extraction_note": (
            str(raw["data_extraction_note"]).strip() if raw.get("data_extraction_note") else None
        ),
    }
    existing = state.agents.get(agent_id, {})
    already_existed = bool(existing)
    # Preserve existing system_prompt if we're updating a seeded agent and no
    # structural field changed materially enough to warrant a redraft.
    if "_existing_system_prompt" in existing:
        record["_existing_system_prompt"] = existing["_existing_system_prompt"]
    state.agents[agent_id] = record
    verb = "Updated" if already_existed else "Added"
    return _tool_result(
        tool_use_id, f"{verb} agent '{agent_id}'. Total agents: {len(state.agents)}."
    )


def _record_custom_tool(
    raw: dict[str, Any], tool_use_id: str, state: _InterviewState
) -> dict[str, Any]:
    params_raw = raw.get("parameters") or []
    params = [
        ToolParameter(
            name=str(p["name"]),
            type=str(p["type"]),
            description=str(p["description"]),
            required=bool(p.get("required", True)),
        )
        for p in params_raw
    ]
    tool = ToolSpec(
        id=str(raw["id"]).strip(),
        description=str(raw["description"]).strip(),
        parameters=params,
    )
    state.custom_tools[tool.id] = tool
    return _tool_result(tool_use_id, f"Registered custom tool '{tool.id}'.")


def _tool_result(tool_use_id: str, content: str, *, is_error: bool = False) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }
    if is_error:
        block["is_error"] = True
    return block


def _draft_all_personas(client: LLMClient, state: _InterviewState, io: UserIO) -> list[AgentConfig]:
    """Draft `system_prompt` for each agent. Reuses an existing prompt if present."""
    # Build a provisional CrewConfig WITHOUT prompts so the drafter has crew-level context.
    provisional_agents: list[AgentConfig] = []
    for record in state.agents.values():
        provisional_agents.append(
            AgentConfig(
                id=record["id"],
                name=record["name"],
                role=record["role"],
                tools=list(record["tools"]),
                can_delegate_to=list(record["can_delegate_to"]),
                system_prompt=_placeholder_prompt(record),
                model=record.get("model"),
                data_extraction_note=record.get("data_extraction_note"),
            )
        )
    provisional_crew = CrewConfig(
        name=state.crew_name or "crew",
        description=state.crew_description or "",
        agents=provisional_agents,
        custom_tools=list(state.custom_tools.values()),
    )

    drafted: list[AgentConfig] = []
    for record in state.agents.values():
        agent_stub = next(a for a in provisional_crew.agents if a.id == record["id"])
        if "_existing_system_prompt" in record:
            prompt_body = record["_existing_system_prompt"]
        else:
            io.info(f"  drafting: {agent_stub.id}")
            prompt_body = _draft_one_persona(client, agent_stub, provisional_crew)

        drafted.append(
            AgentConfig(
                id=agent_stub.id,
                name=agent_stub.name,
                role=agent_stub.role,
                tools=list(agent_stub.tools),
                can_delegate_to=list(agent_stub.can_delegate_to),
                system_prompt=prompt_body,
                model=agent_stub.model,
                data_extraction_note=agent_stub.data_extraction_note,
            )
        )
    return drafted


def _placeholder_prompt(record: dict[str, Any]) -> str:
    """Enough text to satisfy AgentConfig's min-length check while drafting."""
    return (
        f"PLACEHOLDER for {record['id']}. If you are seeing this in a generated YAML, "
        "the drafting step failed or was skipped. Re-run `crewdefine update-agent` to redraft."
    )


def _draft_one_persona(client: LLMClient, agent: AgentConfig, crew: CrewConfig) -> str:
    user_prompt = persona_user_prompt(agent, crew)
    response = client.messages(
        system=PERSONA_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=4096,
    )
    text = _extract_text(response.content_blocks)
    if not text.strip():
        raise InterviewError(f"Persona drafter returned empty output for {agent.id}.")
    return text.strip()


def _extract_text(blocks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for b in blocks:
        if b.get("type") == "text":
            parts.append(str(b.get("text", "")))
    return "\n".join(parts)


def dumps_state(state: _InterviewState) -> str:
    """Debug helper — JSON-serializable snapshot of interview state."""
    return json.dumps(
        {
            "crew_name": state.crew_name,
            "crew_description": state.crew_description,
            "agents": state.agents,
            "custom_tools": {k: v.model_dump() for k, v in state.custom_tools.items()},
            "finalized": state.finalized,
        },
        indent=2,
        default=str,
    )
