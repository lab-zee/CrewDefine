from __future__ import annotations

import pytest
from pydantic import ValidationError

from crewdefine.schema import AgentConfig, CrewConfig, ToolParameter, ToolSpec


def test_agent_id_must_be_snake_case() -> None:
    with pytest.raises(ValidationError, match="snake_case"):
        AgentConfig(
            id="CamelCase",
            name="x",
            role="y who does z",
            system_prompt="a" * 100,
        )


def test_agent_id_must_start_with_letter() -> None:
    with pytest.raises(ValidationError, match="snake_case"):
        AgentConfig(
            id="1_bad",
            name="x",
            role="y who does z",
            system_prompt="a" * 100,
        )


def test_system_prompt_too_short_rejected() -> None:
    with pytest.raises(ValidationError, match="too short"):
        AgentConfig(
            id="ok_id",
            name="Name",
            role="Role who does things",
            system_prompt="too short",
        )


def test_agent_cannot_self_delegate() -> None:
    with pytest.raises(ValidationError, match="itself in can_delegate_to"):
        AgentConfig(
            id="loopy",
            name="Loopy",
            role="Agent who loops",
            system_prompt="a" * 100,
            can_delegate_to=["loopy"],
        )


def test_blank_name_rejected() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        AgentConfig(
            id="ok",
            name="   ",
            role="Role who does things",
            system_prompt="a" * 100,
        )


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentConfig.model_validate(
            {
                "id": "ok",
                "name": "Ok",
                "role": "Role who does things",
                "system_prompt": "a" * 100,
                "unexpected_field": "boom",
            }
        )


def test_to_yaml_dict_preserves_field_order(minimal_agent: AgentConfig) -> None:
    keys = list(minimal_agent.to_yaml_dict().keys())
    # system_prompt must come last (matches LabZ's hand-written convention).
    assert keys[-1] == "system_prompt"
    assert keys[0] == "id"


def test_to_yaml_dict_omits_none_optionals(minimal_agent: AgentConfig) -> None:
    out = minimal_agent.to_yaml_dict()
    assert "model" not in out
    assert "data_extraction_note" not in out


def test_crew_name_must_be_kebab_case() -> None:
    with pytest.raises(ValidationError, match="kebab-case"):
        CrewConfig(name="BadName", description="x", agents=[])


def test_crew_rejects_duplicate_agent_ids(minimal_agent: AgentConfig) -> None:
    dupe = AgentConfig(
        id=minimal_agent.id,
        name="Another",
        role="Another who does things",
        system_prompt="a" * 100,
    )
    with pytest.raises(ValidationError, match="Duplicate agent ids"):
        CrewConfig(name="crew", description="x", agents=[minimal_agent, dupe])


def test_tool_spec_rejects_bad_parameter_type() -> None:
    with pytest.raises(ValidationError, match="Unsupported JSON Schema type"):
        ToolSpec(
            id="t",
            description="x",
            parameters=[ToolParameter(name="p", type="blob", description="d")],
        )
