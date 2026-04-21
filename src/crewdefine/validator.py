"""Validation for both in-memory `CrewConfig`s and on-disk crew directories.

Three layers:
1. Pydantic field validation (handled by `schema.py`, incl. self-delegation).
2. Cross-agent checks: delegation targets exist, tool references resolve.
   Note: cycles in the delegation graph are legitimate in LabZ — the
   director ↔ specialist hub-and-spoke pattern creates them by design, and
   LabZ's runtime handles actual runaway loops via depth limits. We do NOT
   reject cycles here.
3. Round-trip: dump each agent to YAML, `yaml.safe_load` it back, and confirm
   the dict matches what LabZ's `AgentConfig` dataclass expects field-for-field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from crewdefine.schema import AgentConfig, CrewConfig, ToolSpec
from crewdefine.tools_catalog import BUILTIN_TOOL_IDS
from crewdefine.yaml_format import dump_agent_yaml

# Fields that LabZ's `AgentConfig` dataclass accepts. Anything else in the
# emitted YAML will be silently dropped by LabZ — we error instead.
LABZ_AGENT_FIELDS: frozenset[str] = frozenset(
    {
        "id",
        "name",
        "role",
        "tools",
        "can_delegate_to",
        "system_prompt",
        "model",
        "data_extraction_note",
    }
)
LABZ_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {"id", "name", "role", "tools", "can_delegate_to", "system_prompt"}
)


class ValidationError(Exception):
    """Raised when a crew fails validation. `errors` has structured details."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors) if errors else "Validation failed")
        self.errors = errors


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.errors:
            raise ValidationError(self.errors)


def validate_crew(crew: CrewConfig) -> ValidationReport:
    """Validate an in-memory crew. Pydantic has already checked per-field shape."""
    report = ValidationReport()
    agent_ids = {a.id for a in crew.agents}
    builtin_plus_custom: set[str] = set(BUILTIN_TOOL_IDS) | {t.id for t in crew.custom_tools}

    for agent in crew.agents:
        _check_tool_refs(agent, builtin_plus_custom, report)
        _check_delegation_targets(agent, agent_ids, report)
        _check_round_trip(agent, report)

    _check_custom_tools_used(crew.custom_tools, crew.agents, report)
    return report


def validate_crew_dir(crew_dir: Path) -> ValidationReport:
    """Validate an on-disk crew directory by loading every agent YAML."""
    report = ValidationReport()
    agents_dir = crew_dir / "agents"
    if not agents_dir.is_dir():
        report.errors.append(f"No agents/ directory under {crew_dir}.")
        return report

    loaded: list[AgentConfig] = []
    for path in sorted(agents_dir.glob("*.yaml")):
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            report.errors.append(f"{path.name}: YAML parse error: {e}")
            continue
        if not isinstance(raw, dict):
            report.errors.append(
                f"{path.name}: top-level must be a mapping, got {type(raw).__name__}."
            )
            continue
        try:
            agent = AgentConfig.model_validate(raw)
        except Exception as e:  # pydantic.ValidationError, etc.
            report.errors.append(f"{path.name}: {e}")
            continue
        _check_labz_field_parity(raw, path.name, report)
        loaded.append(agent)

    if report.errors:
        return report

    # We don't know custom tools from the dir alone; allow any tool id not in
    # BUILTIN_TOOL_IDS but surface it as a warning so a human can verify.
    agent_ids = {a.id for a in loaded}
    for agent in loaded:
        for tool_id in agent.tools:
            if tool_id not in BUILTIN_TOOL_IDS:
                report.warnings.append(
                    f"{agent.id}: tool {tool_id!r} is not a built-in LabZ tool — ensure a stub exists under tools/."
                )
        _check_delegation_targets(agent, agent_ids, report)

    return report


def _check_tool_refs(agent: AgentConfig, known_tools: set[str], report: ValidationReport) -> None:
    for tool_id in agent.tools:
        if tool_id not in known_tools:
            report.errors.append(
                f"{agent.id}: references unknown tool {tool_id!r}. "
                "Add it to custom_tools or use a built-in tool id."
            )


def _check_delegation_targets(
    agent: AgentConfig, agent_ids: set[str], report: ValidationReport
) -> None:
    for target in agent.can_delegate_to:
        if target not in agent_ids:
            report.errors.append(
                f"{agent.id}: can_delegate_to includes {target!r}, which is not a defined agent id."
            )


def _check_round_trip(agent: AgentConfig, report: ValidationReport) -> None:
    emitted = dump_agent_yaml(agent.to_yaml_dict())
    reloaded = yaml.safe_load(emitted)
    if reloaded != agent.to_yaml_dict():
        report.errors.append(
            f"{agent.id}: YAML round-trip changed the data. "
            "This usually means a string contains characters that break block literal style."
        )


def _check_labz_field_parity(raw: dict[str, Any], label: str, report: ValidationReport) -> None:
    extra = set(raw) - LABZ_AGENT_FIELDS
    if extra:
        report.errors.append(
            f"{label}: unknown fields {sorted(extra)} — LabZ's AgentConfig will drop these silently."
        )
    missing = LABZ_REQUIRED_FIELDS - set(raw)
    if missing:
        report.errors.append(f"{label}: missing required fields {sorted(missing)}.")


def _check_custom_tools_used(
    custom_tools: list[ToolSpec], agents: list[AgentConfig], report: ValidationReport
) -> None:
    used: set[str] = set()
    for a in agents:
        used.update(a.tools)
    for tool in custom_tools:
        if tool.id not in used:
            report.warnings.append(
                f"Custom tool {tool.id!r} is defined but no agent references it."
            )
