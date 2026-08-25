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

from crewdefine.schema import (
    KNOWN_ANSWER_MODE_IDS,
    AgentConfig,
    AnswerModeOption,
    CrewConfig,
    OutputComposition,
    ToolSpec,
    apply_manifest_defaults,
)
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

    _check_infrastructure_agents(agent_ids, report)

    for agent in crew.agents:
        _check_tool_refs(agent, builtin_plus_custom, report)
        _check_delegation_targets(agent, agent_ids, report)
        _check_round_trip(agent, report)

    _check_custom_tools_used(crew.custom_tools, crew.agents, report)
    _check_manifest_fields(crew, report)
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
        if path.name in {"crew.yaml"} or path.name.startswith("_"):
            continue
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

    agent_ids = {a.id for a in loaded}
    _check_infrastructure_agents(agent_ids, report)

    tools_dir = crew_dir / "tools"
    custom_tool_ids = (
        {path.stem for path in tools_dir.glob("*.py")} if tools_dir.is_dir() else set()
    )
    for agent in loaded:
        for tool_id in agent.tools:
            if tool_id not in BUILTIN_TOOL_IDS and tool_id not in custom_tool_ids:
                report.errors.append(
                    f"{agent.id}: custom tool {tool_id!r} has no matching tools/{tool_id}.py."
                )
        _check_delegation_targets(agent, agent_ids, report)

    _check_manifest_file(crew_dir, report)
    _maybe_run_zero_validator(crew_dir, report)
    return report


def _maybe_run_zero_validator(crew_dir: Path, report: ValidationReport) -> None:
    """If ZERO_BACKEND is set, also run Zero's canonical validate_crew.py."""
    import os
    import subprocess
    import sys

    zero_backend = os.environ.get("ZERO_BACKEND", "").strip()
    if not zero_backend:
        return
    script = Path(zero_backend) / "scripts" / "validate_crew.py"
    if not script.is_file():
        # Also accept backend/scripts layout
        alt = Path(zero_backend) / "backend" / "scripts" / "validate_crew.py"
        script = alt if alt.is_file() else script
    if not script.is_file():
        report.warnings.append(
            f"ZERO_BACKEND={zero_backend!r} set but validate_crew.py not found — skipped Zero parity check."
        )
        return
    try:
        proc = subprocess.run(
            [sys.executable, str(script), str(crew_dir)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except Exception as e:
        report.warnings.append(f"Zero validator subprocess failed: {e}")
        return
    if proc.returncode != 0:
        detail = (proc.stdout or proc.stderr or "").strip()
        report.errors.append(
            f"Zero validator failed (exit {proc.returncode}): {detail[:500] or 'no output'}"
        )


def _check_infrastructure_agents(agent_ids: set[str], report: ValidationReport) -> None:
    for required in ("director", "synthesizer"):
        if required not in agent_ids:
            report.errors.append(
                f"Missing required infrastructure agent id {required!r}. "
                "Zero's registry hard-codes these names."
            )


def _check_manifest_fields(crew: CrewConfig, report: ValidationReport) -> None:
    filled = apply_manifest_defaults(crew)
    if not filled.answer_modes:
        report.errors.append("answer_modes must be non-empty after defaults.")
        return
    mode_ids = {m.id for m in filled.answer_modes}
    if filled.default_answer_mode not in mode_ids:
        report.errors.append(
            f"default_answer_mode {filled.default_answer_mode!r} not in answer_modes."
        )


def _check_manifest_file(crew_dir: Path, report: ValidationReport) -> None:
    manifest_path = crew_dir / "crew.yaml"
    if not manifest_path.exists():
        # Also accept agents/crew.yaml (Zero accepts both)
        alt = crew_dir / "agents" / "crew.yaml"
        if alt.exists():
            manifest_path = alt
        else:
            report.warnings.append(
                "No crew.yaml found — Zero will fall back to built-in answer modes. "
                "Emit crew.yaml for seamless handoff."
            )
            return

    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        report.errors.append(f"crew.yaml: YAML parse error: {e}")
        return
    if not isinstance(raw, dict):
        report.errors.append("crew.yaml: top-level must be a mapping.")
        return

    for key in ("name", "display_name", "description", "default_answer_mode", "answer_modes"):
        if key not in raw:
            report.errors.append(f"crew.yaml: missing required field {key!r}.")

    modes = raw.get("answer_modes") or []
    if not isinstance(modes, list) or not modes:
        report.errors.append("crew.yaml: answer_modes must be a non-empty list.")
        return

    mode_ids: list[str] = []
    for i, mode in enumerate(modes):
        if not isinstance(mode, dict):
            report.errors.append(f"crew.yaml: answer_modes[{i}] must be a mapping.")
            continue
        try:
            opt = AnswerModeOption.model_validate(mode)
            mode_ids.append(opt.id)
        except Exception as e:
            report.errors.append(f"crew.yaml: answer_modes[{i}]: {e}")

    default = raw.get("default_answer_mode")
    if default and default not in mode_ids and default in KNOWN_ANSWER_MODE_IDS:
        report.errors.append(
            f"crew.yaml: default_answer_mode {default!r} is not listed in answer_modes."
        )
    elif default and default not in KNOWN_ANSWER_MODE_IDS:
        report.errors.append(f"crew.yaml: default_answer_mode {default!r} is not a known mode id.")

    if "output_composition" in raw and raw["output_composition"] is not None:
        try:
            OutputComposition.model_validate(raw["output_composition"])
        except Exception as e:
            report.errors.append(f"crew.yaml: output_composition: {e}")


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
