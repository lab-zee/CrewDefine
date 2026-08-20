"""Pydantic models that mirror LabZ's agent/crew shape.

LabZ loads each agent YAML into a plain dataclass (`AgentConfig` at
`backend/src/agents/base.py`). It does not validate beyond that, so CrewDefine
owns validation: required fields, id formatting, delegation-graph sanity, and
the OpenAI function-calling shape for tool stubs.

Crew-level metadata is emitted as `crew.yaml` (see Zero's CREW_HANDOFF.md).
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AGENT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
TOOL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

KNOWN_ANSWER_MODE_IDS: frozenset[str] = frozenset(
    {"summary", "light", "extended", "project_plan", "roadmap"}
)

CrewArchetype = Literal["strategy", "lightweight"]


class ToolParameter(BaseModel):
    """Single parameter in an OpenAI function-calling tool definition."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: str = Field(
        description='JSON Schema type: "string", "integer", "number", "boolean", "array", "object".'
    )
    description: str
    required: bool = True

    @field_validator("type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        allowed = {"string", "integer", "number", "boolean", "array", "object"}
        if v not in allowed:
            raise ValueError(f"Unsupported JSON Schema type: {v!r}. Allowed: {sorted(allowed)}")
        return v


class ToolSpec(BaseModel):
    """A custom tool CrewDefine will emit as a Python stub.

    Mirrors LabZ's `TOOL_DEFINITIONS` entry shape (OpenAI function-calling
    format). The generated stub imports are intentionally minimal so a dev
    can fill in the body without fighting the scaffold.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Snake_case tool identifier used by agents.")
    description: str = Field(description="One-to-two sentence summary of what the tool does.")
    parameters: list[ToolParameter] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id_format(cls, v: str) -> str:
        if not TOOL_ID_PATTERN.match(v):
            raise ValueError(f"Tool id must be snake_case, got {v!r}.")
        return v


class AnswerModeOption(BaseModel):
    """One selectable answer mode exposed in LabZ's AnswerModeSelector."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Fixed mode id: summary|light|extended|project_plan|roadmap.")
    label: str = Field(description="UI label (domain-specific wording encouraged).")
    description: str = Field(description="One-line explanation shown under the label.")

    @field_validator("id")
    @classmethod
    def _known_mode_id(cls, v: str) -> str:
        if v not in KNOWN_ANSWER_MODE_IDS:
            raise ValueError(
                f"Answer mode id {v!r} is not supported. Allowed: {sorted(KNOWN_ANSWER_MODE_IDS)}."
            )
        return v

    @field_validator("label", "description")
    @classmethod
    def _nonblank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must be non-empty.")
        return v


class OutputComposition(BaseModel):
    """How the synthesizer should compose rich answers (tabs, citations, tools)."""

    model_config = ConfigDict(extra="forbid")

    tabs: list[str] = Field(
        default_factory=lambda: ["summary"],
        description="UI tabs: summary, raw_data, visualizations, references.",
    )
    citations: Literal["required", "optional", "none"] = "optional"
    charts: Literal["none", "when_quantitative", "always"] = "none"
    tables: Literal["none", "when_structured", "always"] = "when_structured"
    images: Literal["none", "when_requested", "synthesizer_summary"] = "none"
    synthesizer_tools: list[str] = Field(default_factory=list)


class AgentConfig(BaseModel):
    """One agent YAML file. Field order and names match LabZ's dataclass."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Snake_case unique identifier.")
    name: str = Field(description="Human-readable display name.")
    role: str = Field(
        description='One-line purpose, e.g. "Strategic director who coordinates specialists".'
    )
    tools: list[str] = Field(default_factory=list)
    can_delegate_to: list[str] = Field(default_factory=list)
    system_prompt: str = Field(description="Full persona + instructions. Typically 300-1500 words.")
    model: str | None = Field(default=None, description="Optional per-agent LLM model override.")
    data_extraction_note: str | None = Field(
        default=None,
        description="Optional one-line hint about structuring extracted data.",
    )

    @field_validator("id")
    @classmethod
    def _id_format(cls, v: str) -> str:
        if not AGENT_ID_PATTERN.match(v):
            raise ValueError(f"Agent id must be snake_case, got {v!r}.")
        return v

    @field_validator("name", "role")
    @classmethod
    def _nonblank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must be non-empty.")
        return v

    @field_validator("system_prompt")
    @classmethod
    def _prompt_substantive(cls, v: str) -> str:
        if len(v.strip()) < 80:
            raise ValueError(
                "system_prompt looks too short to be useful (< 80 chars). "
                "Agent prompts should define role, responsibilities, and output expectations."
            )
        return v

    @model_validator(mode="after")
    def _no_self_delegation(self) -> AgentConfig:
        if self.id in self.can_delegate_to:
            raise ValueError(f"Agent {self.id!r} lists itself in can_delegate_to.")
        return self

    def to_yaml_dict(self) -> dict[str, Any]:
        """Serialize in LabZ's canonical field order, omitting None optionals."""
        out: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "tools": list(self.tools),
            "can_delegate_to": list(self.can_delegate_to),
        }
        if self.data_extraction_note is not None:
            out["data_extraction_note"] = self.data_extraction_note
        if self.model is not None:
            out["model"] = self.model
        out["system_prompt"] = self.system_prompt
        return out


# ---------------------------------------------------------------------------
# Manifest defaults by archetype
# ---------------------------------------------------------------------------

_STRATEGY_MODES: list[dict[str, str]] = [
    {"id": "summary", "label": "Summary", "description": "Concise executive briefing"},
    {"id": "light", "label": "One-Pager", "description": "Balanced memo with key evidence"},
    {"id": "extended", "label": "Report", "description": "Comprehensive analysis"},
    {"id": "project_plan", "label": "30-60-90", "description": "Phased project plan"},
    {"id": "roadmap", "label": "Roadmap", "description": "Framework + implementation roadmap"},
]

_LIGHTWEIGHT_MODES: list[dict[str, str]] = [
    {"id": "summary", "label": "Quick Answer", "description": "Short, actionable response"},
    {"id": "light", "label": "Full Answer", "description": "Balanced response with key detail"},
]

_STRATEGY_COMPOSITION = OutputComposition(
    tabs=["summary", "raw_data", "visualizations", "references"],
    citations="required",
    charts="when_quantitative",
    tables="when_structured",
    images="synthesizer_summary",
    synthesizer_tools=[
        "visualizer",
        "extract_citations_structured",
        "swot",
        "generate_recommendations",
        "image_generator",
    ],
)

_LIGHTWEIGHT_COMPOSITION = OutputComposition(
    tabs=["summary", "raw_data"],
    citations="optional",
    charts="none",
    tables="when_structured",
    images="none",
    synthesizer_tools=[],
)


def infer_archetype(crew: CrewConfig) -> CrewArchetype:
    """Heuristic: multi-specialist research crews → strategy; else lightweight."""
    if crew.archetype is not None:
        return crew.archetype
    agent_ids = {a.id for a in crew.agents}
    specialists = agent_ids - {"director", "synthesizer"}
    blob = f"{crew.name} {crew.description}".lower()
    strategy_keywords = (
        "strategy",
        "strategic",
        "advisory",
        "competitive",
        "market",
        "business",
        "research",
        "intel",
    )
    if "synthesizer" in agent_ids and len(specialists) >= 2:
        return "strategy"
    if any(k in blob for k in strategy_keywords):
        return "strategy"
    return "lightweight"


def apply_manifest_defaults(crew: CrewConfig) -> CrewConfig:
    """Fill display_name / answer_modes / output_composition when unset."""
    archetype = infer_archetype(crew)
    modes_src = _STRATEGY_MODES if archetype == "strategy" else _LIGHTWEIGHT_MODES
    composition = _STRATEGY_COMPOSITION if archetype == "strategy" else _LIGHTWEIGHT_COMPOSITION

    display_name = crew.display_name
    if not display_name or not display_name.strip():
        display_name = crew.name.replace("-", " ").title()

    answer_modes = crew.answer_modes
    if not answer_modes:
        answer_modes = [AnswerModeOption.model_validate(m) for m in modes_src]

    default_mode = crew.default_answer_mode
    mode_ids = {m.id for m in answer_modes}
    if not default_mode or default_mode not in mode_ids:
        default_mode = "light" if "light" in mode_ids else next(iter(mode_ids))

    output_composition = crew.output_composition or composition

    return crew.model_copy(
        update={
            "display_name": display_name,
            "default_answer_mode": default_mode,
            "answer_modes": answer_modes,
            "output_composition": output_composition,
            "archetype": archetype,
        }
    )


class CrewConfig(BaseModel):
    """A full crew: agents + any custom tools + metadata (→ crew.yaml)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Kebab-case crew name, used as the output directory.")
    description: str = Field(description="One-paragraph summary of the crew's purpose.")
    display_name: str | None = Field(
        default=None, description="UI label for chat header. Defaults from name."
    )
    default_answer_mode: str | None = Field(
        default=None, description="Known mode id; must be in answer_modes."
    )
    answer_modes: list[AnswerModeOption] | None = Field(
        default=None, description="Non-empty list of selectable answer modes."
    )
    output_composition: OutputComposition | None = Field(
        default=None, description="Drives synthesizer prompt drafting and rich tabs."
    )
    archetype: CrewArchetype | None = Field(
        default=None,
        description="strategy | lightweight — selects manifest defaults. Not written to YAML.",
    )
    agents: list[AgentConfig]
    custom_tools: list[ToolSpec] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_format(cls, v: str) -> str:
        if not re.match(r"^[a-z][a-z0-9-]*$", v):
            raise ValueError(f"Crew name must be kebab-case, got {v!r}.")
        return v

    @field_validator("default_answer_mode")
    @classmethod
    def _default_mode_known(cls, v: str | None) -> str | None:
        if v is not None and v not in KNOWN_ANSWER_MODE_IDS:
            raise ValueError(
                f"default_answer_mode {v!r} is not supported. "
                f"Allowed: {sorted(KNOWN_ANSWER_MODE_IDS)}."
            )
        return v

    @model_validator(mode="after")
    def _unique_agent_ids(self) -> CrewConfig:
        ids = [a.id for a in self.agents]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"Duplicate agent ids: {sorted(dupes)}")
        return self

    @model_validator(mode="after")
    def _unique_tool_ids(self) -> CrewConfig:
        ids = [t.id for t in self.custom_tools]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"Duplicate custom tool ids: {sorted(dupes)}")
        return self

    @model_validator(mode="after")
    def _default_mode_in_list(self) -> CrewConfig:
        if self.default_answer_mode and self.answer_modes:
            ids = {m.id for m in self.answer_modes}
            if self.default_answer_mode not in ids:
                raise ValueError(
                    f"default_answer_mode {self.default_answer_mode!r} "
                    f"must be one of answer_modes ids: {sorted(ids)}."
                )
        return self

    def to_manifest_dict(self) -> dict[str, Any]:
        """Serialize crew.yaml fields (archetype omitted)."""
        filled = apply_manifest_defaults(self)
        out: dict[str, Any] = {
            "name": filled.name,
            "display_name": filled.display_name,
            "description": filled.description,
            "default_answer_mode": filled.default_answer_mode,
            "answer_modes": [m.model_dump() for m in filled.answer_modes or []],
        }
        if filled.output_composition is not None:
            out["output_composition"] = filled.output_composition.model_dump()
        return out
