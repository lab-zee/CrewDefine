"""Pydantic models that mirror LabZ's agent/crew shape.

LabZ loads each agent YAML into a plain dataclass (`AgentConfig` at
`backend/src/agents/base.py`). It does not validate beyond that, so CrewDefine
owns validation: required fields, id formatting, delegation-graph sanity, and
the OpenAI function-calling shape for tool stubs.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AGENT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
TOOL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


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


class CrewConfig(BaseModel):
    """A full crew: agents + any custom tools + metadata."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Kebab-case crew name, used as the output directory.")
    description: str = Field(description="One-paragraph summary of the crew's purpose.")
    agents: list[AgentConfig]
    custom_tools: list[ToolSpec] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_format(cls, v: str) -> str:
        if not re.match(r"^[a-z][a-z0-9-]*$", v):
            raise ValueError(f"Crew name must be kebab-case, got {v!r}.")
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
