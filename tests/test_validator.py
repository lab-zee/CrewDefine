from __future__ import annotations

from crewdefine.schema import AgentConfig, CrewConfig, ToolSpec
from crewdefine.validator import validate_crew


def test_basic_crew_passes(basic_crew: CrewConfig) -> None:
    report = validate_crew(basic_crew)
    assert report.ok, report.errors


def test_unknown_delegation_target_fails() -> None:
    bad = AgentConfig(
        id="director",
        name="Director",
        role="Director who orchestrates",
        system_prompt="a" * 100,
        can_delegate_to=["ghost"],
    )
    crew = CrewConfig(name="crew", description="x", agents=[bad])
    report = validate_crew(crew)
    assert not report.ok
    assert any("ghost" in e for e in report.errors)


def test_unknown_tool_fails() -> None:
    agent = AgentConfig(
        id="director",
        name="Director",
        role="Director who orchestrates",
        system_prompt="a" * 100,
        tools=["nonexistent_tool"],
    )
    crew = CrewConfig(name="crew", description="x", agents=[agent])
    report = validate_crew(crew)
    assert not report.ok
    assert any("nonexistent_tool" in e for e in report.errors)


def test_custom_tool_resolves(crew_with_custom_tool: CrewConfig) -> None:
    report = validate_crew(crew_with_custom_tool)
    assert report.ok, report.errors


def test_unused_custom_tool_warns() -> None:
    orphan = ToolSpec(id="orphan_tool", description="nobody calls me")
    director = AgentConfig(
        id="director",
        name="Director",
        role="Director who orchestrates",
        system_prompt="a" * 100,
    )
    crew = CrewConfig(
        name="crew",
        description="x",
        agents=[director],
        custom_tools=[orphan],
    )
    report = validate_crew(crew)
    assert report.ok
    assert any("orphan_tool" in w for w in report.warnings)


def test_delegation_cycle_detected() -> None:
    a = AgentConfig(
        id="a",
        name="A",
        role="A who delegates to B",
        system_prompt="a" * 100,
        can_delegate_to=["b"],
    )
    b = AgentConfig(
        id="b",
        name="B",
        role="B who delegates back to A",
        system_prompt="a" * 100,
        can_delegate_to=["a"],
    )
    crew = CrewConfig(name="cycle-crew", description="x", agents=[a, b])
    report = validate_crew(crew)
    assert not report.ok
    assert any("cycle" in e.lower() for e in report.errors)


def test_round_trip_produces_stable_yaml(basic_crew: CrewConfig) -> None:
    # The round-trip check inside validate_crew should pass on a well-formed crew.
    report = validate_crew(basic_crew)
    assert report.ok, report.errors
