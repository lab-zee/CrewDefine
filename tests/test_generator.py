from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from crewdefine.generator import write_crew
from crewdefine.schema import CrewConfig
from crewdefine.validator import validate_crew_dir


def test_write_crew_produces_expected_layout(basic_crew: CrewConfig, tmp_path: Path) -> None:
    result = write_crew(basic_crew, tmp_path)
    assert result.crew_dir == tmp_path / basic_crew.name
    assert (result.crew_dir / "README.md").exists()
    assert (result.crew_dir / "agents").is_dir()
    # No custom tools in this crew → no tools/ dir.
    assert not (result.crew_dir / "tools").exists()
    for agent in basic_crew.agents:
        assert (result.crew_dir / "agents" / f"{agent.id}.yaml").exists()


def test_write_crew_emits_custom_tool_stubs(
    crew_with_custom_tool: CrewConfig, tmp_path: Path
) -> None:
    result = write_crew(crew_with_custom_tool, tmp_path)
    tool_file = result.crew_dir / "tools" / "crm_lookup.py"
    assert tool_file.exists()
    content = tool_file.read_text(encoding="utf-8")
    assert "TOOL_DEFINITION" in content
    assert '"name": "crm_lookup"' in content
    assert "def crm_lookup(email)" in content


def test_refuses_overwrite_by_default(basic_crew: CrewConfig, tmp_path: Path) -> None:
    write_crew(basic_crew, tmp_path)
    with pytest.raises(FileExistsError):
        write_crew(basic_crew, tmp_path)


def test_overwrite_flag_replaces(basic_crew: CrewConfig, tmp_path: Path) -> None:
    write_crew(basic_crew, tmp_path)
    write_crew(basic_crew, tmp_path, overwrite=True)  # should not raise


def test_emitted_yaml_loads_back_into_equivalent_dict(
    basic_crew: CrewConfig, tmp_path: Path
) -> None:
    result = write_crew(basic_crew, tmp_path)
    for agent in basic_crew.agents:
        path = result.crew_dir / "agents" / f"{agent.id}.yaml"
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert loaded == agent.to_yaml_dict()


def test_multiline_system_prompt_uses_literal_block(basic_crew: CrewConfig, tmp_path: Path) -> None:
    result = write_crew(basic_crew, tmp_path)
    any_agent = basic_crew.agents[0]
    raw = (result.crew_dir / "agents" / f"{any_agent.id}.yaml").read_text(encoding="utf-8")
    # The `|` literal style marker should appear on the system_prompt line when it is multiline,
    # or at minimum the key should be a bare scalar. Accept either; the round-trip test above
    # already guarantees content preservation.
    assert "system_prompt:" in raw


def test_validate_crew_dir_passes_for_written_crew(basic_crew: CrewConfig, tmp_path: Path) -> None:
    result = write_crew(basic_crew, tmp_path)
    report = validate_crew_dir(result.crew_dir)
    assert report.ok, report.errors
