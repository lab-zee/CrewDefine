from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

import crewdefine.cli as cli
from crewdefine.config import Settings
from crewdefine.generator import write_crew
from crewdefine.schema import AgentConfig, CrewConfig

runner = CliRunner()


def _settings() -> Settings:
    return Settings(
        api_key="test",
        model="test-model",
        max_turns=10,
        max_agents_per_crew=5,
        debug_log_path=None,
    )


def _offline_cli(monkeypatch: pytest.MonkeyPatch, crew: CrewConfig) -> None:
    monkeypatch.setattr(cli, "load_settings", _settings)
    monkeypatch.setattr(cli, "LLMClient", lambda _settings: object())
    monkeypatch.setattr(cli, "run_interview", lambda *_args, **_kwargs: crew)


def test_version_and_list_tools_commands() -> None:
    with pytest.raises(typer.Exit):
        cli._main(version=True)

    tools = runner.invoke(cli.app, ["list-tools"])
    assert tools.exit_code == 0
    assert "web_search" in tools.stdout


def test_validate_command_accepts_maintained_crew() -> None:
    result = runner.invoke(
        cli.app,
        ["validate", "crews/dinner-planning-crew"],
    )
    assert result.exit_code == 0
    assert "passes validation" in result.stdout


def test_validate_command_reports_invalid_directory(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["validate", str(tmp_path)])
    assert result.exit_code == 1
    assert "No agents/ directory" in result.stdout


def test_new_command_runs_offline_and_writes_crew(
    basic_crew: CrewConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _offline_cli(monkeypatch, basic_crew)
    result = runner.invoke(cli.app, ["new", "--out", str(tmp_path), "--seed", "test seed"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / basic_crew.name / "crew.yaml").exists()
    assert "Crew written" in result.stdout


def test_add_agent_command_loads_existing_crew(
    basic_crew: CrewConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crew_dir = write_crew(basic_crew, tmp_path).crew_dir
    _offline_cli(monkeypatch, basic_crew)
    result = runner.invoke(cli.app, ["add-agent", str(crew_dir)])
    assert result.exit_code == 0, result.output
    assert "Crew written" in result.stdout


def test_update_agent_rejects_unknown_id(
    basic_crew: CrewConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crew_dir = write_crew(basic_crew, tmp_path).crew_dir
    _offline_cli(monkeypatch, basic_crew)
    result = runner.invoke(cli.app, ["update-agent", str(crew_dir), "missing"])
    assert result.exit_code == 1
    assert "No agent 'missing'" in result.stdout


def test_update_agent_command_rewrites_existing_agent(
    basic_crew: CrewConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crew_dir = write_crew(basic_crew, tmp_path).crew_dir
    _offline_cli(monkeypatch, basic_crew)
    result = runner.invoke(cli.app, ["update-agent", str(crew_dir), "director"])
    assert result.exit_code == 0, result.output
    assert "Crew written" in result.stdout


def test_console_io_handles_number_custom_and_skipped_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers = iter(["2", "custom", "", "plain"])
    monkeypatch.setattr(cli.Prompt, "ask", lambda *_args, **_kwargs: next(answers))
    io = cli.ConsoleIO()
    assert io.ask("Pick", ["one", "two"], allow_skip=True) == "two"
    assert io.ask("Pick", ["one", "two"], allow_skip=False) == "custom"
    assert io.ask("Optional", None, allow_skip=True) == "(skipped)"
    assert io.ask("Required", None, allow_skip=False) == "plain"
    io.info("info")
    io.warn("warn")


def test_settings_and_validation_failures_exit_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli, "load_settings", lambda: (_ for _ in ()).throw(RuntimeError("missing"))
    )
    with pytest.raises(typer.Exit):
        cli._load_settings_or_exit()

    invalid = CrewConfig(
        name="invalid",
        description="missing infrastructure",
        agents=[
            AgentConfig(
                id="worker",
                name="Worker",
                role="Worker who works",
                system_prompt="a" * 100,
            )
        ],
    )
    with pytest.raises(typer.Exit):
        cli._validate_and_write(invalid, tmp_path, overwrite=False)


def test_load_crew_requires_agents_directory(tmp_path: Path) -> None:
    with pytest.raises(typer.Exit):
        cli._load_crew_from_dir(tmp_path)


def test_new_command_reports_interview_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "load_settings", _settings)
    monkeypatch.setattr(cli, "LLMClient", lambda _settings: object())

    def fail(*_args: Any, **_kwargs: Any) -> CrewConfig:
        raise cli.InterviewError("offline failure")

    monkeypatch.setattr(cli, "run_interview", fail)
    result = runner.invoke(cli.app, ["new", "--seed", "test"])
    assert result.exit_code == 1
    assert "Interview failed" in result.stdout
