"""Typer CLI entry points.

Commands:
    crewdefine new [--out DIR]
    crewdefine add-agent CREW_DIR
    crewdefine update-agent CREW_DIR AGENT_ID
    crewdefine validate CREW_DIR
    crewdefine list-tools
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from crewdefine import __version__
from crewdefine.config import Settings, load_settings
from crewdefine.generator import write_crew
from crewdefine.interview import InterviewError, run_interview
from crewdefine.llm import LLMClient
from crewdefine.schema import AgentConfig, CrewConfig
from crewdefine.tools_catalog import BUILTIN_TOOLS
from crewdefine.validator import ValidationError, validate_crew, validate_crew_dir

app = typer.Typer(
    name="crewdefine",
    help="Interactive CLI that helps you author LabZ agent crews as YAML.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.callback()
def _main(
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
) -> None:
    if version:
        console.print(f"crewdefine {__version__}")
        raise typer.Exit(0)


# ---------------------------------------------------------------------------
# User I/O adapter
# ---------------------------------------------------------------------------


class ConsoleIO:
    """Rich-backed UserIO used by the interview loop."""

    def ask(self, question: str, options: list[str] | None, allow_skip: bool) -> str:
        self._render_question(question)
        if options:
            table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 1))
            for i, opt in enumerate(options, 1):
                table.add_row(f"[bold cyan]{i}[/bold cyan][dim].[/dim]", opt)
            console.print(table)
            hint = "[bold cyan]›[/bold cyan] [cyan]Enter a number or type your own answer[/cyan]"
            if allow_skip:
                hint += " [dim](blank to skip)[/dim]"
            raw = Prompt.ask(hint, default="" if allow_skip else None)
            raw = (raw or "").strip()
            if raw.isdigit():
                idx = int(raw) - 1
                if 0 <= idx < len(options):
                    return options[idx]
            return raw or "(skipped)"

        hint = "[bold cyan]›[/bold cyan] [cyan]Your answer[/cyan]"
        if allow_skip:
            hint += " [dim](blank to skip)[/dim]"
        raw = Prompt.ask(hint, default="" if allow_skip else None)
        return (raw or "").strip() or "(skipped)"

    @staticmethod
    def _render_question(question: str) -> None:
        """Render the question as Markdown so embedded tables, bold, code
        fences, and bullet lists display properly. The LLM legitimately uses
        these — e.g. a roster summary before finalizing — and raw markdown
        syntax in a plain Panel is unreadable."""
        console.print()  # leading blank line so successive questions don't stack visually
        console.print(Panel(Markdown(question), border_style="cyan", padding=(0, 1)))

    def info(self, message: str) -> None:
        console.print(f"[italic dim]{message}[/italic dim]")

    def warn(self, message: str) -> None:
        console.print(f"[bold yellow]![/bold yellow] [yellow]{message}[/yellow]")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("new")
def cmd_new(
    out: Path = typer.Option(
        Path("crews"),
        "--out",
        "-o",
        help="Directory that will contain the new crew folder.",
        file_okay=False,
        dir_okay=True,
    ),
    seed: str | None = typer.Option(
        None,
        "--seed",
        "-s",
        help="Optional seed message for the interviewer (e.g. 'A crew for competitive intel').",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite the crew directory if it already exists."
    ),
) -> None:
    """Create a new crew from scratch via guided interview."""
    settings = _load_settings_or_exit()
    client = LLMClient(settings)
    io = ConsoleIO()

    seed_message = seed or (
        "I want to design a new LabZ agent crew from scratch. Please start by asking me "
        "about the crew's purpose, then help me figure out which specialists I need."
    )
    console.print(
        Panel.fit(
            f"[bold cyan]CrewDefine[/bold cyan] [dim]v{__version__}[/dim] — starting a new crew.\n"
            f"[dim]Model:[/dim] [cyan]{settings.model}[/cyan]  [dim]•[/dim]  "
            f"[dim]Max turns:[/dim] [cyan]{settings.max_turns}[/cyan]",
            border_style="cyan",
            padding=(0, 1),
        )
    )

    try:
        crew = run_interview(client, settings, io, seed_user_message=seed_message)
    except InterviewError as e:
        console.print(f"[red]Interview failed:[/red] {e}")
        raise typer.Exit(1) from e

    _validate_and_write(crew, out, overwrite=overwrite)


@app.command("add-agent")
def cmd_add_agent(
    crew_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Path to an existing crew directory.",
    ),
    seed: str | None = typer.Option(None, "--seed", "-s", help="Seed message for the interview."),
) -> None:
    """Add a new agent to an existing crew."""
    settings = _load_settings_or_exit()
    client = LLMClient(settings)
    io = ConsoleIO()

    existing = _load_crew_from_dir(crew_dir)
    seed_message = seed or (
        f"I want to add a new agent to the existing crew '{existing.name}'. "
        f"Here's what the crew already has:\n\n"
        + "\n".join(f"- {a.id} ({a.name}): {a.role}" for a in existing.agents)
        + "\n\nAsk me what new specialist I need and help me define it."
    )

    try:
        updated = run_interview(
            client, settings, io, seed_user_message=seed_message, existing_crew=existing
        )
    except InterviewError as e:
        console.print(f"[red]Interview failed:[/red] {e}")
        raise typer.Exit(1) from e

    # Write to the same dir with overwrite=True since the crew already exists.
    _validate_and_write(updated, crew_dir.parent, overwrite=True)


@app.command("update-agent")
def cmd_update_agent(
    crew_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Path to an existing crew directory.",
    ),
    agent_id: str = typer.Argument(..., help="Snake_case id of the agent to update."),
    seed: str | None = typer.Option(None, "--seed", "-s", help="Seed message for the interview."),
) -> None:
    """Revise an existing agent's prompt, tools, or delegation."""
    settings = _load_settings_or_exit()
    client = LLMClient(settings)
    io = ConsoleIO()

    existing = _load_crew_from_dir(crew_dir)
    target = next((a for a in existing.agents if a.id == agent_id), None)
    if target is None:
        console.print(f"[red]No agent '{agent_id}' found in {crew_dir}.[/red]")
        raise typer.Exit(1)

    seed_message = seed or (
        f"I want to update the agent '{agent_id}' in crew '{existing.name}'. "
        f"Its current role is: {target.role}. "
        f"Ask me what I want to change (role, tools, delegation, or prompt focus), "
        "then call `record_agent` with the same id to overwrite it."
    )

    # Drop the existing agent's cached prompt so the drafter redraws it.
    filtered = CrewConfig(
        name=existing.name,
        description=existing.description,
        agents=[a for a in existing.agents if a.id != agent_id],
        custom_tools=list(existing.custom_tools),
    )

    try:
        updated = run_interview(
            client, settings, io, seed_user_message=seed_message, existing_crew=filtered
        )
    except InterviewError as e:
        console.print(f"[red]Interview failed:[/red] {e}")
        raise typer.Exit(1) from e

    _validate_and_write(updated, crew_dir.parent, overwrite=True)


@app.command("validate")
def cmd_validate(
    crew_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Path to a crew directory.",
    ),
) -> None:
    """Schema + round-trip + delegation-graph validation on a crew directory."""
    report = validate_crew_dir(crew_dir)
    if report.warnings:
        for w in report.warnings:
            console.print(f"[yellow]warn:[/yellow] {w}")
    if report.errors:
        for e in report.errors:
            console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(1)
    console.print(f"[green]OK[/green] {crew_dir} passes validation.")


@app.command("list-tools")
def cmd_list_tools() -> None:
    """Show the built-in LabZ tool catalog CrewDefine knows about."""
    table = Table(title="Built-in LabZ tools", show_lines=False)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Description")
    for t in BUILTIN_TOOLS:
        table.add_row(t.id, t.summary)
    console.print(table)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_settings_or_exit() -> Settings:
    try:
        return load_settings()
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e


def _validate_and_write(crew: CrewConfig, out_root: Path, *, overwrite: bool) -> None:
    report = validate_crew(crew)
    if report.warnings:
        for w in report.warnings:
            console.print(f"[bold yellow]![/bold yellow] [yellow]{w}[/yellow]")
    try:
        report.raise_if_failed()
    except ValidationError as e:
        body = "\n".join(f"[red]•[/red] {err}" for err in e.errors)
        console.print(
            Panel(
                body,
                title="[bold red]Validation failed — not writing any files[/bold red]",
                border_style="red",
                padding=(0, 1),
            )
        )
        raise typer.Exit(1) from e

    out_root.mkdir(parents=True, exist_ok=True)
    result = write_crew(crew, out_root, overwrite=overwrite)
    body = (
        f"[bold]{len(result.agent_files)}[/bold] agent YAMLs"
        f"  •  [bold]{len(result.tool_files)}[/bold] tool stubs\n"
        f"[dim]Path:[/dim] {result.crew_dir}"
    )
    console.print(
        Panel(
            body,
            title="[bold green]✓ Crew written[/bold green]",
            border_style="green",
            padding=(0, 1),
        )
    )


def _load_crew_from_dir(crew_dir: Path) -> CrewConfig:
    """Reconstruct a `CrewConfig` from an existing crew directory.

    This is deliberately simple — we do not persist crew-level metadata
    anywhere yet, so we derive name/description from the directory name
    and the README's first paragraph.
    """
    agents_dir = crew_dir / "agents"
    if not agents_dir.is_dir():
        console.print(f"[red]No agents/ directory under {crew_dir}.[/red]")
        raise typer.Exit(1)

    agents: list[AgentConfig] = []
    for path in sorted(agents_dir.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        agents.append(AgentConfig.model_validate(raw))

    description = "Existing crew loaded from disk."
    readme_path = crew_dir / "README.md"
    if readme_path.exists():
        for line in readme_path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                description = line.strip()
                break

    return CrewConfig(name=crew_dir.name, description=description, agents=agents)


if __name__ == "__main__":
    app()
