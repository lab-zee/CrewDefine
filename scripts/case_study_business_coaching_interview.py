#!/usr/bin/env python3
"""Timed, brief-driven CrewDefine interview for the business-coaching case study.

Runs a real Anthropic interview; answers come from a fixed product brief so the
session is reproducible. Logs Q&A + wall-clock timing to stdout and a JSON file.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from crewdefine.cli import _validate_and_write
from crewdefine.config import load_settings
from crewdefine.interview import InterviewError, run_interview
from crewdefine.llm import LLMClient

console = Console()
ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "crews"
LOG_PATH = ROOT / "crews" / "_case_study_business_coaching_interview.json"

SEED = """I want a Lab Z crew for business coaching aimed at founders and operators
of early-to-growth-stage companies (B2B SaaS and services, roughly seed to Series B).

The crew should:
1. Diagnose the business (positioning, ICP, funnel health, org bottlenecks).
2. Research market and competitors with cited sources.
3. Stress-test unit economics and runway with calculations.
4. Produce a concrete 90-day action plan with milestones and resources.
5. Synthesize a coaching report with charts, tables, and citations.

Required roles (please create these; director + synthesizer are mandatory for Zero):
- director — orchestrate, check info sufficiency, ask clarifying follow-ups
- diagnostician — SWOT + internal assessment; use knowledge_base / document / swot
- market_researcher — web_search, news_search, scrape_website, citations
- financial_analyst — calculator + web_search for comps / pricing sanity
- action_planner — generate_recommendations + follow-up questions for the plan
- synthesizer — visualizer + extract_citations for the final report

Also define these custom tools (stubs are fine — we will implement later):
- industry_kpi_benchmark — given industry + stage, return typical KPI ranges
  (CAC, LTV, churn, gross margin, burn multiple) for coaching comparisons
- runway_and_scenarios — given cash, burn, and optional growth/churn rates,
  return runway months and 2–3 scenario tables
- competitor_snapshot — given a competitor URL or name, return a structured
  profile (positioning, pricing signals, ICP cues, recent news hooks)

Prefer built-in LabZ tools wherever they already cover the need. Use answer
modes summary / light / extended. Output composition should include summary,
charts when useful, tables when structured, and citations required.
"""


@dataclass
class BriefIO:
    """Answer interview questions from the fixed brief + keyword heuristics."""

    transcript: list[dict] = field(default_factory=list)
    _confirm_seen: int = 0

    def ask(self, question: str, options: list[str] | None, allow_skip: bool) -> str:
        answer = self._answer(question, options, allow_skip)
        self.transcript.append(
            {
                "question": question,
                "options": options,
                "answer": answer,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )
        console.print(f"[dim]Q:[/dim] {question[:240]}{'…' if len(question) > 240 else ''}")
        console.print(f"[cyan]A:[/cyan] {answer}\n")
        return answer

    def info(self, message: str) -> None:
        console.print(f"[italic dim]{message}[/italic dim]")

    def warn(self, message: str) -> None:
        console.print(f"[bold yellow]![/bold yellow] [yellow]{message}[/yellow]")

    def _answer(self, question: str, options: list[str] | None, allow_skip: bool) -> str:
        q = question.lower()

        # Final roster / proceed confirmations
        if options:
            joined = " | ".join(options).lower()
            if any(x in joined for x in ("yes", "looks good", "proceed", "confirm", "finalize")):
                for i, opt in enumerate(options):
                    if re.search(r"yes|looks good|proceed|confirm|finalize|ship", opt, re.I):
                        return opt
                return options[0]
            if "answer mode" in q or "verbosity" in q:
                for opt in options:
                    if re.search(r"summary|light|extended|all", opt, re.I):
                        return opt
                return options[0]
            if "chart" in q or "visual" in q or "citation" in q or "composition" in q:
                for opt in options:
                    if re.search(r"yes|required|when|include|chart", opt, re.I):
                        return opt
                return options[-1] if options else "yes"
            # Prefer the option that mentions our specialists / tools when present
            for opt in options:
                if re.search(
                    r"diagnostic|market|financial|action|kpi|runway|competitor|coach",
                    opt,
                    re.I,
                ):
                    return opt
            return options[0]

        if any(w in q for w in ("confirm", "looks good", "ready to finalize", "proceed", "happy with")):
            self._confirm_seen += 1
            return "yes — ship it"

        if "purpose" in q or "what does this crew" in q or "crew do" in q or "who is it for" in q:
            return (
                "Business coaching for founders and operators (seed–Series B B2B SaaS/services). "
                "Diagnose the business, research market/competitors with citations, stress-test "
                "unit economics and runway, then deliver a 90-day action plan coaching report."
            )

        if "specialist" in q or "agents" in q or "roles" in q or "who should" in q:
            return (
                "director; diagnostician (SWOT/internal); market_researcher; "
                "financial_analyst; action_planner; synthesizer. "
                "Director orchestrates; specialists do deep work; synthesizer writes the report."
            )

        if "custom tool" in q or "new tool" in q or "plugin" in q:
            return (
                "Yes — three custom tools: industry_kpi_benchmark, runway_and_scenarios, "
                "and competitor_snapshot. Stubs OK."
            )

        if "tool" in q and ("builtin" in q or "built-in" in q or "which tools" in q or "assign" in q):
            if "diagnostic" in q:
                return "swot, knowledge_base, document, validate_information_sufficiency"
            if "market" in q:
                return "web_search, news_search, scrape_website, extract_citations_structured, competitor_snapshot"
            if "financial" in q:
                return "calculator, web_search, industry_kpi_benchmark, runway_and_scenarios"
            if "action" in q or "plan" in q:
                return "generate_recommendations, generate_followup_questions"
            if "synth" in q:
                return "visualizer, extract_citations"
            if "director" in q:
                return "validate_information_sufficiency, generate_followup_questions"
            return (
                "director: validate_information_sufficiency, generate_followup_questions; "
                "diagnostician: swot, knowledge_base, document; "
                "market_researcher: web_search, news_search, scrape_website, "
                "extract_citations_structured, competitor_snapshot; "
                "financial_analyst: calculator, web_search, industry_kpi_benchmark, runway_and_scenarios; "
                "action_planner: generate_recommendations, generate_followup_questions; "
                "synthesizer: visualizer, extract_citations"
            )

        if "delegat" in q or "collaborate" in q or "graph" in q:
            return (
                "director delegates to all specialists then synthesizer. "
                "Specialists return only to director. No specialist-to-specialist edges."
            )

        if "answer mode" in q or "verbosity" in q or "output mode" in q:
            return "Include summary, light, and extended. Default to light."

        if "composition" in q or "tabs" in q or "chart" in q or "citation" in q or "table" in q:
            return (
                "Tabs: summary + raw_data. Citations required. Charts when useful. "
                "Tables when structured. Synthesizer tools: visualizer, extract_citations."
            )

        if "name" in q and "crew" in q:
            return "business-coaching-crew"

        if "display" in q:
            return "Business Coach"

        if allow_skip and ("optional" in q or "anything else" in q or "skip" in q):
            return ""

        # Default: restate the brief so the interviewer stays on rails
        return (
            "Stay with the business-coaching brief: director, diagnostician, market_researcher, "
            "financial_analyst, action_planner, synthesizer; custom tools industry_kpi_benchmark, "
            "runway_and_scenarios, competitor_snapshot; rich cited report with charts/tables."
        )


def main() -> None:
    settings = load_settings()
    client = LLMClient(settings)
    io = BriefIO()

    console.print(
        f"[bold]Case study interview[/bold] — model={settings.model} max_turns={settings.max_turns}"
    )
    t0 = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    try:
        crew = run_interview(client, settings, io, seed_user_message=SEED)
    except InterviewError as e:
        console.print(f"[red]Interview failed:[/red] {e}")
        raise SystemExit(1) from e
    interview_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    # Force a stable folder name for the case study
    crew = crew.model_copy(update={"name": "business-coaching-crew", "display_name": "Business Coach"})
    _validate_and_write(crew, OUT_ROOT, overwrite=True)
    write_s = time.perf_counter() - t1
    total_s = time.perf_counter() - t0
    ended = datetime.now(timezone.utc).isoformat()

    payload = {
        "started_at": started,
        "ended_at": ended,
        "model": settings.model,
        "interview_seconds": round(interview_s, 1),
        "write_validate_seconds": round(write_s, 1),
        "total_seconds": round(total_s, 1),
        "questions_asked": len(io.transcript),
        "agents": [{"id": a.id, "name": a.name, "tools": a.tools} for a in crew.agents],
        "custom_tools": [t.id for t in crew.custom_tools],
        "transcript": io.transcript,
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(
        f"\n[bold green]Done[/bold green] in {total_s:.1f}s "
        f"(interview {interview_s:.1f}s, write {write_s:.1f}s) → {OUT_ROOT / 'business-coaching-crew'}"
    )
    console.print(f"Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
