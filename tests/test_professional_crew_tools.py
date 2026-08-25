"""Smoke tests for deterministic tools shipped with professional example crews."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from crewdefine.validator import validate_crew_dir

ROOT = Path(__file__).parents[1]
MAINTAINED_CREWS = (
    "business-coaching-crew",
    "dinner-planning-crew",
    "technical-due-diligence",
    "research-evidence-synthesis",
    "incident-analysis",
)


def _load(relative_path: str) -> ModuleType:
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("crew_name", MAINTAINED_CREWS)
def test_maintained_crew_passes_repository_validation(crew_name: str) -> None:
    report = validate_crew_dir(ROOT / "crews" / crew_name)
    assert report.ok, report.errors


def test_competitor_snapshot_builds_offline_name_profile() -> None:
    module = _load("crews/business-coaching-crew/tools/competitor_snapshot.py")
    result = module.competitor_snapshot(" Acme ")
    assert result["competitor_input"] == "Acme"
    assert result["name"] == "Acme"
    assert result["domain"] is None
    assert result["page_excerpt"] is None


def test_competitor_snapshot_normalizes_url_and_excerpt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load("crews/business-coaching-crew/tools/competitor_snapshot.py")
    monkeypatch.setattr(module, "_try_fetch_text", lambda _url: "<title>Evidence</title>")
    result = module.competitor_snapshot("https://www.acme.example/about")
    assert result["name"] == "Acme"
    assert result["domain"] == "www.acme.example"
    assert result["positioning"]["notes_from_page"] == "<title>Evidence</title>"


@pytest.mark.parametrize(
    ("industry", "stage", "normalized_industry", "normalized_stage"),
    [
        ("B2B software", "pre-seed", "b2b_saas", "seed"),
        ("Consulting agency", "Series A", "services", "series_a"),
        ("Professional services", "Series B", "services", "series_b"),
        ("unknown", "unknown", "b2b_saas", "seed"),
    ],
)
def test_industry_kpi_benchmark_normalizes_inputs(
    industry: str,
    stage: str,
    normalized_industry: str,
    normalized_stage: str,
) -> None:
    module = _load("crews/business-coaching-crew/tools/industry_kpi_benchmark.py")
    result = module.industry_kpi_benchmark(industry, stage)
    assert result["normalized_industry"] == normalized_industry
    assert result["normalized_stage"] == normalized_stage
    assert result["kpis"]["gross_margin_pct"]["p50"] > 0


def test_runway_scenarios_flag_short_runway_and_order_cases() -> None:
    module = _load("crews/business-coaching-crew/tools/runway_and_scenarios.py")
    result = module.runway_and_scenarios(
        cash=500_000,
        monthly_burn=100_000,
        growth_rate=0.05,
        churn_rate=0.01,
        monthly_revenue=100_000,
    )
    scenarios = {item["name"]: item for item in result["scenarios"]}
    assert result["coaching_flags"] == {"under_6_months": True, "under_12_months": True}
    assert scenarios["optimistic"]["runway_months"] > scenarios["base"]["runway_months"]
    assert scenarios["pessimistic"]["runway_months"] < scenarios["base"]["runway_months"]


def test_runway_scenarios_handle_cash_flow_positive_case() -> None:
    module = _load("crews/business-coaching-crew/tools/runway_and_scenarios.py")
    result = module.runway_and_scenarios(cash=100_000, monthly_burn=0)
    assert result["base_runway_months"] == "infinite_or_cash_flow_positive"
    assert result["coaching_flags"] == {"under_6_months": False, "under_12_months": False}


def test_dependency_risk_matrix_prioritizes_explicit_risk_signals() -> None:
    module = _load("crews/technical-due-diligence/tools/dependency_risk_matrix.py")
    result = module.dependency_risk_matrix(
        [
            {
                "name": "legacy",
                "known_vulnerability_severity": "critical",
                "maintenance_status": "abandoned",
                "latest_release_age_days": 900,
                "license": "AGPL",
            },
            {
                "name": "current",
                "known_vulnerability_severity": "none",
                "maintenance_status": "active",
                "latest_release_age_days": 20,
                "license": "MIT",
            },
        ]
    )
    assert result["dependencies"][0]["name"] == "legacy"
    assert result["dependencies"][0]["risk_tier"] == "critical"
    assert result["counts_by_tier"]["low"] == 1


def test_evidence_coverage_flags_unsupported_high_severity_findings() -> None:
    module = _load("crews/technical-due-diligence/tools/evidence_coverage_score.py")
    result = module.evidence_coverage_score(
        [
            {"id": "F1", "claim": "Observed", "evidence_refs": ["doc:1"], "severity": "low"},
            {"id": "F2", "claim": "Unverified", "evidence_refs": [], "severity": "high"},
        ]
    )
    assert result["evidence_coverage"] == 0.5
    assert result["unsupported_high_severity"] == ["F2"]


def test_study_appraisal_reports_criteria_instead_of_composite_score() -> None:
    module = _load("crews/research-evidence-synthesis/tools/study_quality_assessment.py")
    result = module.study_quality_assessment(
        [
            {
                "id": "S1",
                "study_design": "randomized",
                "sample_size": 20,
                "controls": "placebo",
                "preregistered": False,
                "attrition_rate": 0.25,
                "missing_data": "reported",
                "conflicts": "none",
                "reported_limitations": "short follow-up",
            }
        ]
    )
    flags = result["studies"][0]["criterion_flags"]
    assert "not preregistered" in flags
    assert "attrition exceeds 20%" in flags
    assert "quality_score" not in result["studies"][0]


def test_evidence_table_builder_detects_duplicate_candidates() -> None:
    module = _load("crews/research-evidence-synthesis/tools/evidence_table_builder.py")
    finding = {"source_id": "S1", "outcome": "latency", "estimate": "20 ms"}
    result = module.evidence_table_builder([finding, finding])
    assert result["row_count"] == 2
    assert result["duplicate_candidates"][0]["possible_duplicate_of"] == 1


def test_event_timeline_builder_sorts_and_reports_large_gaps() -> None:
    module = _load("crews/incident-analysis/tools/event_timeline_builder.py")
    result = module.event_timeline_builder(
        [
            {"timestamp": "2026-01-01T01:00:00Z", "source": "alerts", "description": "B"},
            {"timestamp": "2026-01-01T00:00:00Z", "source": "deploy", "description": "A"},
        ]
    )
    assert result["events"][0]["description"] == "A"
    assert result["gaps_over_30_minutes"][0]["gap_seconds"] == 3600


def test_causal_matrix_does_not_select_a_root_cause() -> None:
    module = _load("crews/incident-analysis/tools/causal_hypothesis_matrix.py")
    result = module.causal_hypothesis_matrix(
        [{"id": "H1", "statement": "Deployment regression"}],
        [{"id": "E1", "supports": "H1"}, {"id": "E2", "contradicts": "H1"}],
    )
    assert result["contested_hypotheses"] == ["H1"]
    assert "root_cause" not in result


def test_action_validator_requires_verification_and_ownership() -> None:
    module = _load("crews/incident-analysis/tools/action_item_validator.py")
    result = module.action_item_validator(
        [{"description": "Add alert", "verification_method": "done"}]
    )
    assert result["unverifiable_action_ids"] == [1]
    assert "owner" in result["actions"][0]["missing_fields"]
