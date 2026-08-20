"""Industry KPI benchmarks for coaching comparisons.

Heuristic ranges for early-stage B2B SaaS / services — not audited market data.
Agents should label these as coaching baselines, not financial advice.
"""

from __future__ import annotations

from typing import Any


TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "industry_kpi_benchmark",
        "description": (
            "Given an industry vertical and company stage, returns typical KPI ranges "
            "(CAC, LTV, churn, gross margin, burn multiple) for coaching comparisons."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "industry": {
                    "type": "string",
                    "description": "Industry vertical, e.g. 'B2B SaaS', 'professional services'.",
                },
                "stage": {
                    "type": "string",
                    "description": "Company stage, e.g. 'seed', 'Series A', 'Series B'.",
                },
            },
            "required": ["industry", "stage"],
        },
    },
}


_STAGE_KEYS = ("seed", "series a", "series_a", "a", "series b", "series_b", "b", "growth")


def _normalize_stage(stage: str) -> str:
    s = (stage or "").strip().lower().replace("-", " ")
    if "seed" in s or "pre" in s:
        return "seed"
    if "b" in s.split() or "series b" in s:
        return "series_b"
    if "a" in s.split() or "series a" in s:
        return "series_a"
    return "seed"


def _normalize_industry(industry: str) -> str:
    i = (industry or "").strip().lower()
    if "saas" in i or "software" in i:
        return "b2b_saas"
    if "service" in i or "agency" in i or "consult" in i:
        return "services"
    return "b2b_saas"


# Approximate coaching baselines (monthly churn where noted).
_TABLE: dict[str, dict[str, dict[str, Any]]] = {
    "b2b_saas": {
        "seed": {
            "cac_usd": {"p25": 800, "p50": 1500, "p75": 3500},
            "ltv_usd": {"p25": 4000, "p50": 9000, "p75": 18000},
            "ltv_cac": {"p25": 2.0, "p50": 3.0, "p75": 5.0},
            "monthly_logo_churn_pct": {"p25": 1.5, "p50": 3.0, "p75": 5.0},
            "gross_margin_pct": {"p25": 65, "p50": 75, "p75": 85},
            "burn_multiple": {"p25": 1.5, "p50": 2.5, "p75": 4.0},
            "net_revenue_retention_pct": {"p25": 90, "p50": 100, "p75": 115},
        },
        "series_a": {
            "cac_usd": {"p25": 1200, "p50": 2500, "p75": 5000},
            "ltv_usd": {"p25": 12000, "p50": 25000, "p75": 45000},
            "ltv_cac": {"p25": 3.0, "p50": 4.0, "p75": 6.0},
            "monthly_logo_churn_pct": {"p25": 1.0, "p50": 2.0, "p75": 3.5},
            "gross_margin_pct": {"p25": 70, "p50": 78, "p75": 88},
            "burn_multiple": {"p25": 1.2, "p50": 2.0, "p75": 3.0},
            "net_revenue_retention_pct": {"p25": 100, "p50": 110, "p75": 125},
        },
        "series_b": {
            "cac_usd": {"p25": 2000, "p50": 4000, "p75": 8000},
            "ltv_usd": {"p25": 25000, "p50": 50000, "p75": 90000},
            "ltv_cac": {"p25": 3.5, "p50": 5.0, "p75": 7.0},
            "monthly_logo_churn_pct": {"p25": 0.7, "p50": 1.5, "p75": 2.5},
            "gross_margin_pct": {"p25": 72, "p50": 80, "p75": 90},
            "burn_multiple": {"p25": 1.0, "p50": 1.5, "p75": 2.5},
            "net_revenue_retention_pct": {"p25": 105, "p50": 120, "p75": 140},
        },
    },
    "services": {
        "seed": {
            "cac_usd": {"p25": 500, "p50": 1200, "p75": 2500},
            "ltv_usd": {"p25": 8000, "p50": 20000, "p75": 40000},
            "ltv_cac": {"p25": 3.0, "p50": 5.0, "p75": 8.0},
            "monthly_logo_churn_pct": {"p25": 2.0, "p50": 4.0, "p75": 8.0},
            "gross_margin_pct": {"p25": 40, "p50": 55, "p75": 70},
            "burn_multiple": {"p25": 1.0, "p50": 2.0, "p75": 3.5},
            "utilization_pct": {"p25": 55, "p50": 70, "p75": 80},
        },
        "series_a": {
            "cac_usd": {"p25": 800, "p50": 1800, "p75": 3500},
            "ltv_usd": {"p25": 15000, "p50": 35000, "p75": 70000},
            "ltv_cac": {"p25": 4.0, "p50": 6.0, "p75": 10.0},
            "monthly_logo_churn_pct": {"p25": 1.5, "p50": 3.0, "p75": 6.0},
            "gross_margin_pct": {"p25": 45, "p50": 60, "p75": 72},
            "burn_multiple": {"p25": 0.8, "p50": 1.5, "p75": 2.5},
            "utilization_pct": {"p25": 60, "p50": 72, "p75": 82},
        },
        "series_b": {
            "cac_usd": {"p25": 1000, "p50": 2200, "p75": 4500},
            "ltv_usd": {"p25": 25000, "p50": 50000, "p75": 100000},
            "ltv_cac": {"p25": 4.5, "p50": 7.0, "p75": 12.0},
            "monthly_logo_churn_pct": {"p25": 1.0, "p50": 2.5, "p75": 5.0},
            "gross_margin_pct": {"p25": 50, "p50": 62, "p75": 75},
            "burn_multiple": {"p25": 0.6, "p50": 1.2, "p75": 2.0},
            "utilization_pct": {"p25": 65, "p50": 75, "p75": 85},
        },
    },
}


def industry_kpi_benchmark(industry: str, stage: str) -> dict[str, Any]:
    ind = _normalize_industry(industry)
    st = _normalize_stage(stage)
    kpis = _TABLE[ind][st]
    return {
        "industry_input": industry,
        "stage_input": stage,
        "normalized_industry": ind,
        "normalized_stage": st,
        "kpis": kpis,
        "disclaimer": (
            "Heuristic coaching baselines for discussion only — not audited benchmarks "
            "or investment advice. Compare founder-reported metrics against ranges, then cite."
        ),
        "how_to_use": (
            "Flag metrics worse than p75 (for churn/burn) or below p25 (for LTV/CAC, margin, NRR) "
            "as coaching priorities."
        ),
    }
