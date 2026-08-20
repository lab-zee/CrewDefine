"""Runway and scenario tables for founder coaching."""

from __future__ import annotations

from typing import Any


TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "runway_and_scenarios",
        "description": (
            "Given current cash, monthly burn, and optional growth and churn rates, "
            "returns projected runway in months plus base / optimistic / pessimistic scenarios."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "cash": {"type": "number", "description": "Current cash on hand in USD."},
                "monthly_burn": {
                    "type": "number",
                    "description": "Monthly net burn rate in USD.",
                },
                "growth_rate": {
                    "type": "number",
                    "description": "Expected monthly revenue growth as a decimal (e.g. 0.05).",
                },
                "churn_rate": {
                    "type": "number",
                    "description": "Monthly logo churn as a decimal (e.g. 0.02).",
                },
                "monthly_revenue": {
                    "type": "number",
                    "description": "Optional starting monthly revenue (USD) for scenario burn netting.",
                },
            },
            "required": ["cash", "monthly_burn"],
        },
    },
}


def _months_runway(cash: float, burn: float) -> float | None:
    if burn <= 0:
        return None
    return round(cash / burn, 1)


def _scenario(
    name: str,
    cash: float,
    monthly_burn: float,
    burn_factor: float,
    growth_rate: float,
    churn_rate: float,
    monthly_revenue: float,
) -> dict[str, Any]:
    # Simple coaching model: net burn adjusts with growth vs churn pressure on revenue.
    revenue = max(0.0, monthly_revenue)
    # Net effect on burn: growth reduces effective burn; churn increases it.
    adjusted_burn = monthly_burn * burn_factor
    if revenue > 0:
        net_growth = growth_rate - churn_rate
        # Each +1% net growth reduces effective burn by ~0.5% of revenue (heuristic).
        adjusted_burn = max(0.0, adjusted_burn - revenue * net_growth * 0.5)
    runway = _months_runway(cash, adjusted_burn)
    return {
        "name": name,
        "monthly_burn_usd": round(adjusted_burn, 0),
        "runway_months": runway if runway is not None else "infinite_or_cash_flow_positive",
        "assumptions": {
            "burn_factor": burn_factor,
            "growth_rate": growth_rate,
            "churn_rate": churn_rate,
            "monthly_revenue_usd": monthly_revenue,
        },
    }


def runway_and_scenarios(
    cash: float,
    monthly_burn: float,
    growth_rate: float | None = None,
    churn_rate: float | None = None,
    monthly_revenue: float | None = None,
) -> dict[str, Any]:
    cash_f = float(cash)
    burn_f = float(monthly_burn)
    g = float(growth_rate) if growth_rate is not None else 0.03
    c = float(churn_rate) if churn_rate is not None else 0.02
    rev = float(monthly_revenue) if monthly_revenue is not None else 0.0

    base = _scenario("base", cash_f, burn_f, 1.0, g, c, rev)
    optimistic = _scenario("optimistic", cash_f, burn_f, 0.85, g + 0.02, max(0.0, c - 0.01), rev)
    pessimistic = _scenario("pessimistic", cash_f, burn_f, 1.2, max(0.0, g - 0.02), c + 0.015, rev)

    return {
        "cash_usd": cash_f,
        "stated_monthly_burn_usd": burn_f,
        "base_runway_months": base["runway_months"],
        "scenarios": [base, optimistic, pessimistic],
        "coaching_flags": {
            "under_6_months": isinstance(base["runway_months"], (int, float))
            and base["runway_months"] < 6,
            "under_12_months": isinstance(base["runway_months"], (int, float))
            and base["runway_months"] < 12,
        },
        "disclaimer": "Simplified scenario math for coaching conversations — not a forecast model.",
    }
