"""Normalize extracted research findings into comparable evidence rows."""

from __future__ import annotations

from typing import Any

TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "evidence_table_builder",
        "description": (
            "Normalize extracted findings for cross-study comparison. Returns rows, "
            "incompatible-field warnings, duplicate candidates, and coverage counts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "description": (
                        "Findings with source_id, population, intervention_or_exposure, comparator, "
                        "outcome, estimate, uncertainty, study_design, and notes."
                    ),
                },
            },
            "required": ["findings"],
        },
    },
}


_FIELDS = (
    "source_id",
    "population",
    "intervention_or_exposure",
    "comparator",
    "outcome",
    "estimate",
    "uncertainty",
    "study_design",
    "notes",
)


def _text(value: Any) -> str | None:
    normalized = str(value).strip() if value is not None else ""
    return normalized or None


def evidence_table_builder(findings: list[dict[str, Any]]) -> dict[str, Any]:
    """Normalize field names and identify comparison limits without interpreting effects."""
    rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    seen: dict[tuple[str | None, str | None, str | None], int] = {}
    duplicates: list[dict[str, Any]] = []
    coverage = {field: 0 for field in _FIELDS}

    for index, raw in enumerate(findings or []):
        finding = raw if isinstance(raw, dict) else {}
        row = {field: _text(finding.get(field)) for field in _FIELDS}
        row["row_id"] = index + 1
        rows.append(row)

        missing = [field for field in _FIELDS[:-1] if row[field] is None]
        if missing:
            warnings.append({"row_id": index + 1, "missing_fields": missing})
        for field in _FIELDS:
            if row[field] is not None:
                coverage[field] += 1

        key = (row["source_id"], row["outcome"], row["estimate"])
        if all(key):
            if key in seen:
                duplicates.append(
                    {"row_id": index + 1, "possible_duplicate_of": seen[key], "key": list(key)}
                )
            else:
                seen[key] = index + 1

    outcome_units: dict[str, set[str]] = {}
    for row in rows:
        if row["outcome"] and row["estimate"]:
            outcome_units.setdefault(row["outcome"].lower(), set()).add(row["estimate"])
    incomparable = [
        {
            "outcome": outcome,
            "reason": "estimates require unit/scale review",
            "estimate_count": len(values),
        }
        for outcome, values in outcome_units.items()
        if len(values) > 1
    ]

    return {
        "rows": rows,
        "row_count": len(rows),
        "field_coverage": coverage,
        "row_warnings": warnings,
        "duplicate_candidates": duplicates,
        "comparison_warnings": incomparable,
    }
