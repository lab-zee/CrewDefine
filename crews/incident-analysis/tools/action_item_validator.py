"""Validate corrective actions for ownership, coverage, and verifiability."""

from __future__ import annotations

from typing import Any

TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "action_item_validator",
        "description": (
            "Validate corrective actions for required fields, verification criteria, duplicates, "
            "and coverage by failure mode."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "description": (
                        "Actions with description, owner, due_date, failure_mode, "
                        "verification_method, completion_evidence, and priority."
                    ),
                },
            },
            "required": ["actions"],
        },
    },
}


_REQUIRED = ("description", "owner", "due_date", "failure_mode", "verification_method")


def _fingerprint(description: Any) -> str:
    return " ".join(str(description or "").lower().split())


def action_item_validator(actions: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply mechanical checks; action effectiveness still requires domain review."""
    rows: list[dict[str, Any]] = []
    duplicates: list[dict[str, int]] = []
    seen: dict[str, int] = {}
    coverage: dict[str, int] = {}
    unverifiable: list[int] = []

    for index, raw in enumerate(actions or []):
        action = raw if isinstance(raw, dict) else {}
        action_id = index + 1
        missing = [field for field in _REQUIRED if not str(action.get(field) or "").strip()]
        verification = str(action.get("verification_method") or "").strip().lower()
        if not verification or verification in {"done", "complete", "check", "tbd"}:
            unverifiable.append(action_id)

        failure_mode = str(action.get("failure_mode") or "unspecified").strip()
        coverage[failure_mode] = coverage.get(failure_mode, 0) + 1
        fingerprint = _fingerprint(action.get("description"))
        if fingerprint:
            if fingerprint in seen:
                duplicates.append(
                    {"action_id": action_id, "possible_duplicate_of": seen[fingerprint]}
                )
            else:
                seen[fingerprint] = action_id

        rows.append(
            {
                "action_id": action_id,
                "description": action.get("description"),
                "owner": action.get("owner"),
                "due_date": action.get("due_date"),
                "failure_mode": failure_mode,
                "priority": action.get("priority"),
                "missing_fields": missing,
                "has_completion_evidence": bool(
                    str(action.get("completion_evidence") or "").strip()
                ),
            }
        )

    return {
        "actions": rows,
        "action_count": len(rows),
        "unverifiable_action_ids": unverifiable,
        "duplicate_candidates": duplicates,
        "coverage_by_failure_mode": coverage,
        "complete_action_count": sum(1 for row in rows if not row["missing_fields"]),
        "caveat": "Completeness and verifiability checks do not prove risk reduction.",
    }
