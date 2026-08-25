"""Deterministic evidence-coverage audit for technical findings."""

from __future__ import annotations

from typing import Any

TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "evidence_coverage_score",
        "description": (
            "Audit finding objects for evidence coverage, unsupported high-severity claims, "
            "confidence distribution, and inference labeling."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "description": (
                        "Finding objects with id, claim, evidence_refs, severity, confidence, "
                        "and inference."
                    ),
                },
            },
            "required": ["findings"],
        },
    },
}


def _has_evidence(value: Any) -> bool:
    if isinstance(value, list):
        return any(str(item).strip() for item in value)
    return bool(str(value or "").strip())


def evidence_coverage_score(findings: list[dict[str, Any]]) -> dict[str, Any]:
    """Return coverage statistics without treating quantity of citations as truth."""
    rows = [item if isinstance(item, dict) else {} for item in (findings or [])]
    covered = [row for row in rows if _has_evidence(row.get("evidence_refs"))]
    unsupported_high: list[str] = []
    audit_flags: list[dict[str, str]] = []
    confidence = {"high": 0, "medium": 0, "low": 0, "unspecified": 0}

    for index, row in enumerate(rows):
        finding_id = str(row.get("id") or f"finding-{index + 1}")
        severity = str(row.get("severity") or "unspecified").lower()
        confidence_key = str(row.get("confidence") or "unspecified").lower()
        if confidence_key not in confidence:
            confidence_key = "unspecified"
            audit_flags.append({"id": finding_id, "flag": "unrecognized confidence value"})
        confidence[confidence_key] += 1

        has_evidence = _has_evidence(row.get("evidence_refs"))
        if severity in {"high", "critical"} and not has_evidence:
            unsupported_high.append(finding_id)
        if not str(row.get("claim") or "").strip():
            audit_flags.append({"id": finding_id, "flag": "missing claim"})
        if bool(row.get("inference")) and not has_evidence:
            audit_flags.append({"id": finding_id, "flag": "unsupported inference"})
        if not has_evidence:
            audit_flags.append({"id": finding_id, "flag": "no evidence reference"})

    total = len(rows)
    return {
        "finding_count": total,
        "findings_with_evidence": len(covered),
        "evidence_coverage": round(len(covered) / total, 3) if total else None,
        "unsupported_high_severity": unsupported_high,
        "confidence_distribution": confidence,
        "audit_flags": audit_flags,
        "caveat": "Coverage measures attached references, not evidence quality or claim validity.",
    }
