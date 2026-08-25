"""Deterministic dependency-risk triage."""

from __future__ import annotations

from typing import Any

TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "dependency_risk_matrix",
        "description": (
            "Deterministic triage of dependency records. Returns normalized risk rows, "
            "reasons, and aggregate counts by risk tier."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dependencies": {
                    "type": "array",
                    "description": (
                        "Dependency records with name, version, ecosystem, license, "
                        "latest_release_age_days, known_vulnerability_severity, and "
                        "maintenance_status."
                    ),
                },
            },
            "required": ["dependencies"],
        },
    },
}


_SEVERITY_POINTS = {"none": 0, "low": 1, "medium": 2, "high": 4, "critical": 6}
_MAINTENANCE_POINTS = {"active": 0, "maintained": 0, "limited": 2, "stale": 3, "abandoned": 5}
_RESTRICTIVE_LICENSES = {"agpl", "gpl", "sspl", "bsl", "unknown", "unlicensed"}


def _tier(score: int) -> str:
    if score >= 7:
        return "critical"
    if score >= 4:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


def dependency_risk_matrix(dependencies: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply transparent rules to dependency metadata; no external lookup is performed."""
    rows: list[dict[str, Any]] = []
    counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}

    for index, raw in enumerate(dependencies or []):
        record = raw if isinstance(raw, dict) else {}
        name = str(record.get("name") or f"dependency-{index + 1}")
        severity = str(record.get("known_vulnerability_severity") or "none").lower()
        maintenance = str(record.get("maintenance_status") or "unknown").lower()
        license_id = str(record.get("license") or "unknown").lower()
        age = record.get("latest_release_age_days")

        score = _SEVERITY_POINTS.get(severity, 1)
        reasons: list[str] = []
        if severity not in {"", "none"}:
            reasons.append(f"reported vulnerability severity: {severity}")

        maintenance_score = _MAINTENANCE_POINTS.get(maintenance, 1)
        score += maintenance_score
        if maintenance_score:
            reasons.append(f"maintenance status: {maintenance}")

        try:
            age_days = max(0, int(age)) if age is not None else None
        except (TypeError, ValueError):
            age_days = None
            reasons.append("invalid release-age value")
        if age_days is None:
            score += 1
            reasons.append("release age not supplied")
        elif age_days > 730:
            score += 3
            reasons.append(f"latest release is {age_days} days old")
        elif age_days > 365:
            score += 1
            reasons.append(f"latest release is {age_days} days old")

        normalized_license = license_id.split("-")[0]
        if normalized_license in _RESTRICTIVE_LICENSES:
            score += 2
            reasons.append(f"license requires review: {license_id}")

        tier = _tier(score)
        counts[tier] += 1
        rows.append(
            {
                "name": name,
                "version": record.get("version"),
                "ecosystem": record.get("ecosystem"),
                "risk_score": score,
                "risk_tier": tier,
                "reasons": reasons or ["no rule-based risk signal"],
            }
        )

    rows.sort(key=lambda row: (-row["risk_score"], row["name"].lower()))
    return {
        "dependencies": rows,
        "counts_by_tier": counts,
        "total": len(rows),
        "method": "rule-based metadata triage; confirm against authoritative advisories",
    }
