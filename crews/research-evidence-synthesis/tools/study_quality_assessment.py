"""Criterion-level study appraisal without a synthetic quality score."""

from __future__ import annotations

from typing import Any

TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "study_quality_assessment",
        "description": (
            "Perform a structured methodological appraisal. Returns criterion-level flags and "
            "missing fields without collapsing quality to one authoritative grade."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "studies": {
                    "type": "array",
                    "description": (
                        "Studies with id, study_design, sample_size, controls, preregistered, "
                        "attrition_rate, missing_data, conflicts, and reported_limitations."
                    ),
                },
            },
            "required": ["studies"],
        },
    },
}


_EXPECTED_FIELDS = (
    "study_design",
    "sample_size",
    "controls",
    "preregistered",
    "attrition_rate",
    "missing_data",
    "conflicts",
    "reported_limitations",
)


def study_quality_assessment(studies: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply transparent appraisal checks while preserving methodological nuance."""
    appraisals: list[dict[str, Any]] = []
    aggregate_flags: dict[str, int] = {}

    for index, raw in enumerate(studies or []):
        study = raw if isinstance(raw, dict) else {}
        study_id = str(study.get("id") or f"study-{index + 1}")
        missing = [field for field in _EXPECTED_FIELDS if study.get(field) in (None, "", [])]
        flags: list[str] = []

        if study.get("preregistered") is False:
            flags.append("not preregistered")
        try:
            sample_size = (
                int(study["sample_size"]) if study.get("sample_size") is not None else None
            )
        except (TypeError, ValueError):
            sample_size = None
            flags.append("invalid sample_size")
        if sample_size is not None and sample_size < 30:
            flags.append("small sample; assess domain-specific power")

        try:
            attrition = (
                float(study["attrition_rate"]) if study.get("attrition_rate") is not None else None
            )
        except (TypeError, ValueError):
            attrition = None
            flags.append("invalid attrition_rate")
        if attrition is not None and attrition > 0.2:
            flags.append("attrition exceeds 20%")

        controls = str(study.get("controls") or "").lower()
        if controls in {"none", "no", "uncontrolled"}:
            flags.append("no reported control condition")
        if not str(study.get("reported_limitations") or "").strip():
            flags.append("no reported limitations")
        if str(study.get("conflicts") or "").lower() not in {"", "none", "no", "not reported"}:
            flags.append("reported conflict of interest")

        for flag in flags:
            aggregate_flags[flag] = aggregate_flags.get(flag, 0) + 1
        appraisals.append(
            {
                "id": study_id,
                "study_design": study.get("study_design"),
                "missing_fields": missing,
                "criterion_flags": flags,
                "appraisal_status": "incomplete" if missing else "reviewed",
            }
        )

    return {
        "studies": appraisals,
        "study_count": len(appraisals),
        "aggregate_flag_counts": aggregate_flags,
        "method": "criterion-level completeness and risk flags; no composite quality score",
    }
