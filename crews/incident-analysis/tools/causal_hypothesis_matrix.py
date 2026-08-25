"""Compare incident hypotheses with explicitly linked evidence."""

from __future__ import annotations

from typing import Any

TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "causal_hypothesis_matrix",
        "description": (
            "Compare causal hypotheses with available evidence without asserting root cause. "
            "Returns support and contradiction links, unsupported hypotheses, and evidence gaps."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "hypotheses": {
                    "type": "array",
                    "description": "Hypotheses with id and statement.",
                },
                "evidence": {
                    "type": "array",
                    "description": (
                        "Evidence with id, supports and/or contradicts hypothesis IDs, and relevance."
                    ),
                },
            },
            "required": ["hypotheses", "evidence"],
        },
    },
}


def _ids(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None or not str(value).strip():
        return []
    return [str(value)]


def causal_hypothesis_matrix(
    hypotheses: list[dict[str, Any]], evidence: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build a relation matrix; confidence remains a human/model judgment."""
    hypothesis_rows: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(hypotheses or []):
        hypothesis = raw if isinstance(raw, dict) else {}
        hypothesis_id = str(hypothesis.get("id") or f"hypothesis-{index + 1}")
        hypothesis_rows[hypothesis_id] = {
            "id": hypothesis_id,
            "statement": hypothesis.get("statement"),
            "supporting_evidence": [],
            "contradicting_evidence": [],
        }

    unlinked: list[str] = []
    unknown_links: list[dict[str, str]] = []
    for index, raw in enumerate(evidence or []):
        item = raw if isinstance(raw, dict) else {}
        evidence_id = str(item.get("id") or f"evidence-{index + 1}")
        links = 0
        for hypothesis_id in _ids(item.get("supports")):
            if hypothesis_id in hypothesis_rows:
                hypothesis_rows[hypothesis_id]["supporting_evidence"].append(evidence_id)
                links += 1
            else:
                unknown_links.append({"evidence_id": evidence_id, "hypothesis_id": hypothesis_id})
        for hypothesis_id in _ids(item.get("contradicts")):
            if hypothesis_id in hypothesis_rows:
                hypothesis_rows[hypothesis_id]["contradicting_evidence"].append(evidence_id)
                links += 1
            else:
                unknown_links.append({"evidence_id": evidence_id, "hypothesis_id": hypothesis_id})
        if not links:
            unlinked.append(evidence_id)

    matrix = list(hypothesis_rows.values())
    unsupported = [row["id"] for row in matrix if not row["supporting_evidence"]]
    unresolved = [
        row["id"] for row in matrix if row["supporting_evidence"] and row["contradicting_evidence"]
    ]
    return {
        "hypotheses": matrix,
        "unsupported_hypotheses": unsupported,
        "contested_hypotheses": unresolved,
        "unlinked_evidence": unlinked,
        "unknown_hypothesis_links": unknown_links,
        "discriminating_evidence_needed": unresolved + unsupported,
        "caveat": "Relation counts do not establish causality or identify a root cause.",
    }
