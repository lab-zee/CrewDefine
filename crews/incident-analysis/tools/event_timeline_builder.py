"""Normalize, order, and audit incident events."""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import pairwise
from typing import Any

TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "event_timeline_builder",
        "description": (
            "Normalize and order incident events. Returns sorted events, timestamp conflicts, "
            "duplicate candidates, gap intervals, and source coverage."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "events": {
                    "type": "array",
                    "description": (
                        "Events with timestamp, source, actor_or_system, event_type, description, "
                        "and confidence."
                    ),
                },
            },
            "required": ["events"],
        },
    },
}


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def event_timeline_builder(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Sort parseable ISO-8601 events and surface ambiguity rather than guessing."""
    normalized: list[dict[str, Any]] = []
    invalid_timestamps: list[int] = []
    source_counts: dict[str, int] = {}

    for index, raw in enumerate(events or []):
        event = raw if isinstance(raw, dict) else {}
        parsed = _parse_timestamp(event.get("timestamp"))
        if parsed is None:
            invalid_timestamps.append(index + 1)
        source = str(event.get("source") or "unspecified")
        source_counts[source] = source_counts.get(source, 0) + 1
        normalized.append(
            {
                "event_id": index + 1,
                "timestamp": event.get("timestamp"),
                "source": source,
                "actor_or_system": event.get("actor_or_system"),
                "event_type": event.get("event_type"),
                "description": event.get("description"),
                "confidence": event.get("confidence"),
                "_parsed": parsed,
            }
        )

    normalized.sort(
        key=lambda row: (
            row["_parsed"] is None,
            row["_parsed"] or datetime.max,
            row["event_id"],
        )
    )
    duplicate_candidates: list[dict[str, Any]] = []
    timestamp_conflicts: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []

    for left, right in pairwise(normalized):
        if (
            left["timestamp"] == right["timestamp"]
            and left["description"]
            and left["description"] == right["description"]
        ):
            duplicate_candidates.append(
                {"event_ids": [left["event_id"], right["event_id"]], "timestamp": left["timestamp"]}
            )
        if (
            left["timestamp"] == right["timestamp"]
            and left["description"] != right["description"]
            and left["source"] != right["source"]
        ):
            timestamp_conflicts.append(
                {"event_ids": [left["event_id"], right["event_id"]], "timestamp": left["timestamp"]}
            )
        if left["_parsed"] and right["_parsed"]:
            seconds = (right["_parsed"] - left["_parsed"]).total_seconds()
            if seconds > 1800:
                gaps.append(
                    {
                        "after_event_id": left["event_id"],
                        "before_event_id": right["event_id"],
                        "gap_seconds": int(seconds),
                    }
                )

    for row in normalized:
        row.pop("_parsed", None)
    return {
        "events": normalized,
        "event_count": len(normalized),
        "invalid_timestamp_event_ids": invalid_timestamps,
        "timestamp_conflicts": timestamp_conflicts,
        "duplicate_candidates": duplicate_candidates,
        "gaps_over_30_minutes": gaps,
        "source_coverage": source_counts,
    }
