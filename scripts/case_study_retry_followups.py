#!/usr/bin/env python3
"""Retry follow-up queries on OpenAI provider; append timings to case study log."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# reuse helpers
import importlib.util

spec = importlib.util.spec_from_file_location(
    "tq",
    "/Users/davidinwald/Documents/GitHub/CrewDefine/scripts/case_study_time_queries.py",
)
tq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tq)

OUT = Path("/Users/davidinwald/Documents/GitHub/CrewDefine/crews/_case_study_business_coaching_queries.json")
data = json.loads(OUT.read_text())
org_id = data["org_id"]
thread_id = data["thread_id"]

QUESTIONS = [
    {
        "id": "q2_light_followup_openai",
        "answer_mode": "light",
        "message": (
            "Follow-up: stress-test runway with base/optimistic/pessimistic scenarios "
            "and compare our CAC/churn to industry KPI benchmarks for seed B2B SaaS. "
            "Use industry_kpi_benchmark and runway_and_scenarios tools."
        ),
    },
    {
        "id": "q3_competitor_openai",
        "answer_mode": "light",
        "message": (
            "Sketch a competitor snapshot for Looker as a positioning foil for an "
            "analytics SaaS selling to ops leaders — what we should not try to beat them on. "
            "Use competitor_snapshot if helpful."
        ),
    },
]

for q in QUESTIONS:
    print(f"\n=== {q['id']} ===")
    row = {"id": q["id"], "answer_mode": q["answer_mode"], "message": q["message"], "provider": "openai"}
    timing = tq.stream_chat(org_id, thread_id, q["message"], q["answer_mode"])
    row.update(timing)
    print(f"done in {timing['total_seconds']}s error={timing.get('error')}")
    print((timing.get("response_preview") or "")[:500])
    data.setdefault("queries", []).append(row)
    data["ended_at"] = datetime.now(timezone.utc).isoformat()
    data["followup_note"] = "q2/q3 retried with LLM_PROVIDER=openai after Gemini thought_signature tool errors"
    OUT.write_text(json.dumps(data, indent=2), encoding="utf-8")

print(f"\nWrote {OUT}")
