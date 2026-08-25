#!/usr/bin/env python3
"""Time Zero chat turns against the loaded business-coaching crew."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

API = "http://localhost:3001"
CREDS = Path("/tmp/case_study_creds.txt").read_text().strip().splitlines()
API_KEY = CREDS[2]
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
OUT = Path(
    "/Users/davidinwald/Documents/GitHub/CrewDefine/crews/_case_study_business_coaching_queries.json"
)


def current_user_id() -> int:
    r = requests.get(f"{API}/api/auth/verify", headers=HEADERS, timeout=30)
    r.raise_for_status()
    return int(r.json()["id"])


QUESTIONS = [
    {
        "id": "q1_summary",
        "answer_mode": "summary",
        "message": (
            "I'm the founder of a B2B SaaS analytics product at seed stage. "
            "ARR is about $180k, monthly burn $45k, cash $320k, 4-person team. "
            "ICP is mid-market ops leaders but our win rate is weak vs Tableau-ish tools. "
            "Monthly logo churn ~3.5%, CAC ~$2,800. Give a concise coaching snapshot: "
            "top risks, KPI vs typical seed SaaS, and the single highest-leverage 90-day move."
        ),
    },
    {
        "id": "q2_light_followup",
        "answer_mode": "light",
        "message": (
            "Follow-up: stress-test runway with base/optimistic/pessimistic scenarios "
            "and compare our CAC/churn to industry KPI benchmarks for seed B2B SaaS. "
            "Use the financial tools if available."
        ),
    },
    {
        "id": "q3_competitor",
        "answer_mode": "light",
        "message": (
            "One more: sketch a competitor snapshot for 'Looker' as a positioning foil "
            "for an analytics SaaS selling to ops leaders, and what we should not try to beat them on."
        ),
    },
]


def create_org(user_id: int) -> int:
    r = requests.post(
        f"{API}/api/organizations",
        headers=HEADERS,
        params={"user_id": user_id},
        json={"name": "Case Study Coaching Co", "description": "Business coaching case study"},
        timeout=30,
    )
    if not r.ok:
        raise RuntimeError(f"create org failed: {r.status_code} {r.text}")
    return r.json()["id"]


def create_thread(org_id: int) -> int:
    r = requests.post(
        f"{API}/api/threads",
        headers=HEADERS,
        json={"title": "Business coaching case study", "organization_id": org_id},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["id"]


def stream_chat(org_id: int, thread_id: int, message: str, answer_mode: str) -> dict:
    t0 = time.perf_counter()
    first_event_s = None
    first_tokenish_s = None
    done_s = None
    events: list[dict] = []
    response_text = ""
    error = None

    with requests.post(
        f"{API}/api/llm/chat/stream",
        headers=HEADERS,
        json={
            "message": message,
            "organization_id": org_id,
            "thread_id": thread_id,
            "chat_mode": "agentic",
            "answer_mode": answer_mode,
        },
        stream=True,
        timeout=900,
    ) as resp:
        resp.raise_for_status()
        buffer = ""
        for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
            if not chunk:
                continue
            if first_event_s is None:
                first_event_s = time.perf_counter() - t0
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw:
                    continue
                try:
                    ev = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                events.append({"type": ev.get("type"), "t": round(time.perf_counter() - t0, 2)})
                et = ev.get("type")
                if (
                    et in {"response", "token", "content", "progress_update", "trace_update"}
                    and first_tokenish_s is None
                ):
                    first_tokenish_s = time.perf_counter() - t0
                if et == "response":
                    data = ev.get("data") or {}
                    response_text = data.get("response") or data.get("message") or str(data)[:2000]
                if et == "error":
                    error = ev.get("data")
                if et == "done":
                    done_s = time.perf_counter() - t0

    total = time.perf_counter() - t0
    return {
        "total_seconds": round(done_s or total, 1),
        "ttfb_seconds": round(first_event_s, 1) if first_event_s is not None else None,
        "first_progress_seconds": round(first_tokenish_s, 1)
        if first_tokenish_s is not None
        else None,
        "event_types": [e["type"] for e in events],
        "event_count": len(events),
        "response_preview": (response_text or "")[:1200],
        "error": error,
    }


def main() -> None:
    user_id = current_user_id()
    org_id = create_org(user_id)
    thread_id = create_thread(org_id)
    results = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "org_id": org_id,
        "thread_id": thread_id,
        "queries": [],
    }
    print(f"org={org_id} thread={thread_id}")
    for q in QUESTIONS:
        print(f"\n=== {q['id']} ({q['answer_mode']}) ===")
        row = {"id": q["id"], "answer_mode": q["answer_mode"], "message": q["message"]}
        try:
            timing = stream_chat(org_id, thread_id, q["message"], q["answer_mode"])
            row.update(timing)
            print(
                f"done in {timing['total_seconds']}s "
                f"(ttfb={timing['ttfb_seconds']}, progress={timing['first_progress_seconds']})"
            )
            if timing.get("error"):
                print("ERROR", timing["error"])
            else:
                print((timing.get("response_preview") or "")[:400])
        except Exception as exc:
            row["error"] = str(exc)
            print("FAILED", exc)
        results["queries"].append(row)
        OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

    results["ended_at"] = datetime.now(timezone.utc).isoformat()
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
