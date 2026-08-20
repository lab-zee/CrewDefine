"""Structured competitor snapshot for coaching context.

Uses optional web scrape when a URL is provided; otherwise returns a structured
skeleton the market researcher can fill from search tools.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse


TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "competitor_snapshot",
        "description": (
            "Given a competitor name or URL, returns a structured competitive profile "
            "including positioning, pricing signals, ICP cues, and recent news hooks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "competitor": {
                    "type": "string",
                    "description": "Competitor name or URL to profile.",
                },
            },
            "required": ["competitor"],
        },
    },
}


def _looks_like_url(value: str) -> bool:
    return bool(re.match(r"^https?://", value.strip(), re.I))


def _try_fetch_text(url: str, limit: int = 4000) -> str | None:
    try:
        import urllib.request

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "LabZ-competitor-snapshot/1.0"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:  # noqa: S310
            raw = resp.read(limit * 2)
        text = raw.decode("utf-8", errors="ignore")
        # Crude tag strip
        text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.I)
        text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:limit] if text else None
    except Exception as exc:  # noqa: BLE001 — tool should never crash the agent
        return f"[fetch_failed: {type(exc).__name__}: {exc}]"


def competitor_snapshot(competitor: str) -> dict[str, Any]:
    raw = (competitor or "").strip()
    is_url = _looks_like_url(raw)
    name = raw
    domain = None
    page_excerpt = None

    if is_url:
        parsed = urlparse(raw)
        domain = parsed.netloc
        name = domain.replace("www.", "").split(".")[0].title() if domain else raw
        page_excerpt = _try_fetch_text(raw)

    return {
        "competitor_input": raw,
        "name": name,
        "domain": domain,
        "positioning": {
            "one_liner": None,
            "category": None,
            "notes_from_page": page_excerpt[:500] if isinstance(page_excerpt, str) and not page_excerpt.startswith("[fetch_failed") else None,
        },
        "pricing_signals": {
            "public_pricing": None,
            "motion_guess": None,  # PLG / sales-led / hybrid
            "evidence": [],
        },
        "icp_cues": {
            "company_size": None,
            "buyer_titles": [],
            "verticals": [],
        },
        "recent_news_hooks": [],
        "page_excerpt": page_excerpt,
        "coach_next_steps": [
            "Fill null fields via web_search / news_search / scrape_website.",
            "Compare ICP and pricing motion to the founder's stated positioning.",
            "Extract citations with extract_citations_structured before synthesis.",
        ],
        "disclaimer": (
            "Skeleton profile for coaching structure. Page text (if any) is raw and may be incomplete."
        ),
    }
