"""Catalog of built-in LabZ tool IDs.

Kept deliberately read-only and data-only so it's easy to diff against
LabZ's `backend/src/agents/tools/__init__.py` when LabZ adds new tools.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BuiltinTool:
    id: str
    summary: str


BUILTIN_TOOLS: tuple[BuiltinTool, ...] = (
    BuiltinTool("web_search", "General web search via DuckDuckGo."),
    BuiltinTool("news_search", "Current-events / news search."),
    BuiltinTool("calculator", "Financial and mathematical calculations."),
    BuiltinTool("document", "Search within uploaded organizational documents."),
    BuiltinTool("knowledge_base", "Semantic search across the org knowledge base."),
    BuiltinTool("swot", "Generate a SWOT analysis from supplied context."),
    BuiltinTool("visualizer", "Create ECharts data visualizations."),
    BuiltinTool("extract_citations", "Format source citations as text."),
    BuiltinTool("extract_citations_structured", "Extract citations as structured JSON."),
    BuiltinTool("generate_recommendations", "Suggest readings/resources based on a topic."),
    BuiltinTool("image_generator", "Generate images via Gemini."),
    BuiltinTool(
        "validate_information_sufficiency", "Check whether enough context exists to answer."
    ),
    BuiltinTool(
        "generate_followup_questions", "Generate follow-up prompts for deeper exploration."
    ),
    BuiltinTool("scrape_website", "Extract company/page info from a URL."),
)

BUILTIN_TOOL_IDS: frozenset[str] = frozenset(t.id for t in BUILTIN_TOOLS)


def describe_tool(tool_id: str) -> str | None:
    """Return the one-line summary for a known tool id, or None."""
    for t in BUILTIN_TOOLS:
        if t.id == tool_id:
            return t.summary
    return None
