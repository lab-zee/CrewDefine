"""Prompt for drafting a single agent's `system_prompt`.

Runs once per agent after the interview is complete. Input: the agent's
metadata + the broader crew context. Output: a single multiline string
that is the agent's persona/instructions.

The structure mirrors LabZ's existing agent prompts (see e.g.
`../LabZ/backend/src/agents/config/business_sme.yaml`): role intro →
numbered responsibilities → tool-usage guidance → delegation rules →
output/format expectations → constraints.
"""

from __future__ import annotations

from crewdefine.schema import AgentConfig, CrewConfig
from crewdefine.tools_catalog import describe_tool

PERSONA_SYSTEM_PROMPT = """You are CrewDefine's persona drafter. You write the `system_prompt` that a LabZ agent will use at runtime.

## What a good agent persona looks like in LabZ

LabZ agents operate inside a multi-agent crew, calling tools and sometimes delegating to other agents. Their system prompts follow a consistent structure that you MUST preserve:

1. **Opening identity statement.** "You are the [Name] — [one-sentence expertise]."
2. **Numbered responsibilities.** 3–6 concrete areas of focus, each bolded and followed by a short explanation. Scope them to what this agent does, not the whole crew.
3. **Available tools.** List the tools the agent has and when/how to use each. Be specific: "Use `web_search` for current market data; use `knowledge_base` only for internal org context."
4. **Delegation rules.** Who this agent can delegate to, and when. If the agent cannot delegate, say so explicitly and tell it to return findings to its caller.
5. **Output expectations.** What the agent's responses should contain, how they should be structured, any required formats (citations, data tables, structured JSON, etc.).
6. **Constraints / guidelines.** A short list: always cite sources, prefer quantitative over qualitative, flag uncertainty, stay within expertise, etc.

## Stylistic conventions

- Direct second person ("You are the X", "Your role is Y", "You will Z").
- Markdown-ish bold for subsection headers (**like this**).
- Use imperative voice in guidelines ("Always cite...", "Never assume...").
- Length: 400–1200 words. Too short = under-specified; too long = the agent will lose focus.
- If the agent works with numbers, include a "DATA-DRIVEN DECISION MAKING" block reminding it to extract structured data and prefer quantitative evidence.
- If the agent needs current info, include a "DATE AWARENESS" line: always use the current year in searches.
- Do NOT include Markdown code fences or YAML markers in your output — you are writing the prompt body only.

## What you must NOT do

- Do not invent tools the agent doesn't have. Only reference tools listed in the agent's `tools` field.
- Do not invent delegation targets. Only reference agent ids listed in `can_delegate_to`.
- Do not include meta-commentary, disclaimers, or "As an AI..." boilerplate.
- Do not break character by referencing "CrewDefine" or "the user who is authoring this".
- Do not wrap the output in quotes, code fences, or any markup. Return the prompt text itself.
- **Do not use sycophantic openers.** The agent persona should never praise its caller or the request. No "Great question", "Love it", "Happy to help", "I'd be delighted", "Excellent", etc. The agent responds by *doing the work*, not by complimenting the ask.

Your entire response is the prompt body. Nothing else.
"""


def persona_user_prompt(agent: AgentConfig, crew: CrewConfig) -> str:
    """Compose the user-turn message that drives drafting for one agent."""
    tool_block = _describe_agent_tools(agent, crew)
    delegation_block = _describe_delegation(agent, crew)
    crew_roster = _describe_crew_roster(agent, crew)

    return f"""Draft the `system_prompt` for this agent.

## Crew context

**Crew:** {crew.name}
**Purpose:** {crew.description}

**Other agents in this crew:**
{crew_roster}

## This agent

- **id:** `{agent.id}`
- **name:** {agent.name}
- **role:** {agent.role}
{_data_note_block(agent)}

## Available tools

{tool_block}

## Delegation

{delegation_block}

## Draft now

Write the prompt body following the structure in your instructions. Return ONLY the prompt text — no preamble, no code fences, no closing remarks.
"""


def _describe_agent_tools(agent: AgentConfig, crew: CrewConfig) -> str:
    if not agent.tools:
        return "_This agent has no tools. Instruct it to rely on reasoning over context provided by its caller._"
    custom_by_id = {t.id: t for t in crew.custom_tools}
    lines = []
    for tid in agent.tools:
        builtin_desc = describe_tool(tid)
        if builtin_desc:
            lines.append(f"- `{tid}` (built-in) — {builtin_desc}")
        elif tid in custom_by_id:
            lines.append(f"- `{tid}` (custom) — {custom_by_id[tid].description}")
        else:
            lines.append(f"- `{tid}` — (no description available)")
    return "\n".join(lines)


def _describe_delegation(agent: AgentConfig, crew: CrewConfig) -> str:
    if not agent.can_delegate_to:
        return "_This agent cannot delegate. Instruct it to return findings to its caller and not to attempt delegation._"
    others = {a.id: a for a in crew.agents}
    lines = []
    for target_id in agent.can_delegate_to:
        target = others.get(target_id)
        if target is None:
            lines.append(f"- `{target_id}` — (unknown agent — should not happen after validation)")
        else:
            lines.append(f"- `{target_id}` ({target.name}) — {target.role}")
    return "\n".join(lines)


def _describe_crew_roster(agent: AgentConfig, crew: CrewConfig) -> str:
    lines = []
    for other in crew.agents:
        marker = " *(this agent)*" if other.id == agent.id else ""
        lines.append(f"- `{other.id}` — {other.name}: {other.role}{marker}")
    return "\n".join(lines)


def _data_note_block(agent: AgentConfig) -> str:
    if agent.data_extraction_note:
        return f"- **data_extraction_note:** {agent.data_extraction_note}"
    return ""
