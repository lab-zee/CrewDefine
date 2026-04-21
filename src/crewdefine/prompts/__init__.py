"""Prompt library for the interviewer and persona drafter.

These are the most load-bearing strings in the project — edit with care.
Any change here deserves a note in the PR describing the user-facing impact,
ideally with a before/after example of generated output.
"""

from crewdefine.prompts.interviewer import INTERVIEWER_SYSTEM_PROMPT, INTERVIEWER_TOOLS
from crewdefine.prompts.persona import PERSONA_SYSTEM_PROMPT, persona_user_prompt

__all__ = [
    "INTERVIEWER_SYSTEM_PROMPT",
    "INTERVIEWER_TOOLS",
    "PERSONA_SYSTEM_PROMPT",
    "persona_user_prompt",
]
