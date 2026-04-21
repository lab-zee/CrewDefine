"""Shared fixtures."""

from __future__ import annotations

import pytest

from crewdefine.schema import AgentConfig, CrewConfig, ToolParameter, ToolSpec


@pytest.fixture
def minimal_agent() -> AgentConfig:
    return AgentConfig(
        id="director",
        name="Strategic Director",
        role="C-level orchestrator who coordinates specialists",
        tools=[],
        can_delegate_to=[],
        system_prompt=(
            "You are the Strategic Director. You coordinate a team of specialists "
            "to produce comprehensive strategic analysis. Delegate to the right "
            "specialist, then synthesize their findings into a coherent recommendation."
        ),
    )


@pytest.fixture
def basic_crew(minimal_agent: AgentConfig) -> CrewConfig:
    researcher = AgentConfig(
        id="researcher",
        name="Market Researcher",
        role="Analyst who gathers market intelligence from public sources",
        tools=["web_search", "news_search"],
        can_delegate_to=[],
        system_prompt=(
            "You are the Market Researcher. Gather market intelligence using your search "
            "tools, extract quantitative data where possible, and cite sources. Return "
            "your findings to the director."
        ),
    )
    director = AgentConfig(
        id=minimal_agent.id,
        name=minimal_agent.name,
        role=minimal_agent.role,
        tools=[],
        can_delegate_to=["researcher"],
        system_prompt=minimal_agent.system_prompt,
    )
    return CrewConfig(
        name="market-intel",
        description="A small crew for market-intelligence tasks.",
        agents=[director, researcher],
    )


@pytest.fixture
def crew_with_custom_tool(basic_crew: CrewConfig) -> CrewConfig:
    crm_tool = ToolSpec(
        id="crm_lookup",
        description="Look up a customer record in the internal CRM by email.",
        parameters=[
            ToolParameter(name="email", type="string", description="The customer's email address.")
        ],
    )
    agents = list(basic_crew.agents)
    # Swap researcher to use the custom tool.
    for i, a in enumerate(agents):
        if a.id == "researcher":
            agents[i] = AgentConfig(
                id=a.id,
                name=a.name,
                role=a.role,
                tools=["web_search", "crm_lookup"],
                can_delegate_to=a.can_delegate_to,
                system_prompt=a.system_prompt,
            )
    return CrewConfig(
        name=basic_crew.name,
        description=basic_crew.description,
        agents=agents,
        custom_tools=[crm_tool],
    )
