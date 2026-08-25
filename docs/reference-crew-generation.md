# Professional reference crews: generation record

Three maintained reference configurations were produced through CrewDefine on 24 August 2026.
They were not assembled by copying YAML from the existing examples.

## Procedure

1. Supply a dense requirements seed to `crewdefine new --seed`.
2. Allow the interview agent to record crew metadata, agents, delegation, custom tool schemas,
   answer modes, and output composition.
3. Review the generated roster at CrewDefine's final confirmation step.
4. Let CrewDefine draft all eight agent personas.
5. Implement the deterministic custom-tool stubs emitted by CrewDefine.
6. Run CrewDefine validation, focused tool tests, Zero validation, and a Zero registry load check.

Interview model: `claude-sonnet-4-6` (the configured CrewDefine default at generation time).

## Configurations

| Configuration | Agents | Custom tools | Primary constraint |
| --- | ---: | ---: | --- |
| Technical Due Diligence | 8 | 2 | Separate observed evidence, inference, and missing evidence |
| Research Evidence Synthesis | 8 | 2 | Expose search limits, methods concerns, contradictions, and uncertainty |
| Incident Analysis | 8 | 3 | Remain blameless and retain competing hypotheses until evidence discriminates |

All three expose five domain-specific answer modes and require citations. Quantitative
visualizations are conditional; structured tables are required; generated images are disabled.

## Verification

```bash
crewdefine validate crews/business-coaching-crew
crewdefine validate crews/dinner-planning-crew
crewdefine validate crews/technical-due-diligence
crewdefine validate crews/research-evidence-synthesis
crewdefine validate crews/incident-analysis

pytest
```

The parametrized reference-crew tests execute these validations in local and CI runs. The full
test command also exercises deterministic crew tools and enforces the repository's coverage floor.

Zero's `backend/scripts/validate_crew.py` was run against each package with its plugin directory.
Each package was then instantiated through Zero's `AgentRegistry`; all loaded eight agents and five
answer modes.

The deterministic tools intentionally report coverage, rule-based flags, normalization warnings,
or evidence relations. They do not claim to establish truth, methodological quality, causality, or
risk reduction.
