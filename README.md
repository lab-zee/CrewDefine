# CrewDefine

> Interactive CLI that helps you author [LabZ](https://github.com/davidinwald/LabZ) agent crews as YAML.

CrewDefine runs a guided, LLM-driven interview and produces a ready-to-drop-in crew directory for LabZ: one YAML per agent, Python stubs for any new tools, and a crew-level README. You start with nothing, answer a few questions, and walk away with a well-structured, validated crew.

## Status

**Alpha.** v1 targets the core loop: create a crew from scratch, add or update agents on an existing crew, validate a crew. Expect rough edges.

## Installation

```bash
pip install crewdefine
```

Or from source:

```bash
git clone https://github.com/lab-zee/CrewDefine
cd CrewDefine
pip install -e ".[dev]"
```

## Quick start

```bash
export ANTHROPIC_API_KEY=sk-ant-...
crewdefine new
```

You'll be asked about the crew's purpose, the specialists it needs, and how they should collaborate. Output lands in `./crews/<crew-name>/`:

```
crews/strategy-crew/
├── crew.yaml             # Manifest: answer modes + output composition (Zero)
├── README.md
├── agents/
│   ├── director.yaml     # required
│   ├── synthesizer.yaml  # required
│   └── financial.yaml
└── tools/
    └── competitor_lookup.py
```

### Load into Zero (LabZ)

From the [Zero](https://github.com/davidinwald/Zero) repo:

```bash
./scripts/load-crew.sh ../CrewDefine/crews/strategy-crew
docker compose restart backend
```

`crew.yaml` drives Zero's answer-mode selector (`GET /api/crew`). Contract: see Zero's `docs/CREW_HANDOFF.md`.

## Commands

| Command | Purpose |
| --- | --- |
| `crewdefine new` | Create a new crew from scratch via guided interview. |
| `crewdefine add-agent <crew-dir>` | Add a new agent to an existing crew. |
| `crewdefine update-agent <crew-dir> <agent-id>` | Revise an existing agent's prompt, tools, or delegation. |
| `crewdefine validate <crew-dir>` | Schema + round-trip + delegation-graph + crew.yaml validation. |
| `crewdefine list-tools` | Show the built-in LabZ tool IDs CrewDefine knows about. |

## How it works

1. **Interview.** An Anthropic Claude model conducts an adaptive conversation. It can only act through a constrained tool interface (`ask_user`, `record_agent`, `finish`), so it cannot wander off-task or produce unstructured output. There's a hard cap on turns.
2. **Draft.** After the interview, CrewDefine prompts Claude to draft each agent's `system_prompt` — grounded in LabZ's prompt conventions (numbered responsibilities, tool-usage guidance, delegation rules, output format). Synthesizer drafts include `output_composition` guidance for charts, tables, and citations.
3. **Validate.** Every generated YAML is loaded back through Pydantic, cross-checked against LabZ's `AgentConfig` shape, and the delegation graph is checked for unknown agent IDs. `director` and `synthesizer` are required.
4. **Emit.** Files are written only after validation passes, including root `crew.yaml`.

## LabZ / Zero compatibility

CrewDefine targets LabZ's agent config schema: `id`, `name`, `role`, `tools`, `can_delegate_to`, `system_prompt`, optional `model` and `data_extraction_note`. See [`src/crewdefine/schema.py`](src/crewdefine/schema.py).

The crew manifest (`crew.yaml`) adds: `display_name`, `default_answer_mode`, `answer_modes` (fixed ids: `summary`, `light`, `extended`, `project_plan`, `roadmap`), and `output_composition`.

Known LabZ tool IDs (selected via the interview) are catalogued in [`src/crewdefine/tools_catalog.py`](src/crewdefine/tools_catalog.py). New custom tools are emitted as Python stubs matching LabZ's plugin format (`TOOL_DEFINITION` + callable).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and PRs welcome.

## License

MIT. See [LICENSE](LICENSE).
