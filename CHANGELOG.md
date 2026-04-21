# Changelog

All notable changes to this project will be documented here. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org/).

## [Unreleased]

### Added
- Initial v0.1 scaffolding: pyproject, CI, lint/format/type/test tooling.
- `crewdefine new` — guided crew creation via LLM-driven interview.
- `crewdefine add-agent` / `crewdefine update-agent` — incremental agent authoring.
- `crewdefine validate` — schema, round-trip, and delegation-graph checks on a crew directory.
- `crewdefine list-tools` — introspect the built-in LabZ tool catalog.
- Pydantic schema mirroring LabZ's `AgentConfig`.
- Python tool-stub emission in LabZ's `TOOL_DEFINITIONS` format.
