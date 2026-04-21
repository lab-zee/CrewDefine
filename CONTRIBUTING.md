# Contributing to CrewDefine

Thanks for your interest. CrewDefine is a small, focused tool — contributions that keep it that way are welcome.

## Development setup

```bash
git clone https://github.com/lab-zee/CrewDefine
cd CrewDefine
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # add your ANTHROPIC_API_KEY
```

## Running checks

```bash
ruff check src tests
ruff format --check src tests
mypy src/crewdefine
pytest
```

CI runs these on Python 3.10, 3.11, and 3.12. Match them locally before opening a PR.

## Before you open a PR

- Add or update tests for any behavior change.
- Keep the CLI surface stable — new commands are fine, renames and removals need a migration note in `CHANGELOG.md`.
- If you change the emitted YAML shape, update the schema in `src/crewdefine/schema.py` *and* the round-trip tests.
- Don't commit `.env` or API keys. Don't commit generated crews under `/crews/`.

## Design principles

1. **Validation is owned by this repo.** LabZ's loader is lax on purpose; CrewDefine is where we enforce shape, required fields, and delegation-graph sanity.
2. **Guardrails over cleverness.** The interview LLM acts through a constrained tool interface. Adding a new open-ended generation step should come with turn limits and a schema-check on the output.
3. **Emit nothing until validation passes.** Partial/broken crews on disk are worse than a clear error message.
4. **The prompt library is a first-class artifact.** High-quality persona drafts are the core value of the tool. Changes to `src/crewdefine/prompts/` deserve review and, ideally, side-by-side examples in the PR.

## Reporting bugs

Include: OS, Python version, CrewDefine version, the command you ran, and the full error output. Redact any prompt text you don't want public.
