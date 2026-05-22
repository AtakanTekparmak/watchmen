# watchmen-daycare

Student-teacher skill distillation: evolve watchmen SKILL.md bundles via
iterative LLM mutation against derived eval sets.

See `../DAYCARE_SPEC.md` for the full design.

## Install (dev)

```
uv sync --extra dev
uv run pytest tests/ -v
uv run daycare doctor
```
