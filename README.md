# kai-skills

Skill-folder evolution pipeline for the Hermes agent. Evolves `SKILL.md` folders against a curated subset of the TBLite benchmark using three approaches:

- **Track A** — Autoreason A/B/AB tournament with greedy convergence.
- **Track B** — OpenEvolve MAP-Elites + islands + LLM patches.
- **Track D** — Sequential: Track B explores, Track A refines.

Asymmetric model setup: Sonnet 4.6 for mutation/meta LLM calls, MiniMax M2.7 for inner Hermes agent rollouts. Everything routes through OpenRouter.

## Prerequisites

- Python 3.11+
- `uv` (project-mode package manager — `brew install uv` or see <https://docs.astral.sh/uv>)
- Docker daemon running (colima works on macOS — `brew install colima && colima start`)
- An OpenRouter API key

## Install

```bash
git clone https://github.com/firstbatchxyz/kai-skills
cd kai-skills

# Hermes-agent and OpenEvolve are vendored dependencies;
# fetch them alongside the repo root.
git clone https://github.com/NousResearch/hermes-agent hermes-agent
git clone https://github.com/codelion/openevolve openevolve

# Install dependencies (uv reads pyproject.toml + uv.lock)
uv sync
```

The `hermes-agent[cli]` extra ships the `hermes` binary; run `uv sync --reinstall-package hermes-agent` if `hermes` is missing after a fresh checkout.

## Configure

Keep your OpenRouter key in a sourced env file (not committed):

```bash
echo 'export OPENROUTER_API_KEY=sk-or-v1-...' > ~/.openrouter_env
source ~/.openrouter_env
```

Hermes reads its model config from `~/.hermes/config.yaml`:

```yaml
model:
  provider: openrouter
  default: minimax/minimax-m2.7
  temperature: 0.0
```

## Run

### Evaluate a single skills folder

```bash
uv run python -m skill_evolve.evaluator \
  --skills seed_skills_empty/ \
  --max-workers 7 \
  --model minimax/minimax-m2.7 \
  --repeats 1 \
  --sources tblite \
  --json
```

`--max-workers N` runs up to N tasks concurrently via Docker. `--repeats K` does K independent rollouts per task and aggregates (recommended K >= 2 for variance control — single-eval composite variance is ~+/-0.2 on this benchmark).

### Track A — autoreason

```bash
uv run python -m skill_evolve.track_a.runner \
  --seed seed_skills_empty/ \
  --out runs/track_a_$(date +%s)/ \
  --max-passes 10 \
  --outer-model anthropic/claude-sonnet-4.6 \
  --inner-model minimax/minimax-m2.7 \
  --max-workers 7 \
  --repeats 1
```

### Track B — openevolve

```bash
uv run python -m skill_evolve.track_b.run \
  --seed seed_skills_empty/ \
  --out runs/track_b_$(date +%s)/ \
  --num-generations 15 \
  --num-islands 3 \
  --migration-interval 5 \
  --outer-model anthropic/claude-sonnet-4.6 \
  --inner-model minimax/minimax-m2.7 \
  --max-workers 7 \
  --repeats 1
```

### Track D — sequential B -> A

```bash
uv run python -m skill_evolve.track_d.run \
  --seed seed_skills_empty/ \
  --out runs/track_d_$(date +%s)/ \
  --b-generations 10 \
  --a-max-passes 8 \
  --num-islands 3 \
  --outer-model anthropic/claude-sonnet-4.6 \
  --inner-model minimax/minimax-m2.7 \
  --max-workers 7 \
  --repeats 1
```

### Synthetic smoke (no API calls, no Docker)

Every track accepts `--force-synthetic` for a free end-to-end smoke:

```bash
uv run python -m skill_evolve.track_a.runner \
  --seed seed_skills_empty/ \
  --out /tmp/track_a_smoke \
  --max-passes 2 \
  --force-synthetic
```

## Repository layout

```
skill_evolve/         # Main package (evaluator, sandbox, verifiers, tracks)
  benchmark/          # Task manifest + HF dataset hydration
  track_a/            # Autoreason tournament
  track_b/            # OpenEvolve MAP-Elites + islands
  track_c/            # Hybrid (deprecated; keeps compiling, not actively used)
  track_d/            # Sequential B -> A
  tests/              # Unit tests (143 passing, 5 skipped without Docker)

seed_skills/          # Claude-written 5-skill seed (baseline)
seed_skills_empty/    # 1-skill placeholder for ablation
seed_skills_realworld/# OSS 5-skill seed (obra/superpowers + hermes shipped + ...)

# Skills-with-code ablation seeds (Phase 4; scripts+prose, hermes-native subdirs)
seed_skills_empty_code/ # E1: 1 placeholder skill, no scripts (code-allowed baseline)
seed_skills_1_code/     # E2: 1 upstream skill with script (obra systematic-debugging)
seed_skills_5_code/     # E3: 5 upstream skills (systematic-debugging, verification-before-completion, analyzing-persistence-mechanisms-in-linux, mcp-builder, webapp-testing)

results.md            # Running write-up of all evolution runs
logs.md               # Hourly check-in log across runs
NEXT_STEPS.md         # Current priorities
runs/                 # Per-run artifacts (gitignored)
```

## Skills-with-code seeds (Phase 4)

Three additional seed folders feed the skills-with-code ablation, where
skills may ship executable helpers alongside their `SKILL.md`. All three
use hermes-native subdirs (`scripts/`, `references/`, `templates/`,
`assets/`) and round-trip through `SkillFolder` + `FolderArtifact`.

- **`seed_skills_empty_code/`** (E1) — single placeholder skill, no
  scripts. "Code allowed but none provided" baseline.
- **`seed_skills_1_code/`** (E2) — `systematic-debugging` from
  [obra/superpowers](https://github.com/obra/superpowers) (MIT) with
  the verbatim `find-polluter.sh` bisection helper.
- **`seed_skills_5_code/`** (E3) — five upstream SWE skills covering
  bug-hunting, pre-ship verification, Linux persistence scanning, MCP
  server authoring, and Playwright webapp testing. Sourced from
  obra/superpowers (MIT), mukul975/Anthropic-Cybersecurity-Skills
  (Apache-2.0), and anthropics/skills (Apache-2.0). Each skill carries
  `source_url:` and `license:` in its SKILL.md frontmatter;
  `verification-before-completion` is prose-only upstream and is
  flagged `source: upstream-prose-only`.

## Tests

```bash
uv run pytest skill_evolve/ -x -q \
  --ignore=skill_evolve/tests/test_hermes_docker_integration.py
```

143 tests pass; the integration test requires a live Docker daemon.

## Known caveats

- **Single-eval composite variance is ~+/-0.2** on the current 10-task TBLite manifest. Any claim relying on a single composite is unreliable; use `--repeats 3` for decision-gating evaluations.
- **Cascade short-circuit** (stage-1 gate in the evaluator) will truncate evals when all stage-1 tasks fail binary. The current `manifest.json` puts all 10 tasks at stage 1 to defuse this; if you add new tasks, prefer stage 1 unless the task is reliably slow AND reliably passing.
- **TBLite tasks vary wildly on `/app` vs `/workdir` verification paths.** 48 of 100 tasks in `NousResearch/openthoughts-tblite` assert on `/app/*`; the rest are incompatible with our mount geometry. Check `test_outputs.py` before adding.
