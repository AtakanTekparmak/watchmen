# skill_evolve — shared substrate for evolving Hermes-agent skills

This package gives the three downstream tracks (autoreason loop, openevolve
fork, hybrid) a common eval surface:

```
skill_evolve/
├── __init__.py
├── sandbox.py                 # per-eval HERMES_HOME isolation
├── benchmark/
│   ├── __init__.py
│   ├── manifest.json          # curated 9-task subset (5 TBLite + 4 SWE-bench)
│   └── load.py                # hydrates manifest from HF datasets
├── verifiers.py               # real test.sh + swebench harness verifiers
├── evaluator.py               # evaluate(skills_folder) -> EvalResult
├── tests/                     # pytest unit + (opt-in) docker tests
└── README.md
```

## Setup

```bash
# from repo root
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ./hermes-agent
uv pip install datasets pyyaml swebench pytest
```

### System requirements for real verifiers

The verifiers (introduced after the v0 plumbing-only evaluator) need:

| Requirement | Used for | Failure mode |
|---|---|---|
| **Docker daemon running** | TBLite test.sh + SWE-bench harness | Verifier returns `passed=None`, `verifier_status="verifier_unavailable"`. Score treats it as fail but the per-task report flags it. |
| **`docker` CLI on PATH** | TBLite container start + cp/exec; SWE-bench harness | Same as above. |
| **`git` CLI on PATH** | SWE-bench repo clone + `git diff` patch extraction | `repo_missing`/`empty_patch` per-task statuses. |
| **`swebench` python package** | SWE-bench harness invocation | `verifier_unavailable` on SWE-bench tasks; TBLite still works. |
| **Outbound network** (first run) | `git clone` GitHub repos, pull TBLite Docker Hub images, fetch HF datasets | Stage-fail per task; `staging_error` in `notes`. |

First TBLite run pulls a docker image (~200-500 MB per task; cached after).
First SWE-bench run pulls the harness's per-instance images (~1 GB each;
also cached). Plan disk + bandwidth accordingly.

`hermes-agent/` is a sibling clone of
[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent).
We deliberately do NOT modify it — the sandbox sets `HERMES_HOME` and lets
hermes pick up the candidate skill folder via `skills.external_dirs` in
the temp `config.yaml`.

## Required environment variables

For real LLM calls, **both** the outer (mutation) and inner (Hermes agent
rollout) LLMs route through OpenRouter via the OpenAI-compatible SDK.
Set the key in the parent shell (all are forwarded into the sandbox):

| Variable | What it unlocks |
|---|---|
| `OPENROUTER_API_KEY`     | OpenRouter — used by **both** the outer mutation LLM (critic / op-planner / body-writer / synthesizer / Track B patch generator) and the inner Hermes agent rollouts per task |
| `OPENAI_API_KEY`         | Direct OpenAI (evaluator-side only) |
| `NOUS_API_KEY` / `NOUS_PORTAL_API_KEY` | Nous Portal (evaluator-side only) |

Model strings follow OpenRouter naming — e.g. `anthropic/claude-sonnet-4.6`,
`minimax/minimax-m2.7`, `openai/gpt-5.1`. OpenRouter routes the request
to the correct upstream provider based on the slug, so no `--provider`
flag is needed.

## Model configuration (outer vs inner)

Every track (A, B, C) and the compare harness accept a split pair of model
flags so you can use a high-quality LLM for the few outer mutation calls
and a cheap LLM for the many inner agent rollouts:

| Flag | Role | Default | Who calls it |
|---|---|---|---|
| `--outer-model` | **Mutation LLM** | `anthropic/claude-sonnet-4.6` | Critic, op-planner, body-writer, synthesizer (Track A); patch generator (Track B); tournament mutation client (Track C) |
| `--inner-model` | **Agent rollout LLM** | `minimax/minimax-m2.7` | `SkillFolderEvaluator` → `run_agent.py --model=...` for each Hermes task rollout |

Rationale: the outer LLM makes O(10s) of calls per pass and drives the
whole improvement signal, so quality matters. The inner LLM makes
O(hundreds) of calls (N_tasks × candidates × passes), so cost matters
more than peak quality.

The pre-v2.1 `--model` flag still works as a **deprecated alias** that
sets both to the same value and emits a deprecation warning. Prefer the
split flags in all new scripts.

Example (compare harness, three tracks, explicit split):

```bash
uv run python -m skill_evolve.compare.run_all \
  --seed seed_skills/ \
  --out runs/v21/ \
  --budget-passes 4 \
  --tracks a,b,c \
  --outer-model anthropic/claude-sonnet-4.6 \
  --inner-model minimax/minimax-m2.7
```

The tracks persist the chosen models into their history / `best_meta.json`
records (`outer_model`, `inner_model` fields in Track A's `history.json`;
alongside aggregate metrics in each track's `best_meta.json`).

As of v2.1 `best_meta.json` also carries
`eval_artifacts.per_task` — a JSON list of per-task outcomes
(`task_id`, `verified`, `success`, `elapsed_s`, `tool_calls`, …) so
downstream tooling can tell exactly which tasks a candidate fixed or
regressed, not just the aggregate `composite` / `success_rate`.

Optional:

| Variable | Purpose |
|---|---|
| `SKILL_EVOLVE_MODEL`     | Override default model string (evaluator) |
| `HERMES_HOME`            | Ignored — sandbox always overrides |
| `HERMES_DISABLE_TELEMETRY` | Set automatically by sandbox |

If **no** key is set the evaluator returns a **synthetic placeholder
EvalResult** (success_rate = 0, synthetic = True) so the rest of the
pipeline can be exercised without spend. The mutation clients
(Track A's `LLMClient`, Track B's `OpenRouterLLM`) similarly fall back
to their synthetic modes when invoked with `--force-synthetic` or when
`OPENROUTER_API_KEY` is unset.

## Invoking each piece

### Sandbox smoke test

```bash
python -m skill_evolve.sandbox /path/to/skills_folder
# stands up a temp HERMES_HOME, writes config.yaml with skills.external_dirs,
# shells `batch_runner.py --help` to verify env wiring, then cleans up.
```

### Inspect benchmark

```bash
# offline (no HF round-trip): just IDs/structure
python -m skill_evolve.benchmark.load --offline

# fully hydrate (needs HF cache or network)
python -m skill_evolve.benchmark.load                 # all sources
python -m skill_evolve.benchmark.load --sources tblite
```

In code:

```python
from skill_evolve.benchmark import load_subset
tasks = load_subset()                # list[dict]
tasks = load_subset(sources=["tblite"])
```

### Evaluator

```bash
# Synthetic dry run — verifies plumbing, no LLM spend, no Docker
python -m skill_evolve.evaluator --skills /path/to/skills --force-synthetic

# Live run with REAL verifiers (Docker required), single source/worker
python -m skill_evolve.evaluator --skills /path/to/skills --sources tblite

# Skip real verification — fast plumbing test with live LLM but no Docker
python -m skill_evolve.evaluator --skills /path/to/skills --sources tblite --no-verify

# Parallel (each worker gets its own sandbox + workspace)
python -m skill_evolve.evaluator --skills /path/to/skills --max-workers 4

# JSON output for downstream tooling
python -m skill_evolve.evaluator --skills /path/to/skills --json
```

In code:

```python
from pathlib import Path
from skill_evolve.evaluator import evaluate

result = evaluate(
    Path("/path/to/skills"),
    cascade=True,
    max_workers=4,
)
print(result.composite, result.success_rate)
```

## Tests

```bash
# Unit tests only (no Docker, no network) — ~1s
pytest skill_evolve/tests/

# Docker-marked integration tests (needs running daemon, pulls image) — ~30s
pytest skill_evolve/tests/ -m docker

# Everything
pytest skill_evolve/tests/ -m "docker or not docker"
```

`tests/test_verifiers.py` covers:

* `extract_git_diff` round-trips against a synthetic git repo (modified
  file, untracked file, empty diff, non-git dir).
* `verify_task` dispatcher contract: unknown kind → `passed=None`,
  swebench without staged repo → `passed=False, status="repo_missing"`,
  tblite without docker → `passed=None, status="verifier_unavailable"`.
* (`-m docker`) TBLite `broken-python` end-to-end: untouched workspace
  → verified fail; staged workspace → verifier returns a definite
  True/False (not None) — proves the docker round-trip works.

## Composite score

```
composite = success_rate - 0.05 * normalized_tool_call_overhead
where normalized_tool_call_overhead = clamp(avg_tool_calls / 10, 0, 1)
```

Documented inline in `evaluator.py`. Downstream tracks can swap by
overriding `compute_composite` or by post-processing `EvalResult`.

## Cascade

Stage-1 contains the two cheapest TBLite tasks. If both fail
(`success_rate == 0`), the rest of the suite is skipped — saves ~80% of
wall time on dead candidates (broken SKILL.md, infinite loops, etc.).
Disable with `--no-cascade` or `cascade=False`.

## Subprocess vs in-process

We invoke `hermes-agent/run_agent.py` as a subprocess per task. AIAgent
mutates global state at import time (registry, signal handlers,
provider selection); per-process isolation matches the per-`HERMES_HOME`
isolation we already need for parallel evaluation. Cost: ~300 ms cold
start per task, dwarfed by LLM latency.

## Trajectory parsing

`run_agent.py --save_trajectories=True` appends to
`trajectory_samples.jsonl` (success) and `failed_trajectories.jsonl`
(failure) in the **current working directory** — the sandbox sets
`cwd = HERMES_HOME/run/`. We parse that JSONL after the subprocess
exits, count `<tool_call>` blocks per turn, and pick out
`name == "skill_view"` invocations to populate `skills_invoked`.

## Real verifiers

`evaluator.py` now wires real pass/fail verifiers via
`skill_evolve/verifiers.py`:

| `success_check_kind` | What runs | Pass = |
|---|---|---|
| `tblite_test_sh` | `docker run -v workspace:/app <image>` then `bash test.sh` inside | `reward.txt == "1"` (or pytest exit 0) |
| `swebench_patch_tests` | `git diff` from the staged repo → `python -m swebench.harness.run_evaluation` | instance_id appears in harness `resolved_ids` |

### Workspace staging

Before the agent runs, `_run_one_task` calls `stage_workspace(task, ...)`:

  * **TBLite**: pulls the task image (cached after first run), `docker create`s
    a stopped container, `docker cp /app/.` into the workspace. The agent
    runs with `cwd = workspace` so its edits land in the same place that
    test.sh will read at verify time (we mount the workspace back to `/app`
    in a fresh container then exec test.sh).
  * **SWE-bench**: `git clone https://github.com/<repo>.git` and
    `git checkout <base_commit>` into `workspace/repo`. The agent edits
    files there; we extract the unified diff afterward via `git diff` and
    feed it to the official harness.

### Three-state success

`TaskOutcome.verified` is now `True` / `False` / `None`:

  * `True` — verifier ran and the task passed.
  * `False` — verifier ran and the task failed.
  * `None` — verifier could not run (Docker missing, harness install
    missing, `--no-verify`, synthetic placeholder, unknown
    `success_check_kind`). For scoring it's counted as 0 (success_rate
    treats it as fail), but `EvalResult.unverified_count` and the
    per-task `verifier_status` field let callers tell apart "agent
    failed" from "we couldn't verify".

### `--no-verify` and `--force-synthetic`

Both bypass real verification but for different reasons:

  * `--no-verify` — keeps live LLM calls but skips Docker/harness
    invocation. `success` falls back to the legacy "agent finished a
    saved trajectory" signal. Useful when iterating on agent prompts /
    skill folders without paying Docker pull + test.sh time.
  * `--force-synthetic` — emits a hard-coded zero-success placeholder
    without spending any LLM credits OR running Docker. Useful for
    pipeline plumbing tests.

### Cascade short-circuit

Stage-1 still runs first. If 0/N of its tasks succeeds (on the *primary*
success signal — verifier pass when `verify=True`, trajectory-exists
when `verify=False`), stages 2+ are skipped. Real Docker/SWE-bench
invocations are slow, so the cascade saves substantial wall time on
broken candidate skill folders.

### What's still NOT in scope

- **Modifying upstream `hermes-agent/`** — left untouched on purpose so
  we can rebase against upstream cleanly. The agent runs on the host
  with cwd = staged workspace; we don't drop the agent into the docker
  container itself (TBLite-style).
- **Running the agent inside SWE-bench's per-instance containers** — we
  rely on the agent doing repo-relative edits via the standard file
  tools, then export those as a patch. Matches the SWE-bench
  Verified evaluation contract.
