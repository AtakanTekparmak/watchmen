# Plan 0 — Claude Haiku 4.5 + Skills Evolution on SkillsBench (Revision 3)

> Revised after R2 audit + further user decisions on 2026-04-29 (20-task subset, single seed run, repeats=2 inner). Every line range, CLI flag, and symbol below was re-verified against the live tree on 2026-04-29. New / changed sections are flagged with **[v3]**.

## 1. Goal **[v3]**

Baseline Claude Haiku 4.5 inside the official SkillsBench harness (with-skills vs no-skills) over a **deterministic 20-task subset** (NOT the full 84), then evolve **one** skill bundle from `seed_skills_empty/` against the same SkillsBench task universe using our existing Track B pipeline, and produce a head-to-head comparison report — all reproducible from a fresh `uv sync`. The 20-task subset is chosen by a documented diverse-domain sampler (one task per SkillsBench domain bucket until 20) and published as a committed artifact at `skill_evolve/skillsbench/subset_20.json`. **Both baseline and evolution unify on the official `bench` CLI** (per D-1) — the bench CLI's built-in `claude-code` agent IS what runs Claude Code with skills mounted, so using it everywhere gives leaderboard parity for free and removes a dual-track maintenance burden.

## 2. TL;DR Table **[v3]**

Per-trial cost assumption (carried over): Haiku 4.5 with 5–15 tool turns and ~10 K total tokens averages **~$0.05/trial** (input $1/M, output $5/M; cache‑aware; verified empirically against recent Track B Hermes runs in `runs/`). Re-measure during Phase A smoke and overwrite this constant before Phase D commits budget.

**Budget ceiling: $200 (D-4).** Cost arithmetic below assumes the **D-13 resolution** (20-task subset for baseline + re-baseline; single seed run from `seed_skills_empty/`; inner-loop `--repeats 2`). Total **~$87** with **~$113 headroom**. See §3 D-13 for the math.

| Phase | What it does | Trials | $/trial | Wall (est.) | Cost (est.) |
|---|---|---|---|---|---|
| A. Vendor SkillsBench + smoke | Submodule + `benchflow` install + 1 task end-to-end via `bench eval create -f scenes/smoke.yaml -t <task_dir> -a claude-code -m claude-haiku-4-5` | 4 | ~$0.05 | 1–2 h | ~$1–2 |
| B. Agent backend ABC | `BenchCliBackend` + behavior-preserving `HermesBackend` refactor (no `ClaudeCodeBackend`, no SDK fallback) | 0 (mocked tests) | — | 1–2 h | <$1 |
| C. SkillsBench task source adapter | New `source="skillsbench"` branch in `load.py` + verifier shim + anonymizer mirrored from existing in‑file functions + `select_subset_20.py` execution | 0 | — | 2–3 h | <$1 |
| D. Baseline run (Haiku 4.5) | **20** tasks × 5 trials × 2 conditions (with-skills + no-skills) via `BenchCliBackend` | **200** | ~$0.05 | 2–4 h | **~$10** |
| E. Evolution run — `seed_skills_empty/` | Track B → SkillsBench → bench-cli inner; 20 gen × 3 island × **2 repeats** × 12-task hot subset × 1 seed | **1,440** inner | ~$0.05 | 12–20 h evolve | **~$72** |
| F. Re-baseline + comparison | 5 trials × 20 with-skills (evolved) via `BenchCliBackend` (no-skills not re-run; lift = evolved_with_skills − baseline_no_skills) + 2-way comparison | **100** | ~$0.05 | 2–4 h + 1 h | **~$5 + ~$1 = ~$6** |
| **Total** | | **~1,744** | | **~2–3 days wall** | **~$87** |

Spend headroom: $200 ceiling − $87 estimate = **~$113 reserve**. Each phase sets a hard ceiling via `--max-budget-usd`; the bench CLI subprocess wrapper caps by counting `total_cost_usd` from the per-task JSON and exiting early.

## 3. Decisions **[v3]**

D-1, D-3, D-4, D-5, D-6, D-9, D-10, D-13 are **resolved by the user** (folded into the plan). D-14 NEW (subset selection method) is also resolved with a recommendation. D-2a, D-2b, D-7, D-8, D-11, D-12 stand from prior revision.

### Resolved (user-confirmed 2026-04-29)

- **D-1. Backend choice. RESOLVED → "use the official bench cli".** Unify on `bench` CLI for both baseline AND evolution. Drop the `claude-agent-sdk` Python wrapper. Inner evolution trial = `bench eval create -f <yaml> -t <task_dir> -a claude-code -m claude-haiku-4-5`. The bench CLI's built-in `claude-code` agent IS what runs Claude Code with skills — that's the entire benefit of using their official harness.
- **D-3. Trials. RESOLVED → 5 trials baseline + 5 trials re-baseline + repeats=2 inner.** Baseline Phase D and re-baseline Phase F use `--trials 5`. Inner-loop evolution uses `--repeats 2` (was 5 in user's verbatim D-3 answer; reduced to 2 per D-13 reconciliation against the $200 ceiling).
- **D-4. Budget. RESOLVED → $200 total ceiling.** Per-phase soft caps: D=$15, E=$80, F=$10 (sum $105 → $95 headroom against the $200 ceiling). Plumbed through `--max-budget-usd` per phase; bench CLI subprocess wrapper caps by counting `total_cost_usd`.
- **D-5. Anonymization. RESOLVED → ON, mandatory.** Implementation pattern picked: option (c) — string-substitute in evaluator artifacts only (the same approach the existing tblite path uses at `skill_evolve/track_b/openevolve_skills/evaluator.py:108-156`). The bench CLI subprocess invocations themselves use real `-t <task_dir>` paths because they're not outer-LLM-visible. The existing 3-chokepoint anonymization (seed scrub → per-eval artifact scrub → patch-write leak guard) extends naturally to SkillsBench task IDs. Options (a) symlink staging and (b) copy-and-rename were considered and rejected — see §6 Group C step 6 and R-5.
- **D-6. Skill pool. RESOLVED → ONLY empty seed.** Per user decision (2026-04-29), the second evolution run from `seed_skills_task_matched/` is **dropped**. Reason: cost was the gating factor at 5-trial inner; even after reducing to 2-repeat inner, scoping to one seed simplifies the comparison surface and frees budget for higher-trial-count baseline + re-baseline. Group F is now a **two-way comparison** (baseline vs evolved-from-empty), not three-way.
- **D-9. Leaderboard tolerance. RESOLVED → skip gate.** Record actual no-skills and with-skills rates; emit to `summary.md`; do NOT halt regardless of delta from leaderboard 27.7%/11.0%. The previous ±3pp acceptance gate is removed.
- **D-10. Multi-skill growth. RESOLVED → unbounded, capped only at `MAX_FILES=60`.** Allow evolution to split skills freely. Cap inherits from `skill_evolve/track_b/openevolve_skills/folder_artifact.py:43` (already 60). No explicit "max 5 skills" sub-cap.
- **D-13. Budget reconciliation. RESOLVED → 20-task subset + empty seed + repeats=2 inner.** Final arithmetic:
  - Baseline: 5 trials × 20 tasks × 2 conditions = **200 trials × $0.05 = $10**
  - Inner evolution: 12 hot tasks × 20 gens × 3 islands × 2 repeats × 1 seed = **1,440 trials × $0.05 = $72**
  - Re-baseline: 5 trials × 20 tasks × 1 condition (with-skills only) = **100 trials × $0.05 = $5**
  - Phase A (smoke + rate-probe): ~$1–2
  - Phases B+C (mocked tests): <$1
  - **Total: ~$87** vs **$200 ceiling = $113 headroom**.

  This replaces the earlier R2 D-13 trade-table; the user has now selected the parameters directly. The rationale for going to 20 tasks (vs full 84) is that 12-of-the-20 are reused for the inner loop, so the same task universe drives evolution AND headline measurement, and the diverse-domain sampler ensures all 11 SkillsBench domains are represented.

- **D-14. NEW. Subset selection method. RESOLVED → diverse-domain sampler.** Algorithm (deterministic, reproducible):
  1. Walk `vendor/skillsbench/tasks/` and parse each `task.toml` for the `domain` field. If `domain` is absent, infer from `tags` or `category`; if all are absent, bucket as `"misc"`.
  2. Group tasks by domain.
  3. Sort domains alphabetically; within each domain, sort tasks alphabetically by task ID.
  4. Round-robin: pop one task from each non-empty domain bucket in order, append to the subset, until the subset has 20 entries (or all buckets empty — should not happen with 11 domains × 84 tasks).
  5. Emit the resulting list to `skill_evolve/skillsbench/subset_20.json` (a JSON array of task ID strings) and commit.
  
  This guarantees: (a) subset is reproducible from the vendored SkillsBench commit + the script source; (b) all SkillsBench domains have at least one representative; (c) when a domain has fewer than ~2 representatives in 20, it still gets coverage. The script `scripts/select_subset_20.py` is run once at vendor time and the output committed.

- **Evolution generations. RESOLVED → 20** (matches v6/v7 history).

### Open (must be answered before /implement)

- [ ] **D-2a. Auth — pre‑Phase‑A.** Confirm `ANTHROPIC_API_KEY` is exported in the user's shell (`echo "${ANTHROPIC_API_KEY:0:10}…"`). Phase A's smoke calls `bench eval create -m claude-haiku-4-5 -a claude-code` which dispatches direct to api.anthropic.com from the host process — immediate requirement, does not interact with Hermes Docker plumbing.
- [ ] **D-2b. Auth — post‑Phase‑A.** After D-2a is confirmed and Phase A completes, decide whether the Hermes/Docker code path also needs `ANTHROPIC_API_KEY` forwarded. Concretely: edit `skill_evolve/hermes_docker.py:109` to extend `forward_env = ["OPENROUTER_API_KEY"]` to `["OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"]`. **Out of scope for the unified bench-cli path (D-1)** — the Hermes/TBLite code is only kept for behavior preservation on the existing tblite source; bench CLI never goes through Hermes.
- [ ] **D-7. Concurrency.** `bench eval create` parallelism — 2, 4, or 8? Measured during Phase A (D-12), not assumed. Default cap until measured: **2**.
- [ ] **D-8. `--bare` posture.** The bench CLI's `claude-code` agent runs Claude Code in some mode — confirm during Phase A whether it's already isolated (no `~/.claude/skills/` leak) or whether we need an explicit `--bare`-equivalent flag in the YAML scene config. Inspect `vendor/skillsbench/experiments/*.yaml` for the canonical posture.
- [ ] **D-11. Docker plumbing.** SkillsBench tasks ship `tasks/<id>/environment/Dockerfile`; bench builds and runs that container. Hermes Docker (TBLite path) uses `nousresearch/tblite-*` images via `TERMINAL_DOCKER_*` env bridge in `skill_evolve/hermes_docker.py:109-117`. **Recommendation: keep the two paths fully separate.** When `--task-source skillsbench`, route through `BenchCliBackend` (uses bench's own Docker). The "single Docker context" approach (mount both into one) is rejected — too brittle. SkillsBench verification runs `pytest tests/test_outputs.py` in bench's container; the agent runs in bench's "agent container" via `bench eval create`'s built-in dispatch. No cross-mount.
- [ ] **D-12. Empirical rate-probe.** At end of Phase A, kick off 8 parallel `bench eval create -f scenes/smoke.yaml` calls and measure 429 rate. Memory's "tier-1 = 4 RPS" assumption is unverified for the user's actual tier. Output: `runs/smoke/rate_probe.json` `{"observed_rps": N, "max_concurrent_no_429": K}`. K becomes the Phase D `--concurrency` default.

## 4. Requirements **[v3]**

The implementation must satisfy:

- **R-1.** Vendor SkillsBench under `skill_evolve/benchmark/vendor/skillsbench/` (git submodule, pinned commit recorded in `.gitmodules` + `skill_evolve/benchmark/SKILLSBENCH_VERSION`).
- **R-2.** Add `source="skillsbench"` to `skill_evolve/benchmark/load.py:191-237` (the existing `for entry in manifest["tasks"]` loop in `load_subset`) without touching the existing `_hydrate_tblite` (line 109) / `_hydrate_swebench` (line 155) branches; existing tblite eval must remain bit-identical.
- **R-3. [v2: bench-cli only]** New agent backend ABC at `skill_evolve/agents/base.py`. Two concrete implementations ship: `BenchCliBackend` (subprocess-wraps `bench eval create -f <yaml> -t <task_dir> -a claude-code -m <model>`) and `HermesBackend` (behavior-preserving relocation of current `_run_one_task` from `skill_evolve/evaluator.py:314`). **No `ClaudeCodeBackend`. No `claude-agent-sdk` dependency. No SDK CLI fallback.** The ABC stays for future-proofing.
- **R-4.** *(Removed per D-9.)* Baseline runner records actual no-skills and with-skills rates without halting on leaderboard delta. **Confirmed still removed in R3.**
- **R-5. [v2]** Anonymization parity through bench CLI: build a SkillsBench-aware id_map analogous to `build_task_id_map` (live at `skill_evolve/track_b/openevolve_skills/evaluator.py:56`) and apply at the same three chokepoints used today: seed scrub (`controller.py:92-104`), per-eval artifact scrub (lives in `evaluator.py` between lines 108 and 156, called from the hot path in `SkillFolderEvaluator`), patch-write leak guard (`iteration.py:105-107`). **Anonymization happens in evaluator artifacts only** (option (c) from D-5 discussion); bench CLI subprocess invocations use real `-t` paths because they're not outer-LLM-visible. Note: id_map domain is now **20 task IDs** (subset_20.json) instead of 84.
- **R-6. [v3]** `seed_skills_empty/` parity check operates over the **20-task subset** (not 84): Haiku 4.5 + empty seed should yield ≈ leaderboard "no-skills" 11.0% — sanity gate before evolution. **Recorded, not enforced** (per D-9 — no halt).
- **R-7.** All non-trivial parallel agents in `/implement` are Opus, never Sonnet.
- **R-8.** All Anthropic API traffic goes direct (`ANTHROPIC_API_KEY`), never OpenRouter.
- **R-9.** Reproducibility: `git clone <repo> && uv sync && bash scripts/setup_skillsbench.sh && uv run python -m skill_evolve.skillsbench.baseline` and `… .skillsbench.evolve` must each be a single command. The 20-task subset is committed (R-14) so any clone gets the same task IDs.
- **R-10.** No regression in existing Track B tblite path: `pytest skill_evolve/track_b/tests/` still passes (the existing `test_anonymize_tasks.py` continues to lock the tblite path).
- **R-11.** Per-task timeout (default 600 s) and per-run budget cap honored; partial results are checkpointed to disk after every task.
- **R-12.** Anti-leakage: evolved skills are scanned for SkillsBench task-specific filenames, paths, magic numbers, exact commands.
- **R-13.** Output JSON shape preserved across the new runner — same per-task and top-level keys as Hermes evaluator (`EvalResult` at `skill_evolve/evaluator.py:161`, `TaskOutcome` at line 120), so existing tooling still parses results.
- **R-14. NEW [v3].** **20-task subset is chosen deterministically and committed.** `scripts/select_subset_20.py` runs the diverse-domain sampler (D-14) at vendor time; output written to `skill_evolve/skillsbench/subset_20.json` (a JSON array of 20 task ID strings). The file is checked into git so any clone reproduces the same subset. A unit test (`test_subset_selection.py`) asserts the file has exactly 20 entries and every ID corresponds to an existing task dir.

## 5. Files To Modify / Create **[v3]**

All paths absolute under `/Users/atakantekparmak/Desktop/work/kai-skills/`. Verified line ranges as of 2026-04-29.

### New files

- `skill_evolve/benchmark/vendor/skillsbench/` — vendored upstream (submodule). Pinned commit recorded in `.gitmodules` + `skill_evolve/benchmark/SKILLSBENCH_VERSION`.
- `skill_evolve/benchmark/skillsbench_loader.py` — `hydrate_one(task_dir: Path) -> Task` walks `tasks/<id>/`, parses `task.toml`, reads `instruction.md`, locates `environment/`, `tests/`, `environment/skills/`. Returns the `Task` dataclass already defined in `skill_evolve/benchmark/load.py:42`.
- `skill_evolve/benchmark/skillsbench_verifier.py` — `verify(workdir, task) -> VerifyResult`. **Resolved 2026-04-29**: benchflow 0.3.2's CLI exposes only `eval create`/`eval list`/`eval retrieve` — no `eval verify` subcommand exists. The verifier shim therefore goes straight to the Dockerfile path: `docker build` the task's `environment/Dockerfile` into `skillsbench-<id>:local`, then mount `<workdir>` into the resulting container and run `tests/test.sh`. Returns the `VerifyResult` dataclass from `skill_evolve/verifiers.py:59`.
- `skill_evolve/benchmark/skillsbench_anonymize.py` — defines `build_skillsbench_id_map(task_records) -> Dict[str, str]`, `sanitize_text_skillsbench(text, id_map)`, `sanitize_artifact_skillsbench(artifact, id_map)`, `find_leaked_skillsbench_names(artifact, id_map)`. Mirrors the in-file functions at `skill_evolve/track_b/openevolve_skills/evaluator.py:56` (`build_task_id_map`), `:81` (`sanitize_text`), `:108` (`sanitize_artifact`), `:131` (`find_leaked_names`). Operates over the 20-task subset's IDs (not all 84).
- `skill_evolve/agents/__init__.py` — agent backend registry: `get_backend(name) -> AgentBackend`. Names registered: `"hermes"`, `"bench-cli"`. **No `"claude-code"`.**
- `skill_evolve/agents/base.py` — `AgentBackend` ABC: `run_task(task, skills_dir, *, model, timeout_s, budget_usd, anonymize_map) -> TrajectoryResult`.
- `skill_evolve/agents/hermes.py` — refactor of `_run_one_task` (currently at `skill_evolve/evaluator.py:314`) into the ABC. Behavior-preserving — same Hermes/Docker subprocess invocation; just relocated behind the interface.
- `skill_evolve/agents/bench_cli.py` — `BenchCliBackend`. Materializes a per-task scene YAML at `runs/<run>/scenes/<task_id>.yaml`, then `subprocess.run(["bench","eval","create","-f", str(yaml_path), "-t", str(task_dir), "-a","claude-code","-m",model], ...)`. **No `-s` flag** — that flag does not exist in benchflow 0.3.x (verified). Skills are wired through scene YAML.
- `skill_evolve/skillsbench/__init__.py` — entry-point package for SkillsBench-targeted runs.
- `skill_evolve/skillsbench/subset_20.json` **NEW [v3]** — JSON array of exactly 20 task ID strings, generated by `scripts/select_subset_20.py` at vendor time and committed. Single source of truth for which tasks Phase D and Phase F use.
- `skill_evolve/skillsbench/scenes/` — holds YAML scene templates that the bench CLI consumes via `-f`.
  - `scenes/baseline_with_skills.yaml.tmpl` — placeholders for `<task_dir>`, `<model>`, `<skills_dir>`. The runner string-substitutes per task and writes the materialized YAML.
  - `scenes/baseline_no_skills.yaml.tmpl` — same shape, omits the skills mount.
  - `scenes/smoke.yaml.tmpl` — single-task, single-trial config used by `scripts/smoke_skillsbench.sh`.
  - The YAML shape (per benchflow docs): `task_dir`, top-level + `scenes:` list with `roles:` + `turns:`. Exact contents confirmed during Phase A by examining upstream `skillsbench/experiments/*.yaml` examples after the submodule is cloned.
- `skill_evolve/skillsbench/baseline.py` — `python -m skill_evolve.skillsbench.baseline` CLI. Reads `subset_20.json` (or accepts `--task-list <path>`); iterates tasks × trials × conditions; for each, materializes a scene YAML and dispatches via `BenchCliBackend`.
- `skill_evolve/skillsbench/evolve.py` — `python -m skill_evolve.skillsbench.evolve` CLI: thin wrapper around `skill_evolve.track_b.run` with `--task-source skillsbench --agent-backend bench-cli --anonymize-tasks` defaults and `--task-list <hot_12.json>` injection. Defaults `--seed seed_skills_empty/`.
- `skill_evolve/skillsbench/compare.py` — `python -m skill_evolve.skillsbench.compare --baseline <path> --evolved <path>`. **Two-way comparison** (per D-6 ONLY empty).
- `scripts/setup_skillsbench.sh` — one-shot: `git submodule update --init`, `uv sync`, `uv pip install 'benchflow>=0.3.0a7'`, `bench tasks init`. **Does NOT install `claude-agent-sdk`.** Also runs `scripts/select_subset_20.py` if `subset_20.json` is missing.
- `scripts/smoke_skillsbench.sh` — runs ONE task end-to-end via `bench eval create -f scenes/smoke.yaml -t <task_dir> -a claude-code -m claude-haiku-4-5`.
- `scripts/expand_skillsbench_manifest.py` — vendor-time generator that walks `vendor/skillsbench/tasks/`, emits 84 concrete manifest entries (one per task) appended to `manifest.json`. (All 84 stay in the manifest; only 20 are referenced by Phase D + Phase F via `subset_20.json`.)
- `scripts/select_subset_20.py` **NEW [v3]** — implements the D-14 diverse-domain sampler. Reads `vendor/skillsbench/tasks/`, parses each `task.toml` for the `domain` field (falls back to `tags` or `category`), buckets tasks by domain, runs the round-robin sampler, emits `skill_evolve/skillsbench/subset_20.json`. Run once at vendor time and commit. Idempotent — re-running over the same vendored commit yields identical output.
- `scripts/select_hot_12.py` **NEW [v3]** — reads a baseline `summary.json` + `subset_20.json`, picks 12 task IDs by score quartile mix. Default bin distribution (documented in the script docstring): 4 from "failing" (with-skills score < 0.2), 4 from "partial" (0.2 ≤ score < 0.8), 4 from "passing" (score ≥ 0.8). Within each bin, sort by task ID alphabetically and take the first N. If a bin has < 4 entries, pull from the next bin (priority order: partial > failing > passing). Emits `runs/<run>/hot_12.json` (a JSON array of 12 task ID strings) for that evolution run.
- `skill_evolve/track_b/tests/test_skillsbench_adapter.py` — unit tests for loader, verifier shim, anonymizer (mirrors existing `test_anonymize_tasks.py`).
- `skill_evolve/track_b/tests/test_bench_cli_backend.py` — mocked subprocess tests; canned bench CLI JSON output; no real API calls.
- `skill_evolve/track_b/tests/test_subset_selection.py` **NEW [v3]** — verifies `subset_20.json` has exactly 20 entries, all referenced task dirs exist; verifies `select_hot_12.py` picks correctly given a fixture `summary.json` (e.g. construct a synthetic summary with known per-task scores, run the bin selector, assert the 12 returned IDs match the expected mix).
- `plans/plan_0.md` — this file.

### Modified files (verified line ranges)

- `skill_evolve/benchmark/load.py:191-237` — extend the `for entry in manifest["tasks"]` loop in `load_subset` (line 191) with a third branch: `elif entry["source"] == "skillsbench": tasks.append(_hydrate_skillsbench(entry))`. Add the `_hydrate_skillsbench` helper directly above `load_subset`, mirroring `_hydrate_tblite` at line 109 and `_hydrate_swebench` at line 155.
- `skill_evolve/benchmark/manifest.json` — append (do not replace) **84 concrete entries** generated by `scripts/expand_skillsbench_manifest.py`. Each entry has the same shape as existing tblite entries (`{task_id, source: "skillsbench", dataset_task_name, success_check_kind: "skillsbench_test_sh", timeout_s, stage}`). Wildcard envelopes are NOT supported by `load_subset` (verified — it iterates per-entry and dispatches on `entry["source"]`).
- `skill_evolve/evaluator.py:314` — `_run_one_task` gains an `agent_backend: AgentBackend` param; existing default = `HermesBackend()` (current behavior bit-identical). Internal calls to `stage_workspace` (which lives at `skill_evolve/verifiers.py:140`) and `verify_task` (lives at `skill_evolve/verifiers.py:735`) are unchanged.
- `skill_evolve/verifiers.py:140-167` — `stage_workspace` gains a third branch: `if kind == "skillsbench_test_sh": return _stage_skillsbench_workspace(payload, workspace)`. Add the `_stage_skillsbench_workspace` helper next to `_stage_tblite_workspace` (line 169) and `_stage_swebench_workspace` (line 220). Implementation: `cp -r vendor/skillsbench/tasks/<id>/environment/ <workspace>/` (no Hermes-style double-mount; bench CLI's container handles execution per **D-11**).
- `skill_evolve/verifiers.py:729-744` — `_VERIFIERS` dict (line 729) gains `"skillsbench_test_sh": verify_skillsbench`; `verify_task` dispatch already routes via this dict (line 735), no further change needed. Implement `verify_skillsbench` next to `verify_tblite` (line 369) and `verify_swebench` (line 616).
- `skill_evolve/track_b/run.py` — add `--task-source {tblite,skillsbench}` (default `tblite`), `--agent-backend {hermes,bench-cli}` (default `hermes`), `--task-list <path>` (default None, takes a JSON array of task IDs to subset to); thread all three through `Controller`/`Evaluator`/`SkillFolderEvaluator`. **No `claude-code` choice.**
- `skill_evolve/track_b/openevolve_skills/controller.py:92-104` — extend the seed scrub block (currently calls `_sanitize_artifact` from the in-file `evaluator.py` module) to dispatch on `task_source`: when `skillsbench`, import from `skill_evolve.benchmark.skillsbench_anonymize`. Anonymizer module is selected once at `Controller.__init__` and stored on the evaluator instance.
- `skill_evolve/track_b/openevolve_skills/evaluator.py:108-156` — per-eval artifact scrub (the `sanitize_artifact` function definition at line 108; it is called from `SkillFolderEvaluator` later). Same parameterization; `sanitize_text` / `sanitize_artifact` calls become `self._anonymizer.sanitize_text(...)`.
- `skill_evolve/track_b/openevolve_skills/iteration.py:105-107` — patch-write leak guard. Same parameterization: `self._anonymizer.find_leaked_names(...)`.
- `skill_evolve/sandbox.py` — when `agent_backend == "bench-cli"`, write the candidate skill bundle to a path the materialized scene YAML references (Claude Code's discovered location inside bench's container).
- `skill_evolve/hermes_docker.py:109` — **D-2b only.** Extend `forward_env = ["OPENROUTER_API_KEY"]` to `["OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"]` IFF Hermes path will use Anthropic (out of scope for the unified bench-cli path). Surfaced via `TERMINAL_DOCKER_FORWARD_ENV` on line 117.
- `pyproject.toml` — add `benchflow>=0.3.0a7` to dependencies; add console scripts for `…baseline`, `…evolve`, `…compare`. **Do NOT add `claude-agent-sdk`.**
- `.gitmodules` — register the SkillsBench submodule.

## 6. Implementation Phases (Parallelizable Groups) **[v3]**

### Group A — Vendor SkillsBench + smoke (independent)

1. Create `scripts/setup_skillsbench.sh` — clones repo as a git submodule under `skill_evolve/benchmark/vendor/skillsbench/`, pinned to a known-good SHA recorded in `SKILLSBENCH_VERSION`.
2. Add `benchflow>=0.3.0a7` to `pyproject.toml`; `uv sync`. **Do NOT add `claude-agent-sdk`.**
3. Run `bench tasks init`.
4. **D-2a check**: `[[ -n "$ANTHROPIC_API_KEY" ]] || exit 99`. Hard-fail if not set.
5. Inspect `vendor/skillsbench/experiments/*.yaml` and pick the simplest passing example as the canonical scene shape; copy a sanitized version to `skill_evolve/skillsbench/scenes/smoke.yaml.tmpl`.
6. Pick one task (suggest `forensics_scanner` per memory — already cracked 3/3 by our pipeline; high-confidence smoke).
7. Materialize `scenes/smoke.yaml` and run `bench eval create -f /tmp/smoke.yaml -t vendor/skillsbench/tasks/forensics_scanner -a claude-code -m claude-haiku-4-5`. Capture output JSON to `runs/smoke/no_skills.json`.
8. Repeat with the same task's `environment/skills/` referenced inside the YAML. Capture to `runs/smoke/with_skills.json`. Assert with-skills score > no-skills score, both runs total cost < $5.
9. **D-12 rate-probe**: kick off 8 parallel `bench eval create` calls. Time them; record 429 count. Write `{"observed_rps": N, "max_concurrent_no_429": K}` to `runs/smoke/rate_probe.json`. K becomes the Phase D `--concurrency` default.
10. **Re-measure $/trial**: read `total_cost_usd` from each smoke result. Average over the 4 (2 conditions × 2 runs). Overwrite the `~$0.05/trial` constant at the top of §2 if the measured value differs by >25%.
11. Document end-to-end in `scripts/smoke_skillsbench.sh` for one-command repro.

### Group B — Agent backend ABC (independent of A; mockable)

1. Define `skill_evolve/agents/base.py::AgentBackend` ABC with `run_task(self, task: Task, skills_dir: Path | None, *, model: str, timeout_s: int, budget_usd: float, anonymize_map: dict | None) -> TrajectoryResult`. `TrajectoryResult` mirrors the existing `TaskOutcome` shape from `evaluator.py:120` (R-13).
2. Implement `bench_cli.py::BenchCliBackend` — see §5 description. Per-task scene YAML materialization; `subprocess.run(["bench","eval","create","-f", yaml_path, "-t", task_dir, "-a","claude-code","-m",model])`. Parse the per-task JSON bench writes. Enforce per-task timeout via `subprocess.run(timeout=...)`. Budget cap by reading `total_cost_usd` from the JSON and short-circuiting in the runner loop.
3. Implement `agents/hermes.py` as a refactor of current `_run_one_task` (`skill_evolve/evaluator.py:314`) — relocate inside the ABC; behavior-preserving.
4. Wire `agents/__init__.py::get_backend(name)` for `"hermes"` and `"bench-cli"` only.
5. Tests at `test_bench_cli_backend.py`: mock `subprocess.run` returning a canned bench JSON payload, verify the `-f`/`-t`/`-a`/`-m` argv shape, verify `TrajectoryResult` shape matches `TaskOutcome.to_dict()` keys via fixture, verify timeout + budget cap respected.

### Group C — SkillsBench task source adapter (depends on A) **[v3]**

1. `skill_evolve/benchmark/skillsbench_loader.py::hydrate_one(task_dir) -> Task`: parse `task.toml`, read `instruction.md`, embed `environment_dir`/`tests_dir`/`skills_dir` paths into `success_check_payload`. Return the standard `Task` dataclass from `load.py:42`.
2. `skill_evolve/benchmark/skillsbench_verifier.py::verify(task, run_dir) -> VerifyResult`: **resolved 2026-04-29** — benchflow 0.3.2 has no `bench eval verify` subcommand, so the shim goes straight to the Dockerfile path. Mount workdir into `tasks/<id>/environment/Dockerfile`-built image and run `tests/test.sh`. Parse exit code + pytest summary into `VerifyResult(passed, status, detail)` matching the tblite verifier shape. ``FileNotFoundError`` on the docker subprocess returns `status='verifier_unavailable'`.
3. `skill_evolve/benchmark/load.py:191-237` — add `_hydrate_skillsbench(entry)` helper above `load_subset`; add `elif entry["source"] == "skillsbench":` branch in the iteration loop.
4. `skill_evolve/verifiers.py:140` — `stage_workspace` gains a `kind == "skillsbench_test_sh"` branch; `_stage_skillsbench_workspace` `cp -r`s `environment/` into `workspace`.
5. `skill_evolve/verifiers.py:729` — register `"skillsbench_test_sh": verify_skillsbench` in the `_VERIFIERS` dict.
6. **Anonymizer at `skill_evolve/benchmark/skillsbench_anonymize.py` (per D-5 option (c))**: ports the existing `build_task_id_map` / `sanitize_text` / `sanitize_artifact` / `find_leaked_names` from `skill_evolve/track_b/openevolve_skills/evaluator.py:56-156` — same algorithm, sourced over the **20 SkillsBench task IDs from `subset_20.json`** (alpha-sorted, `task_001..task_020`), redacting both bare segment (`forensics-disk-recovery`) and fully-qualified form (`skillsbench/forensics-disk-recovery`). **Anonymization is applied only to evaluator-visible artifacts** — bench CLI subprocess invocations use real `-t <task_dir>` paths because subprocess argv is not outer-LLM-visible. This is the same chokepoint pattern the tblite path already uses.
7. **Manifest expansion**:
   - `scripts/expand_skillsbench_manifest.py`: walks `vendor/skillsbench/tasks/`, parses each `task.toml`, emits 84 concrete `{task_id: "skillsbench/<id>", source: "skillsbench", dataset_task_name: "<id>", success_check_kind: "skillsbench_test_sh", timeout_s: 600, stage: 1}` entries.
   - Run once at vendor time; commit the resulting expanded `manifest.json` (additive — does not modify tblite block).
8. **Subset-20 generation [v3 NEW step]**: run `python scripts/select_subset_20.py` (after submodule clone). Verify the emitted file at `skill_evolve/skillsbench/subset_20.json` has exactly 20 entries; verify each ID corresponds to an existing dir under `vendor/skillsbench/tasks/`. Commit the JSON file.
9. Tests at `test_skillsbench_adapter.py`: load 1 fixture task, verify hydration shape matches the `Task` dataclass, verify anonymizer round-trip (`sanitize → desanitize` is a no-op modulo `task_NNN` aliases), verify verifier shim returns 1.0 on the bundled `solution/solve.sh` path and 0.0 on an empty workdir.
10. Tests at `test_subset_selection.py` **[v3 NEW]**: assert subset_20.json exists and has exactly 20 entries; assert each task ID exists under `vendor/skillsbench/tasks/`; assert the diverse-domain sampler is deterministic (running it twice yields identical output); test `select_hot_12.py` against a fixture summary.json.

### Group D — Baseline runner (depends on A + B + C) **[v3]**

0. **NEW step [v3]: subset-20 verification.** Before any `bench` invocation: assert `skill_evolve/skillsbench/subset_20.json` exists; load it; assert exactly 20 entries; assert each task ID has a corresponding dir under `vendor/skillsbench/tasks/`. If `subset_20.json` is missing, run `python scripts/select_subset_20.py` to generate it. Hard-fail otherwise.
1. `skill_evolve/skillsbench/baseline.py::main(argv)` — argparse: `--task-list skill_evolve/skillsbench/subset_20.json` (default), `--model claude-haiku-4-5`, `--trials 5`, `--conditions with-skills,no-skills`, `--agent-backend bench-cli` (only choice for SkillsBench), `--max-budget-usd 15` (per D-4 soft cap), `--out runs/skillsbench_baseline_<ts>/`, `--concurrency` (default = output of D-12 rate-probe, fallback 2).
2. Iterate **only** the 20 tasks in `subset_20.json` × trials × conditions. For each, materialize the relevant scene YAML template into `runs/<run>/scenes/<task>_<cond>_<trial>.yaml`, then dispatch via `BenchCliBackend.run_task`. Checkpoint to `results.jsonl` after every task. Catch budget exceeded → flush + exit cleanly.
3. Aggregate: per-task pass rate (mean over 5 trials), per-condition mean, lift (with-skills − no-skills). **Launch shape: 5 trials × 20 tasks × 2 conditions = 200 trials.**
4. Emit `summary.json` (same keys as `EvalResult.to_dict()` per R-13) plus `summary.md`.
5. **[v3: D-9 gate dropped — confirmed still removed.]** Record actual no-skills and with-skills rates; emit to `summary.md`; do NOT halt regardless of delta from leaderboard 27.7%/11.0%.
6. **R-6 empty-seed parity check**: a `--condition empty-seed` mode runs `BenchCliBackend` with `seed_skills_empty/` over the 20-task subset and records the result (does not gate — per D-9).
7. **Stability check (former oracle parity)**: re-route the full 20-task subset at 1 trial each, twice. Confirm aggregate within ±3pp and per-task disagreement ≤4 tasks. >4 disagreements halts and triggers determinism audit.

### Group E — Evolution Run (depends on C + D) — `seed_skills_empty/` **[v3]**

1. `skill_evolve/track_b/run.py`: add `--task-source` (default `tblite`), `--agent-backend` (default `hermes`), `--task-list <path>` (defaults to None = full manifest). When `--task-source skillsbench`, force `--agent-backend bench-cli` unless explicit override; force `--anonymize-tasks` on (warn if user disables — D-5 says mandatory).
2. Thread `task_source`, `agent_backend`, and `task_list` through `Controller.__init__`, `SkillFolderEvaluator.__init__`, all the way to `_run_one_task`.
3. **Anonymizer dispatch**: at `Controller.__init__`, build the id_map once (call into `skill_evolve.benchmark.skillsbench_anonymize.build_skillsbench_id_map(task_records)` when `task_source=="skillsbench"`, else the existing in-file `build_task_id_map`). Store on the evaluator. The three chokepoints — `controller.py:92-104`, `evaluator.py:108-156`, `iteration.py:105-107` — call `evaluator.anonymizer.sanitize_text(...)` / `find_leaked_names(...)`.
4. **Anti-leakage scanner (R-12)**: at `evaluate_artifact` exit, scan the candidate SKILL.md + scripts (per `folder_artifact.py` `*.md` and `*/scripts/*.{sh,py}`) for: real task IDs, full task dir names, file paths from the task's `environment/`, magic numbers from `tests/test_outputs.py`, exact commands from `solution/solve.sh`. Scanner runs against **the bench CLI's exposed surface — the materialized scene YAML + skill folder** that `BenchCliBackend` mounts. Same algorithm as before. If any hit, attach `leak_warning` to `EvaluationResult.artifacts` and zero the score (configurable via `--leak-policy {warn,zero,raise}`, default `zero`).
5. `skill_evolve/skillsbench/evolve.py` — preset wrapper:
   ```bash
   python -m skill_evolve.track_b.run \
     --task-source skillsbench \
     --agent-backend bench-cli \
     --anonymize-tasks \
     --task-list runs/skillsbench_evolve_v0/hot_12.json \
     --num-generations 20 \
     --num-islands 3 \
     --repeats 2 \
     --max-workers 2 \
     --seed seed_skills_empty/ \
     --out runs/skillsbench_evolve_v0/
   ```
   `--inner-model` and `--outer-model` are NOT flags here — they are baked into the bench scene YAML config (inner) and the existing Track B outer-model config respectively. `--repeats 2` is the user-confirmed final value (D-13).
6. **Inner-loop hot-12 selection [v3 NEW step]**: at evolution-run start, after Group D produces `runs/skillsbench_baseline_v0/summary.json`, run:
   ```bash
   python scripts/select_hot_12.py \
     runs/skillsbench_baseline_v0/summary.json \
     skill_evolve/skillsbench/subset_20.json \
     -o runs/skillsbench_evolve_v0/hot_12.json
   ```
   This materializes the 12-task hot subset (a strict subset of the 20) into the run directory before evolution begins. The selection algorithm (mix of pass/fail/timeout from baseline) is documented in §5 entry for `scripts/select_hot_12.py`. Total inner trial count: **12 × 20 × 3 × 2 × 1 = 1,440 trials × $0.05 = $72**.
7. Smoke: run a 2-generation, 1-island, 1-repeat evolution end-to-end on a 4-task subset (`SKILLSBENCH_TASK_LIMIT=4` env, or pass a fixture `--task-list smoke_4.json`). Assert `best/` populated; anti-leakage scanner reports zero leaks on the seed.

### Group F — Re-baseline + comparison (depends on D + E) **[v3]**

1. **F.1 Re-baseline evolved bundle**: 5 trials × 20 tasks × **1 condition (with-skills only)** via `BenchCliBackend` against `runs/skillsbench_evolve_v0/track_b/best/`. → `runs/skillsbench_evolved_rebaseline_v0/`. **Confirmed:** the no-skills second condition is NOT re-run because the standard SkillsBench "lift" measurement is `evolved_with_skills − baseline_no_skills` — the baseline no-skills number from Phase D is reused. Phase F trial count: 5 × 20 × 1 = **100 trials = $5**.
2. **F.2 `skill_evolve/skillsbench/compare.py::main(--baseline <dir> --evolved <dir>)`** — load BOTH `summary.json`s (baseline + evolved-rebaseline). **Two-way comparison**, not three-way.
3. **Per-task table**: `task_id`, `baseline_no_skills_rate`, `baseline_with_skills_rate`, `evolved_rate` (=evolved_with_skills_rate from F.1), `delta_vs_no_skills` (= evolved_rate − baseline_no_skills_rate, the headline lift), `delta_vs_with_skills` (= evolved_rate − baseline_with_skills_rate, evolution lift over the seed-equivalent baseline), `n_*`.
4. Aggregate: composite delta, mean_score delta, scored_task_count delta.
5. Significance: paired bootstrap (10k resamples) over the **20 task-level rates** → 95% CI on each delta.
6. Exploit checks: re-run anti-leakage scanner over the evolved bundle; for any task where evolved >> baseline, flag for manual review.
7. Emit `comparison.md` (narrow markdown table per memory pref — Slack-readable) + `comparison.json`.

## 7. Concrete CLI Invocations **[v3]**

```bash
# One-time setup
bash scripts/setup_skillsbench.sh
# (runs git submodule update --init, uv sync, bench tasks init, scripts/select_subset_20.py)

# Smoke (Phase A end)
bash scripts/smoke_skillsbench.sh
# Equivalent manual invocation:
bench eval create \
  -f skill_evolve/skillsbench/scenes/smoke.yaml \
  -t skill_evolve/benchmark/vendor/skillsbench/tasks/forensics_scanner \
  -a claude-code \
  -m claude-haiku-4-5

# Baseline (Phase D) — 20-task subset
uv run python -m skill_evolve.skillsbench.baseline \
  --task-list skill_evolve/skillsbench/subset_20.json \
  --model claude-haiku-4-5 \
  --trials 5 \
  --conditions with-skills,no-skills \
  --agent-backend bench-cli \
  --concurrency "${RATE_PROBE_K:-2}" \
  --max-budget-usd 15 \
  --out runs/skillsbench_baseline_v0/

# Hot-12 selection (between Phase D and Phase E)
uv run python scripts/select_hot_12.py \
  runs/skillsbench_baseline_v0/summary.json \
  skill_evolve/skillsbench/subset_20.json \
  -o runs/skillsbench_evolve_v0/hot_12.json

# Evolution run — empty seed (Phase E)
uv run python -m skill_evolve.skillsbench.evolve \
  --task-list runs/skillsbench_evolve_v0/hot_12.json \
  --num-generations 20 \
  --num-islands 3 \
  --repeats 2 \
  --max-workers 2 \
  --seed seed_skills_empty/ \
  --max-budget-usd 80 \
  --out runs/skillsbench_evolve_v0/

# Re-baseline evolved bundle (Phase F.1) — 20-task subset, with-skills only
uv run python -m skill_evolve.skillsbench.baseline \
  --task-list skill_evolve/skillsbench/subset_20.json \
  --model claude-haiku-4-5 --trials 5 \
  --conditions with-skills \
  --skills-dir runs/skillsbench_evolve_v0/track_b/best/ \
  --max-budget-usd 10 \
  --out runs/skillsbench_evolved_rebaseline_v0/

# Two-way comparison (Phase F.2)
uv run python -m skill_evolve.skillsbench.compare \
  --baseline runs/skillsbench_baseline_v0/summary.json \
  --evolved runs/skillsbench_evolved_rebaseline_v0/summary.json \
  --out runs/skillsbench_compare_v0/
```

Single-command reproducibility (R-9):
- Baseline: `bash scripts/setup_skillsbench.sh && uv run python -m skill_evolve.skillsbench.baseline`
- Evolution + compare end-to-end: scripted in `scripts/run_skillsbench_e2e.sh` (NEW).

## 8. Testing Strategy **[v3]**

**Per-phase verification:**

- **A (smoke):** `bench eval create -f scenes/smoke.yaml -t <task_dir> -a claude-code` returns nonzero score on known-passable task with-skills, returns lower without-skills, total wall <10 min, total cost <$5. D-12 rate-probe writes a single integer to `runs/smoke/rate_probe.json`.
- **B (backend):** `pytest skill_evolve/track_b/tests/test_bench_cli_backend.py` — all mocked, no API calls. Verify subprocess argv shape (`bench eval create -f -t -a claude-code -m`); verify `TrajectoryResult` shape matches Hermes shape; verify timeout raises cleanly; verify budget cap short-circuits.
- **C (adapter):** `pytest skill_evolve/track_b/tests/test_skillsbench_adapter.py` — load fixture task, anonymizer round-trip, verifier returns 1.0 on `solution/solve.sh` and 0.0 on empty. Plus **`pytest skill_evolve/track_b/tests/test_subset_selection.py`** [v3]: assert subset_20.json has exactly 20 entries, all referenced dirs exist; assert hot-12 selection picks the correct mix from a fixture summary.
- **D (baseline):** **No leaderboard ±3pp gate (D-9 dropped, R3 confirmed).** Record actual rates over the **20-task subset**. R-6 empty-seed parity recorded over the 20-task subset (not enforced). **Stability check** (former oracle parity): 20-task subset run twice, ±3pp aggregate, ≤4 per-task disagreements. Trial count: **200**.
- **E (evolution):** Smoke run on 4 tasks × 2 gens × 1 island finishes <30 min; `best/` populated; anti-leakage scanner reports zero leaks on the seed. Full-run trial count: **1,440** (12 × 20 × 3 × 2 × 1).
- **F (comparison):** Trial count: **100** (5 × 20 × 1). Paired bootstrap CI over 20 task rates excludes zero before claiming lift; anti-leakage scanner clean on the evolved bundle.

**Regression gate:** `pytest skill_evolve/track_b/tests/` (full) must pass. The existing `test_anonymize_tasks.py` continues to test the tblite path; the new `test_skillsbench_adapter.py` tests the SkillsBench path; the new `test_subset_selection.py` tests subset reproducibility.

**Budget guards:** Each runner enforces `--max-budget-usd`; on exceed, flush partial results to `results.jsonl` and exit 2. Caller scripts treat exit 2 as "partial OK, do not advance phase."

## 9. Risks **[v3]**

- **Auth.** `ANTHROPIC_API_KEY` not yet plumbed through `forward_env` at `skill_evolve/hermes_docker.py:109`. Mitigation: gated only when `agent_backend == "hermes"`; bench-cli path (the only path used for SkillsBench) bypasses Hermes entirely (see D-1, D-2b).
- **Cost.** Spend already at ~$497-512/$500 (memory) on prior project budget. New estimate: **$87 vs $200 ceiling = $113 headroom** (was $44 in R2 under three-way / 84-task plan). The extra $69 of headroom directly funds (a) deeper inner repeats (2 instead of 1) and (b) re-running phases if hot-12 mix is unrepresentative. Hard `--max-budget-usd` per phase; pre-flight dry-run with `--trials 1 --concurrency 1` on 4 tasks before committing the full run; per-trial cost re-measured at end of Phase A.
- **20-task subset may not be representative of full 84.** **NEW [v3].** Headline lift number is computed over 20 tasks, not the full leaderboard surface. Mitigation: (a) the diverse-domain sampler (D-14) covers all 11 SkillsBench domains so no domain is silently dropped; (b) `subset_20.json` is committed and reproducible — anyone can reconstruct identical subset from the vendored commit; (c) if Phase F results look suspiciously strong/weak vs leaderboard, the comparison can be re-run on the full 84 for $42 + $21 = $63 additional spend, well within the $113 headroom. Also surface this caveat explicitly in `comparison.md`.
- **Rate limits.** Anthropic tier unverified. Mitigation: D-12 empirical probe at end of Phase A overrides tier-1 = 4 RPS assumption.
- **Dataset access.** SkillsBench is Apache-2.0 public — no access risk.
- **Anonymization parity.** SkillsBench task dir names ARE the leak vector (analogue of tblite manifest IDs). R-5 plus the `skillsbench_id_map` mirroring `build_task_id_map` directly addresses this. Per D-5 we use option (c) — substitute in evaluator artifacts only — same as the tblite path, because bench CLI's subprocess argv is not outer-LLM-visible. **id_map domain is now 20 IDs instead of 84** — smaller substitution table, simpler, but still sufficient because evolution is over those 20 (well, the hot-12 strict subset).
- **Time.** ~2-3 days wall total (single evolution run instead of two). Phase E 12-20 h. Mitigation: kick off as `run_in_background` Bash jobs, monitor via Track B's existing `history.jsonl`.
- **Contamination.** SkillsBench may overlap with our prior `seed_skills_*` corpora. By restricting D-6 to the empty seed, we eliminate the contamination risk path entirely — the evolved bundle starts from no skill text whatsoever. (Cost: we lose the "max-achievable" upper bound that `seed_skills_task_matched/` would have given. Acceptable tradeoff.)
- **Harness variance.** Bench CLI's `claude-code` agent may surface its own variance. Mitigation: 5-trial mean is the headline metric (D-3); per-task variance reported.
- **Bench CLI version churn.** `benchflow` is alpha (`>=0.3.0a7`). Mitigation: pin exact version in `pyproject.toml`; version recorded in `SKILLSBENCH_VERSION`.
- **Implicit context loading.** Bench CLI's `claude-code` agent may or may not isolate `~/.claude/skills/` and `CLAUDE.md`. **D-8 must be answered during Phase A** by inspecting `vendor/skillsbench/experiments/*.yaml` for the canonical isolation posture.
- **Docker plumbing complexity (D-11).** Decision: keep bench's container and Hermes's TBLite container fully separate; route per task source. Risk: if a SkillsBench task somehow needs Hermes's TBLite-image binaries, this fails. Mitigation: that's not how SkillsBench tasks work; their containers are self-contained.
- **`MAX_FILES=60` cap (D-10).** Evolution can grow up to 60 files unbounded (per user decision). Risk: bundle bloat → context overhead at inner-trial time. Mitigation: anti-leakage scanner already inspects `*.md` and `*/scripts/*.{sh,py}`; `folder_artifact.py:155` raises if exceeded. Monitor file count growth in `history.jsonl`.

## 10. Linting / Formatting / Conventions

- **uv project mode.** All commands use `uv run python -m ...`; never bare `python`.
- **Lint.** If `ruff` is configured in `pyproject.toml`, run `uv run ruff check skill_evolve/` after each phase. Match existing module style otherwise (snake_case, type hints on public functions, dataclasses for structured returns).
- **Tests live at:** `skill_evolve/track_b/tests/`. Pattern: `test_<feature>.py`. Run with `uv run pytest skill_evolve/track_b/tests/ -x -v`.
- **No emojis** in any source / docs / commit messages (per user pref).
- **Commit style** (sample from recent log): lowercase type prefix, e.g. `feat(skillsbench): vendor SkillsBench + bench CLI smoke (Phase A)`. One commit per Phase A-F.
- **Branch:** stay on `master` OR create `feat/skillsbench` if user prefers; ask in /implement before first commit.
- **/implement dispatch.** Each Group A-F → its own Opus subagent (R-7). Groups A, B can run in parallel. C depends on A. D depends on A+B+C. E depends on C+D. F depends on D+E.
- **Audit loop.** /implement should audit each group's output via a separate Opus auditor before merging — particularly Group C (touching three anonymization chokepoints) and Group E (which can silently break R-5 anonymization parity).
- **pyproject.toml additions:** **only `benchflow>=0.3.0a7`** (NOT `claude-agent-sdk`).

---

## Audit response

- **Item 1.** Replaced fabricated `_LIVE_KEY_ENV_VARS` / `FORWARD_ENV` with the real symbol path: local `forward_env = ["OPENROUTER_API_KEY"]` list inside `build_env_patch()` at `skill_evolve/hermes_docker.py:109`, surfaced via `TERMINAL_DOCKER_FORWARD_ENV` on line 117.
- **Item 2.** Removed the broken `skill_evolve/benchmark/anonymize.py` reference. New `skill_evolve/benchmark/skillsbench_anonymize.py` mirrors the in-file functions at `skill_evolve/track_b/openevolve_skills/evaluator.py:56-156`.
- **Item 3.** Removed every `bench eval create … -s <skills>` invocation. Replaced with YAML scene config + `-f` flag plus required `-t <task_dir> -a claude-code -m <model>`.
- **Item 4.** Dropped SDK API surface verification (no longer relevant — `claude-agent-sdk` removed from the plan per D-1).
- **Item 5.** D-10 rewritten per user decision: unbounded growth, capped only at `MAX_FILES=60`.
- **Item 6.** Per-entry manifest expansion approach (84 concrete entries appended to `manifest.json`).
- **Item 7.** Fixed all line ranges. Re-verified 2026-04-29: `verify_task` at `verifiers.py:735`, `_VERIFIERS` dict at `:729`, `stage_workspace` at `:140`, `_stage_tblite_workspace` at `:169`, `_stage_swebench_workspace` at `:220`, `verify_tblite` at `:369`, `verify_swebench` at `:616`. Hermes `_run_one_task` at `evaluator.py:314`. `EvalResult` at `evaluator.py:161`, `TaskOutcome` at `:120`. Anonymizer functions: `build_task_id_map` at `evaluator.py:56`, `sanitize_text` at `:81`, `sanitize_artifact` at `:108`, `find_leaked_names` at `:131`. Anonymization chokepoints: `controller.py:92-104`, `iteration.py:105-107`. `MAX_FILES=60` at `folder_artifact.py:43`. `load_subset` iteration loop at `load.py:191-237`.
- **Item 8.** D-11 (Docker plumbing) explicit recommendation: separate paths.
- **Item 9.** Per-trial cost assumption (~$0.05/trial). Recomputed §2 totals.
- **Item 10.** Oracle parity becomes determinism stability check under unified bench-cli backend (D-1).
- **Item 11.** D-12 empirical rate-probe at end of Phase A.
- **Item 12.** D-2 split into D-2a (pre-A) and D-2b (post-A). D-2b out of scope under unified bench-cli path.

## User-decision-folded **[v2]**

- **D-1 → unified on bench CLI.** Dropped `claude-agent-sdk` dependency, `skill_evolve/agents/claude_code.py`, SDK CLI fallback. Group B builds only `BenchCliBackend` + behavior-preserving `HermesBackend`. Agent ABC stays for future-proofing.
- **D-3 → 5 repeats baseline (final headline) + repeats=? inner.** Inner repeats reconciled in v3 as `repeats=2`.
- **D-4 → $200 total ceiling.** Plumbed through `--max-budget-usd` per phase.
- **D-5 → ON, mandatory.** Anonymization implementation pattern picked: option (c), evaluator-artifact-only substitution (same as tblite path; bench CLI subprocess argv is not outer-LLM-visible).
- **D-9 → gate dropped.** No halt on leaderboard delta. Group D step 5 records actual rates; step 7 reframed as determinism stability check.
- **D-10 → unbounded growth at `MAX_FILES=60`.** Evolution may split skills freely.
- **Evolution generations → 20** (matches v6/v7 history).

## User-decision-folded **[v3]**

- **20-task subset for baseline + re-baseline (D-14 NEW).** Replaces the 84-task baseline. Diverse-domain sampler (one task per SkillsBench domain bucket, round-robin until 20) emits `skill_evolve/skillsbench/subset_20.json` at vendor time, committed to git. Sampler implemented in `scripts/select_subset_20.py`.
- **Single seed run (empty only — D-6 narrowed).** The earlier R2 plan ran TWO evolution runs (`seed_skills_empty/` and `seed_skills_task_matched/`); v3 runs ONLY the empty seed. Group E.5 is removed entirely. Group F is now a 2-way comparison (baseline vs evolved-from-empty). Reason: cost was the gating factor at 5-trial inner, and even after reducing to 2-repeat inner, scoping to one seed simplifies the comparison surface. The "max-achievable" upper bound that `seed_skills_task_matched/` would have given is sacrificed.
- **Inner repeats=2 (D-13 resolved).** User confirmed `--repeats 2` for the inner loop (between R2's "1 or 5" alternatives). 12 hot tasks × 20 gens × 3 islands × 2 repeats × 1 seed = **1,440 inner trials × $0.05 = $72**.
- **Hot-12 derived from baseline.** New `scripts/select_hot_12.py` reads the Phase D baseline `summary.json` + `subset_20.json` and picks 12 task IDs by score quartile mix. The 12 hot tasks are a strict subset of the 20.
- **New cost: ~$87** (vs $200 ceiling, $113 headroom). Was $156/$44 in R2.

## Task List

> Scope for this `/implement` invocation: **Groups A, B, C only** ("stop before baselining"). Implementers WRITE all code/scripts/templates/tests, but DO NOT execute paid actions: smoke run (`bench eval create`), rate-probe, manifest expansion, subset_20 generation. Those are gated behind the user explicitly running them after review. Submodule clone + `uv sync` + dev-only steps are OK.

- [x] **Group A — Vendor SkillsBench + smoke (code only, no paid execution)**
  - [x] A.1 Create `scripts/setup_skillsbench.sh` (submodule init + `uv sync` + `uv pip install 'benchflow>=0.3.0a7'` + `bench tasks init` + conditional subset_20 generation; D-2a key check inline)
  - [x] A.2 Add `benchflow>=0.3.0a7` to `pyproject.toml` `[project.dependencies]`; do NOT add `claude-agent-sdk`
  - [x] A.3 Add SkillsBench git submodule under `skill_evolve/benchmark/vendor/skillsbench/` (pinned commit recorded in `.gitmodules` + `skill_evolve/benchmark/SKILLSBENCH_VERSION`)
  - [x] A.4 Create `skill_evolve/skillsbench/__init__.py` package marker
  - [x] A.5 Create `skill_evolve/skillsbench/scenes/{smoke,baseline_with_skills,baseline_no_skills}.yaml.tmpl` (placeholder shapes; mark "Phase A confirms exact schema")
  - [x] A.6 Create `scripts/smoke_skillsbench.sh` (script body that materializes scene YAML + invokes `bench eval create` for `forensics_scanner` with-skills + no-skills + writes `runs/smoke/*.json`; DO NOT execute it here)
  - [x] A.7 Add console-script entries `skill-evolve-skillsbench-baseline`, `skill-evolve-skillsbench-evolve`, `skill-evolve-skillsbench-compare` to `pyproject.toml [project.scripts]` (target stubs in `skill_evolve.skillsbench.{baseline,evolve,compare}:main`)

- [x] **Group B — Agent backend ABC (independent of A, mockable)**
  - [x] B.1 Create `skill_evolve/agents/__init__.py` with `get_backend(name) -> AgentBackend` registry (`"hermes"`, `"bench-cli"` only)
  - [x] B.2 Create `skill_evolve/agents/base.py` with `AgentBackend` ABC (`run_task(task, skills_dir, *, model, timeout_s, budget_usd, anonymize_map) -> TrajectoryResult`) and `TrajectoryResult` dataclass mirroring `TaskOutcome` shape from `skill_evolve/evaluator.py:120`
  - [x] B.3 Create `skill_evolve/agents/bench_cli.py` `BenchCliBackend` (per-task scene YAML materialization → `subprocess.run(["bench","eval","create","-f", yaml_path, "-t", task_dir, "-a","claude-code","-m",model])` → parse JSON → return `TrajectoryResult`; honor `timeout_s` and `budget_usd`)
  - [x] B.4 Create `skill_evolve/agents/hermes.py` `HermesBackend` (refactor of `_run_one_task` from `skill_evolve/evaluator.py:314` into the ABC; behavior-preserving — original path stays callable)
  - [x] B.5 Create `skill_evolve/track_b/tests/test_bench_cli_backend.py` with mocked `subprocess.run` returning canned bench JSON; verify argv shape, `TrajectoryResult` fields, timeout/budget enforcement

- [x] **Group C — SkillsBench task source adapter (code only, no paid execution)**
  - [x] C.1 Create `skill_evolve/benchmark/skillsbench_loader.py` `hydrate_one(task_dir) -> Task` (parse `task.toml`, read `instruction.md`, embed `environment_dir`/`tests_dir`/`skills_dir` paths into `success_check_payload`; return `Task` from `skill_evolve/benchmark/load.py:42`)
  - [x] C.2 Create `skill_evolve/benchmark/skillsbench_verifier.py` `verify(task, run_dir) -> VerifyResult` (Dockerfile build + `tests/test.sh` is the sole path — benchflow 0.3.2 ships no `bench eval verify` subcommand; return `VerifyResult` from `skill_evolve/verifiers.py`)
  - [x] C.3 Create `skill_evolve/benchmark/skillsbench_anonymize.py` mirroring functions from `skill_evolve/track_b/openevolve_skills/evaluator.py:56-156` (`build_skillsbench_id_map`, `sanitize_text_skillsbench`, `sanitize_artifact_skillsbench`, `find_leaked_skillsbench_names`); operates over `subset_20.json` IDs
  - [x] C.4 Modify `skill_evolve/benchmark/load.py` (add `_hydrate_skillsbench(entry)` helper above `load_subset` mirroring `_hydrate_tblite` at line 109; add `elif entry["source"] == "skillsbench":` branch in iteration loop at line 213)
  - [x] C.5 Modify `skill_evolve/verifiers.py` (add `_stage_skillsbench_workspace` helper near line 220; add `kind == "skillsbench_test_sh"` branch in `stage_workspace` at line 140; add `verify_skillsbench` near line 616; register `"skillsbench_test_sh": verify_skillsbench` in `_VERIFIERS` dict at line 729)
  - [x] C.6 Create `scripts/expand_skillsbench_manifest.py` (walks `vendor/skillsbench/tasks/`, parses each `task.toml`, emits 84 concrete manifest entries appended to `skill_evolve/benchmark/manifest.json`; idempotent; do NOT execute here)
  - [x] C.7 Create `scripts/select_subset_20.py` (D-14 diverse-domain sampler: parse `domain` field with fallback to `tags`/`category`/`"misc"`, alpha-sort domains + tasks within each, round-robin pop until 20; emit `skill_evolve/skillsbench/subset_20.json`; idempotent; do NOT execute here)
  - [x] C.8 Create `scripts/select_hot_12.py` (reads baseline `summary.json` + `subset_20.json`; picks 12 task IDs by score quartile mix per docstring: 4 failing/4 partial/4 passing with bin-fallback rules; emits `runs/<run>/hot_12.json`)
  - [x] C.9 Create `skill_evolve/track_b/tests/test_skillsbench_adapter.py` (fixture task hydration, anonymizer round-trip, verifier shim returns 1.0 on `solution/solve.sh` and 0.0 on empty workdir)
  - [x] C.10 Create `skill_evolve/track_b/tests/test_subset_selection.py` (assert subset_20.json shape via fixture; assert sampler determinism — run twice, compare; test hot-12 picker against synthetic summary fixture)
