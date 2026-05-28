# Plan 0 — Consolidate daycare + skill_evolve evolution into a single program

## 1. Goal

Consolidate the daycare and skill_evolve evolution lineages into a single evolution program living in `skill_evolve`, lifting daycare's full-bundle sentinel-block mutation, smoke-test gating, MAX_BUNDLE_TOKENS guard, DeepSeek thinking-mode content/reasoning short-circuit fallback, and adding an `--eval-source` flag (skillsbench default, behavioral optional) plus train/validation split; shrink daycare to a standalone corpus-mining / eval-build / corpus-audit tool. (FINDINGS_v2.md at `/Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare/FINDINGS_v2.md` is a separately tracked deliverable — not in scope here.)

## 2. Requirements

### Functional (in scope)

- skill_evolve gains full-bundle mutation via sentinel-block patch format:
  - `ADD_FILE`, `EDIT_FILE`, `DELETE_FILE`, `REWRITE_FOLDER` operating on any path inside a bundle (SKILL.md, scripts/*, references/*).
  - Explicit `<<<END_FILE>>>` enforcement with a "CRITICAL" warning in the proposer system prompt.
  - HARD LIMITS block (max-file count, max bytes per file, MAX_SKILL_TOKENS, MAX_BUNDLE_TOKENS) embedded verbatim in the prompt.
- DeepSeek thinking-mode handling — **TWO DISTINCT MECHANISMS**, do not conflate:
  - **(a) `reasoning_max_tokens` budget cap.** track_b ALREADY has this at `track_b/openevolve_skills/llm_client.py` (constructor param at line 116, stored at line 125, passed to OpenRouter via `extra_body["reasoning"]["max_tokens"]` at lines 151-154). Group A must port this to `track_a/llm.py` if track_a's chat client does not already cap reasoning tokens (verify on read; if track_a sits behind a different client wrapper add the same `extra_body` shape).
  - **(b) `content or reasoning or ""` short-circuit.** track_b currently extracts `content` and `reasoning` as TWO SEPARATE variables at `track_b/openevolve_skills/llm_client.py:157-159` (`content = msg.content or ""`, then `reasoning = getattr(msg, "reasoning", None) or ""`) and uses them divergently for raw-log + return paths. Daycare's verifier uses the unified short-circuit `text = msg.get("content") or msg.get("reasoning") or ""` at `watchmen-fukara/daycare/src/daycare/verifier.py:189`. The port adds the unified short-circuit to (i) `track_a/llm.py`'s response-extraction path AND (ii) `skill_evolve/behavioral/adapter.py` (judge consumer). track_b's existing two-field extraction stays as-is for its raw-logging behaviour — Group A explicitly documents in the PR description WHY track_b keeps two fields (raw-log needs both separately) but the new unified-fallback helper lives in shared utilities so future call sites get the right behavior by default.
- Smoke-test gating before scoring:
  - `py_compile` on every `*.py` in the candidate bundle.
  - `bash -n` on every `*.sh` in the candidate bundle.
  - Reject bundle (no eval call) on failure; record reason in the artifact.
- `MAX_BUNDLE_TOKENS = 60000` and `MAX_SKILL_TOKENS = 3000` guards rejecting oversize candidates before eval.
- macOS `._` AppleDouble metadata files filtered inline in `_bundle_tokens()` and `list_scripts()` (not via a `find -delete` shell command — must be tolerant of being uploaded already).
- New CLI flag `--eval-source {skillsbench,behavioral}` plumbed through:
  - `skill_evolve/track_b/run.py` (primary)
  - `skill_evolve/skillsbench/evolve.py` (preset wrapper)
  - `skill_evolve/track_a/runner.py` (unified UX, even though track_a's main flow stays skillsbench-default)
- New CLI flag `--eval-set <path>` (path to a daycare-style `eval_set.jsonl`) required when `--eval-source behavioral`; behavioral adapter scores each candidate via judge LLM (port of `daycare/verifier.score_single`).
- New CLI flag `--validation-task-list <path>` for held-out scoring after a winner is accepted on the training set; train hot_5 / validate subset_17 is the canonical config. Task-set name → JSON file path resolution table:
  - `hot_5` → `/Users/atakantekparmak/Desktop/work/kai-skills/runs/skillsbench_baseline_v2/hot_5.json`
  - `subset_17` → `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/skillsbench/subset_17.json` (verify exact path on read; if absent, the canonical config must point at wherever subset_17 lives — Group B audits and pins).
  - Custom paths are passed through verbatim. Names are NOT a fixed enum; if `--task-set <name>` is given and not in the resolution table, treat as a path.
- Daycare CLI shrinks: drop `run`, `promote`, `daemon` subcommands; keep `eval-build`, `doctor`, `runs`.
- Daycare modules deleted: `anchor.py`, `controls.py`, `finalize.py`, `watchdog.py`, `daemon.py`. `evolve.py` deleted after its harvested constants/prompts/helpers land in skill_evolve. `mutator.py`, `leak_scanner.py` ported to `skill_evolve/shared/`, then daycare copies deleted.
- Daycare keeps: `eval_builder.py`, `behavioral_builder.py`, `synth_builder.py`, `corpus.py`, `runner.py`, `providers.py`, `selector.py`, `anonymize.py`, `verifier.py`.
- Test coverage:
  - Round-trip ADD_FILE / EDIT_FILE / DELETE_FILE / REWRITE_FOLDER (apply → re-parse → identical bundle).
  - Smoke-test rejection paths (broken `.py`, broken `.sh`, ok bundle), PLUS `--no-smoke-test` bypass path (smoke not called, candidate evaluated regardless of script syntax).
  - MAX_BUNDLE_TOKENS / MAX_SKILL_TOKENS enforcement edge cases (just under, exactly at, over).
  - DeepSeek `content=None / reasoning="..."` fallback test in track_a/llm.
  - Behavioral adapter wired against a tiny fixture eval_set.jsonl.
  - One-iter end-to-end integration test on a mock SkillsBench fixture.
- Canonical consolidated config flag-set documented in skill_evolve README (qwen3.6-27b inner, deepseek-v4-pro proposer, no LLM judge, hot_5 train + subset_17 validate, full-bundle mutation, 6h budget / $50 cap / 12 iters).

### Anti-requirements (explicitly out of scope)

- Variance experiment on `r_d2079019` bundle (tracked separately in `/Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare/next_steps.md`, runs against a separate branch).
- FINDINGS_v2.md content — mentioned only.
- Reviving daycare `run`/`promote`/`daemon` subcommands or their tests in any form.
- Converting daycare from Click to argparse, or skill_evolve from argparse to Click.
- Touching `skill_evolve/benchmark/vendor/**` (pre-existing lint violations live there — leave alone).
- Bumping daycare to Python 3.12 or skill_evolve to Python 3.11 (each keeps its own version pin).
- New outer LLM clients / new bench backends.
- Replacing track_a's JSON-op proposer format with sentinel blocks — sentinel mode is added behind a `--patch-format sentinel-blocks` flag; the existing JSON ops stay default for track_a's other call sites.

## 3. Files to modify / create

### Important verified line/symbol references (read these before starting)

- daycare `mutator.py` exports `apply_ops` at line 208 (NOT `apply_file_ops`). The port renames `apply_ops → apply_file_ops` in `skill_evolve/shared/bundle_ops.py`. Original-name → new-name mapping:
  - `daycare.mutator.apply_ops` → `skill_evolve.shared.bundle_ops.apply_file_ops`
  - `daycare.mutator.parse_and_apply` → `skill_evolve.shared.bundle_ops.parse_and_apply`
  - `daycare.evolve._bundle_tokens` → `skill_evolve.shared.bundle_ops.bundle_tokens`
  - `daycare.evolve.list_scripts` → `skill_evolve.shared.bundle_ops.list_scripts`
- daycare `evolve.py` proposer template lives at lines 620-681 (`_PROPOSER_SYSTEM_PROMPT_TEMPLATE`). The `.format(...)` rendering happens at lines 682-685 (`_PROPOSER_SYSTEM_PROMPT = ...format(...)`). The runtime assertion block is at lines 686-692 (two `assert "{N}" in _PROPOSER_SYSTEM_PROMPT` statements).
- track_b `patch_parser.py` exports `parse_patch(text) -> List[Operation]` (line 89), `Operation = AddFile | EditFile | DeleteFile | RewriteFolder` (line 76), and `PatchParseError(ValueError)` (line 47). It does NOT export `parse_sentinel_blocks` or `FileOp` — those are daycare-side names.
- track_b `llm_client.py:104` is a COMMENT about `reasoning_max_tokens` budget. The actual content/reasoning extraction is at lines 157-159 (two separate variables, not the unified short-circuit pattern).
- track_a `ops.py` dispatch is at line 595 `apply_op()` (name-based `if/elif` chain). There is NO `_OP_REGISTRY` symbol.
- daycare `verifier.py:140` defines `score_single(...) -> float | None`. The DeepSeek content/reasoning short-circuit is at line 189. `_parse_judge_response(text)` parses a float score out of the LLM JSON.
- daycare `verifier.py` DOES still contain `score_bundle` at line 357 — Group C must NOT assume it's gone. The plan retains verifier.py; whether `score_bundle` survives is decided in C.9 by grepping its callers.

### skill_evolve — shared utilities (consumed by Groups A, B, D)

- **CREATE** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/__init__.py` — empty marker.
- **CREATE** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/patch_parser.py` — sentinel-block parser refactored out of `track_b/openevolve_skills/patch_parser.py`. **PRIMARY** exports use daycare naming: `parse_sentinel_blocks(text: str, existing_paths: Optional[set[str]] = None) -> list[FileOp]`, `FileOp` dataclass (`op_kind`, `path`, `body`, `line_start`), `SentinelParseError(ValueError)`. **BACKWARD-COMPAT aliases** for track_b consumers: `parse_patch = parse_sentinel_blocks`, `Operation = FileOp`, `PatchParseError = SentinelParseError`. (Note: `Operation` was a union type alias in track_b; the alias becomes a dataclass — `isinstance(x, Operation)` keeps working because every parser output is a `FileOp` subclass.) The per-kind names `AddFile`, `EditFile`, `DeleteFile`, `RewriteFolder` are exported as **distinct empty subclasses of `FileOp`** (NOT simple aliases) — `track_b/tests/test_patch_parser.py` asserts both `isinstance(ops[0], AddFile)` AND `type(o).__name__ == "EditFile" / "DeleteFile"`, neither of which the alias scheme satisfies. The parser dispatches on the parsed op string and instantiates the matching subclass at construction time. Group A owns the move; Group B/D consume.
- **CREATE** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/bundle_ops.py` — apply/validate logic: `apply_file_ops(bundle_dir, ops) -> ApplyResult`, `validate_scripts(bundle_dir) -> SmokeResult`, `bundle_tokens(bundle_dir) -> int`, `list_scripts(bundle_dir) -> list[Path]`, `hash_bundle(bundle_dir) -> str`, `shebang_insurance(bundle_dir)`, `parse_and_apply(parent_bundle_dir, llm_text, candidate_dir)`. Ported from `watchmen-fukara/daycare/src/daycare/mutator.py` (`apply_ops` at line 208 renamed to `apply_file_ops`, `parse_and_apply` at line 408) + `evolve.py` lines 79-102 (`_bundle_tokens`) and 105-115 (`list_scripts`). Constants `MAX_SKILL_TOKENS=3000`, `MAX_BUNDLE_TOKENS=60000`, `APPLE_DOUBLE_PREFIX="._"` live here.
- **CREATE** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/leak_scanner.py` — direct copy of `watchmen-fukara/daycare/src/daycare/leak_scanner.py`. Daycare's copy is then deleted (see Group C). Re-exported from `skill_evolve.shared` so existing track_b callers in `iteration.py` resolve through the shared module rather than reaching into watchmen-fukara.
- **MODIFY** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/openevolve_skills/patch_parser.py` — replace body with `from skill_evolve.shared.patch_parser import *` shim PLUS explicit re-exports of the back-compat alias names so star-import semantics behave; existing tests at `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_patch_parser.py` MUST still pass.

### skill_evolve — Group A (sentinel-block patch capability + smoke guard + DeepSeek fallback)

- **MODIFY** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_a/ops.py` — add four new op classes `AddFileOp`, `EditFileOp`, `DeleteFileOp`, `RewriteFolderOp`. Each adapts the `FileOp` dataclass from `shared/patch_parser.py` to the existing `Op` protocol used by `apply_op()`. Place after the existing `apply_rewrite_content` (currently ends ~line 393). Register via additional `elif name == "add_file": …` (etc.) branches inside the existing `apply_op()` dispatch at line 595 — there is no `_OP_REGISTRY` symbol; dispatch is a name-based `if/elif` chain.
- **MODIFY** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_a/prompts.py` — add `SENTINEL_PROPOSER_SYSTEM_PROMPT` (ported from `watchmen-fukara/daycare/src/daycare/evolve.py` template lines 620-681, rendered via `.format(...)` at lines 682-685) selected when `patch_format == "sentinel-blocks"`. Existing JSON-op prompt remains default. Port the assertion block from daycare evolve.py lines 686-692, BUT convert from bare `assert` statements to explicit `if MAX_SKILL_TOKENS_STR not in SENTINEL_PROPOSER_SYSTEM_PROMPT: raise RuntimeError(...)` checks called from a module-level `_assert_prompt_well_formed()` invoked on import. Reason: bare `assert` is stripped under `python -O`.
- **MODIFY** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_a/runner.py` — add `--patch-format {json-ops,sentinel-blocks}` argparse arg (default `json-ops`) inside the new `_add_patch_args(parser)` helper (see section 7d). When `sentinel-blocks`, the proposer LLM output is parsed via `shared.patch_parser.parse_sentinel_blocks` and applied via `shared.bundle_ops.apply_file_ops`. Add `--smoke-test/--no-smoke-test` (default on) wiring through to candidate scoring.
- **MODIFY** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_a/llm.py` — add unified DeepSeek reasoning fallback to the chat response extraction. Pattern: `text = msg.get("content") or msg.get("reasoning") or ""` (the daycare verifier.py:189 idiom — NOT the track_b two-field split). Also ensure track_a's chat call carries `extra_body={"reasoning": {"max_tokens": reasoning_max_tokens}}` if track_a's client wrapper doesn't already; mirror the warning log shape used in `track_b/openevolve_skills/llm_client.py`.
- **MODIFY** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_a/folder.py` — insert smoke + token-cap check inside `SkillFolder.write(dest)` at line 187. Specifically: right before `return dest` at line 206. On failure (smoke or token cap): raise `BundleRejected(reason)` (new exception defined in `shared.bundle_ops`); upstream caller in track_a's pass loop (find via `git grep '.write(' track_a/`) catches this, logs reason, records on artifact, returns a sentinel "rejected" candidate without an eval call.
- **MODIFY** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/openevolve_skills/iteration.py` — call the same smoke + token guards from `shared/bundle_ops.py` BETWEEN the `mutate(...)` call at line 92 and the `evaluator.evaluate_artifact(...)` call at line 143. Concretely: insert a `_run_smoke_gate(child_artifact)` invocation immediately after the anonymize block ends (current line 139) and before line 141's "5. Evaluate child." comment. On rejection, return an `IterationResult` with `op_type="parse_error"` and `notes=f"smoke_rejected: {reason}"`. (track_b currently applies patches without smoke gating.)

### skill_evolve — Group B (`--eval-source` + behavioral adapter + validation split)

- **CREATE** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/behavioral/__init__.py` — empty marker.
- **CREATE** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/behavioral/adapter.py` — `score_bundle_behavioral(bundle_dir, eval_set_path, judge_model, *, repeats=1) -> EvalResult`. Loads `eval_set.jsonl` (daycare format), runs each item through judge LLM, returns a `skill_evolve.evaluator.EvalResult`-shaped object with `success_rate`, `composite`, `mean_score`, `per_task`, `failures`. Internals port from `watchmen-fukara/daycare/src/daycare/verifier.py:score_single` (line 140, returns `float | None`) using the content/reasoning short-circuit at line 189. The judge LLM client uses `track_b/openevolve_skills/llm_client.py:OpenRouterLLM` (no new client). When iterating bundle files for prompt rendering, USE `skill_evolve.shared.bundle_ops.list_scripts()` / `bundle_tokens()` — do NOT walk the directory directly: those helpers filter `._*` AppleDouble files and `__pycache__/`.
- **CREATE** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/behavioral/eval_set.py` — small loader returning typed records from a `.jsonl` (no daycare import).
- **MODIFY** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/evaluator.py` — `evaluate(...)` gains `eval_source: Literal["skillsbench","behavioral"] = "skillsbench"`, `eval_set_path: Optional[Path] = None`, `judge_model: Optional[str] = None`. When `eval_source == "behavioral"`, dispatch to `skill_evolve.behavioral.adapter.score_bundle_behavioral`. Existing skillsbench/tblite paths unchanged.
- **MODIFY** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/run.py` — add `--eval-source`, `--eval-set`, `--judge-model`, `--validation-task-list` argparse args. Plumb through `track_b/openevolve_skills/controller.py` → `iteration.py` → `evaluator.evaluate`. After a winner is accepted on the training set, if `--validation-task-list` is set, run a second eval over the held-out list and record the validation score on the artifact (no re-acceptance gate — recorded only).
- **MODIFY** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/skillsbench/evolve.py` — pass the new flags through; default `--eval-source skillsbench --task-set hot_5 --validation-task-list /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/skillsbench/subset_17.json` when invoked with the new `--canonical` preset. (Verify the subset_17.json path exists during B.8 — if not, pin to wherever it actually lives.)
- **MODIFY** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_a/runner.py` — add the same four flags to `_cli()` (line 579) for UX parity inside the new `_add_eval_source_args(parser)` helper (Group A also touches this file — see Cross-group interfaces, section 7d).
- **MODIFY** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/openevolve_skills/iteration.py` — plumb `eval_source`, `eval_set_path`, `judge_model`, `validation_task_list` through to the evaluator call at line 143 (Group A also touches this file at the smoke-gate insertion site between lines 139 and 141 — coordinated in section 7e).
- **MODIFY** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/openevolve_skills/controller.py` — surface the new args from CLI to iteration.

### daycare — Group C (shrink)

- **DELETE** `/Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare/src/daycare/anchor.py`
- **DELETE** `/Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare/src/daycare/controls.py`
- **DELETE** `/Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare/src/daycare/finalize.py`
- **DELETE** `/Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare/src/daycare/watchdog.py`
- **DELETE** `/Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare/src/daycare/daemon.py`
- **DELETE** `/Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare/src/daycare/evolve.py` (after Group A has harvested the prompts/constants; gated via section 7g).
- **DELETE** `/Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare/src/daycare/mutator.py` and `/Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare/src/daycare/leak_scanner.py` (after Group A copies them into `skill_evolve/shared/`).
- **MODIFY** `/Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare/src/daycare/cli.py` — drop `run`, `promote`, `daemon` Click subcommands and their imports. Keep `eval-build`, `doctor`, `runs`. Remove any `from daycare.evolve import …` / `from daycare.anchor import …` / `from daycare.daemon import …` etc.
- **MODIFY** `/Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare/src/daycare/__init__.py` — drop re-exports for deleted modules.
- **MODIFY** `/Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare/pyproject.toml` — drop unused deps that came in only via deleted modules (audit imports of each deleted file first); drop console-script entries for `run/promote/daemon` if they were registered separately.
- **DELETE** test files exclusively covering dropped commands/modules:
  - `/Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare/tests/test_daemon_incremental.py`
  - `/Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare/tests/test_evolve_script_mutations.py`
  - `/Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare/tests/test_mutator.py`
  - `/Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare/tests/test_leak_scanner.py` (if it only exercised daycare's copy — Group C verifies; if it covers the surviving import path keep it).
- **MODIFY** `/Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare/tests/test_verifier.py` — keep tests for `score_single`. **C.9 must grep verifier.py FIRST** to confirm whether `score_bundle` is still present after the run-loop callers leave (verifier.py line 357 currently has it). If `score_bundle` has no surviving callers in the keep-list, remove it and its tests; otherwise keep both. Do NOT assume removal without verification.
- **NO CHANGE** to daycare's `eval_builder.py`, `behavioral_builder.py`, `synth_builder.py`, `corpus.py`, `runner.py`, `providers.py`, `selector.py`, `anonymize.py`, `verifier.py` (modulo the `score_bundle` audit above).

### skill_evolve / daycare — Group D (tests + fixtures)

- **CREATE** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests/__init__.py`
- **CREATE** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests/test_patch_parser_roundtrip.py` — ADD_FILE / EDIT_FILE / DELETE_FILE / REWRITE_FOLDER round-trip.
- **CREATE** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests/test_bundle_ops_smoke.py` — smoke-test rejection paths, MAX_BUNDLE_TOKENS / MAX_SKILL_TOKENS guards, macOS `._` filter, `--no-smoke-test` bypass.
- **CREATE** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests/test_sentinel_edge_cases.py` — unterminated `<<<END_FILE>>>`, path traversal (`../etc/passwd`), absolute paths, empty body, duplicate ADD_FILE in same patch, ADD_FILE on path that already exists in parent bundle, REWRITE_FOLDER token-cap behavior.
- **CREATE** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_a/tests/test_llm_deepseek_reasoning.py` — content=None+reasoning="…" returns the reasoning string; both empty returns "".
- **CREATE** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/behavioral/tests/__init__.py`
- **CREATE** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/behavioral/tests/test_adapter_smoke.py` — feed a 2-record fixture `eval_set.jsonl` plus a stub judge LLM, assert EvalResult shape.
- **CREATE** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/behavioral/tests/fixtures/tiny_eval_set.jsonl` — two daycare-format records.
- **CREATE** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_iteration_smoke_guard.py` — synthetic LLM emits a bad-script ADD_FILE; iteration loop rejects without calling the evaluator.
- **CREATE** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_validation_holdout.py` — synthetic LLM produces a winner; validation list scored separately and recorded on artifact.
- **CREATE** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/fixtures/mock_skillsbench/` — minimal fake scene/task layout (3 task IDs, deterministic pass/fail) for the end-to-end integration test.
- **CREATE** `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_end_to_end_one_iter.py` — runs one iteration of `track_b/run.py` end-to-end against `mock_skillsbench/` with stub OpenRouter + sentinel-block patch format; asserts artifact written, smoke gate ran, validation list scored.

## 4. Implementation steps (4 parallel groups)

Each group runs on its own git branch (see section 8). Coordination points are listed in section 7 — outside of those, each group is independently parseable and executable.

Group A merges first into `script-mutation`. Groups B and D.1-D.5 build against the section-7-locked API in parallel and can run from day 0. D.6-D.9 are gated on A's signal AND B's CLI-flag commits. Group C runs in two phases: C1 (parallel from day 0) and C2 (gated on A's signal commit). Section 8 has the explicit ordering diagram.

---

### Group A — skill_evolve gains sentinel-block patch capability + smoke guard + DeepSeek fallback

Owner branch: `script-mutation/group-a` (squash-merged into `script-mutation` first).

A.1. Read `/Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare/src/daycare/mutator.py` end-to-end and `/Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare/src/daycare/evolve.py` lines 66-115 (constants + `_bundle_tokens` + `list_scripts`) and 620-692 (template lines 620-681, rendering lines 682-685, assertions lines 686-692).

A.2. Read `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/openevolve_skills/patch_parser.py` end-to-end. Compare against daycare's parser. The shared module's primary names follow daycare conventions (`parse_sentinel_blocks`, `FileOp`, `SentinelParseError`); track_b's existing names (`parse_patch`, `Operation`, `PatchParseError`) become aliases per section 7a. Verify by running `python3 -m pytest /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_patch_parser.py -x` AFTER the shim lands (step A.4) — it must pass without edits to the test file.

A.3. Create `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/__init__.py` and `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/patch_parser.py`. Move the parser body into shared; primary names = daycare style; add the back-compat aliases (`parse_patch = parse_sentinel_blocks`, `Operation = FileOp`, `PatchParseError = SentinelParseError`). For the per-kind names (`AddFile`, `EditFile`, `DeleteFile`, `RewriteFolder`) define DISTINCT empty subclasses of `FileOp` — NOT simple aliases — because `track_b/tests/test_patch_parser.py` asserts both `isinstance(ops[0], AddFile)` (which requires distinct classes to discriminate) AND `[type(o).__name__ for o in ops] == ["EditFile", "DeleteFile"]` (which requires `type(o).__name__` to actually be the kind name). The parser must dispatch on the parsed op string and instantiate the matching subclass at construction time so `type(instance).__name__` matches the op kind. Concretely:

```python
@dataclass
class FileOp:
    op: Literal["ADD_FILE", "EDIT_FILE", "DELETE_FILE", "REWRITE_FOLDER"]
    path: str
    content: str | None = None
    files: dict[str, str] | None = None

@dataclass
class AddFile(FileOp): pass
@dataclass
class EditFile(FileOp): pass
@dataclass
class DeleteFile(FileOp): pass
@dataclass
class RewriteFolder(FileOp): pass
# parser dispatches op string → subclass at construction time
```

Daycare-side consumers can keep using `FileOp` (parent catches all); track_b's `isinstance(x, AddFile)` and `type(x).__name__` checks work because each instance is constructed as the proper subclass. Replace `track_b/openevolve_skills/patch_parser.py` with a `from skill_evolve.shared.patch_parser import *  # re-export` shim plus explicit `__all__` covering both naming sets.

A.4. Run `python3 -m pytest /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_patch_parser.py -x` — MUST pass with the shim. If it fails because a known alias is missing, add the alias and re-run; do NOT modify the existing test file.

A.5. Create `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/bundle_ops.py`. Port:
- `_bundle_tokens` (daycare evolve.py:79-102) → `bundle_tokens(bundle_dir: Path) -> int`. Inline-filter `._*` files and `__pycache__` dirs inside the walk. Add `MAX_SKILL_TOKENS=3000`, `MAX_BUNDLE_TOKENS=60000` module-level constants.
- `list_scripts` (daycare evolve.py:105-115) → keep filter for `._*`.
- `apply_ops` (daycare mutator.py:208) → renamed `apply_file_ops`. Also port `parse_and_apply` (daycare mutator.py:408), `shebang_insurance`, `validate_scripts`, `hash_bundle` — adjust imports to use `skill_evolve.shared.patch_parser` for the `FileOp` dataclass.
- `validate_scripts` returns a `SmokeResult` dataclass with `ok: bool`, `failures: list[(Path, str)]`. Use `py_compile.compile(path, doraise=True)` for `.py` and `subprocess.run(["bash","-n",str(p)], capture_output=True)` for `.sh`. Collect ALL failures (not first-fail short-circuit) for the artifact.
- Define `BundleRejected(Exception)` here too — used by `folder.write()` to signal smoke/cap rejection.

A.5b. Create `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/leak_scanner.py` as a verbatim copy of `watchmen-fukara/daycare/src/daycare/leak_scanner.py`. Re-export it from `skill_evolve.shared.__init__` so any track_b caller that previously reached into daycare resolves locally.

A.6. Port the proposer system prompt from daycare evolve.py template lines 620-681 (rendered via `.format(...)` at lines 682-685) into `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_a/prompts.py` as `SENTINEL_PROPOSER_SYSTEM_PROMPT`. Include the HARD LIMITS block (verbatim `MAX_SKILL_TOKENS` / `MAX_BUNDLE_TOKENS` numbers) and the CRITICAL `<<<END_FILE>>>` warning. Convert the daycare assertion block at lines 686-692 from bare `assert` to explicit `raise RuntimeError(...)`:

```
def _assert_prompt_well_formed() -> None:
    if str(MAX_SKILL_TOKENS) not in SENTINEL_PROPOSER_SYSTEM_PROMPT:
        raise RuntimeError("MAX_SKILL_TOKENS must appear verbatim in the proposer prompt")
    if str(MAX_BUNDLE_TOKENS) not in SENTINEL_PROPOSER_SYSTEM_PROMPT:
        raise RuntimeError("MAX_BUNDLE_TOKENS must appear verbatim in the proposer prompt")

_assert_prompt_well_formed()
```

Reason: bare `assert` is stripped under `python -O`; we want this check live in all runtime modes.

A.7. Add four new op classes in `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_a/ops.py` (`AddFileOp`, `EditFileOp`, `DeleteFileOp`, `RewriteFolderOp`). Each `apply()` delegates to `skill_evolve.shared.bundle_ops.apply_file_ops` with a single-item list. Place after `apply_rewrite_content` (~line 393). Register in the dispatch at `apply_op()` (line 595) by adding `elif name == "add_file": …`, `elif name == "edit_file": …`, etc. — there is NO `_OP_REGISTRY` symbol; dispatch is a name-based `if/elif` chain.

A.8. Add `--patch-format {json-ops,sentinel-blocks}` arg in `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_a/runner.py` `_cli()` (line 579) inside a new `_add_patch_args(parser)` helper called from `_cli()` (see section 7d). When sentinel-blocks: select `SENTINEL_PROPOSER_SYSTEM_PROMPT`, route the LLM response through `shared.patch_parser.parse_sentinel_blocks`, then `shared.bundle_ops.apply_file_ops`.

A.9. Add `--smoke-test / --no-smoke-test` arg (default on). Insert smoke + token-cap check inside `SkillFolder.write(dest)` in `track_a/folder.py` (line 187, returns `dest` at line 206). Specifically: right before `return dest` at line 206, if `smoke_test_enabled`, call `validate_scripts(dest)` and `bundle_tokens(dest)`. On failure: raise `BundleRejected(reason)`. Catch in the runner's pass loop, log reason, write to artifact, return a sentinel "rejected" candidate (no eval call). The `--no-smoke-test` bypass simply skips the validate/bundle_tokens call and lets the candidate proceed unchanged.

A.10. Add unified DeepSeek reasoning fallback to `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_a/llm.py` chat-response extraction. Pattern: `text = msg.get("content") or msg.get("reasoning") or ""`. Mirror the warning log shape from `track_b/openevolve_skills/llm_client.py` (around lines 157-159). Cover both OpenAI-style (`msg.content`) and OpenRouter-style (`msg["content"]`) response shapes. ALSO add (if missing) the `extra_body={"reasoning": {"max_tokens": …}}` budget cap mirroring track_b/llm_client.py lines 151-154 — verify on read whether track_a's client wrapper already does this; if so, skip the cap part and just add the short-circuit.

A.11. In `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/openevolve_skills/iteration.py`, insert smoke + token-cap gate BETWEEN `mutate(...)` (line 92) and `evaluator.evaluate_artifact(...)` (line 143). Concretely: call `_run_smoke_gate(child_artifact) -> Optional[str]` immediately AFTER the existing anonymize block ends (current line 139, the `# kai-skills patch end` comment) and BEFORE line 141's `# 5. Evaluate child.` comment. Return shape on rejection: `IterationResult(op_type="parse_error", notes=f"smoke_rejected: {reason}", score_delta=0.0, child_id=None)`.

A.12. Run `uvx ruff check /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_a /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b --fix` then `uvx ruff format` on the same paths.

A.13. Run `python3 -m pytest /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_a/tests /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests -x`.

A.14. Push the **signal commit** to `script-mutation` (not the subbranch) with the exact message specified in section 7g, so Group C can advance from Phase C1 to Phase C2.

**Acceptance criteria — Group A:**

- `python3 -m pytest /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_patch_parser.py` passes (re-export shim + alias names work).
- `python3 -m pytest /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_a/tests` passes.
- `uvx ruff check /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_a` returns clean.
- `python3 -c "from skill_evolve.shared.bundle_ops import bundle_tokens, validate_scripts, MAX_BUNDLE_TOKENS; assert MAX_BUNDLE_TOKENS == 60000"` exits 0.
- `python3 -c "from skill_evolve.track_a.prompts import SENTINEL_PROPOSER_SYSTEM_PROMPT; assert '<<<END_FILE>>>' in SENTINEL_PROPOSER_SYSTEM_PROMPT and '60000' in SENTINEL_PROPOSER_SYSTEM_PROMPT"` exits 0.
- Signal commit landed on `script-mutation` per section 7g.

---

### Group B — `--eval-source` flag + behavioral adapter + validation split

Owner branch: `script-mutation/group-b`. Depends on the dataclass + module-name agreements in section 7 but NOT on Group A's actual file moves; Group B builds against agreed import paths and verifies via mocks until Group A merges.

B.1. Read `/Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare/src/daycare/verifier.py` end-to-end (especially `score_single` at line 140 — note the return type is `float | None`, not a dict — and the content/reasoning short-circuit at line 189). Read `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/evaluator.py` `evaluate(...)` (~line 639) and `EvalResult` dataclass.

B.2. Create `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/behavioral/__init__.py`, `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/behavioral/eval_set.py`, `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/behavioral/adapter.py`.

B.3. In `behavioral/eval_set.py`: `EvalItem` dataclass (id, prompt, rubric, expected_action, metadata) plus `load_eval_set(path: Path) -> list[EvalItem]`. JSONL one-record-per-line. Skip blank lines. Raise `EvalSetFormatError` (defined locally) on missing required keys.

B.4. In `behavioral/adapter.py`: `score_bundle_behavioral(bundle_dir, eval_set_path, judge_model, *, repeats=1, llm=None) -> EvalResult`. For each item, render the SKILL.md + scripts into a prompt — use `skill_evolve.shared.bundle_ops.list_scripts()` and `bundle_tokens()` to iterate bundle files (which filter `._*` AppleDouble files and `__pycache__/`); DO NOT walk the directory directly. Call the judge LLM (default: `track_b.openevolve_skills.llm_client.OpenRouterLLM`), apply `msg.get("content") or msg.get("reasoning") or ""` short-circuit (mirroring daycare verifier.py:189), parse a continuous `score: float` in [0, 1] from the judge response — port `_parse_judge_response` from daycare verifier.py. The judge returns a float per item (NOT a binary verdict). Aggregate `success_rate` (count of `score >= threshold` / N), `composite`, `mean_score`, `per_task`, `failures` matching `skill_evolve.evaluator.EvalResult`. Composite fitness derives from mean of `score` floats.

B.5. Modify `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/evaluator.py` `evaluate(...)`:
- Add params `eval_source: Literal["skillsbench","behavioral"]="skillsbench"`, `eval_set_path: Optional[Path]=None`, `judge_model: Optional[str]=None`.
- Dispatch: `if eval_source == "behavioral": return score_bundle_behavioral(...)`. Else fall through to existing skillsbench path.
- Validate: if `eval_source == "behavioral"` but `eval_set_path is None`, raise `ValueError("--eval-set required when --eval-source behavioral")`.

B.6. Modify `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/run.py`:
- Add `--eval-source`, `--eval-set`, `--judge-model`, `--validation-task-list` argparse args.
- Plumb to `controller.py` → `iteration.py` → `evaluator.evaluate`.
- After a winner is accepted on the training set, if `--validation-task-list` is set, re-evaluate the accepted bundle against the validation list with the *same* `eval_source` and record `artifact.validation_score`. No re-acceptance gate.

B.7. Modify `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/openevolve_skills/controller.py` and `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/openevolve_skills/iteration.py` to thread the four new args. In iteration.py, B's edit goes at the `evaluator.evaluate_artifact(...)` call site (line 143); A's smoke gate insertion is upstream at lines 139-140. Both groups MUST cite their exact insertion line in their PR description.

B.8. Modify `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/skillsbench/evolve.py`:
- Add `--canonical` flag. When set, inject the consolidated config. Default task-set names resolve via the table in section 2: `hot_5` → `/Users/atakantekparmak/Desktop/work/kai-skills/runs/skillsbench_baseline_v2/hot_5.json`; `subset_17` → `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/skillsbench/subset_17.json` (verify on read; pin to the actual location if different).
- Canonical args: `--inner-model qwen/qwen3.6-27b --proposer-model deepseek/deepseek-v4-pro --eval-source skillsbench --task-set hot_5 --validation-task-list <subset_17 resolved path> --patch-format sentinel-blocks --smoke-test --max-iters 12 --budget-hours 6 --budget-usd 50`.
- Forward the new flags from the user side too.

B.9. Modify `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_a/runner.py` `_cli()` (line 579) — add the same four flags for UX parity inside the new `_add_eval_source_args(parser)` helper. (Group A also edits this file; see section 7d for conflict resolution.)

B.10. Run `uvx ruff check /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/behavioral /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/skillsbench --fix && uvx ruff format` on the same paths.

B.11. Run `python3 -m pytest /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/behavioral/tests -x`.

**Acceptance criteria — Group B:**

- `python3 -m pytest /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests` passes.
- `python3 -c "from skill_evolve.behavioral.adapter import score_bundle_behavioral; from skill_evolve.evaluator import evaluate; import inspect; assert 'eval_source' in inspect.signature(evaluate).parameters"` exits 0.
- `python3 /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/skillsbench/evolve.py --canonical --dry-run` prints the consolidated config without error.
- `--eval-source behavioral` without `--eval-set` exits non-zero with the documented error.

---

### Group C — daycare shrink (two phases)

Owner branch: `daycare-shrink` (in `watchmen-fukara` repo).

**Phase C1 — parallel from day 0 (does NOT depend on any other group):**

C1.1. Read `/Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare/src/daycare/cli.py` end-to-end. Catalogue every import + every Click subcommand. Confirm `run`, `promote`, `daemon` are the three to drop; confirm `eval-build`, `doctor`, `runs` are the three to keep.

C1.2. For each module listed for deletion (anchor.py, controls.py, finalize.py, watchdog.py, daemon.py), grep across the surviving keep-list (eval_builder, behavioral_builder, synth_builder, corpus, runner, providers, selector, anonymize, verifier, cli) for imports. If any surviving module imports a doomed module, list it in the PR description as a blocker.

C1.3. Delete `anchor.py`, `controls.py`, `finalize.py`, `watchdog.py`, `daemon.py` (always safe — never imported by the surviving keep-list per the audit in C1.2; if C1.2 surfaces a coupling, fix it first).

C1.4. Modify `/Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare/src/daycare/cli.py`: drop `run`, `promote`, `daemon` Click commands. Drop all imports for deleted modules. Verify `@cli.command()` decorators for `eval-build`, `doctor`, `runs` still present.

C1.5. Modify `/Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare/src/daycare/__init__.py`: drop re-exports for deleted modules.

C1.6. Audit `/Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare/pyproject.toml`: walk every dep, grep the surviving codebase, drop any dep no longer imported. Drop entry-point console scripts for the three deleted subcommands if they were separately registered.

C1.7. Delete test files: `tests/test_daemon_incremental.py`. (test_evolve_script_mutations.py and test_mutator.py stay until Phase C2 because they exercise evolve/mutator/leak_scanner which still exist on disk in Phase C1.)

**Phase C2 — gated on Group A's signal commit (see section 7g):**

C2.1. Wait for Group A to land the signal commit `feat(shared): port daycare mutator + sentinel prompt — group-c green light` on `script-mutation` (not on the subbranch). Confirm via `git log script-mutation --oneline`.

C2.2. Delete `evolve.py`, `mutator.py`, `leak_scanner.py`.

C2.3. Delete `tests/test_evolve_script_mutations.py`, `tests/test_mutator.py`. Verify `tests/test_leak_scanner.py` — if it tests daycare's removed copy, delete; if it covers the now-shared path, delete (skill_evolve has its own copy of the tests). 

C2.4. Grep verifier.py for `score_bundle` (line 357 currently). Grep the keep-list for `verifier.score_bundle` callers. If no surviving callers, remove `score_bundle` and any test cases in `tests/test_verifier.py` exercising it. If callers remain, leave both in.

C2.5. Run `uvx ruff check /Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare --fix && uvx ruff format /Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare`.

C2.6. Run `uvx ty check /Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare/src/daycare`. Fix any newly-surfaced `Unknown name` errors caused by the deletions.

C2.7. From `/Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare`: `uv run --extra dev pytest`.

C2.8. Manual smoke: `uv run daycare --help` lists only `eval-build`, `doctor`, `runs`. `uv run daycare eval-build --help` works. `uv run daycare doctor` exits 0 on a clean checkout. `uv run daycare runs list` exits 0.

**Acceptance criteria — Group C:**

- `uv run --extra dev pytest` in daycare passes.
- `uv run daycare --help` shows exactly `eval-build`, `doctor`, `runs` and no others.
- `uvx ruff check /Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare` returns clean.
- `python3 -c "import importlib; [importlib.import_module(f'daycare.{m}') for m in ['cli','eval_builder','behavioral_builder','synth_builder','corpus','runner','providers','selector','anonymize','verifier']]"` exits 0.
- The five Phase-C1 deleted module files do not exist on disk; the three Phase-C2 deleted module files do not exist on disk; `git status` shows their deletions staged.

---

### Group D — tests + integration fixture (two phases)

Owner branch: `script-mutation/group-d`. D.1-D.5 run in parallel with A and B (only depend on the section-7-locked API). D.6-D.9 are gated on A's signal commit AND B's CLI-flag commits. Group D merges last on the skill_evolve side so it asserts the final integrated behavior.

**Phase D1 — parallel from day 0:**

D.1. Create `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests/__init__.py` and `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests/test_patch_parser_roundtrip.py`. Cases:
- ADD_FILE creates a file with the exact body bytes (trailing newline preserved per parser contract).
- EDIT_FILE replaces an existing file's full contents.
- DELETE_FILE removes the file; subsequent `parse_and_apply` re-emit produces no record.
- REWRITE_FOLDER wipes a directory and writes a multi-file body delimited per parser spec.
- Round-trip: apply N ops → serialize bundle → re-parse → equal `hash_bundle` digest.

D.2. Create `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests/test_bundle_ops_smoke.py`. Cases:
- Broken `.py` (`def foo(:`) → `validate_scripts` returns `ok=False`, failure entry mentions the path.
- Broken `.sh` (`if then`) → `bash -n` fails; reported.
- Clean bundle → ok.
- `MAX_SKILL_TOKENS` exceeded → `bundle_tokens` returns total and the higher-level helper raises `BundleRejected`.
- `MAX_BUNDLE_TOKENS` exceeded → same.
- REWRITE_FOLDER body that pushes bundle over `MAX_BUNDLE_TOKENS` → rejected by the token cap before write succeeds (or rejected post-write before eval — confirm order with Group A and assert the actual behavior).
- macOS `._foo.py` AppleDouble file in scripts/ → ignored by `bundle_tokens` and `list_scripts`, even though it exists on disk.
- `--no-smoke-test` bypass path: smoke checks not invoked, evaluator runs even on syntactically broken scripts. (Verified via mocking or a thin wrapper helper that consumes the flag.)

D.3. Create `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests/test_sentinel_edge_cases.py`. The sentinel API locked in section 7a is:

```
parse_sentinel_blocks(text: str, existing_paths: Optional[set[str]] = None) -> list[FileOp]
```

raises `SentinelParseError(kind)` with `kind ∈ {"unterminated", "path_traversal", "absolute_path", "add_existing", "edit_missing", "mixed_content"}`. Cases:
- Unterminated `<<<END_FILE>>>` → `SentinelParseError("unterminated")`.
- Path traversal `../../etc/passwd` → `SentinelParseError("path_traversal")` before any filesystem write.
- Absolute path `/tmp/foo` → `SentinelParseError("absolute_path")`.
- Empty body ADD_FILE → file created empty (no error).
- Duplicate ADD_FILE on same path in the same patch → second invocation raises `SentinelParseError("add_existing")` (locked behavior — second-wins is NOT permitted).
- ADD_FILE on a path that already exists in the PARENT bundle (passed via `existing_paths`) → `SentinelParseError("add_existing")` before any write.
- EDIT_FILE on a path that does NOT exist in the parent bundle → `SentinelParseError("edit_missing")`.
- Mixed content under REWRITE_FOLDER (e.g. content outside `--- file:` headers) → `SentinelParseError("mixed_content")`.
- Multiple ops in one response → parsed as a list in order.

D.4. Create `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_a/tests/test_llm_deepseek_reasoning.py`. Mock the OpenRouter HTTP layer. Assert:
- `{"content": "answer"}` returns `"answer"`.
- `{"content": None, "reasoning": "thoughts"}` returns `"thoughts"` plus a warning log.
- `{"content": "", "reasoning": ""}` returns `""`.

D.5. Create `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/behavioral/tests/__init__.py`, `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/behavioral/tests/fixtures/tiny_eval_set.jsonl` (two records: one expected-high-score, one expected-low-score), and `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/behavioral/tests/test_adapter_smoke.py`. Cases:
- Stub judge LLM returning `{"score": 0.9, "reasoning": "..."}` for both → adapter `EvalResult.mean_score ≈ 0.9`, `per_task` length 2. (Note: judge returns a continuous score float in [0, 1], NOT a binary verdict — matching daycare `verifier.score_single` which returns `float | None`.)
- Stub judge LLM returning DeepSeek-style `{"content": None, "reasoning": "{\"score\": 0.5}"}` → still parsed (reasoning short-circuit wired).
- Missing `--eval-set` raises `ValueError` from `evaluate(...)`.

**Phase D2 — gated on Group A signal commit AND Group B CLI-flag commits:**

D.6. Create `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_iteration_smoke_guard.py`. Use the existing synthetic LLM fixture pattern from `track_b/tests/test_iteration_synthetic.py`. The synthetic LLM emits an ADD_FILE with a broken `.py`. Assert the iteration loop:
- Records a smoke-failure on the artifact.
- Does NOT call the evaluator (mock the evaluator and assert call_count == 0).

D.7. Create `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_validation_holdout.py`. Synthetic LLM produces a clean winner. Run with `--validation-task-list <path-to-2-task-fixture>`. Assert `artifact.validation_score` is recorded and is distinct from `artifact.train_score`.

D.8. Create `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/fixtures/mock_skillsbench/` with:
- 3 fake task IDs (`mock_task_a`, `mock_task_b`, `mock_task_c`).
- A deterministic backend stub (registered via the existing `--agent-backend` plumbing) where `mock_task_a` and `mock_task_b` pass with the right SKILL.md present, `mock_task_c` always fails.
- A `train_3.json` and `val_2.json` task list.

D.9. Create `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_end_to_end_one_iter.py`. Drives `track_b/run.py:main()` programmatically (via a Python entry point, not subprocess) with:
- `--patch-format sentinel-blocks`
- `--eval-source skillsbench`
- `--task-list .../train_3.json`
- `--validation-task-list .../val_2.json`
- Stubbed OpenRouter LLM emitting a one-op ADD_FILE patch.
- Assert: iteration completes, smoke gate ran, evaluator scored, validation score recorded, artifact JSON written to a tmp path.

D.10. Run the full test sweep:
- `python3 -m pytest /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/behavioral/tests /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_a/tests /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests -x`.
- `uvx ruff check /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve --fix --exclude benchmark/vendor`.
- `uvx ruff format /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve --exclude benchmark/vendor`.

**Acceptance criteria — Group D:**

- All new test files exist at the paths listed in section 3.
- Full pytest sweep (line 1 of D.10) exits 0.
- The end-to-end test runs in under 30s (no real network calls).
- `uvx ruff check` (with the vendor exclusion) is clean.

---

## 4b. Second-pass groups (E/F/G/H) — SkillOpt no-gradient ports

Groups E/F/G/H run as a SECOND IMPLEMENTATION PASS after Groups A/B/C/D have landed (and Group A/B/D have passed audit). They port four no-gradient mechanisms from the SkillOpt paper (Microsoft Research, May 2026, arXiv 2605.23904) on top of the consolidation already specified above.

CRITICAL CONTEXT for implementers: SkillOpt is NOT an SFT method. The paper uses deep-learning vocabulary (epochs, lr, validation gate, gradient) but the mechanism is pure text-space optimization with a frozen target model and a frontier optimizer LLM that proposes structured edits to a single `best_skill.md`. Every mechanism below is compatible with the no-gradient constraint that governs this codebase. No model weights are touched.

Sequencing: Round 1 of pass 2 runs E + F in parallel (max 2 agents). Round 2 runs G + H in parallel (max 2 agents). G depends on E (rejected buffer) and F (edit budget) on top of A + B. H depends on A (parser) and F (merged patches feed L_t clip). See section 8's "Second pass" diagram.

Implementers MUST treat Groups A/B/C/D sections (above) as frozen — append-only edits to shared modules; never modify the symbols or semantics those groups locked.

**Cost projection.** Group H doubles proposer calls per iter (+$15-25/run on canonical 12-iter run). Group G adds ~3 consolidator calls per run (+$3-5). Group F's `clip_ops` uses paper's fallback path (parser order, no rank LLM call) — no added cost. Group E uses no extra LLM. **Estimated canonical post-amendment cost: $70-80/run vs ~$50 baseline.** If budget is a concern, set `--reflection-mode single` (disables H) and/or `--slow-update-every 12` (effectively disables G in canonical 12-iter run) to recover the $50 budget. The `--canonical` preset opts INTO the higher cost as the recommended config.

**Wallclock estimate.** Canonical 12-iter run wallclock target: ~4-5h of the 6h budget under qwen3.6-27b thinking-mode (`feedback_qwen_calibration_slow.md` documents 2-5 min/call). Group H's `ThreadPoolExecutor(max_workers=2)` parallelizes the doubled proposer cost so wallclock impact is ~1.1× not 2×. Group G's 3 consolidator fires add ~10-15 min serial. If wallclock pressure observed mid-run, fall back to `--reflection-mode single` (recovers ~30 min of proposer overhead) and/or `--slow-update-every 12` (effectively disables consolidator on canonical 12-iter run).

**Replicability protocol.** Per `feedback_evolution_denominator.md` and `feedback_writeup_retractions.md`, no E/F/G/H lift claim is promotable from a single-roll run. Group E modifies the existing `--rng-seed` arg (track_b/run.py:67 default=0, track_a/runner.py:725 default=None) as the first second-pass group affected by replicability. Group E updates the default to `None` (unseeded by default — opt-in seeding) AND threads `args.rng_seed` into the proposer LLM client's `seed` param (currently the flag only seeds the controller's RNG, not the proposer). F/G/H consume `args.rng_seed` from the controller without further re-declaration. Canonical validation protocol for any reported lift claim: **n ≥ 3 seeded runs minimum**; report median delta and per-seed deltas. Plan_0 §E acceptance criteria (smoke-of-12-iters under strict gate) is a SMOKE check, NOT a promotion gate — promotion requires multi-seed median.

**Context budget.** Canonical-mode proposer prompt is bounded by an explicit token cap to avoid blowing qwen3.6-27b's 128k context window. Worst-case canonical prompt:
- HARD LIMITS + EDIT BUDGET + META-SKILL + RECENT REJECTIONS + CRITICAL preamble: ~3k tokens
- Bundle context (capped at `MAX_BUNDLE_TOKENS=60k`): ≤60k tokens
- Rejected buffer (10 entries × ~500 tokens): ~5k tokens
- Meta-skill (20-iter tail markdown): ~5k tokens
- Failure + success reflection minibatches (≤5 tasks per side × ~1k tokens/trajectory): ~10k tokens
- **Total worst-case: ~83k tokens, leaving ~45k headroom for proposer thinking-mode reasoning output.**

Group E adds `MAX_PROPOSER_PROMPT_TOKENS=90000` (configurable via `--max-proposer-prompt-tokens INT`) as a hard pre-call check. If the rendered prompt exceeds the cap, the rendering helper degrades gracefully in this order: (1) truncate rejected-buffer entries to oldest-first removal until under cap, (2) truncate meta-skill tail by oldest-iter removal, (3) if still over cap, log `prompt_oversize` warning and TRUNCATE bundle context to fit (keeping SKILL.md + frontmatter + first N scripts/* by `list_scripts` order). Per `feedback_qwen_calibration_slow.md`, qwen3.6-27b thinking mode can emit ~5-10k reasoning tokens; the 45k headroom is safely above this.

---

### Group E — Strict validation gate + rejected-edit buffer

Owner branch: `script-mutation/group-e` (second pass; cut from `script-mutation` after A/B/D have merged). Depends on Group B (validation-task-list plumbing already lands the held-out scoring path; E flips its semantics from record-only to gate). Parallel with F.

Motivated by: `feedback_gap_closed_interpretation.md` (gap_closed brittleness on n<30 holdouts — 1 flip ≈ 9-22pp; strict-gate suppresses silent drift acceptance) and `feedback_evolution_denominator.md` (hot_5 5-task denominator amplifies single-roll noise; rejected buffer feeds proposer prior failures, not just successes, so the search exploits both axes).

Paper anchor: `skillopt/evaluation/gate.py:31-73` — strict-> on both `cand_hard > current_score` AND `cand_hard > best_score`; ties REJECTED to prevent silent drift.

E.1. Read `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/openevolve_skills/iteration.py` end-to-end (post-Group-A/B state). Locate the existing acceptance path around the smoke-gate insertion (line 192, see digest) and the validation call site at the `evaluator.evaluate_validation(...)` call at line 260. Confirm that B currently records the validation score without gating (this is what E changes).

E.2. CREATE `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/rejected_buffer.py`. Define:

```
@dataclass(frozen=True)
class RejectedEdit:
    patch_text: str
    delta_train: float
    delta_val: float
    rejection_reason: Literal["train_no_improve", "val_not_strict_gt", "val_tie", "smoke_rejected", "token_cap", "parse_error"]
    iteration: int

class RejectedBuffer:
    def __init__(self, capacity: int = 10): ...
    def push(self, edit: RejectedEdit) -> None: ...          # bounded ring, FIFO eviction
    def render_for_prompt(self) -> str: ...                  # ## RECENT REJECTIONS section body
    def to_jsonl(self, path: Path) -> None: ...              # append-only per-iteration persist
    @classmethod
    def from_jsonl(cls, path: Path, capacity: int = 10) -> "RejectedBuffer": ...
```

Persist format: one JSON object per line with keys `patch_text`, `delta_train`, `delta_val`, `rejection_reason`, `iteration`.

E.3. MODIFY `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/openevolve_skills/iteration.py`. Flip the validation semantics from record-only to strict acceptance gate. Concretely, after the existing `evaluator.evaluate_artifact(...)` call (line 207) computes `train_score` and the existing `evaluator.evaluate_validation(...)` call (line 260) computes `val_score`, branch on the new `--validation-gate` mode threaded through from `track_b/run.py`:
- `strict` (default): accept iff `train_score >= parent_train_score` AND `val_score > best_val_score_seen_so_far`. Ties on val_score REJECTED. Track `best_val_score_seen_so_far` on the run-level state object (controller-owned; see section 7j).
- `record`: preserve the plan_0 Group B behavior — record val_score on artifact, accept on train criterion only.
- `relaxed`: accept iff `train_score >= parent_train_score` AND `val_score >= best_val_score_seen_so_far` (ties accepted).

On rejection, build a `RejectedEdit` from the candidate's parsed patch text + deltas + reason (one of `train_no_improve`, `val_not_strict_gt`, `val_tie` — `smoke_rejected` / `token_cap` / `parse_error` are pushed from their own existing rejection sites added in Group A), push to the run-level `RejectedBuffer`, and persist via `buffer.to_jsonl(run_dir / "rejected_buffer.jsonl")` immediately so a crashed run still has the history.

E.4. MODIFY `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_a/prompts.py`. Add `## RECENT REJECTIONS — DO NOT REPEAT` section to `SENTINEL_PROPOSER_SYSTEM_PROMPT`. The section body is a `{recent_rejections}` `.format()` slot rendered by `RejectedBuffer.render_for_prompt()`. When the buffer is empty render the literal string `(none yet)` so the section never collapses to a bare header. Extend `_assert_prompt_well_formed()` (currently at line 514) to also check `"## RECENT REJECTIONS" in SENTINEL_PROPOSER_SYSTEM_PROMPT`.

E.5. MODIFY `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/run.py`. Add argparse args inside the existing arg block (lines 191-220, after the `--validation-task-list` arg landed by Group B):
- `--validation-gate {strict,record,relaxed}` default `strict`.
- `--rejected-buffer-size INT` default `10`.
- `--max-proposer-prompt-tokens INT` default `90000`. Gates the proposer prompt builder via the context-budget logic from the §4b preamble: the rendering helper enforces this cap and degrades gracefully (rejected-buffer truncate → meta-skill tail truncate → bundle truncate) before any LLM call. Threaded through `controller.py` → `iteration.py` and consumed by the prompt-rendering helper that G's consolidator path and H's reflection paths both call.
- MODIFY existing `--rng-seed` (track_b/run.py:67) default 0 → None and thread into proposer LLM client `seed` param. The flag itself already exists; this step changes its semantics from RNG-only seeding to RNG + LLM seeding. When set, controller seeds `random.seed(args.rng_seed)` AND threads the seed into the proposer LLM client's `seed` param (for OpenRouter request-level determinism where the provider supports it). Per the replicability paragraph in §4b preamble, F/G/H reuse this flag without re-declaring it.

Thread through `controller.py` → `iteration.py`. The controller instantiates a single `RejectedBuffer(capacity=args.rejected_buffer_size)` and passes the instance into each `run_iteration(...)` call.

E.6. MODIFY `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/skillsbench/evolve.py`. Extend the `--canonical` preset (flag at line 199, main() at line 526) to inject `--validation-gate strict` and `--rejected-buffer-size 10`.

E.7. CREATE `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests/test_rejected_buffer.py`. Cases: capacity-bounded ring (push 12, len == 10, oldest evicted), JSONL round-trip via `to_jsonl` + `from_jsonl`, `render_for_prompt` returns `(none yet)` when empty, render contains each pushed `patch_text` head (first 200 chars) when non-empty.

E.8. CREATE `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_validation_gate.py`. Mirror the existing `track_b/tests/test_iteration_smoke_guard.py` pattern (monkeypatch `SyntheticLLM`, mock evaluator). Cases:
- strict: train_up + val_up → accepted; train_up + val_tie → rejected; train_up + val_down → rejected; train_down → rejected.
- record: train_up + val_tie → accepted (plan_0 behavior preserved).
- relaxed: train_up + val_tie → accepted; train_up + val_down → rejected.
- Each rejection path pushes exactly one `RejectedEdit` with the expected `rejection_reason`.

E.9. Run `uvx ruff check /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_a /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/skillsbench --fix --exclude benchmark/vendor` then `uvx ruff format` on the same paths.

E.10. Run `python3 -m pytest /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests -x`.

**Acceptance criteria — Group E:**

- `python3 -m pytest /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests/test_rejected_buffer.py /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_validation_gate.py -x` passes (E.7 + E.8 cover bounded ring, JSONL round-trip, all three gate modes, push-per-rejection).
- Existing `python3 -m pytest /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_validation_holdout.py -x` (Group D's record-mode test) still passes with `--validation-gate record` explicit (E preserves D's plan_0 contract under `record`).
- `uvx ruff check /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b --exclude benchmark/vendor` returns clean.
- `python3 -c "from skill_evolve.shared.rejected_buffer import RejectedBuffer, RejectedEdit; b = RejectedBuffer(capacity=3); [b.push(RejectedEdit(patch_text='x', delta_train=0.0, delta_val=0.0, rejection_reason='val_tie', iteration=i)) for i in range(5)]; assert b.render_for_prompt() != '(none yet)'"` exits 0.
- `python3 -c "from skill_evolve.track_a.prompts import SENTINEL_PROPOSER_SYSTEM_PROMPT; assert '## RECENT REJECTIONS' in SENTINEL_PROPOSER_SYSTEM_PROMPT"` exits 0.
- `python3 -c "from track_b.run import build_parser; p = build_parser(); ns = p.parse_args(['skill', '--max-proposer-prompt-tokens', '50000']); assert ns.max_proposer_prompt_tokens == 50000"` exits 0 (verifies the new `--max-proposer-prompt-tokens` flag parses correctly and binds to `args.max_proposer_prompt_tokens`).
- Smoke run of 12 iters under `--validation-gate strict` on `hot_5` + `subset_17` produces ≥1 acceptance OR the run logs an explicit `gate_stuck` warning recommending `--validation-gate relaxed`.
- Cross-group gate: E ships before G can consume `RejectedBuffer` for the consolidator path (see 7l).

---

### Group F — Bounded edit budget L_t + scheduler

Owner branch: `script-mutation/group-f` (second pass; parallel with E). Depends on Group A (sentinel parser is the source of the parsed-ops list that gets clipped). Independent of E.

Paper anchor: `skillopt/optimizer/clip.py` (`rank_and_select(edits, max_edits)`) + `skillopt/optimizer/scheduler.py` (`CosineScheduler._compute_lr`: `lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t))`).

F.1. Read `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/patch_parser.py` to confirm `parse_sentinel_blocks(...)` returns a `list[FileOp]` whose length is the count subject to clipping (this is the post-A landed contract).

F.2. CREATE `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/edit_budget.py`. Define:

```
ScheduleSpec = str  # "constant:N" | "linear:N->M" | "cosine:N->M"

def parse_schedule(spec: str) -> tuple[Literal["constant","linear","cosine"], int, int]:
    """Parse 'constant:8' -> ('constant', 8, 8); 'linear:8->2' -> ('linear', 8, 2); 'cosine:8->2' -> ('cosine', 8, 2)."""

def compute_lt(spec: str, iter_n: int, max_iters: int) -> int:
    """
    constant: returns N.
    linear:   floor(N + (M - N) * t), t = iter_n / max(1, max_iters - 1), clamped [min(N,M), max(N,M)].
    cosine:   M + 0.5 * (N - M) * (1 + cos(pi * t)), t in [0, 1], rounded to nearest int, clamped.
    Mirrors paper's _compute_lr but returns int (edit count, not lr).
    """

def clip_ops(ops: list, lt: int) -> list:
    """Mirror skillopt.optimizer.clip.rank_and_select fallback: if len(ops) <= lt return ops unchanged; else truncate to first lt. No optimizer-rank pass — we keep the paper's fallback path (parser order = proposer order = priority order)."""
```

Validation in `parse_schedule`: reject malformed input (raises `ValueError("invalid edit-budget spec: <spec>")`). Cosine + linear with `N == M` collapse to constant behavior, NOT an error.

F.3. MODIFY `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/openevolve_skills/iteration.py`. After sentinel parsing (parser is called inside the existing pre-smoke flow) and BEFORE the smoke gate at line 192, call `clip_ops(parsed_ops, compute_lt(args.edit_budget, iter_n, max_iters))`. If `len(parsed_ops) > lt`, log a single warning `edit_budget: clipped {parsed_count} -> {lt} ops at iter {iter_n}` and persist the clipped count + original count on the iteration artifact. The clipped list (NOT the original) feeds the apply path.

F.4. MODIFY `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_a/prompts.py`. Add a `## EDIT BUDGET` block to `SENTINEL_PROPOSER_SYSTEM_PROMPT` with a `{edit_budget_line}` `.format()` slot rendered by the caller as e.g. `Current L_t = 6 (cosine:8->2 at iter 3 of 12). Emit at most 6 file ops; surplus ops will be silently truncated in order.` Extend `_assert_prompt_well_formed()` (line 514) to also assert `"## EDIT BUDGET" in SENTINEL_PROPOSER_SYSTEM_PROMPT`.

F.5. MODIFY `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/run.py`. Add argparse arg `--edit-budget STR` default `cosine:8->2` (matching paper default). Thread through `controller.py` → `iteration.py`.

F.6. MODIFY `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/skillsbench/evolve.py`. Extend the `--canonical` preset to inject `--edit-budget cosine:8->2`.

F.7. CREATE `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests/test_edit_budget.py`. Cases:
- `parse_schedule("constant:5") == ("constant", 5, 5)`.
- `parse_schedule("linear:8->2") == ("linear", 8, 2)`; `parse_schedule("cosine:8->2") == ("cosine", 8, 2)`.
- `parse_schedule("garbage")` raises `ValueError`.
- `compute_lt("constant:5", k, 12) == 5` for k in {0, 6, 11}.
- `compute_lt("linear:8->2", 0, 12) == 8`; `compute_lt("linear:8->2", 11, 12) == 2`; midpoint within [2, 8].
- `compute_lt("cosine:8->2", 0, 12) == 8`; `compute_lt("cosine:8->2", 11, 12) == 2`; cosine midpoint ≈ 5.
- `clip_ops([a,b,c,d], 2) == [a, b]`; `clip_ops([a,b], 5) == [a, b]` (length-stable when under budget).
- `compute_lt("cosine:5->5", 3, 12) == 5` (degenerate range collapses to constant).

F.8. CREATE `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_iteration_edit_budget.py`. Monkeypatch `SyntheticLLM` to emit a 5-op sentinel patch; run iteration with `--edit-budget constant:2`; assert the apply path saw exactly 2 ops, the iteration artifact records `edit_budget.parsed = 5` and `edit_budget.applied = 2`, and the evaluator was called once (clipping does NOT short-circuit the evaluator the way smoke rejection does).

F.9. Run `uvx ruff check /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_a /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/skillsbench --fix --exclude benchmark/vendor` then `uvx ruff format`.

F.10. Run `python3 -m pytest /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests -x`.

**Acceptance criteria — Group F:**

- `python3 -m pytest /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests/test_edit_budget.py /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_iteration_edit_budget.py -x` passes (F.7 + F.8 cover all three schedules, both endpoints, midpoints, degenerate range, and the iteration-level clip + artifact-record behavior).
- `uvx ruff check /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b --exclude benchmark/vendor` returns clean.
- `python3 -c "from skill_evolve.shared.edit_budget import compute_lt; assert compute_lt('cosine:8->2', 0, 12) == 8 and compute_lt('cosine:8->2', 11, 12) == 2"` exits 0.
- `python3 -c "from skill_evolve.track_a.prompts import SENTINEL_PROPOSER_SYSTEM_PROMPT; assert '## EDIT BUDGET' in SENTINEL_PROPOSER_SYSTEM_PROMPT"` exits 0.
- Cross-group gate: F ships before G (G's fence-respecting consolidator clip and H's failure/success merge BOTH consume `clip_ops` + `compute_lt`).

---

### Group G — Slow-update protected region + meta-skill consolidator

Owner branch: `script-mutation/group-g` (second pass; depends on A, B, E, F). Parallel with H.

Motivated by: `feedback_skill_md_saturation.md` — both daycare runs plateau immediately after their first win (9 subsequent iters → 0 winners, ceiling ~0.40-0.41). Group G directly attacks this saturation plateau by introducing a separate slow-update channel (meta_skill.md + protected SKILL.md region) that accumulates cross-iteration lessons the step-level proposer cannot overwrite.

DESIGN DECISIONS (locked, do NOT relitigate during implementation):
- Protected region in SKILL.md ONLY — paper-faithful. `scripts/*` and `references/*` stay fully mutable so Group A's full-bundle mutation is preserved.
- Fence markers, EXACT paper strings: `<!-- SLOW_UPDATE_START -->` and `<!-- SLOW_UPDATE_END -->`.
- Position: END of SKILL.md (paper-faithful; mirrors `skillopt/optimizer/slow_update.py:inject_empty_slow_update_field` which appends via `return skill.rstrip() + block`). The inner model still sees the fence because it consumes the full SKILL.md context. Concrete injection shape: `return skill.rstrip() + "\n\n" + SLOW_UPDATE_START + "\n" + content + "\n" + SLOW_UPDATE_END + "\n"`.
- Step-level (fast) proposer is FORBIDDEN from edits whose target lines fall inside the fence. The sentinel parser detects the violation and raises `SentinelParseError("slow_update_violation")` — a new error kind added to the section 7b enum (see section 7l for the extended enum).
- Consolidator (slow) proposer fires every K iters (`--slow-update-every K`, default K=4). Receives accepted + rejected history (from `RejectedBuffer` + accepted-edit log) and the current SKILL.md, writes a SINGLE update that lives ENTIRELY inside the fence. Goes through the strict validation gate from Group E (same accept/reject path as fast edits).
- Separate `meta_skill.md` at bundle ROOT. NOT deployed at inference. Prepended to the proposer system prompt only. Accumulates edit-pattern stats + persistent failure signatures across iterations.
- At promotion / validation deployment: `meta_skill.md` is excluded from the deployed bundle.

Paper anchor: `skillopt/optimizer/slow_update.py` (`SLOW_UPDATE_START`, `SLOW_UPDATE_END`, `has_slow_update_field`, `inject_empty_slow_update_field`, `extract_slow_update_field`, `replace_slow_update_field`) + `skillopt/optimizer/skill.py:_is_in_slow_update_region` (step edits whose target falls inside fence are skipped with `report["status"] = "skipped_protected_slow_update_region"`).

G.1. Read paper's `skillopt/optimizer/slow_update.py` (per the digest's verbatim function names) and `skillopt/optimizer/skill.py:_is_in_slow_update_region`. Confirm the marker strings ARE `<!-- SLOW_UPDATE_START -->` / `<!-- SLOW_UPDATE_END -->` and not a variant.

G.2. CREATE `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/slow_update.py`. Define:

```
SLOW_UPDATE_START = "<!-- SLOW_UPDATE_START -->"
SLOW_UPDATE_END = "<!-- SLOW_UPDATE_END -->"

def has_slow_update_field(skill_md_text: str) -> bool: ...

def inject_empty_slow_update_field(skill_md_text: str) -> str:
    """If absent, append the empty fence block at the END of SKILL.md (paper-faithful;
    mirrors paper's `return skill.rstrip() + block`). Concrete shape:
    `return skill.rstrip() + "\n\n" + SLOW_UPDATE_START + "\n" + content + "\n" + SLOW_UPDATE_END + "\n"`
    where `content` is an empty string for the initial inject."""

def extract_slow_update_field(skill_md_text: str) -> str:
    """Return the content strictly between the markers. Raise ValueError if exactly one marker is present."""

def replace_slow_update_field(skill_md_text: str, new_content: str) -> str:
    """Replace content between markers; preserve markers verbatim. Inject if absent first."""

def is_in_slow_update_region(skill_md_text: str, target_line: int) -> bool:
    """Return True iff target_line (1-based) falls between the markers."""
```

Mirror SkillOpt's signatures one-to-one. All five functions are pure-string; no filesystem.

G.3. MODIFY `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/patch_parser.py`. Extend `parse_sentinel_blocks` (line 232) to detect SLOW_UPDATE violations: for any EDIT_FILE / DELETE_FILE / REWRITE_FOLDER op whose `path` resolves to the bundle's SKILL.md AND whose effective edit range overlaps the fence, raise `SentinelParseError("slow_update_violation")`. The parser receives the parent bundle's SKILL.md text via a new optional kwarg `parent_skill_md: Optional[str] = None` — when None (back-compat path used by the existing daycare-side callers), the check is skipped. Add `"slow_update_violation"` to the error-kinds enum at line 68 (extending the existing `{"unterminated", "path_traversal", "absolute_path", "add_existing", "edit_missing", "mixed_content"}` set). See section 7l.

G.4. CREATE `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/meta_skill.py`. Define:

```
@dataclass
class MetaSkillEntry:
    iteration: int
    summary: str                         # one-line natural-language note written by consolidator
    edit_pattern_stats: dict[str, int]   # e.g. {"ADD_FILE_scripts": 4, "EDIT_FILE_SKILL.md": 2}
    persistent_failures: list[str]       # task ids that have failed >= N consecutive iters

class MetaSkill:
    def __init__(self, path: Path): ...
    def append(self, entry: MetaSkillEntry) -> None: ...      # markdown append (## Iteration N section)
    def render_for_prompt(self) -> str: ...                   # produces the ## META-SKILL block body
    @classmethod
    def load(cls, path: Path) -> "MetaSkill": ...
```

Persistence: `meta_skill.md` is a literal markdown file at bundle root. The `render_for_prompt` method reads the file. The `append` method appends a `## Iteration N` section with bullet points. Storage is markdown (NOT JSONL) so the consolidator can edit it directly; rolling history is bounded to the last `--meta-skill-max-iters` entries (default 20) via tail-truncation on `append`.

G.5. MODIFY `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_a/prompts.py`. Two changes:
- Add a `## META-SKILL` block to `SENTINEL_PROPOSER_SYSTEM_PROMPT` with a `{meta_skill_body}` `.format()` slot, rendered from `MetaSkill.render_for_prompt()`. When meta_skill.md does not yet exist (first iteration), render `(empty — no consolidated patterns yet)`.
- Add a new top-level constant `CONSOLIDATOR_PROPOSER_SYSTEM_PROMPT` — the slow-update proposer's system prompt. It must (a) explicitly cite the SLOW_UPDATE_START/END markers verbatim, (b) instruct the LLM to emit EXACTLY ONE `EDIT_FILE SKILL.md` sentinel block whose body is the full SKILL.md with new content inside the fence, (c) cite the accepted + rejected edit history slot `{edit_history}`, (d) cite the persistent-failures slot `{persistent_failures}`.
- Extend `_assert_prompt_well_formed()` (line 514) to assert: `"## META-SKILL" in SENTINEL_PROPOSER_SYSTEM_PROMPT`, `"<!-- SLOW_UPDATE_START -->" in CONSOLIDATOR_PROPOSER_SYSTEM_PROMPT`, `"<!-- SLOW_UPDATE_END -->" in CONSOLIDATOR_PROPOSER_SYSTEM_PROMPT`.

G.6. MODIFY `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/openevolve_skills/iteration.py`. Every K iterations (where K = `args.slow_update_every`), AFTER the normal fast-edit accept/reject branch but BEFORE the next iteration loops, run the consolidator path:
- Build the consolidator prompt from `CONSOLIDATOR_PROPOSER_SYSTEM_PROMPT` with `{edit_history}` = recent accepted edits + `RejectedBuffer.render_for_prompt()` and `{persistent_failures}` = task ids failing in ≥ `persistent_failure_window` consecutive iters (default 3).
- Call the consolidator LLM (`args.consolidator_model`, defaulting to the proposer model if unset).
- Parse via `parse_sentinel_blocks(consolidator_text, parent_skill_md=current_skill_md, existing_paths={...})`. The single op MUST be `EDIT_FILE SKILL.md` whose replacement keeps fence markers and only mutates content between them. Any other op shape is rejected with `op_type="parse_error", notes="consolidator_invalid_op_shape"`.
- Apply ONLY inside the fence: programmatically extract the proposed in-fence content via `extract_slow_update_field(proposed_new_skill_md)`, then `replace_slow_update_field(current_skill_md, that_content)`. This double-guard ensures the consolidator cannot accidentally edit outside the fence even if its sentinel block claims to.
- Run the same strict validation gate from Group E. On accept, also `MetaSkill.append(entry)` with summary + stats + persistent failures and persist.
- On reject, push to `RejectedBuffer` with reason `val_not_strict_gt` / `val_tie` / `train_no_improve` exactly as fast edits.

G.7. MODIFY `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/run.py`. Add argparse args:
- `--slow-update-every INT` default `4`.
- `--meta-skill-path PATH` default `<bundle_root>/meta_skill.md`.
- `--consolidator-model SLUG` default `None` (fall back to proposer model).
- `--meta-skill-max-iters INT` default `20`.
- `--persistent-failure-window INT` default `3`.

G.8. MODIFY `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/skillsbench/evolve.py`. Extend the `--canonical` preset to inject `--slow-update-every 4 --meta-skill-max-iters 20 --persistent-failure-window 3`.

G.9. MODIFY the deployment-strip path. Find all bundle-promotion / bundle-deployment call sites (start by grepping for `validate_scripts(` and the validation-eval call site at `track_b/openevolve_skills/evaluator.py:326`). At every site that copies a bundle for scoring or promotion, exclude `meta_skill.md` from the copied tree. Add a single helper `skill_evolve.shared.bundle_ops.copy_for_deployment(src: Path, dst: Path) -> None` that wraps `shutil.copytree` with the `meta_skill.md` exclusion (also keep the existing `._*` and `__pycache__/` filters from `list_scripts`). Update each site to call this helper instead of raw `copytree`.

G.10. CREATE `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests/test_slow_update.py`. Cases:
- `has_slow_update_field` true/false.
- `inject_empty_slow_update_field` appends block at END of SKILL.md (paper-faithful; verify trailing newline + position relative to `rstrip()` of the input).
- `extract_slow_update_field` returns inner content; raises on half-fence (only START or only END).
- `replace_slow_update_field` preserves markers and bytes outside fence.
- `is_in_slow_update_region(text, line)` boundary behavior at marker lines.

G.11. CREATE `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests/test_patch_parser_slow_update.py`. Cases:
- Step-level `EDIT_FILE SKILL.md` whose body would overwrite the fence → `SentinelParseError("slow_update_violation")`.
- Step-level `EDIT_FILE SKILL.md` whose body changes content OUTSIDE the fence → accepted.
- Step-level `EDIT_FILE scripts/foo.py` → unaffected (fence only protects SKILL.md).
- Back-compat: `parent_skill_md=None` → check skipped.

G.12. CREATE `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_consolidator_path.py`. Monkeypatch `SyntheticLLM` to alternate between fast-edit responses and a consolidator response on every Kth iter. Assert: every K iters the consolidator path runs, meta_skill.md is appended, the fence content changes, content outside the fence is byte-identical, the strict validation gate is exercised.

G.13. CREATE `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_deployment_strip.py`. Build a fixture bundle containing `meta_skill.md`, call `copy_for_deployment(...)`, assert `meta_skill.md` is absent in the destination and all other files are byte-identical.

G.14. Run `uvx ruff check /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_a /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/skillsbench --fix --exclude benchmark/vendor` then `uvx ruff format`.

G.15. Run `python3 -m pytest /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests -x`.

**Acceptance criteria — Group G:**

- `python3 -m pytest /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests/test_slow_update.py /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests/test_patch_parser_slow_update.py /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_consolidator_path.py /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_deployment_strip.py -x` passes.
- Existing `python3 -m pytest /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests/test_sentinel_edge_cases.py -x` (Group D) still passes — the new `"slow_update_violation"` enum entry must not break the existing six-kind cases (parser is back-compat under `parent_skill_md=None`).
- `python3 -c "from skill_evolve.shared.slow_update import SLOW_UPDATE_START, SLOW_UPDATE_END; assert SLOW_UPDATE_START == '<!-- SLOW_UPDATE_START -->' and SLOW_UPDATE_END == '<!-- SLOW_UPDATE_END -->'"` exits 0.
- `python3 -c "from skill_evolve.track_a.prompts import CONSOLIDATOR_PROPOSER_SYSTEM_PROMPT; assert '<!-- SLOW_UPDATE_START -->' in CONSOLIDATOR_PROPOSER_SYSTEM_PROMPT and '<!-- SLOW_UPDATE_END -->' in CONSOLIDATOR_PROPOSER_SYSTEM_PROMPT"` exits 0.
- `uvx ruff check /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve --exclude benchmark/vendor` returns clean.
- Cross-group gate: G consumes E's `RejectedBuffer` and F's `clip_ops` — DO NOT start G implementation until both have merged.

---

### Group H — Success/failure minibatch partition reflection

Owner branch: `script-mutation/group-h` (second pass; depends on A, F). Parallel with G.

Motivated by: `feedback_skill_md_saturation.md` (saturation plateau — same shape mutations recur) and `feedback_phase_e_v9.md` ("same Task Management mutation found twice" pattern; failure-trace-conditioned mutation works at zero-scale but doesn't translate to run-level lift). H attacks both by forcing the proposer to consider failure AND success axes simultaneously via parallel reflection calls.

Paper anchor: `skillopt/gradient/reflect.py` (`run_error_analyst_minibatch`, `run_success_analyst_minibatch`) + `skillopt/gradient/aggregate.py` (`_split_minibatches`, `_hierarchical_merge` — parallel via `ThreadPoolExecutor`, failure-priority conflict resolution).

H.1. Read `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/openevolve_skills/iteration.py` (post A/B/E/F state) to confirm the single-proposer-call site that H branches on. The existing call goes through the proposer LLM once per iteration; H replaces this with two parallel calls under `--reflection-mode partition`.

H.2. CREATE `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/reflection.py`. Define:

```
@dataclass(frozen=True)
class Partition:
    failures: list[PerTaskResult]   # items where per-task score < threshold
    successes: list[PerTaskResult]  # items where per-task score >= threshold

def partition_trajectories(eval_result, *, threshold: float = 0.5) -> Partition:
    """Split EvalResult.per_task into failures/successes by threshold (default 0.5)."""

def merge_patches(failure_ops: list[FileOp], success_ops: list[FileOp], lt: int) -> list[FileOp]:
    """Hierarchical merge with FAILURE PRIORITY.
       Rules:
         1. Build keyed index by (op_kind, path) for each side.
         2. On collision (e.g. both contain EDIT_FILE same path): failure-side op wins; success-side op dropped with a recorded note.
         3. Non-colliding ops: concatenate in [failure..., success...] order.
         4. Final list capped to lt via shared.edit_budget.clip_ops (length-stable when under budget).
       Returns the merged list. Caller is responsible for persisting the per-side raw lists + the collision-drop notes on the iteration artifact.
    """
```

H.3. MODIFY `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_a/prompts.py`. Add two new top-level constants:
- `FAILURE_REFLECTION_PROPOSER_SYSTEM_PROMPT` — slot fields `{failure_trajectories}` (rendered failure mini-batch), `{edit_budget_line}` (from Group F), `{recent_rejections}` (from Group E), `{meta_skill_body}` (from Group G, optional — falls back to empty if Group G is not yet merged). Instructs the proposer to emit a sentinel-block patch addressing the failure modes.
- `SUCCESS_REFLECTION_PROPOSER_SYSTEM_PROMPT` — same slot shape; instructs the proposer to emit a sentinel-block patch that reinforces / generalizes the success patterns.
- Both prompts cite L_t in the EDIT BUDGET block exactly as the existing `SENTINEL_PROPOSER_SYSTEM_PROMPT` does.
- Extend `_assert_prompt_well_formed()` (line 514) to assert both new constants contain `<<<END_FILE>>>` and `## EDIT BUDGET`.

H.4. MODIFY `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/openevolve_skills/iteration.py`. Branch on `args.reflection_mode`:
- `single` (back-compat default for non-canonical runs): existing single proposer call path, unchanged.
- `partition` (canonical default): call `partition_trajectories(eval_result, threshold=args.reflection_success_threshold)`. If either partition is empty (all-pass or all-fail), skip the empty side and fall back to a single-side call (NOT to single-mode — still uses the reflection prompt on the non-empty side). Otherwise launch the two proposer calls in parallel via `concurrent.futures.ThreadPoolExecutor(max_workers=2)`. Parse each output via `parse_sentinel_blocks` (with `parent_skill_md` from Group G if available). Merge via `merge_patches(failure_ops, success_ops, lt)` where `lt` comes from `compute_lt(args.edit_budget, iter_n, max_iters)`. Persist both raw per-side op lists and the merged list to the iteration artifact.

H.5. MODIFY `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/run.py`. Add argparse args:
- `--reflection-mode {single,partition}` default `partition`.
- `--reflection-batch-size INT` default `8` (paper `B_m`; effective per-side batch ≤ 5 because hot_5 has 5 tasks total — after partition, both sides typically fit in a single sub-batch, so `B_m=8` just acts as a per-side cap. Flag is still surfaced for non-hot configurations where N > 5).
- `--reflection-success-threshold FLOAT` default `0.5` (matches default behavioral threshold; partition cut-off).

H.6. MODIFY `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/skillsbench/evolve.py`. Extend the `--canonical` preset to inject `--reflection-mode partition --reflection-batch-size 8 --reflection-success-threshold 0.5`. Backward-compat: when a user explicitly passes `--reflection-mode single`, the preset must not override.

H.7. CREATE `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests/test_reflection_partition.py`. Cases:
- `partition_trajectories` on a 4-task result with 2 below + 2 above threshold → `len(failures) == 2`, `len(successes) == 2`.
- All-pass trajectories → `successes` non-empty, `failures` empty.
- All-fail → inverse.
- Boundary score == threshold → goes to `successes` (≥, not >).

H.8. CREATE `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests/test_reflection_merge.py`. Cases:
- Disjoint paths: `merge_patches([EDIT_FILE a], [EDIT_FILE b], 10) == [EDIT_FILE a, EDIT_FILE b]` (order: failures first).
- Collision: `merge_patches([EDIT_FILE x with body F], [EDIT_FILE x with body S], 10)` → result contains the failure-body op only; the dropped success op is reported via the second return value or a side-channel (see implementation).
- L_t clip: 5 failure ops + 5 success ops with `lt=3` → result length 3, all from failure side (failures-priority + parser-order).
- Empty success side: returns failure list clipped to `lt`.
- Empty failure side: returns success list clipped to `lt`.

H.9. CREATE `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_iteration_partition_mode.py`. Monkeypatch `SyntheticLLM` to emit different sentinel patches when the system prompt contains `FAILURE_REFLECTION` vs `SUCCESS_REFLECTION` substrings. Run iteration with `--reflection-mode partition`. Assert: two proposer calls fired (`call_count == 2`), the merged op list contains ops from both sides, the iteration artifact records both raw per-side lists + the merged list.

H.10. CREATE `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_iteration_single_mode.py`. Same fixture but `--reflection-mode single`. Assert: one proposer call (`call_count == 1`), no partition rendered, back-compat path identical to the plan_0 default.

H.11. Run `uvx ruff check /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_a /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/skillsbench --fix --exclude benchmark/vendor` then `uvx ruff format`.

H.12. Run `python3 -m pytest /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests -x`.

**Acceptance criteria — Group H:**

- `python3 -m pytest /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests/test_reflection_partition.py /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests/test_reflection_merge.py /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_iteration_partition_mode.py /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_iteration_single_mode.py -x` passes.
- Existing `python3 -m pytest /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_iteration_smoke_guard.py -x` (Group D) still passes under `--reflection-mode single` (back-compat).
- `python3 -c "from skill_evolve.track_a.prompts import FAILURE_REFLECTION_PROPOSER_SYSTEM_PROMPT, SUCCESS_REFLECTION_PROPOSER_SYSTEM_PROMPT; assert '<<<END_FILE>>>' in FAILURE_REFLECTION_PROPOSER_SYSTEM_PROMPT and '<<<END_FILE>>>' in SUCCESS_REFLECTION_PROPOSER_SYSTEM_PROMPT"` exits 0.
- `uvx ruff check /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve --exclude benchmark/vendor` returns clean.
- Cross-group gate: H consumes F's `clip_ops` + `compute_lt` — DO NOT start H implementation until F has merged. H is decoupled from G — partition mode works even when slow-update is disabled.

---

## 5. Testing strategy

| Group | Primary test commands | Edge cases covered |
| --- | --- | --- |
| A | `python3 -m pytest /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_a/tests /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests -x` | Re-export shim + back-compat aliases preserve existing patch_parser tests. Smoke-gate rejection wired into both track_a and track_b iteration loops. |
| B | `python3 -m pytest /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/behavioral/tests -x` | `--eval-source behavioral` without `--eval-set` raises. `--validation-task-list` recorded separately. Behavioral adapter handles DeepSeek reasoning short-circuit. |
| C | `cd /Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare && uv run --extra dev pytest` | All eight deleted modules absent (5 from C1 + 3 from C2). CLI lists only kept subcommands. ty check clean. `score_bundle` audit complete. |
| D | Full sweep in D.10 | Sentinel edge cases (unterminated, traversal, absolute, empty, duplicate, add_existing, edit_missing, mixed_content). Smoke rejection (broken `.py`, broken `.sh`) + `--no-smoke-test` bypass. Token cap (just under, at, over, REWRITE_FOLDER heavy op). DeepSeek (`content=None+reasoning`, both empty). End-to-end one-iter. |
| E | `python3 -m pytest /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests/test_rejected_buffer.py /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_validation_gate.py -x` | Bounded-ring eviction, JSONL round-trip, empty-buffer prompt rendering, all three gate modes (strict/record/relaxed), tie-rejection under strict, plan_0 Group-D `record` back-compat preserved, context-budget degradation order (rejected-buffer truncate → meta-skill truncate → bundle truncate). Refs E.2, E.3, E.7, E.8. |
| F | `python3 -m pytest /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests/test_edit_budget.py /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_iteration_edit_budget.py -x` | constant/linear/cosine schedule parsing + endpoints + midpoints + degenerate `N == M` collapse. `clip_ops` length-stable under budget; truncates to first L_t over budget. Iteration artifact records parsed vs applied counts; evaluator still called after clip. Refs F.2, F.3, F.7, F.8. |
| G | `python3 -m pytest /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests/test_slow_update.py /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests/test_patch_parser_slow_update.py /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_consolidator_path.py /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_deployment_strip.py -x` | Fence detection (presence, half-fence error, boundary-line membership). Step-level `slow_update_violation` raised for EDIT_FILE SKILL.md crossing the fence; `parent_skill_md=None` back-compat preserved. Consolidator fires every K iters, appends `meta_skill.md`, exercised under strict gate. `meta_skill.md` excluded from deployment copy. Refs G.2-G.6, G.9, G.10-G.13. |
| H | `python3 -m pytest /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests/test_reflection_partition.py /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/shared/tests/test_reflection_merge.py /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_iteration_partition_mode.py /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/test_iteration_single_mode.py -x` | Partition split at threshold (≥ goes to successes), all-pass / all-fail single-side skip. Failure-priority collision resolution on `(op_kind, path)` key; L_t-bounded merge. `partition` mode fires 2 proposer calls + records both raw lists + merged list. `single` mode preserves plan_0 Group-D smoke-guard behavior. Refs H.2, H.4, H.7-H.10. |

Common: every group runs `uvx ruff check --fix` and `uvx ruff format` on its touched paths before claiming acceptance.

## 6. Linting / formatting notes

- daycare (Python 3.11+):
  - `uvx ruff check /Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare --fix`
  - `uvx ruff format /Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare`
  - `uvx ty check /Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare/src/daycare`
  - `cd /Users/atakantekparmak/Desktop/work/kai-skills/watchmen-fukara/daycare && uv run --extra dev pytest`
  - Ruleset is minimal (`F + E9` only); do not promote stricter rules in this PR.
- skill_evolve (Python 3.12+):
  - `uvx ruff check /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve --fix --exclude benchmark/vendor`
  - `uvx ruff format /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve --exclude benchmark/vendor`
  - `uvx ty check /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve` (best-effort; pre-existing issues outside this PR's scope can be ignored).
  - Tests: `python3 -m pytest /Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/<dirs>` or `cd /Users/atakantekparmak/Desktop/work/kai-skills && uv run --extra dev pytest skill_evolve/...`.
  - Parent workspace `pyproject.toml` has stricter ruff config than daycare. `benchmark/vendor/**` has pre-existing violations — DO NOT touch and DO NOT include in lint targets.

## 7. Cross-group interfaces

All groups must agree on the following names and shapes BEFORE starting. These are LOCKED at section-7 read time; downstream groups (B, D especially) can build against them without waiting on Group A's file moves.

**a. Shared module paths (Group A owns; B/C/D consume):**
- `skill_evolve.shared.patch_parser` — primary exports `parse_sentinel_blocks(text: str, existing_paths: Optional[set[str]] = None) -> list[FileOp]`, `FileOp` dataclass, `SentinelParseError`. **Back-compat aliases** for track_b consumers: `parse_patch = parse_sentinel_blocks`, `Operation = FileOp`, `PatchParseError = SentinelParseError`. The per-kind names `AddFile`, `EditFile`, `DeleteFile`, `RewriteFolder` are **distinct empty subclasses of `FileOp`** (NOT aliases) — `track_b/tests/test_patch_parser.py` requires both `isinstance(x, AddFile)` discrimination AND `type(x).__name__ == "EditFile"` / `"DeleteFile"`, which the alias scheme cannot satisfy. The parser dispatches on the parsed op string and instantiates the matching subclass at construction time. Daycare-side consumers keep using `FileOp` (parent catches all subclasses via `isinstance`).
- `skill_evolve.shared.bundle_ops` — exports `apply_file_ops`, `parse_and_apply`, `validate_scripts`, `bundle_tokens`, `list_scripts`, `hash_bundle`, `shebang_insurance`, `SmokeResult`, `ApplyResult`, `BundleRejected`, `MAX_SKILL_TOKENS=3000`, `MAX_BUNDLE_TOKENS=60000`, `APPLE_DOUBLE_PREFIX="._"`.
- `skill_evolve.shared.leak_scanner` — direct copy of `daycare.leak_scanner`.
- `skill_evolve.track_a.prompts.SENTINEL_PROPOSER_SYSTEM_PROMPT` — module-level str constant containing `<<<END_FILE>>>`, `60000`, `3000`, "CRITICAL".

**b. FileOp dataclass + error class (Group A defines; D's parser tests consume):**
```
@dataclass(frozen=True)
class FileOp:
    op_kind: Literal["ADD_FILE","EDIT_FILE","DELETE_FILE","REWRITE_FOLDER"]
    path: str                # relative to bundle root, validated for traversal
    body: str                # empty string for DELETE_FILE
    line_start: int          # for error messages

class SentinelParseError(ValueError):
    def __init__(self, kind: str, *args): ...
    # kind ∈ {"unterminated", "path_traversal", "absolute_path",
    #        "add_existing", "edit_missing", "mixed_content"}
```

**c. New CLI flag names (Groups A + B both add to `track_a/runner.py`; B also adds to `track_b/run.py` and `skillsbench/evolve.py`):**
- `--patch-format {json-ops,sentinel-blocks}` — Group A owns.
- `--smoke-test / --no-smoke-test` — Group A owns.
- `--eval-source {skillsbench,behavioral}` — Group B owns.
- `--eval-set <path>` — Group B owns.
- `--judge-model <slug>` — Group B owns.
- `--validation-task-list <path>` — Group B owns.

**d. `track_a/runner.py` _cli() merge note:** both A and B add argparse args here. Convention: Group A's flags go in a new `_add_patch_args(parser)` helper, Group B's flags go in `_add_eval_source_args(parser)`. Both helpers are called from `_cli()` (line 579) in alphabetical order. This eliminates the textual conflict.

**e. `track_b/openevolve_skills/iteration.py` merge note:** Group A inserts the smoke gate `_run_smoke_gate(child_artifact)` immediately AFTER the anonymize block ends (current line 139, the `# kai-skills patch end` comment) and BEFORE line 141's `# 5. Evaluate child.` comment. Group B's evaluator-call edits happen at line 143 (`evaluator.evaluate_artifact(...)` call site) where the new `eval_source` / `eval_set_path` / `judge_model` kwargs are threaded. Both groups MUST cite the exact insertion line in their PR description. Disjoint line ranges: A touches 139-141, B touches 143.

**f. EvalResult contract (Group B consumes from `skill_evolve.evaluator`):** existing dataclass at `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/evaluator.py` near `evaluate()` (~line 639) — Group B does NOT change its shape; only routes through it. Behavioral judge returns continuous score float in [0, 1] per item; `EvalResult.mean_score` = mean of per-task scores; `EvalResult.success_rate` = fraction where score ≥ threshold (default 0.5 — match daycare verifier conventions). Stub fixtures use shape `{"score": <float>, "reasoning": "..."}` — NOT `{"verdict": "pass"}`.

**g. Daycare deletion gate (Group C ↔ Group A):** Group C Phase C1 lands ANY TIME after Day 0; Phase C2 must wait for Group A's signal commit on the integration branch `script-mutation` (NOT on the `script-mutation/group-a` subbranch) with the exact message: `feat(shared): port daycare mutator + sentinel prompt — group-c green light`. C2 confirms via `git log script-mutation --oneline` before deleting `daycare/src/daycare/evolve.py`, `mutator.py`, `leak_scanner.py`.

**h. Fixture path (Group B + D):** behavioral fixture lives at `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/behavioral/tests/fixtures/tiny_eval_set.jsonl`. SkillsBench mock lives at `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/track_b/tests/fixtures/mock_skillsbench/`. Groups D creates; Groups B reads in its own tests via `pathlib.Path(__file__).parent / "fixtures" / ...`.

**i. Task-set name resolution (Group B owns):**
| name | resolved path |
| --- | --- |
| `hot_5` | `/Users/atakantekparmak/Desktop/work/kai-skills/runs/skillsbench_baseline_v2/hot_5.json` |
| `subset_17` | `/Users/atakantekparmak/Desktop/work/kai-skills/skill_evolve/skillsbench/subset_17.json` (verify on read; if absent, pin to actual location) |
| any other | passed through as a path |

**j. Rejected-edit buffer schema (Group E owns; G consumes):**
- Module: `skill_evolve.shared.rejected_buffer`. Exports `RejectedEdit` (frozen dataclass) and `RejectedBuffer` (bounded ring, FIFO eviction).
- `RejectedEdit` fields: `patch_text: str`, `delta_train: float`, `delta_val: float`, `rejection_reason: Literal["train_no_improve","val_not_strict_gt","val_tie","smoke_rejected","token_cap","parse_error"]`, `iteration: int`.
- `RejectedBuffer(capacity: int = 10)`. Methods: `push(edit) -> None`, `render_for_prompt() -> str` (returns the literal string `(none yet)` when empty), `to_jsonl(path) -> None` (append-only persist; one JSON object per line), `from_jsonl(path, capacity=10) -> RejectedBuffer`.
- Acceptance-gate semantics (locked):
  - `strict` (default): accept iff `train_score >= parent_train_score` AND `val_score > best_val_score_seen_so_far`. Ties on val_score REJECTED.
  - `record`: plan_0 Group-B behavior — accept on train criterion only; record val_score on artifact.
  - `relaxed`: accept iff `train_score >= parent_train_score` AND `val_score >= best_val_score_seen_so_far` (ties accepted).
- Persist path convention: `<run_dir>/rejected_buffer.jsonl` written every iteration immediately after the gate decision (crash-safe).
- Prompt slot: rendered into `## RECENT REJECTIONS — DO NOT REPEAT` section of `SENTINEL_PROPOSER_SYSTEM_PROMPT` via `{recent_rejections}` `.format()` slot. G's `CONSOLIDATOR_PROPOSER_SYSTEM_PROMPT` and H's `FAILURE_REFLECTION_PROPOSER_SYSTEM_PROMPT` / `SUCCESS_REFLECTION_PROPOSER_SYSTEM_PROMPT` consume the same renderer.
- Controller responsibility: instantiate exactly ONE `RejectedBuffer` per run (capacity = `args.rejected_buffer_size`) and pass the instance into every `run_iteration(...)` call.
- **Divergence from paper (gate strictness asymmetry):** the train-side check is `>=` (non-strict) while the val-side check is `>` (strict). Paper uses a single strict `>` on a single score. Rationale: training noise on hot_5 (n=5) is high; requiring strict improvement on BOTH axes would over-reject. Allowing neutral train moves so long as val strictly improves preserves the paper's anti-silent-drift guarantee while accepting the higher train-noise floor of small-n SkillsBench evals.
- **Divergence from paper (`RejectedEdit` schema):** paper's `engine/trainer.py:1340-1347` has fields `op`, `content`, `target`, `score_before`, `score_after` and accumulates within an epoch (no fixed capacity). skill_evolve's schema (`patch_text`, `delta_train`, `delta_val`, `rejection_reason`, `iteration`) is a skill_evolve-specific design adding dual-axis fields for richer feedback. `K=10` is a memory-bounded ring (paper accumulates within-epoch); skill_evolve has no epoch concept distinct from iter so bounding by capacity replaces bounding by epoch.
- **Fallback guidance:** if `strict` produces zero accepts across 12 iters on canonical hot_5/subset_17, fall back to `--validation-gate relaxed` (accepts ties on val_score). If `relaxed` also stalls, fall back to `record` (no gate; just records val_score for post-hoc analysis). The choice is per-run; the canonical preset starts at `strict` but operators can downgrade.
- **Prompt-rendering helper + context-budget enforcement (Group E owns; G and H consume):** Group E owns the proposer prompt-rendering helper that enforces `--max-proposer-prompt-tokens` (default `90000`) before any LLM call. Helper signature is LOCKED here for downstream consumers:
  ```
  def render_proposer_prompt(
      template: str,
      *,
      bundle_text: str,
      rejected_buffer: RejectedBuffer,
      meta_skill_body: str,
      edit_budget_line: str,
      slots: dict[str, str],
      max_tokens: int,
  ) -> tuple[str, dict[str, int]]:
      """Render `template` with the supplied slots, enforcing `max_tokens` via the
      3-stage graceful degradation order:
        1. rejected-buffer truncate (oldest-first removal until under cap),
        2. meta-skill tail truncate (oldest-iter removal until under cap),
        3. bundle truncate (keep SKILL.md + frontmatter + first N scripts/* in
           list_scripts order, drop the rest; log `prompt_oversize` warning).
      Returns (rendered_prompt, counters) where counters tracks how many tokens
      were trimmed at each stage for the iteration artifact.
      """
  ```
  The degradation ORDER is locked — never re-order stages or skip a stage. G's `CONSOLIDATOR_PROPOSER_SYSTEM_PROMPT` call (per §7l "LLM call routing") and H's `FAILURE_REFLECTION_PROPOSER_SYSTEM_PROMPT` / `SUCCESS_REFLECTION_PROPOSER_SYSTEM_PROMPT` calls (per §7m "LLM call routing") BOTH route through this helper for their respective LLM calls. The `max_tokens` arg threads from `args.max_proposer_prompt_tokens` (E's new flag). Per-stage truncation counters are persisted on the iteration artifact under `prompt_budget.{rejected_trimmed,meta_skill_trimmed,bundle_trimmed}` for post-hoc analysis.

**k. Edit-budget schedule parser + L_t API (Group F owns; G + H consume):**
- Module: `skill_evolve.shared.edit_budget`. Exports `ScheduleSpec` (type alias = `str`), `parse_schedule(spec) -> tuple[Literal["constant","linear","cosine"], int, int]`, `compute_lt(spec, iter_n, max_iters) -> int`, `clip_ops(ops, lt) -> list`.
- Spec grammar (locked):
  - `constant:N` — parsed as `("constant", N, N)`. Always returns `N`.
  - `linear:N->M` — parsed as `("linear", N, M)`. `compute_lt` = `floor(N + (M - N) * t)`, `t = iter_n / max(1, max_iters - 1)`, clamped to `[min(N,M), max(N,M)]`.
  - `cosine:N->M` — paper default. `compute_lt` = `round(M + 0.5 * (N - M) * (1 + cos(pi * t)))`, `t in [0,1]`, clamped.
  - `N == M` for linear/cosine collapses to constant behavior (NOT an error).
- `parse_schedule` raises `ValueError("invalid edit-budget spec: <spec>")` on malformed input.
- `clip_ops(ops, lt)`: paper's fallback path only — if `len(ops) <= lt` return ops unchanged; else truncate to first `lt` (parser order = proposer order = priority order). No optimizer-rank pass.
- Iteration plumbing: clip happens AFTER sentinel parsing and BEFORE the smoke gate at `track_b/openevolve_skills/iteration.py:192`. Both the original parsed count and the clipped applied count are persisted on the iteration artifact (`edit_budget.parsed`, `edit_budget.applied`).
- Prompt slot: rendered into `## EDIT BUDGET` section of `SENTINEL_PROPOSER_SYSTEM_PROMPT` via `{edit_budget_line}` `.format()` slot. H's `FAILURE_REFLECTION_PROPOSER_SYSTEM_PROMPT` and `SUCCESS_REFLECTION_PROPOSER_SYSTEM_PROMPT` reuse the same slot shape.
- Modes supported: `constant`, `linear`, `cosine`. `autonomous` (paper's no-limit mode where the proposer chooses the count) is deferred — it needs a separate ranking signal that the deterministic `clip_ops` fallback path does not provide.

**l. Slow-update fence + meta-skill + parser error-kind extension (Group G owns; depends on E + F):**
- Fence markers (VERBATIM paper strings — DO NOT vary): `<!-- SLOW_UPDATE_START -->` and `<!-- SLOW_UPDATE_END -->`. Position: END of SKILL.md (paper-faithful; `inject_empty_slow_update_field` appends via `return skill.rstrip() + block` per `skillopt/optimizer/slow_update.py`). Concrete injection shape: `return skill.rstrip() + "\n\n" + SLOW_UPDATE_START + "\n" + content + "\n" + SLOW_UPDATE_END + "\n"`.
- Module: `skill_evolve.shared.slow_update`. Exports `SLOW_UPDATE_START`, `SLOW_UPDATE_END`, `has_slow_update_field(text) -> bool`, `inject_empty_slow_update_field(text) -> str`, `extract_slow_update_field(text) -> str` (raises `ValueError` on half-fence), `replace_slow_update_field(text, new_content) -> str`, `is_in_slow_update_region(text, target_line) -> bool`.
- Error-kind enum extension (extends section 7b set): NEW kind `"slow_update_violation"`. The full enum after Group G is:
  ```
  {"unterminated", "path_traversal", "absolute_path",
   "add_existing", "edit_missing", "mixed_content",
   "slow_update_violation"}
  ```
  `parse_sentinel_blocks(text, existing_paths=None, parent_skill_md=None)` adds the new optional `parent_skill_md` kwarg. When set, EDIT_FILE / DELETE_FILE / REWRITE_FOLDER ops targeting SKILL.md whose effective range overlaps the fence raise `SentinelParseError("slow_update_violation")`. When `parent_skill_md is None` the check is skipped (back-compat for daycare-side and pre-G callers, e.g. Group D's existing sentinel edge-case tests).
- Meta-skill: `skill_evolve.shared.meta_skill`. `MetaSkillEntry` dataclass (`iteration`, `summary`, `edit_pattern_stats`, `persistent_failures`) + `MetaSkill` class (`append`, `render_for_prompt`, `load`). On-disk format: literal markdown file at `<bundle_root>/meta_skill.md`. Bounded to last `args.meta_skill_max_iters` entries via tail-truncation on `append` (default 20).
- Deployment strip: `skill_evolve.shared.bundle_ops.copy_for_deployment(src, dst)` excludes `meta_skill.md`, `._*` (AppleDouble), and `__pycache__/`. ALL bundle promotion / validation-eval copy sites use this helper. `meta_skill.md` is NEVER deployed at inference.
- Consolidator I/O: separate top-level constant `CONSOLIDATOR_PROPOSER_SYSTEM_PROMPT` in `track_a/prompts.py`. Slots: `{edit_history}`, `{persistent_failures}`. The consolidator emits exactly ONE `EDIT_FILE SKILL.md` sentinel block whose body keeps the fence markers and only mutates content between them. The iteration loop double-guards: it extracts the proposed in-fence content via `extract_slow_update_field` and applies via `replace_slow_update_field` against the current SKILL.md — so even if the consolidator's body lies about out-of-fence content, only in-fence bytes are taken. Consolidator outputs go through the SAME strict validation gate as fast edits (E's gate).
- Cadence: every `args.slow_update_every` iters (default 4). Step-level proposer continues running every iter; the consolidator path is ADDITIONAL, not a replacement.
- **Divergence from paper (silent-skip vs strict-raise):** paper silently skips edits inside the fence with `report["status"] = "skipped_protected_slow_update_region"`. Plan raises `SentinelParseError("slow_update_violation")` to match skill_evolve's all-or-nothing patch contract. The reject-whole-patch behavior is consistent with how the existing `add_existing`, `edit_missing`, `mixed_content`, etc. error kinds work — a single malformed op kills the whole patch.
- **Divergence from paper (consolidator input shape).** Paper's `slow_update.md` (https://raw.githubusercontent.com/microsoft/SkillOpt/main/skillopt/prompts/slow_update.md) expects: previous-epoch's skill, current-epoch's skill, a longitudinal A-vs-B comparison of the SAME training-set rollouts under both skills (categorized into regressions / persistent failures / improvements / stable successes), AND the previous slow-update guidance block for self-reflection. Plan's `CONSOLIDATOR_PROPOSER_SYSTEM_PROMPT` instead receives: recent accepted edits (last K), `RejectedBuffer.render_for_prompt()`, current SKILL.md, and persistent-failure list (tasks failing in ≥ N consecutive iters). This is a fundamentally simpler input contract — no A-vs-B skill comparison, no four-way categorization, no prior-guidance self-reflection slot. Rationale: skill_evolve has no epoch concept distinct from iter; per-iter rollouts are already stored in the artifact tree, but materializing two full skill versions and re-rolling on a shared set every K iters doubles eval cost. The iteration-history + rejected-buffer surrogate is a no-extra-rollout approximation.
- **Divergence from paper (MetaSkill artifact shape).** Paper's `meta_skill.md` prompt (https://raw.githubusercontent.com/microsoft/SkillOpt/main/skillopt/prompts/meta_skill.md) produces a single rolling JSON `{"reasoning": ..., "meta_skill_content": ...}` that is an OPTIMIZER-SIDE coach memo — revised/replaced each epoch, addressing the future optimizer (not the target). Plan's `MetaSkill` is a per-iteration markdown log with `## Iteration N` sections containing `summary`, `edit_pattern_stats`, `persistent_failures`, tail-truncated to last 20 entries; consumer is the proposer system prompt prepend. Different artifact type (audit log vs rolling coach memo), different on-disk format (markdown sections vs JSON-extracted content). Rationale: skill_evolve's existing per-iter artifact discipline (write-once JSON per iter) maps naturally onto a markdown-section log; a rolling-rewritten JSON memo would require a separate state file and overwrite semantics not present elsewhere in skill_evolve. The audit-log shape is what proposer actually needs (prior-iter context), not what paper's coach memo provides.
- **Cadence rationale (K=4):** paper fires the consolidator at epoch boundaries (~once per 10+ steps). skill_evolve has no epoch concept distinct from iter. K=4 means ~3 consolidator fires per canonical 12-iter run, balancing freshness of cross-iter lessons vs added LLM cost. K is user-configurable via `--slow-update-every`.
- **LLM call routing:** all new consolidator LLM calls in this section route through `track_a/llm.py:call_proposer` (or the equivalent llm_client wrapper that A.10 added), which carries the unified `msg.get('content') or msg.get('reasoning') or ''` short-circuit for DeepSeek thinking-mode handling. No new transport code.

**m. Reflection partition + hierarchical merge + reflection prompts (Group H owns; depends on F):**
- Module: `skill_evolve.shared.reflection`. Exports `Partition` (frozen dataclass with `failures: list[PerTaskResult]`, `successes: list[PerTaskResult]`), `partition_trajectories(eval_result, *, threshold: float = 0.5) -> Partition`, `merge_patches(failure_ops, success_ops, lt) -> list[FileOp]`.
- Threshold semantics: per-task score `>= threshold` goes to `successes`; `<` threshold goes to `failures` (locked: boundary score == threshold is a success).
- Merge contract (locked):
  1. Build keyed index by `(op_kind, path)` for each side.
  2. Collisions resolved FAILURE-FIRST: failure-side op wins; the dropped success op is recorded on the iteration artifact (`reflection.collisions_dropped`).
  3. Non-colliding ops concatenated as `[failure_ops..., success_ops...]` in their original parser order.
  4. Final list capped to `lt` via `shared.edit_budget.clip_ops` (length-stable when under budget; truncates to first `lt` over budget).
- Modes (`--reflection-mode`):
  - `single`: existing single proposer call; back-compat path used by non-canonical runs and by Group D's smoke-guard test.
  - `partition` (canonical default): two parallel proposer calls (`ThreadPoolExecutor(max_workers=2)`), one with `FAILURE_REFLECTION_PROPOSER_SYSTEM_PROMPT`, one with `SUCCESS_REFLECTION_PROPOSER_SYSTEM_PROMPT`. Empty partitions are skipped — if all-pass, only the success side fires; if all-fail, only the failure side fires; NEITHER condition falls back to single-mode (the surviving side still uses its reflection prompt).
- Prompt I/O: `FAILURE_REFLECTION_PROPOSER_SYSTEM_PROMPT` and `SUCCESS_REFLECTION_PROPOSER_SYSTEM_PROMPT` in `track_a/prompts.py`. Both consume the same slot shape: `{failure_trajectories}` / `{success_trajectories}`, `{edit_budget_line}` (from F), `{recent_rejections}` (from E), `{meta_skill_body}` (from G, optional — falls back to empty string when G is not merged).
- Artifact persistence: both raw per-side op lists, the merged list, and any `collisions_dropped` entries are written to the iteration artifact under `reflection.{failure_ops,success_ops,merged_ops,collisions_dropped}`.
- **Paper prompt anchors (for fidelity reference — skill_evolve prompts must mirror these instructional bones):**

  From `skillopt/prompts/analyst_error.md` (verbatim head):
  ```
  You are an expert failure-analysis agent for AI agent tasks.

  You will be given MULTIPLE failed agent trajectories from a single minibatch
  and the current skill document.
  Your job is to identify the most important COMMON failure patterns across
  the batch and propose a concise set of skill edits.

  ## Analysis Process
  1. Read ALL trajectories in the minibatch.
  2. Identify the most prevalent, systematic failure patterns across them.
  3. For each pattern, classify its failure type.
  4. Propose skill edits that address the COMMON patterns — not individual edge cases.
  5. Edits must be generalizable; do not hardcode task-specific values.
  6. Only patch gaps in the skill — do not duplicate existing content.

  You will be told the maximum number of edits (the budget L). Produce AT MOST L edits,
  focusing on the highest-impact patterns. You may produce fewer if warranted.

  IMPORTANT: The skill document may contain a section between
  <!-- SLOW_UPDATE_START --> and <!-- SLOW_UPDATE_END --> markers.
  This is a PROTECTED section managed by a separate slow-update process.
  Do NOT propose any edits that target, modify, or delete content within
  these markers.
  ```

  From `skillopt/prompts/analyst_success.md` (verbatim head):
  ```
  You are an expert success-pattern analyst for AI agents.

  You will be given MULTIPLE successful agent trajectories from a single minibatch
  and the current skill document. Your job is to identify generalizable behavior
  patterns that are COMMON across the batch and worth encoding in the skill.

  ## Rules
  - Only propose patches for patterns NOT already covered in the skill.
  - Focus on patterns that appear across MULTIPLE trajectories in the batch.
  - Be concise. Patterns must generalize beyond specific tasks.
  - Prefer reinforcing existing sections over adding new top-level sections.

  You will be told the maximum number of edits (the budget L). Produce AT MOST L edits,
  focusing on the most broadly applicable patterns. You may produce fewer if warranted.

  IMPORTANT: The skill document may contain a section between
  <!-- SLOW_UPDATE_START --> and <!-- SLOW_UPDATE_END --> markers.
  This is a PROTECTED section managed by a separate slow-update process.
  Do NOT propose any edits that target, modify, or delete content within
  these markers.
  ```

  skill_evolve's `FAILURE_REFLECTION_PROPOSER_SYSTEM_PROMPT` and `SUCCESS_REFLECTION_PROPOSER_SYSTEM_PROMPT` must preserve these instructional bones (common-pattern focus, no edge-case hardcoding, generalizability mandate, slow-update protection notice) while swapping the JSON edit-op contract for skill_evolve's sentinel-block format (sentinel ops emit `ADD_FILE` / `EDIT_FILE` / `DELETE_FILE` / `REWRITE_FOLDER`, not the paper's `append` / `insert_after` / `replace` / `delete` JSON shape). The slow-update protection notice is REQUIRED verbatim once Group G has merged.
- **Divergence from paper (hierarchical merge dropped):** paper's `skillopt/gradient/aggregate.py:merge_patches` does THREE LLM merge calls (per-side hierarchical merge via parallel `ThreadPoolExecutor` + a final failure-priority merge). Plan's `merge_patches(failure_ops, success_ops, lt)` is a deterministic keyed-dict collision resolver with NO LLM calls. Rationale: per-side hierarchical merge is degenerate on hot_5 (≤1 sub-batch per side because the eval set has 5 tasks total), so the parallel structure has nothing to parallelize; final failure-priority merge collapses to a keyed-dict resolution. The simplification saves ~3 LLM calls per iter at no quality cost under the canonical config. Re-add LLM-based merge later if quality regression is observed on larger eval sets.
- **LLM call routing:** all new proposer LLM calls (both reflection sides) in this section route through `track_a/llm.py:call_proposer` (or the equivalent llm_client wrapper that A.10 added), which carries the unified `msg.get('content') or msg.get('reasoning') or ''` short-circuit for DeepSeek thinking-mode handling. The `ThreadPoolExecutor(max_workers=2)` wraps two `call_proposer` invocations.
- **Empty-output handling:** if a partition's proposer call returns an empty patch (no ops parsed) or raises `SentinelParseError`, treat that side as empty for the merge (fall through to the other side's ops only). If BOTH sides return empty, log `partition_both_empty` and produce an empty merged patch (no candidate this iter — equivalent to skip). This is distinct from the empty-PARTITION case (all-pass / all-fail) where the side is skipped before any LLM call.

**n. Second-pass co-edit coordination contracts (Groups E/F/G/H all append to shared files):**

Round 1 (E+F parallel) and Round 2 (G+H parallel) both have multiple groups touching the same files: `track_a/prompts.py` template, `track_a/prompts.py:_assert_prompt_well_formed()`, `skillsbench/evolve.py` `_apply_canonical_defaults(args)`, `track_b/run.py` argparse block, and `track_b/openevolve_skills/iteration.py`. The contracts below eliminate textual conflicts via disjoint regions + sequential append-only edits.

- **`track_a/prompts.py` template ordering — `SENTINEL_PROPOSER_SYSTEM_PROMPT`:** after the second pass, section order is (top → bottom):
  1. existing HARD LIMITS block (from Group A)
  2. `## EDIT BUDGET` (F) — with `{edit_budget_line}` slot
  3. `## META-SKILL` (G) — with `{meta_skill_body}` slot
  4. `## RECENT REJECTIONS — DO NOT REPEAT` (E) — with `{recent_rejections}` slot
  5. existing CRITICAL `<<<END_FILE>>>` block (from Group A)

  Mechanism: each group's insertion finds the `## <prev-section> closing marker` (e.g. blank line after the prior block) and prepends its `## <own-section>` block before the next existing section. F lands first (between HARD LIMITS and CRITICAL); E inserts between F's block and CRITICAL; in Round 2, G inserts between F's `## EDIT BUDGET` and E's `## RECENT REJECTIONS` (i.e. at the gap reserved by the final ordering).

  Each group is responsible for inserting its `##`-headed block at the assigned position with the slot name above. NO group modifies another group's section.

- **`_assert_prompt_well_formed()` extension:** each group adds ONE additional clause of the shape `if "<own_marker>" not in SENTINEL_PROPOSER_SYSTEM_PROMPT: raise RuntimeError(...)`. E adds the clause for `## RECENT REJECTIONS`, F for `## EDIT BUDGET`, G for `## META-SKILL`. H does NOT add a `SENTINEL_PROPOSER_SYSTEM_PROMPT` assert (its assertions land on the new `FAILURE_REFLECTION_PROPOSER_SYSTEM_PROMPT` / `SUCCESS_REFLECTION_PROPOSER_SYSTEM_PROMPT` constants only).

- **`skillsbench/evolve.py` canonical preset (`_apply_canonical_defaults(args)`):** each group appends ONE pair of argv tokens, sequential append-only — NO reordering of prior groups' lines:
  - F: append `--edit-budget cosine:8->2`.
  - E: append `--validation-gate strict` (plus `--rejected-buffer-size 10`, `--max-proposer-prompt-tokens 90000`).
  - H: append `--reflection-mode partition` (plus `--reflection-batch-size 8`, `--reflection-success-threshold 0.5`).
  - G: append `--slow-update-every 4` (plus `--meta-skill-max-iters 20`, `--persistent-failure-window 3`).

  List order here = on-disk append order = merge order from §8 (Round 1: F-first, then E rebases; Round 2: H-first, then G rebases).

- **`track_b/run.py` argparse block:** each group appends its argparse arg definitions at the END of the existing block (currently lines 191-229 post-A/B; line numbers drift as groups merge — use the textual end-marker rather than absolute lines). Sequential, append-only — NO group reorders prior groups' args. Append order for second pass: F appends `--edit-budget`; E appends `--validation-gate`, `--rejected-buffer-size`, `--max-proposer-prompt-tokens` (default `90000`) (Existing `--rng-seed` in track_b/run.py:67 and track_a/runner.py:725 — E modifies its default and adds proposer-LLM threading; F/G/H consume `args.rng_seed` without further declaration. NO new declaration.); H appends `--reflection-mode`, `--reflection-batch-size`, `--reflection-success-threshold`; G appends `--slow-update-every`, `--meta-skill-max-iters`, `--persistent-failure-window`. F/G/H consume `args.rng_seed` and `args.max_proposer_prompt_tokens` without re-declaring. List order here = on-disk append order = merge order from §8 (Round 1: F-first, then E rebases; Round 2: H-first, then G rebases).

- **`track_b/openevolve_skills/iteration.py` insertion regions — disjoint by line region:**
  - **F:** clip ops BEFORE smoke gate, at line ~190 (immediately before `_run_smoke_gate` call at line 192). Wraps the result of `parse_sentinel_blocks` with `clip_ops(parsed_ops, compute_lt(...))`.
  - **E:** validation-gate logic AT or AFTER the `evaluator.evaluate_validation(...)` call at line 260. Wraps the existing acceptance check; the new gate branches on `args.validation_gate`.
  - **G:** consolidator branch at the TOP of the iteration loop body — every K iters, the consolidator path runs in ADDITION to (not instead of) the regular fast-edit path. Specifically, after the fast-edit accept/reject branch completes, branch on `iter_n % args.slow_update_every == 0` and call `run_consolidator_iteration(...)`. Disjoint from F/E/H insertion sites.
  - **H:** branch on `args.reflection_mode == "partition"` at the proposer call site (currently around line 122). Under partition mode the single proposer call becomes two parallel calls + merge. Disjoint from F/E/G — the proposer-call site is upstream of F's clip site (which works on the merged op list), E's validation-gate site, and G's consolidator branch.

All four groups MUST cite their exact insertion region (with surrounding 2-line context) in their PR description so the merge auditor can verify disjointness.

- **skill_evolve branches:**
  - Parent integration branch: `script-mutation` cut from `master`.
  - Group A subbranch: `script-mutation/group-a` → squash-merge into `script-mutation` first (provides shared/ + sentinel prompt + DeepSeek fallback). The signal commit per section 7g lands on `script-mutation` directly (via the squash-merge commit message OR via a follow-up commit on `script-mutation` — either way it shows up in `git log script-mutation --oneline`).
  - Group B subbranch: `script-mutation/group-b` → squash-merge after A (depends on shared/ exports).
  - Group D subbranch: `script-mutation/group-d` → squash-merge last (asserts integrated behavior).
  - Final PR `script-mutation` → `master` once all three groups have merged in.
- **daycare branch (watchmen-fukara repo):**
  - Group C lands on `daycare-shrink` cut from `master`. Phase C1 commits land any time. Phase C2 commits land after the Group A signal in section 7g.
  - PR `daycare-shrink` → `master` after Phase C2 completes.
- **Variance experiment (out of scope, mentioned only):** runs against `r_d2079019`'s existing bundle on a separate branch `daycare-v2-variance` cut from the current `daycare-v2-behavioral` snapshot. Not touched by this plan.
- **Commit hygiene:** each group's PR uses prefixed messages — `feat(shared): …`, `feat(track_a): …`, `feat(track_b): …`, `feat(behavioral): …`, `chore(daycare): drop run/promote/daemon`, `test(shared): …`, `test(behavioral): …`. Co-authored-by trailer per existing repo convention. No `--no-verify`; no force-push to `master`.
- **Sequencing diagram:**

```
Day 0: cut script-mutation and daycare-shrink from their respective master.
       Start in parallel:
         Group A      → script-mutation/group-a
         Group B      → script-mutation/group-b (against locked section-7 API)
         Group C.C1   → daycare-shrink (anchor/controls/finalize/watchdog/daemon
                       deletions + cli/__init__/pyproject + test_daemon delete)
         Group D.D1-5 → script-mutation/group-d (sentinel/bundle/llm/behavioral
                       tests against locked section-7 API)

A merges → script-mutation. Signal commit per section 7g lands on script-mutation.
       ↓
B rebases onto script-mutation, merges.
       ↓
D.6-9 land on script-mutation/group-d (now both A and B are present), then D
merges last on the skill_evolve side.
       ↓
script-mutation → master.

In parallel on daycare side, once A's signal commit is on script-mutation:
Group C.C2 lands evolve/mutator/leak_scanner deletions and test cleanups.
       ↓
daycare-shrink → master.
```

  - C1 lands any time after Day 0.
  - C2 gated on A's signal commit.
  - D2 (D.6-D.9) gated on A's signal commit AND B's merge.

- **Second pass (Groups E/F/G/H — SkillOpt no-gradient ports):** runs after first pass A/B/C/D have merged into `master` and Group A/B/D audit is green. Cut a new integration branch `skillopt-port` from `master`. Subbranches `skillopt-port/group-{e,f,g,h}`. Sequencing:

```
Day 0' (after A/B/C/D land on master): cut skillopt-port from master.
        Start Round 1 in parallel (max 2 agents):
          Group E → skillopt-port/group-e (strict gate + RejectedBuffer;
                    depends on B's validation-task-list plumbing)
          Group F → skillopt-port/group-f (L_t scheduler + clip_ops;
                    depends on A's sentinel parser)

Round 1 merge order: F MERGES FIRST (smaller surface — touches prompts.py
EDIT-BUDGET block + run.py + iteration.py clip site + new shared/edit_budget.py),
THEN E rebases onto F's commit (E touches prompts.py RECENT-REJECTIONS block +
run.py + iteration.py validation-gate site + new shared/rejected_buffer.py).
Both groups append to the same _assert_prompt_well_formed(),
_apply_canonical_defaults(), and argparse block; conflicts resolved by
sequential append per section 7n.
        ↓
Round 2 in parallel (max 2 agents), once BOTH E and F are on skillopt-port:
          Group G → skillopt-port/group-g (slow-update fence + meta-skill
                    consolidator; consumes E's RejectedBuffer + F's clip_ops)
          Group H → skillopt-port/group-h (success/failure partition reflection;
                    consumes F's clip_ops + compute_lt; decoupled from G)

Round 2 merge order: H MERGES FIRST (smaller surface — touches prompts.py
new FAILURE_REFLECTION / SUCCESS_REFLECTION constants + run.py + iteration.py
proposer-call site + new shared/reflection.py), THEN G rebases (larger surface —
touches prompts.py META-SKILL block + CONSOLIDATOR_PROPOSER_SYSTEM_PROMPT
+ shared/patch_parser.py extension + shared/slow_update.py + shared/meta_skill.py
+ iteration.py consolidator branch + bundle_ops copy_for_deployment helper).
Both groups append to the same _assert_prompt_well_formed(),
_apply_canonical_defaults(), and argparse block; conflicts resolved by
sequential append per section 7n.
        ↓
G merges → skillopt-port.
        ↓
skillopt-port → master.
```

  - E and F are independent (parallel-safe). G blocks on E + F. H blocks on F only.
  - No daycare-side changes in pass 2 — Group C is already done.
  - Commit message prefixes for pass 2: `feat(shared): port SkillOpt rejected-buffer / edit-budget / slow-update / reflection-merge ...`, `feat(track_a): add CONSOLIDATOR/FAILURE_REFLECTION/SUCCESS_REFLECTION prompts ...`, `feat(track_b): wire --validation-gate / --edit-budget / --slow-update-every / --reflection-mode ...`, `test(shared): ...`, `test(track_b): ...`. Same Co-authored-by trailer convention as pass 1; no `--no-verify`; no force-push to `master`.

## Task List

Implementation strategy: Group A runs ALONE in Round 1 (creates shared/ surface that B/D consume). Groups B, C, D run in PARALLEL in Round 2 (max 3 agents) once A's shared/ exists. Within Group C, the implementer handles its own C1→C2 phase split sequentially; within Group D, the implementer handles D1→D2 phase split sequentially. No actual git branching — agents edit files in place.

- [x] **1. Group A — sentinel-block patch capability + smoke guard + DeepSeek fallback** (skill_evolve) — IMPLEMENTED (audit pending; covered by post-E/F/G/H audit)
  - [ ] A.1 Read daycare mutator.py, evolve.py:66-115 and 620-692
  - [ ] A.2 Read existing track_b patch_parser.py; compare against daycare's
  - [ ] A.3 Create skill_evolve/shared/{__init__.py,patch_parser.py} with daycare-primary names + distinct-subclass back-compat
  - [ ] A.4 Run test_patch_parser.py — MUST pass
  - [ ] A.5 Create shared/bundle_ops.py (apply_file_ops, validate_scripts, bundle_tokens, list_scripts, hash_bundle, shebang_insurance, parse_and_apply, BundleRejected, constants)
  - [ ] A.5b Create shared/leak_scanner.py (verbatim copy from daycare)
  - [ ] A.6 Port proposer prompt → track_a/prompts.py:SENTINEL_PROPOSER_SYSTEM_PROMPT (raise RuntimeError not bare assert)
  - [ ] A.7 Add AddFileOp/EditFileOp/DeleteFileOp/RewriteFolderOp to track_a/ops.py and register in apply_op() dispatch
  - [ ] A.8 Add --patch-format flag in track_a/runner.py _cli() via _add_patch_args() helper
  - [ ] A.9 Add --smoke-test/--no-smoke-test; insert smoke + token-cap in SkillFolder.write() at folder.py:187 before return at line 206
  - [ ] A.10 Add unified content/reasoning fallback to track_a/llm.py + reasoning_max_tokens cap (verify existing)
  - [ ] A.11 Insert smoke gate in track_b/openevolve_skills/iteration.py between lines 139 and 141
  - [ ] A.12 ruff check --fix + ruff format on shared/, track_a/, track_b/
  - [ ] A.13 pytest track_a/tests track_b/tests -x
  - [ ] A.14 Note: signal commit messaging is for real git workflow — for this implementation pass, just ensure A's work is complete before B/C/D start

- [x] **2. Group B — --eval-source flag + behavioral adapter + validation split** (skill_evolve; runs in parallel with C and D after A) — IMPLEMENTED
  - [ ] B.1 Read daycare/verifier.py + skill_evolve/evaluator.py:639+EvalResult
  - [ ] B.2 Create skill_evolve/behavioral/{__init__.py, eval_set.py, adapter.py}
  - [ ] B.3 EvalItem dataclass + load_eval_set in behavioral/eval_set.py
  - [ ] B.4 score_bundle_behavioral in behavioral/adapter.py — use shared.bundle_ops list_scripts/bundle_tokens, parse continuous score float
  - [ ] B.5 Modify evaluator.py evaluate(): add eval_source, eval_set_path, judge_model params + dispatch
  - [ ] B.6 Modify track_b/run.py: add --eval-source/--eval-set/--judge-model/--validation-task-list argparse args; thread to controller/iteration
  - [ ] B.7 Modify track_b/openevolve_skills/{controller,iteration}.py to thread new args; B touches iteration.py at line 143 evaluate_artifact call
  - [ ] B.8 Modify skillsbench/evolve.py: add --canonical flag with task-set resolution (hot_5 → kai-skills/runs/skillsbench_baseline_v2/hot_5.json)
  - [ ] B.9 Modify track_a/runner.py _cli() — add same four flags via _add_eval_source_args() helper
  - [ ] B.10 ruff check --fix + ruff format on behavioral/, track_b/, skillsbench/
  - [ ] B.11 pytest track_b/tests behavioral/tests -x

- [x] **3. Group C — daycare shrink (Phase C1 + Phase C2)** (watchmen-fukara/daycare; runs in parallel with B and D after A) — IMPLEMENTED
  - [ ] C1.1 Catalogue daycare cli.py imports + subcommands; confirm drop/keep lists
  - [ ] C1.2 Grep doomed modules (anchor/controls/finalize/watchdog/daemon) across surviving keep-list
  - [ ] C1.3 Delete anchor.py, controls.py, finalize.py, watchdog.py, daemon.py
  - [ ] C1.4 Trim cli.py: drop run/promote/daemon Click commands; drop imports
  - [ ] C1.5 Drop __init__.py re-exports for deleted modules
  - [ ] C1.6 Audit pyproject.toml: drop newly-unused deps, drop deleted-subcommand console scripts
  - [ ] C1.7 Delete tests/test_daemon_incremental.py
  - [ ] C2.1 Verify A's work is in place (shared/ exists with bundle_ops.py + patch_parser.py + leak_scanner.py)
  - [ ] C2.2 Delete daycare evolve.py, mutator.py, leak_scanner.py
  - [ ] C2.3 Delete tests/test_evolve_script_mutations.py, test_mutator.py; audit test_leak_scanner.py
  - [ ] C2.4 Grep verifier.py for score_bundle (line 357); remove if no callers, else leave
  - [ ] C2.5 ruff check --fix + ruff format on daycare
  - [ ] C2.6 ty check on daycare/src/daycare; fix Unknown name errors from deletions
  - [ ] C2.7 uv run --extra dev pytest from daycare/
  - [ ] C2.8 Manual smoke: daycare --help shows only eval-build, doctor, runs

- [x] **4. Group D — tests + integration fixture (Phase D1 + Phase D2)** (skill_evolve; runs in parallel with B and C after A) — IMPLEMENTED
  - [ ] D.1 shared/tests/test_patch_parser_roundtrip.py (ADD/EDIT/DELETE/REWRITE/round-trip)
  - [ ] D.2 shared/tests/test_bundle_ops_smoke.py (broken .py/.sh, token caps, AppleDouble filter, REWRITE_FOLDER cap, --no-smoke-test bypass)
  - [ ] D.3 shared/tests/test_sentinel_edge_cases.py (unterminated/traversal/absolute/empty/duplicate/add_existing/edit_missing/mixed_content/multi-op)
  - [ ] D.4 track_a/tests/test_llm_deepseek_reasoning.py (content/reasoning fallback paths)
  - [ ] D.5 behavioral/tests/{fixtures/tiny_eval_set.jsonl, test_adapter_smoke.py} (continuous score float, DeepSeek-style fallback, missing eval_set)
  - [ ] D.6 track_b/tests/test_iteration_smoke_guard.py (smoke rejection skips evaluator)
  - [ ] D.7 track_b/tests/test_validation_holdout.py (validation_score recorded distinct from train_score)
  - [ ] D.8 track_b/tests/fixtures/mock_skillsbench/ (3 fake task IDs + stubbed backend + train_3/val_2 lists)
  - [ ] D.9 track_b/tests/test_end_to_end_one_iter.py (in-process main() with stubbed LLM)
  - [ ] D.10 Full test sweep + ruff check --fix --exclude benchmark/vendor + ruff format

- [x] **5. Group E — Strict validation gate + rejected-edit buffer** (skill_evolve; second pass, Round 1, parallel with F) — IMPLEMENTED
  - [ ] E.1 Read track_b/openevolve_skills/iteration.py post-A/B state; confirm validation call site at line 260 and smoke-gate site at line 192
  - [ ] E.2 Create shared/rejected_buffer.py (RejectedEdit frozen dataclass + RejectedBuffer bounded ring + JSONL persist/load + render_for_prompt)
  - [ ] E.3 Modify track_b/openevolve_skills/iteration.py — flip validation semantics to strict acceptance gate; branch on --validation-gate {strict,record,relaxed}; push rejections to RejectedBuffer
  - [ ] E.4 Modify track_a/prompts.py — add ## RECENT REJECTIONS — DO NOT REPEAT section with {recent_rejections} slot; extend _assert_prompt_well_formed()
  - [ ] E.5 Modify track_b/run.py — add --validation-gate, --rejected-buffer-size, --max-proposer-prompt-tokens (default 90000) argparse args; MODIFY existing --rng-seed default (0→None) and thread into proposer LLM client; thread all four through controller/iteration
  - [ ] E.6 Modify skillsbench/evolve.py — extend --canonical preset with --validation-gate strict + --rejected-buffer-size 10
  - [ ] E.7 Create shared/tests/test_rejected_buffer.py (bounded ring, JSONL round-trip, render_for_prompt empty/non-empty)
  - [ ] E.8 Create track_b/tests/test_validation_gate.py (all three modes + tie behavior + push-per-rejection)
  - [ ] E.9 ruff check --fix + ruff format on shared/, track_a/, track_b/, skillsbench/
  - [ ] E.10 pytest shared/tests track_b/tests -x

- [x] **6. Group F — Bounded edit budget L_t + scheduler** (skill_evolve; second pass, Round 1, parallel with E) — IMPLEMENTED
  - [ ] F.1 Read shared/patch_parser.py to confirm parse_sentinel_blocks returns list[FileOp]
  - [ ] F.2 Create shared/edit_budget.py (parse_schedule + compute_lt for constant/linear/cosine + clip_ops fallback)
  - [ ] F.3 Modify track_b/openevolve_skills/iteration.py — clip parsed ops to L_t AFTER parsing and BEFORE smoke gate at line 192; persist parsed/applied counts on artifact
  - [ ] F.4 Modify track_a/prompts.py — add ## EDIT BUDGET block with {edit_budget_line} slot; extend _assert_prompt_well_formed()
  - [ ] F.5 Modify track_b/run.py — add --edit-budget argparse arg default cosine:8->2
  - [ ] F.6 Modify skillsbench/evolve.py — extend --canonical preset with --edit-budget cosine:8->2
  - [ ] F.7 Create shared/tests/test_edit_budget.py (constant/linear/cosine endpoints + midpoints + degenerate N==M + malformed spec)
  - [ ] F.8 Create track_b/tests/test_iteration_edit_budget.py (5 ops + constant:2 → 2 applied + artifact records)
  - [ ] F.9 ruff check --fix + ruff format on shared/, track_a/, track_b/, skillsbench/
  - [ ] F.10 pytest shared/tests track_b/tests -x

- [x] **7. Group G — Slow-update protected region + meta-skill consolidator** (skill_evolve; second pass, Round 2, parallel with H; depends on A, B, E, F) — IMPLEMENTED + AUDIT PASS (consolidator gate wired via shared `_apply_validation_gate` helper)
  - [ ] G.1 Read paper's skillopt/optimizer/slow_update.py + skill.py:_is_in_slow_update_region; confirm verbatim marker strings
  - [ ] G.2 Create shared/slow_update.py (markers + has/inject/extract/replace_slow_update_field + is_in_slow_update_region)
  - [ ] G.3 Modify shared/patch_parser.py — extend parse_sentinel_blocks with parent_skill_md kwarg; raise SentinelParseError("slow_update_violation") on fence overlap; add new error kind to enum at line 68
  - [ ] G.4 Create shared/meta_skill.py (MetaSkillEntry dataclass + MetaSkill class with append/render_for_prompt/load; markdown on-disk; tail-truncate to last K entries)
  - [ ] G.5 Modify track_a/prompts.py — add ## META-SKILL block with {meta_skill_body} slot; add CONSOLIDATOR_PROPOSER_SYSTEM_PROMPT top-level constant citing fence markers; extend _assert_prompt_well_formed()
  - [ ] G.6 Modify track_b/openevolve_skills/iteration.py — every K iters run consolidator path (single EDIT_FILE SKILL.md; double-guard via extract/replace; strict gate; meta-skill append)
  - [ ] G.7 Modify track_b/run.py — add --slow-update-every, --meta-skill-path, --consolidator-model, --meta-skill-max-iters, --persistent-failure-window args
  - [ ] G.8 Modify skillsbench/evolve.py — extend --canonical preset with the new G defaults
  - [ ] G.9 Add shared/bundle_ops.py:copy_for_deployment helper (exclude meta_skill.md + AppleDouble + __pycache__); rewire all bundle-promotion/validation-eval copy sites
  - [ ] G.10 Create shared/tests/test_slow_update.py (presence + half-fence error + inject placement + boundary membership)
  - [ ] G.11 Create shared/tests/test_patch_parser_slow_update.py (fence-overlap raises slow_update_violation; out-of-fence accepted; scripts/* unaffected; parent_skill_md=None back-compat)
  - [ ] G.12 Create track_b/tests/test_consolidator_path.py (every K iters fires + meta_skill appended + fence-only mutation + strict gate exercised)
  - [ ] G.13 Create track_b/tests/test_deployment_strip.py (copy_for_deployment excludes meta_skill.md; other bytes identical)
  - [ ] G.14 ruff check --fix + ruff format on shared/, track_a/, track_b/, skillsbench/
  - [ ] G.15 pytest shared/tests track_b/tests -x

- [x] **8. Group H — Success/failure minibatch partition reflection** (skill_evolve; second pass, Round 2, parallel with G; depends on A, F) — IMPLEMENTED + AUDIT PASS (collisions propagated)
  - [ ] H.1 Read track_b/openevolve_skills/iteration.py post-A/B/E/F state; confirm single-proposer call site that H branches on
  - [ ] H.2 Create shared/reflection.py (Partition dataclass + partition_trajectories + merge_patches with failure-priority + L_t clip via shared.edit_budget.clip_ops)
  - [ ] H.3 Modify track_a/prompts.py — add FAILURE_REFLECTION_PROPOSER_SYSTEM_PROMPT + SUCCESS_REFLECTION_PROPOSER_SYSTEM_PROMPT top-level constants; extend _assert_prompt_well_formed()
  - [ ] H.4 Modify track_b/openevolve_skills/iteration.py — branch on --reflection-mode; partition path runs two parallel proposer calls via ThreadPoolExecutor(max_workers=2); merge via merge_patches; persist raw + merged + collisions on artifact
  - [ ] H.5 Modify track_b/run.py — add --reflection-mode, --reflection-batch-size, --reflection-success-threshold args
  - [ ] H.6 Modify skillsbench/evolve.py — extend --canonical preset with --reflection-mode partition + --reflection-batch-size 8 + --reflection-success-threshold 0.5 (respect explicit user override)
  - [ ] H.7 Create shared/tests/test_reflection_partition.py (threshold split + boundary ≥ → success + all-pass + all-fail)
  - [ ] H.8 Create shared/tests/test_reflection_merge.py (disjoint concat + collision failure-wins + L_t clip + empty-side handling)
  - [ ] H.9 Create track_b/tests/test_iteration_partition_mode.py (2 proposer calls + merged ops include both sides + artifact records all three lists)
  - [ ] H.10 Create track_b/tests/test_iteration_single_mode.py (1 proposer call + back-compat with Group D smoke-guard)
  - [ ] H.11 ruff check --fix + ruff format on shared/, track_a/, track_b/, skillsbench/
  - [ ] H.12 pytest shared/tests track_b/tests -x

