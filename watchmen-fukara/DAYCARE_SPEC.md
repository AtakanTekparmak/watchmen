# watchmen-daycare — design spec (v3)

> **Student-teacher skill distillation.** Real agentic conversations from a strong model
> (Claude Opus 4.7, captured in watchmen corpus) serve as the teacher signal. A weak target
> model (qwen3-30b-a3b or similar) is the student. Skill text is the knowledge bridge that
> closes the capability gap — the same role RL does in synth-self-improve-rl, but via
> iterative text mutation instead of gradient steps.

---

## Changes from v2 (v3 additions)

Three final audits revealed blockers. Every fix below is a hard requirement. New sections and
changes are tagged `[v3]` in their headings.

**Corpus reality (from Phase 1 dry-run on ctf)**
- R1. "1,206 ctf sessions" was wrong by 16×: 1,143 of those are subagents. The real ctf
  working set is **76 non-subagent sessions** (all at `project_dir = /Desktop/work/ctf`;
  sd-zero/ and pedogogical-rl/ are subdirs but sessions run from the ctf root).
  An intermediate audit added a spurious `message_count ≥ 10` gate dropping 43 sessions —
  that gate is NOT in the spec. Phase 1a processes all sessions (short ones just yield
  fewer triples). 97% of user-turn volume still comes from ~5 mega-sessions. (Phase 1a.)
- R2. Semantic dedup gate added to Phase 1d: ≥60 evals is necessary but not sufficient.
  Cap ≤5 evals per anonymized prompt cluster (embedding cosine distance threshold 0.92).
  30 distinct clusters × 2 evals each is the real floor. (Phase 1d.)
- R3. Hard discard triggers added to Phase 1b for live-infra patterns that pass the
  script_gen classifier but are 100% unverifiable: `ssh root@`, `nvidia-smi`,
  `tmux list-sessions`, `tail -f /`. (Phase 1b.)
- R4. pi-agent transcript format (`claude-agent-acp` JSONL schema at `~/.pi/agent/sessions/`)
  documented as a second parser backend needed in `corpus.py`. Phase 0 doctor flags
  session count lost to unknown format. (Phase 1a, corpus.py.)
- R5. AUP-refusal contamination: CTF-specific prompts refused by Anthropic's filters must
  be discarded at Phase 1b (not classified as discard_other — logged separately as
  `discard_safety_refusal` for tracking). Judge calls on these would also hit refusals.

**Framing fix: student-teacher distillation (not deployment optimization)**
- F1. The model mismatch (optimize qwen3-32b, reference from Opus 4.7) is NOT a flaw —
  it IS the design. The corpus captures Opus 4.7 behavior; skill text bridges the weak
  model to that behavior. This mirrors synth-self-improve-rl's student-teacher dynamic
  at the text layer instead of the weight layer. Phase 4 adds a ceiling metric. (Phase 4,
  Comparison table, Non-negotiables.)

**Closed-loop judge defense**
- J1. Phase 4 Baseline A is a hard promotion block: `best.holdout_score ≤ baseline_a +
  2ε` → `daycare promote` is DISALLOWED (was "flag for review"). The empty-bundle floor
  is the only reliable escape valve from judge gaming. (Phase 4a.)
- J2. eval-build runs on all 3 projects (ctf, pi, wmca) before the first evolution run.
  The project with the highest count of semantically distinct surviving evals (post-dedup)
  is selected as the first evolution target. (Phase 0, CLI.)

## Changes from v1 (v2 additions, preserved)

This v2 incorporated three audits (methodology fidelity to synth-self-improve-rl, kai-skills
evolution lessons, watchmen integration). Every fix below is a hard requirement, not a
suggestion.

**Methodology fidelity (synth-self-improve-rl) fixes**
- M1. Genealogy explicitly documented as GEPA rolling-best (not iter_0-anchored model
  weights). The fixed anchor is `eval_set.jsonl` + `weak_model` + sampling config, not the
  bundle itself. (Phase 3d, Non-negotiables.)
- M2. Round-trip sanity gate adds structural assertions: `py_compile`/`bash -n` on
  `script_gen` references; bundle-slug existence check on `skill_invoke` references.
  (Phase 1f.)
- M3. Weakness analysis adds a completion-length quartile reward bucket table; proposer
  must label each cluster's primary failure mode (truncation / reasoning / format).
  (Phase 3a.)
- M4. Run A / Run B renamed "Baseline A — empty bundle (floor)" and "Baseline B — naive
  few-shot". They are baselines, not control trains. New optional "Reproducibility run"
  documented. (Phase 4.)
- M5. Token regularizer calibrated: λ_init=0.00001 with annealing, penalty capped at 0.05
  regardless of iter. (Phase 0, Phase 3c.)
- M6. OR model version pinning is a Phase 0 hard gate: exact version string is captured
  from response headers and abort-on-change is enforced for the run lifetime. (Phase 0.)
- M7. Anonymization moved into the spec body as a non-negotiable. `eval_set.jsonl` rows
  carry an `anonymized_prompt` field; the proposer never sees raw prompts. (Phase 1e,
  Non-negotiables.)
- M8. Minimum evals raised to 60 (→ 30 holdout). Acceptance epsilon =
  `max(0.01, 1/n_holdout)`. (Phase 1d, Phase 3d.)

**Kai-skills evolution lessons (Phase E v6–v9d)**
- K1. Mutation format is sentinel blocks (`<<<ADD_FILE>>>`, `<<<EDIT_FILE>>>`,
  `<<<DELETE_FILE>>>`, `<<<REWRITE_FOLDER>>>`), not unified diffs. EDIT_FILE carries
  full rewritten body. Traversal/duplication/missing-file guards listed. (Phase 3b.)
- K2. Weakness report includes `invocations_render` and `unused_render` first-class:
  separates "skill didn't fire" from "skill fired but didn't help". (Phase 3a.)
- K3. Output-side leak scanner runs on every proposer patch. `leak-policy=zero` is the
  default; any leaked eval identifier → reject patch. (Phase 3b → new step 3b.5.)
- K4. `history.jsonl` records per-attempt outcomes:
  `parse_error | validate_error | eval_error | promoted | rejected_by_fitness`. (Phase 3d.)
- K5. `OPENROUTER_RAW_LOG_DIR` support — every raw LLM response written to disk when env
  var is set. (Providers, Phase 0.)
- K6. Cascade short-circuit: smoke-3 random holdout evals per candidate; if 0/3, skip
  full holdout scoring. (Phase 3c.)
- K7. Shebang insurance: every emitted `.py` gets `#!/usr/bin/env python3`, every `.sh`
  gets `#!/bin/bash`. Auto-fixed if missing. (Phase 3b validate step.)
- K8. Script discipline block added to the proposer system prompt (no pip install,
  ≤150 lines/file, argparse-only, validated with `py_compile` + `bash -n`). (Phase 3b.)
- K9. SIGALRM wall-clock watchdog fires at `budget - 10min`, drains in-flight rollouts,
  proceeds to Phase 4. (Phase 3, Daemon.)
- K10. Subprocess isolation per rollout — each weak-model invocation runs in its own
  subprocess to prevent global state mutation. (Phase 2, Phase 3c.)

**Watchmen integration fixes**
- W1. `skill_name` column may not exist or be populated in `corpus.db`. Phase 0 doctor
  runs `watchmen ingest --full` if missing/all-NULL or falls back to JSONL scanning.
  (Phase 0.)
- W2. `read_session_full()` is a 600-char-truncated string renderer, not a structured
  parser. Daycare ships its own `parse_transcript(path)` that returns
  `(user_turn, assistant_response, tool_calls_in_turn)` triples with no truncation.
  (Phase 1a + new `corpus.py` contract.)
- W3. `transcript_path` files can be GC'd by Claude Code. Phase 1a guards every read
  with `Path.exists()` + skip-with-log. (Phase 1a.)
- W4. `_pinned.json` is a flat JSON array of slug strings, not an object. `daycare
  promote` appends slug to the array. (Phase 5.)
- W5. Token counter is `tiktoken cl100k_base` everywhere. (Phase 0, Phase 3c.)
- W6. `project_dir → bundle` mapping function specified: match session.project_dir's path
  prefix against `projects.json:source_repo`; fall back to basename. (Phase 1a.)
- W7. Proposer K=6 candidates each run as a separate multi-turn `watchmen.Agent`
  invocation, not a single-shot dump. (Phase 3b.)
- W8. `run.json` adds `estimated_cost_usd` computed up-front. (Phase 0.)
- W9. Rollout aggregation specified: mean across rollouts per eval, then mean across
  evals; partial failures (1/5 rollouts errors) use the surviving rollouts; full failure
  (5/5) scores 0.0. (Phase 3c.)

---

## Name rationale

Skills start as raw candidates extracted from conversations (babies), get curated by watchmen
into SKILL.md bundles (children), then **daycare** takes them through an evolution loop to
maturity. Standalone package now; plan to contribute upstream to firstbatchxyz/watchmen.

---

## Source material

- https://github.com/firstbatchxyz/watchmen — the corpus + skill curation system this sits next to
- https://github.com/vivekvkashyap/synthetic-self-improve-rl — the methodology this mirrors
  (optimize skill text the way synth-self-improve-rl optimizes model weights via RL)
- The `skill_evolve/` sibling tree in this repo — the GEPA-on-skillsbench predecessor
  (Phase E v4–v9d) whose operational lessons are baked in throughout.

---

## Decisions baked in

| Question | Decision |
|---|---|
| Integration | Standalone package in `watchmen-fukara/daycare/`, import watchmen as lib, plan to merge upstream |
| Target | All 3 projects (ctf first), project-agnostic from day 1 |
| Eval shape | Hybrid: script/plan-gen + skill-invocation-quality + procedural Q&A, all LLM-judged |
| Corpus access | Daycare ships its own `parse_transcript()`; reads `corpus.db` read-only |
| Mutation scope | Full bundle: SKILL.md + scripts (sentinel-block mutations) |
| Mutation genealogy | GEPA rolling-best (parent = previous winner), fixed-anchor at the **eval set + weak model** level |
| Scale | K=6 candidates, ≥60 evals (≥30 holdout), 5 rollouts/eval, ~2–3h/iter |
| Serving | All OpenRouter (weak model + proposer + judge), version-pinned |
| Skill selection | Traffic×error-rate score + analyst-priority boost, user can override with `--skill` |
| Token counter | `tiktoken cl100k_base` |
| MVP | Full 5-phase pipeline + daemon + CLI |

---

## Comparison table [v3]

| synth-self-improve-rl | watchmen-daycare |
|---|---|
| Teacher: strong LLM generating synthetic data | Teacher: Claude Opus 4.7 behavior captured in watchmen corpus (real conversations) |
| Student: weak model optimized via RL | Student: weak model (qwen3-32b) optimized via skill text mutation |
| Knowledge bridge: synthetic dataset | Knowledge bridge: evolved SKILL.md + scripts |
| Anchor: iter_0 checkpoint (model weights) | Anchor: frozen `eval_set.jsonl` + pinned weak_model version + fixed sampling config |
| Train budget: 100 RL steps | Mutation budget: K × max_iters skill candidates |
| Eval: hub env held-out test | Eval: held-out 50% of derived eval_set, 5 rollouts per eval |
| Substrate: prime-rl + verifiers | Substrate: OpenRouter (weak model + judge), version-pinned |
| Ceiling: teacher's eval score | Ceiling: Opus 4.7 + empty skill score on held-out (Baseline C) |
| Weak model signal: reward from env | Weak model signal: LLM judge score against rubric, ≥0.0 ≤1.0 |
| Cross-model concern: none (one model) | Cross-model concern: intentional student≠teacher by design |
| Genealogy | iter_0-anchored (each step starts from base weights) | GEPA rolling-best (parent = previous winner). Justification: text edits compose; fixed anchor lives at the eval/sampling level, not the bundle. |

---

## Repository layout

```
watchmen-fukara/
├── DAYCARE_SPEC.md                     ← this file
├── daycare/                            ← new standalone package
│   ├── pyproject.toml
│   ├── src/daycare/
│   │   ├── cli.py                      # Click CLI entry point
│   │   ├── corpus.py                   # parse_transcript(); corpus.db queries; project_dir mapper
│   │   ├── eval_builder.py             # Phase 1: eval extraction pipeline
│   │   ├── anonymize.py                # strip identifiers from prompts/refs/completions
│   │   ├── anchor.py                   # Phase 2: iter_0 baseline scoring
│   │   ├── evolve.py                   # Phase 3: mutation loop
│   │   ├── mutator.py                  # sentinel-block parser + validator + applier
│   │   ├── leak_scanner.py             # output-side eval-identifier leak detection
│   │   ├── controls.py                 # Phase 4: Baseline A / Baseline B
│   │   ├── finalize.py                 # Phase 5: promote + summary
│   │   ├── verifier.py                 # LLM judge, eval scoring, rollout aggregation
│   │   ├── selector.py                 # skill selection: traffic×error + analyst priority
│   │   ├── providers.py                # OR wrapper, version pinning, raw-log dir, retry/backoff
│   │   ├── runner.py                   # subprocess-isolated weak-model rollout runner
│   │   ├── watchdog.py                 # SIGALRM wall-clock watchdog
│   │   └── daemon.py                   # launchd/systemd, mirrors watchmen cadence
│   └── tests/
└── watchmen/                           # existing watchmen code (unchanged)
```

**Output home**: `~/.watchmen/daycare/runs/<project>-<UTC>/` — sibling to watchmen's own dirs.

---

## CLI surface [v2]

```
daycare init                              # verify watchmen install, run doctor, configure OpenRouter key
daycare doctor                            # Phase 0 checks, can run standalone
daycare eval-build <project>              # Phase 1 only — dump eval_set.jsonl + report
daycare run <project>                     # Phases 1–5
    [--skill <slug>]                      # override auto-selection
    [--budget 8h]
    [--max-iters 12]
    [--K 6]
    [--rollouts 5]
    [--seed 42]
    [--weak-model qwen/qwen3-32b]
    [--proposer deepseek/deepseek-v4-0324]
    [--judge deepseek/deepseek-v4-0324]
    [--max-workers 2]                     # OR 20MB/hr cap default
    [--smoke-3 / --no-smoke-3]            # cascade short-circuit (default on)
    [--leak-policy zero|warn]             # default zero
    [--anonymize / --no-anonymize]        # default on, --no-anonymize aborts with warning unless --force
    [--reproducibility]                   # optional Phase 4 re-derive from different seed
daycare promote <project> <slug>          # copy best → bundles/…, append to _pinned.json array
daycare runs                              # table of past runs with Δ scores
daycare daemon {install,uninstall,run}
```

---

## Phase 0 — Bootstrap [v2]

1. Confirm `~/.watchmen/corpus.db` exists; count sessions.
2. Confirm `~/.watchmen/bundles/<project>/` exists for target project.
3. **W1**: verify `tool_calls.skill_name` column exists AND has non-NULL rows. If missing
   or 100% NULL, run `watchmen ingest --full` automatically; if that still fails, fall back
   to JSONL scanning (parse skill XML blocks out of transcripts). Document outcome in
   `doctor.json`.
4. **J2 — First-run multi-project eval probe**: if no prior `daycare run` has completed
   for any project, automatically run `daycare eval-build` on ALL enabled projects before
   the first evolution run. Select the project with the highest post-dedup distinct cluster
   count as the first evolution target. Record the survey results in
   `~/.watchmen/daycare/eval_survey.json`. CLI override: `--project <name>` skips the
   survey and runs against the named project directly.

5. **M6**: Ping all three OpenRouter models with a 5-token request. Capture exact version
   string from the response headers (`x-or-model-version` or equivalent; otherwise hash of
   the response metadata). Write `run.json.model_pins = {weak, proposer, judge}`. On every
   subsequent OR call in this run, re-check the version string. **Mismatch → abort the run**
   and write a `model_drift` entry to `run.json.status`.
5. **K5**: if `OPENROUTER_RAW_LOG_DIR` is set, ensure the directory exists and is writable.
6. Auto-select skill slug if `--skill` not provided (see Skill Selection below).
7. **W5**: import `tiktoken`, instantiate `enc = tiktoken.get_encoding("cl100k_base")`. Used
   for every token count below.
8. **W8**: compute `estimated_cost_usd` = `K × n_holdout × rollouts × (avg_input_tokens ×
   weak_in_$/Mtok + avg_output_tokens × weak_out_$/Mtok)` × max_iters + judge call cost.
   Print before evolution starts; require `--yes` if estimate > $50.
9. Create `RUN_DIR = ~/.watchmen/daycare/runs/<project>-<UTC>/`; write `run.json`.

**run.json schema** [v2]:
```json
{
  "project": "<project>",
  "skill_slug": "<slug>",
  "weak_model": "<OR model id>",
  "proposer_model": "<OR model id>",
  "judge_model": "<OR model id>",
  "model_pins": {
    "weak": "<exact version string from OR headers>",
    "proposer": "<...>",
    "judge": "<...>"
  },
  "start_ts": "<iso>",
  "budget_seconds": <int>,
  "max_iters": 12,
  "max_skill_tokens": 2500,
  "lambda_init": 0.00001,
  "lambda_cap": 0.05,
  "epsilon": "max(0.01, 1/n_holdout)",
  "K": 6,
  "rollouts": 5,
  "seed": 42,
  "max_workers": 2,
  "smoke_3_enabled": true,
  "leak_policy": "zero",
  "anonymize": true,
  "estimated_cost_usd": <float>,
  "raw_log_dir": "<path or null>",
  "status": "running"
}
```

**Skill selection** (`selector.py`):

For each skill slug in `~/.watchmen/bundles/<project>/skills/`:
- `traffic_score` = count of `tool_calls` rows with `skill_name = slug` in the last 60 days
  (falls back to JSONL scan per W1 if column unusable)
- `error_boost` = `sessions.tool_error_count / sessions.tool_use_count` for sessions where that
  skill fired (join via session_id from tool_calls where skill_name = slug)
- `analyst_boost` = 1.5× if the slug appears in `analyses/<project>/_running.md` under
  "Skill candidates"
- `priority = traffic_score × (1 + error_boost) × analyst_boost`

Pick highest priority. Log ranking to `RUN_DIR/selector_log.md`.

---

## Phase 1 — Eval Extraction [v2]

Build `RUN_DIR/eval_set.jsonl`. Most project-specific phase; everything downstream is generic.

### 1a. Session pull [v2]

- Query `corpus.db` for sessions in the last 60 days, filtered by `project_dir`.
- **W6 — project_dir mapping**: `sessions.project_dir` is a full path (e.g.
  `/Users/atakantekparmak/Desktop/work/ctf`). `projects.json[project].source_repo` is the
  bundle's source path. Match a session to a project iff
  `session.project_dir == source_repo` OR
  `session.project_dir.startswith(source_repo + "/")` OR
  `basename(session.project_dir) == project`.
  Implementation: `corpus.match_session_to_project(session, projects_json) -> bool`.
- **W3**: for each session, run `Path(transcript_path).exists()`. If missing, append
  `{session_id, reason: "transcript_gone"}` to `eval_extraction_log.md` and skip.
- **W2** — call daycare's own `corpus.parse_transcript(path) -> List[Triple]` (NOT
  watchmen's `read_session_full()`, which truncates to 600 chars). The parser:
  - Reads the JSONL file line by line.
  - Walks the `parentUuid` chain to reconstruct thread order.
  - Emits `Triple(user_turn, assistant_response, tool_calls_in_turn)` with full content
    (no truncation at extraction time).
  - Truncation happens only when building proposer context downstream.
- `next_user_turn` = implicit acceptance signal:
  - **accepted**: next user turn builds on the response (follow-up, acknowledgment, uses the output)
  - **rejected**: next user turn re-asks, quotes-and-corrects, or is clearly a retry
- Boost candidates whose prompts match themes in `analyses/<project>/_running.md` (call
  judge to score theme relevance).

### 1b. Eval type classification

For each candidate triple, classify using the judge model:

| Type | Description | When it applies |
|---|---|---|
| `script_gen` | Assistant wrote shell/Python/JS to accomplish the task | tool_calls include Bash/Edit with substantial code; reference contains a script |
| `skill_invoke` | A skill was invoked and produced an outcome | `tool_calls.skill_name` is set for this session turn |
| `procedural_qa` | Assistant explained a procedure, concept, or made a decision | text-heavy response, no code execution needed |

**Hard discard triggers [v3 — R3]** (applied BEFORE judge classification, logged as
`discard_live_infra` or `discard_safety_refusal`):
- Bash input containing `ssh root@`, `ssh ubuntu@`, `nvidia-smi`, `tmux list-sessions`,
  `tail -f /`, `watch ` — unverifiable live-infra state snapshots that look like `script_gen`
- Next user turn contains API refusal markers: "I cannot assist with", "AUP", "Terms of
  Service" — these will also be refused by the judge on OR
- `discard_safety_refusal`: CTF-specific prompts that triggered Anthropic safety filters in
  the original session (detected by assistant_text matching `"I'm not able to assist with"`
  or `"I cannot help with"` patterns)

After hard discard: pure confirmations ("keep monitoring", "yes", single-word), turns with no
substantive assistant content, anything requiring live infra to verify. Log all discards
to `eval_extraction_log.md`.

### 1c. Verifier config (rubric generation)

Each admitted eval gets a judge-generated `rubric` string (≤150 words). The rubric is
frozen at extraction time and used unchanged across all iters.

- `script_gen`: rubric specifies what the script must accomplish + structural requirements
- `skill_invoke`: rubric specifies expected outcome class + quality bar
- `procedural_qa`: rubric specifies key points that must be covered

**Rubric format**: "Score 0.0–1.0. Award 1.0 if: [criterion 1], [criterion 2]. Award 0.5
if: [partial condition]. Award 0.0 if: [failure condition]."

### 1d. Knowledge-gap calibration (non-negotiable) [v2]

Score every admitted eval once with `<WEAK_MODEL>` + empty skill on OpenRouter:
- `score == 0`: knowledge bottleneck. Discard.
- `score >= 0.9`: already solved. Discard.
- Otherwise: keep.

**M8 — Minimum survival: 60 evals** (→ 30 holdout after 50/50 split). But count alone is
insufficient. Apply semantic dedup before the floor check:

**R2 — Semantic dedup gate (mandatory)**:
1. Embed every surviving eval's `anonymized_prompt` using a lightweight embedding model
   (e.g. OR's `text-embedding-3-small` or a local sentence-transformer).
2. Cluster by cosine similarity threshold 0.92 (tight — only near-duplicates cluster).
3. Cap each cluster at ≤5 evals (keep the 5 with highest `baseline_score` variance within
   the cluster, to preserve the most informative range).
4. After dedup: require ≥30 distinct clusters (not ≥60 total). This is the real floor.

The pod-monitor pattern alone produces 300+ near-duplicates in a single ctf session;
without dedup, the holdout slice scores the same eval 30 times and the fitness signal
is meaningless variance.

If fewer than 30 distinct clusters survive, abort with "insufficient distillable surface —
corpus lacks semantic diversity, not volume."

Log discard counts + cluster distribution to `eval_extraction_log.md`.

### 1e. Train / held-out split + anonymization [v2]

50/50 stratified by eval type + acceptance flag, using `run.json.seed`.

**M7 — anonymization (mandatory)**: for every row, produce `anonymized_prompt`,
`anonymized_reference`, `anonymized_rubric` by running each field through
`anonymize.strip(text)`:
- Replace UUIDs and session IDs with `<SESSION_ID>`
- Replace absolute paths under `/Users/<name>/…` with `/<USER>/…`
- Replace `project_dir` matches with `<PROJECT_ROOT>`
- Replace email addresses, OR keys, tokens, GitHub repo slugs found in
  `projects.json.source_repo` with placeholders
- Replace any literal substring matching the skill slug or other slugs in
  `bundles/<project>/skills/` with `<SKILL_SLUG>`

The **proposer reads only the anonymized fields**. The judge reads the raw `prompt`,
`reference`, `rubric` (it needs them to score). The verifier writes weak-model completions
in raw form to `held_out_log/` (proposer never sees these).

Write rows:

```json
{
  "id": "<sha256[:12]>",
  "split": "train" | "holdout",
  "type": "script_gen" | "skill_invoke" | "procedural_qa",
  "prompt": "<user turn verbatim>",
  "anonymized_prompt": "<prompt with identifiers stripped>",
  "reference": "<assistant turn — text + tool calls summary>",
  "anonymized_reference": "<reference with identifiers stripped>",
  "rubric": "<frozen ≤150-word scoring guide>",
  "anonymized_rubric": "<rubric with identifiers stripped>",
  "baseline_score": <float>,
  "baseline_completion_len_tokens": <int>,
  "accepted": true | false,
  "source_session": "<session_id>",
  "source_skill": "<slug if skill_invoke, else null>"
}
```

### 1f. Round-trip sanity gate [v2]

For 20 random evals (or all if fewer), score the **reference itself** as the candidate.
Must score ≥ 0.90.

**M2 — Structural assertions on top of judge self-consistency**:
- `script_gen`: extract the script from the reference; run
  `python -m py_compile` (for `.py`) or `bash -n` (for `.sh`). Reference must parse
  without crash. If it doesn't, drop the eval (not a "rubric is broken" problem — the
  reference itself is malformed).
- `skill_invoke`: verify `source_skill` corresponds to an existing slug in
  `~/.watchmen/bundles/<project>/skills/`. If the slug has been deleted/renamed since
  the conversation, drop the eval.
- `procedural_qa`: judge self-consistency only.

Any failure = rubric broken or reference stale; fix or drop. Log in
`eval_extraction_log.md`.

---

## Phase 2 — Anchor (iter_0) [v2]

Score `<WEAK_MODEL>` + baseline bundle (current `~/.watchmen/bundles/<project>/skills/<slug>/`)
against held-out slice. Run `rollouts=5` completions per eval; aggregate mean judge score.

**K10 — Each rollout runs in a subprocess (`runner.run_rollout_subprocess(...)`)**
to prevent global state mutation between rollouts. The runner returns
`{score, completion, completion_len_tokens, error}`; the parent collates.

**Bundle baseline** = copy of `skills/<slug>/SKILL.md` + all files in `scripts/`. If slug
doesn't exist yet, bundle is empty → iter_0 IS Baseline A simultaneously, and the run.json
notes `iter_0_is_baseline_a = true`.

Write `RUN_DIR/iter_0/`:
```
bundle/                ← snapshot of the baseline bundle
  SKILL.md
  scripts/
eval_summary.json      ← {holdout_score, n_holdout, by_type, by_accepted, by_length_quartile}
sampling.json          ← {temperature, seed, n_per_eval, model, model_version_pin}
```

The `iter_0.holdout_score` is the number every subsequent iter must beat (by ≥ `epsilon`).

---

## Phase 3 — Evolution Loop [v2]

Each iter `N` from 1, until `elapsed >= budget_seconds` or `iter_index >= max_iters`.

**K9 — SIGALRM watchdog**: at run start, register a SIGALRM that fires at
`budget_seconds - 600`. The handler sets a global `STOP_AFTER_THIS_ITER` flag; the loop
checks it at iter boundaries and exits cleanly into Phase 4. Rollouts in flight at the
flag-set moment are drained, not killed.

### 3a. Weakness analysis (TRAIN slice only) [v2]

Run `<WEAK_MODEL>` + current-best bundle against train slice. 5 rollouts per eval
(subprocess-isolated per K10). Pull bottom-quartile rows by score.

**Proposer reads (all anonymized)**:
- Train-slice failing prompts (`anonymized_prompt`) + references (`anonymized_reference`)
  + rubrics (`anonymized_rubric`).
- Weak model failing completions (cropped to 300 chars, run through `anonymize.strip()`).
- What the reference does differently.

Output: `weakness_report.md` with:
- Cluster name + severity (# failing evals).
- 2–3 paraphrased example prompts per cluster (min 2 — K-skills lesson: too narrow →
  hyper-specific patches).
- Failing model output (cropped to 300 chars, anonymized).
- What reference does differently.

**M3 — length-bucket table** (mandatory section in `weakness_report.md`):

| Length quartile (completion tokens) | n train evals | mean score | dominant failure mode |
|---|---|---|---|
| Q1 (shortest 25%) | | | |
| Q2 | | | |
| Q3 | | | |
| Q4 (longest 25%) | | | |

Length quartiles are computed off `baseline_completion_len_tokens` per row at iter_0
(frozen — same quartile bounds across all iters). The proposer must label each cluster's
primary failure mode as `truncation` / `reasoning` / `format` and reference this column
in mutation proposals.

**K2 — invocation telemetry** (mandatory section in `weakness_report.md`):

```
## invocations_render
<skill_slug>: invoked in N/M holdout rollouts at iter_{N-1}
<other_slug>: invoked in K/M rollouts (cross-skill firing)

## unused_render
<skill_slug>: bundled but never invoked across all holdout rollouts this iter
```

This distinguishes "skill didn't fire" (frontmatter/description problem — fix the
`when_to_use` and description) from "skill fired but didn't help" (content problem — fix
the procedure body or scripts). Implemented by parsing weak-model output for skill XML
blocks / tool calls.

**Proposer may NOT read any held-out slice field** — not prompts, references, rubrics, or
prior held-out eval completion logs.

### 3b. Bundle mutation proposal (K=6) [v2]

**W7 — Each of the K=6 candidates is a separate multi-turn `watchmen.Agent` invocation.**
Not a single-shot dump. The agent loop per candidate:
1. System prompt: skill discipline block (see below) + sentinel-block protocol.
2. User turn 1: weakness_report.md + invocations/unused_render + length-quartile table.
3. User turn 2: current best bundle (SKILL.md + scripts/ listing + each script content,
   anonymized if it leaks identifiers).
4. User turn 3: mutation_log.md from prior iters (anonymized).
5. User turn 4: "Emit your mutation now using sentinel blocks. Target the dominant cluster
   from weakness_report.md." Optional follow-up turn for clarification.

The proposer's K=6 invocations are independent (different temperatures or different
target-cluster assignments) and run with `max_workers ≤ 2` to respect the OR cap.

**K8 — script discipline block in the proposer system prompt** (verbatim):

> Constraints on emitted mutations:
> - No `pip install` calls — only Python stdlib + libraries already declared in
>   `requirements.txt` of the bundle.
> - Each script file ≤ 150 lines.
> - All CLI arguments via `argparse`. No hardcoded paths.
> - Every Python script: `python -m py_compile <file>` must pass.
> - Every bash script: `bash -n <file>` must pass.
> - SKILL.md must stay under `MAX_SKILL_TOKENS = 2500` (tiktoken cl100k_base).
> - Do not include literal session IDs, user names, absolute home paths, or any project
>   identifier from the eval set. Generic guidance only.

**K1 — Sentinel-block mutation format (the ONLY accepted format)**:

```
<<<ADD_FILE path/relative/to/bundle>>>
... full file content ...
<<<END_FILE>>>

<<<EDIT_FILE path/relative/to/bundle>>>
... FULL rewritten body (NOT a diff) ...
<<<END_FILE>>>

<<<DELETE_FILE path/relative/to/bundle>>>

<<<REWRITE_FOLDER scripts>>>
--- file: scripts/foo.py
... content ...
--- file: scripts/bar.sh
... content ...
<<<END_REWRITE>>>
```

`mutator.py` validates each emitted block:
- Reject any `path` containing `..` or starting with `/` (traversal guard).
- Reject `ADD_FILE` for a path that already exists in parent bundle.
- Reject `EDIT_FILE` / `DELETE_FILE` for a path that does not exist in parent bundle.
- Reject unterminated blocks (no matching `<<<END_*>>>`).
- Reject mixed-block content (text outside sentinel scope).

On validation failure: log `parse_error` to `history.jsonl`, discard candidate
(does NOT count against fitness, the candidate slot is simply empty for this iter).

Per candidate: 1-sentence target + reasoning ≤200 words.

**Hard constraints (post-validation, before scoring)**:
- SKILL.md must stay under `MAX_SKILL_TOKENS = 2500`.
- Run `py_compile` and `bash -n` on every emitted script. Failure → `validate_error` in
  `history.jsonl`, discard candidate.
- **K7 — Shebang insurance**: after parsing, if any `.py` file lacks
  `#!/usr/bin/env python3` at line 1, prepend it. Same for `.sh` and `#!/bin/bash`.
- No reading held-out slice.

Apply all surviving K → `iter_N/candidates/c{0..5}/bundle/`.

### 3b.5. Leak scanner [v2 — new step, K3]

`leak_scanner.scan(candidate_bundle, eval_set) -> List[Leak]` runs against every candidate
bundle BEFORE scoring. Scans every file's content for:
- Any verbatim `source_session` ID from `eval_set.jsonl`.
- Any literal absolute path that appears in a holdout `prompt` or `reference`.
- Any literal substring ≥ 12 chars that appears verbatim in any holdout row (n-gram
  signature match).
- Email addresses, OR keys, GitHub repo slugs from `projects.json`.

**Policy** (`run.json.leak_policy`):
- `zero` (default): any leak → reject the candidate, log `validate_error: leak` to
  `history.jsonl`.
- `warn`: log to `leak_log.md`, candidate proceeds.

### 3c. Score candidates (held-out, numeric aggregates only) [v2]

For each surviving candidate:

**K6 — Cascade short-circuit (smoke-3)**: pick 3 random holdout evals (seeded). Run 1
rollout each. If all 3 score 0.0 → skip full holdout scoring, log `smoke_failed` to
`history.jsonl`, candidate fitness = 0.0. Saves ~80% cost on broken bundles.

Otherwise: run `<WEAK_MODEL>` + candidate bundle against full held-out slice, 5 rollouts
per eval (subprocess-isolated per K10), `max_workers ≤ 2`.

**W9 — Rollout aggregation**:
1. For each eval, run 5 rollouts. Each rollout returns `{score, error}`.
2. Per-eval score = mean of non-error scores. If all 5 error → score = 0.0.
3. Per-eval is mean per type (`script_gen` / `skill_invoke` / `procedural_qa`) recorded
   separately.
4. Holdout score = unweighted mean of per-eval scores.

**M5 — Fitness with calibrated regularizer**:
```
λ_N = min(λ_cap, λ_init × (1 + 0.5 × N))     # λ_init = 0.00001, λ_cap = 0.05
δ_tokens = max(0, tokens(SKILL.md) − tokens(parent_SKILL.md))
penalty = λ_N × δ_tokens
fitness = holdout_score − penalty
```

At iter_5 adding 500 tokens now pays `min(0.05, 0.00001 × 3.5 × 500) = 0.0175`,
comfortably below the acceptance epsilon for n_holdout = 30 (`ε = 0.033`).

Per-candidate `eval_summary.json` written:
```json
{
  "holdout_score": <float>,
  "fitness": <float>,
  "tokens_skill_md": <int>,
  "penalty": <float>,
  "lambda_N": <float>,
  "n_holdout": <int>,
  "by_type": {"script_gen": <float>, "skill_invoke": <float>, "procedural_qa": <float>},
  "by_accepted": {"true": <float>, "false": <float>},
  "by_length_quartile": {"q1": <float>, "q2": <float>, "q3": <float>, "q4": <float>},
  "invocations_render": {...},
  "unused_render": [...],
  "smoke_failed": false
}
```

Per-row completions logged to `held_out_log/iter_<N>/c{c}/<eval_id>_<rollout>.json` —
this path is in the proposer-readable blocklist (path isolation; see Information firewall).

### 3d. Accept / discard [v2]

**Genealogy = GEPA rolling-best (M1)**. Parent = previous winner (or iter_0 bundle if
no prior winner). Justification: text edits compose; the fixed anchor lives at the
`eval_set.jsonl` + `weak_model` + sampling-config level, not the bundle. This is
explicitly documented as the divergence from synth-self-improve-rl (which uses
iter_0 model weights every step).

**M8 — Acceptance**: Winner = max-fitness candidate. If
`winner.fitness > parent.fitness + epsilon` where `epsilon = max(0.01, 1 / n_holdout)`:
promote, write `iter_N/bundle/`. Else: parent unchanged, log "no improvement."

**K4 — history.jsonl** (append-only, per-attempt outcomes):
```json
{"iter": N, "candidate": c, "outcome": "parse_error | validate_error | eval_error | smoke_failed | promoted | rejected_by_fitness", "fitness": <float|null>, "ts": "<iso>", "reasoning_preview": "<≤80 chars>"}
```

Append to `iter_N/mutation_log.md`:
- Parent bundle hash (SHA256 of SKILL.md + script concatenation).
- 6 attempts: per-candidate outcome (sentinel-block summary, fitness, verdict).
- Winner + Δ vs parent.
- Proposer reasoning (≤200 words per candidate).
- Verdict per candidate: `accepted | rejected_by_fitness | rejected_by_token_cap |
  parse_error | validate_error | smoke_failed | leak | no_change`.

### 3e. Context reset + continue

Load-bearing state is on disk. Proposer rereads `weakness_report.md`, `mutation_log.md`,
`history.jsonl`, `metrics.json`, `bundle/SKILL.md` fresh each iter. No conversational state
carried over. (Mirrors the `/compact` pattern — disk is the source of truth.)

---

## Phase 4 — Baselines [v3]

After budget expires (or watchdog fires per K9), on its own time. Does not count against
`--budget`.

**BEST_BUNDLE** = bundle with highest fitness across all iters ≥ 1. Write `best_iter.json`.

### 4a. Baseline A — empty bundle (floor, HARD PROMOTION GATE) [v3 — J1]

`<WEAK_MODEL>` + empty SKILL.md + no scripts on held-out, 5 rollouts.

**J1 — Hard promotion block**: if `best.holdout_score ≤ baseline_a.holdout_score + 2ε`
(note: 2ε not ε — absorbs within-run LLM variance), **`daycare promote` is DISALLOWED**.
Write `run.json.promote_blocked = true` and `promote_reason = "best_does_not_beat_floor"`.

The empty-bundle floor is the only reliable escape valve from the judge closed-loop.
Every kai-skills Phase E run that skipped or softened this check produced a retracted
lift. This is a hard gate, not advisory.

If `best.holdout_score > baseline_a + 2ε`: proceed to 4b.

### 4b. Baseline B — naive few-shot [v2 — renamed]

Construct SKILL.md mechanically: 5 (prompt, reference) pairs from train slice, verbatim
(passed through `anonymize.strip()`), no procedural guidance — just examples. Score
`<WEAK_MODEL>` + this skill on held-out, 5 rollouts.
- If `best.holdout_score ≤ baseline_b.holdout_score + epsilon`: GEPA didn't earn its
  compute. Naive distillation is enough. Log; don't crash. Promotion still allowed (passed A).

### 4c. Baseline C — teacher ceiling [v3 — F1, new]

**Student-teacher framing**: the corpus reference answers come from `<TEACHER_MODEL>` (the
dominant model in the project's `corpus.db`, e.g. `claude-opus-4-7` for ctf). Score
`<TEACHER_MODEL>` + empty skill on the held-out slice, 5 rollouts.

This establishes the **distillation ceiling**: how well does the teacher already perform on
these evals without any skill? Write `baseline_c.json`.

The key metric is: **`gap_closed = (best - baseline_a) / (baseline_c - baseline_a)`**.
A gap_closed of 0.5 means the evolved skill enables the weak model to close 50% of the
teacher's capability advantage. This is the primary reported metric for a daycare run.

Note: baseline_c does NOT gate promotion — the teacher ceiling is a measurement, not a
requirement. A gap_closed of 0.0 = no distillation. gap_closed of 1.0 = full distillation.
gap_closed > 1.0 = the skill makes the weak model EXCEED teacher performance on these evals
(rare; usually means the evals are too easy, see baseline_a gate).

Verdicts:
- Pass A (best > floor + 2ε) + gap_closed ≥ 0.3: **real distillation lift** — adopt.
- Pass A + gap_closed 0.1–0.3: **modest lift** — consider adopting, run more iters.
- Pass A + gap_closed < 0.1: **marginal** — log; don't promote unless very cheap to run again.
- Fail A: **bug or judge gaming** — hard blocked from promotion.

### 4c. Reproducibility run (optional, M4 + v2)

If `--reproducibility` is set:
- Re-run Phase 3 from iter_0 with `seed = run.json.seed + 1` (same eval_set.jsonl, same
  weak model pin, same proposer pin, same budget).
- Compare `best.holdout_score` across the two seeds. Write `reproducibility.json` with
  the two scores and Δ.
- Note that this is documentation of divergence from synth-self-improve-rl, not a hard
  requirement. synth-self-improve-rl can reproduce from the iter_0 checkpoint
  deterministically; daycare can't because LLM sampling is not deterministic and the
  proposer is autoregressive. The reproducibility run is the closest analog.

---

## Phase 5 — Finalize [v2]

1. Update `run.json.status` ∈ {`completed`, `budget_exhausted`, `max_iters_reached`,
   `aborted`, `model_drift`}; add `end_ts`, `total_duration_s`, `actual_cost_usd`.
2. Write `RUN_DIR/optimized/<slug>/` = best bundle (SKILL.md + scripts/).
3. Append `## Final summary` to `scratchpad.md`: per-iter score table, A/B verdicts,
   length-quartile movement table, recommended action.
4. Print summary table.

`daycare promote <project> <slug>`:
- Copies `RUN_DIR/optimized/<slug>/` → `~/.watchmen/bundles/<project>/skills/<slug>/`.
- Appends evolution provenance to `_curation_log.md` (run_id, Δ scores, A/B verdicts).
- **W4 — `_pinned.json` is a flat JSON array of slug strings**:
  - If file doesn't exist: create with `[slug]`.
  - If file exists: load (must be a list), append slug if not already present, write back.
  - Do NOT invent a richer schema; this matches watchmen's contract and prevents the
    curator from overwriting our promoted bundle on next run.
- Updates `_manifest.json` mtimes.

---

## Information firewall (non-negotiable) [v2]

The proposer LLM context may contain:

| Data | Allowed | Notes |
|---|---|---|
| Train slice: anonymized prompts + references + rubrics | ✅ | Raw prompts NEVER (M7) |
| Train slice: anonymized weak-model failing completions | ✅ | Cropped 300 chars, anonymized |
| Held-out slice: aggregate score only `{holdout_score, n_holdout, by_type, by_length_quartile}` | ✅ | |
| `invocations_render` / `unused_render` aggregates | ✅ | K2 |
| Prior `mutation_log.md` entries (anonymized) | ✅ | |
| Prior `history.jsonl` aggregates | ✅ | |
| Held-out prompts, references, rubrics (raw OR anonymized) | ❌ | |
| Held-out per-row completions | ❌ | Path-isolated to `held_out_log/` |
| Raw (un-anonymized) prompts, references, completions | ❌ | M7 |

Enforced by:
1. **Path isolation**: held-out completions written to `held_out_log/` dirs never
   included in proposer's context-building reads. Refuse even if user requests override.
2. **Anonymization at eval-set construction time** (Phase 1e): the proposer-readable
   fields are pre-anonymized, so even an accidental read of `eval_set.jsonl` doesn't leak.
3. **Output-side leak scanner** (Phase 3b.5): the proposer's emitted patches are scanned
   for any leaked eval identifier; `leak-policy=zero` rejects on any hit.

---

## Daemon mode [v2]

Standalone launchd/systemd units separate from watchmen's units.

| Cadence | What |
|---|---|
| Every 2h | Re-scan corpus.db for new skill-firing sessions; refresh selector ranking |
| Daily, off-peak (03:00 local) | Full `daycare run` on the highest-priority skill in most-active project |
| Weekly | Re-run Phase 1d calibration to detect eval drift |

`daycare daemon install` writes units. Does not interfere with watchmen's launchd units.
The daemon wraps each `daycare run` with the SIGALRM watchdog (K9); if the budget is
exceeded, the watchdog cleanly exits the loop and runs Phase 4 — the daemon never lets a
job run past `budget + 30min`.

---

## Key non-negotiables [v2]

1. **Information firewall**: proposer never sees held-out slice content, never sees raw
   (un-anonymized) prompts/references/completions. Enforced via path isolation +
   construction-time anonymization + output leak scanner.
2. **Never modify `~/.watchmen/bundles/`, `analyses/`, or `corpus.db`** during evolution.
   Read-only until `promote`.
3. **Knowledge-gap filter is the eval gatekeeper**: reject score=0 (bottleneck) and
   score≥0.9 (solved). Only the middle survives. Minimum 60 evals (M8).
4. **Token regularizer is mandatory, calibrated**: `λ_init=0.00001`, anneals as
   `λ_N = min(λ_cap, λ_init × (1 + 0.5×N))`, `λ_cap=0.05`. Hard token cap 2500
   on SKILL.md.
5. **Fixed anchor across every iter** = same `eval_set.jsonl`, same `weak_model` version
   pin (M6), same temperature, same seed, same rollouts. The bundle genealogy is
   GEPA rolling-best (M1) — this is the documented divergence from
   synth-self-improve-rl, justified because text edits compose and the fixed anchor lives
   at the eval/sampling level.
6. **Phase 4 runs AFTER on its own time**: `--budget` never reserves time for baseline
   runs. SIGALRM watchdog (K9) ensures clean handoff.
7. **Every mutation is deterministic and logged**: `history.jsonl` per-attempt +
   `mutation_log.md` per-iter.
8. **Domain-agnostic**: no assumptions about project type. The eval extractor adapts via
   classification.
9. **Sentinel-block mutations only**: unified diffs are rejected at parse time. Strict
   traversal/duplicate/missing-file guards (K1).
10. **OR model version pinning** (M6): exact version string captured at iter_0; any drift
    aborts the run with `model_drift` status.

---

## Connections to kai-skills evolution work

The `skill_evolve/` system in this repo is the GEPA-on-skillsbench predecessor. Key lessons
that carry forward — and where they're enforced in v2:

- **Anonymize evals from the proposer** (v5/v6 routing-exploit lesson): see M7 + Phase 1e.
  Strip session IDs, project names, user-specific identifiers from prompts before passing
  to the proposer. Now a non-negotiable in the spec body, not just a lesson.
- **Failure-trace-conditioned mutation produces task-specific patches at zero scale**
  (Phase E v9d lesson): minimum 2–3 example prompts per cluster + paraphrasing in
  `weakness_report.md`. (Phase 3a.)
- **Empty baseline before promoting** (writeup retractions lesson): Baseline A (empty
  bundle) is mandatory, not optional. A cheap $0.10 control run saved multiple
  retractions in Phase E.
- **Denominator matters** (evolution denominator critique): with ≥60 evals (≥30 holdout),
  a +0.033 improvement is ≥1 eval flipped. With 5 rollouts per eval, single-roll variance
  is absorbed. M8 raises the floor; epsilon = `max(0.01, 1/n_holdout)`.
- **Max workers ≤ 2 for OpenRouter 20MB/hr cap**: the OR org-level cap hit in Phase E v4.
  daycare defaults to `--max-workers 2` (CLI + run.json).
- **Mandate frontmatter fixes invocation, not score** (Phase E v9d): `invocations_render`
  and `unused_render` in weakness_report.md (K2) separate the two failure axes
  ("didn't fire" vs "fired but didn't help").
- **DeepSeek-v4-pro as proposer** (Phase E v6 unlock): default proposer is the strongest
  available OR model; cheap models tried only after the strong model produces a non-zero
  lift on this skill.
- **OPENROUTER_RAW_LOG_DIR for 12h-run debugging** (Phase E v6+v9d): K5 captures every raw
  response to disk when the env var is set.
- **Sentinel-block mutation format > unified diff** (skill_evolve lesson): K1. Unified
  diffs get mangled by every LLM tested; sentinel blocks parse reliably.
- **Pre-launch ping every API key** (global rule): Phase 0 step 4 — 5-token request
  against every OR model used.

---

## Open questions (to resolve before/during implementation)

1. **Rollout parallelism**: with K=6 candidates × ≥30 held-out evals × 5 rollouts = 900+
   calls per iter, and OR's 20MB/hr cap, what's the actual throughput? Likely: score
   candidates sequentially (6 serial passes), parallelize within each pass at workers=2.
   Smoke-3 short-circuit (K6) should cut this 50–80% on broken candidates.

2. **Context injection mechanism**: how does the weak model "use" the skill? Options:
   - Inject SKILL.md content as a system prompt prefix.
   - Inject as a skill XML block in the user message.
   - Simulate Claude Code's actual skill injection format (prepend skill content to
     system prompt the way Claude Code does at `~/.claude/skills/`).
   Default: system-prompt prefix (simplest, works on all OR models). Measure first
   iter_0 score with each option; pin the choice in `sampling.json`.

3. **OR seed reproducibility**: OpenRouter's `seed` parameter is best-effort (not
   guaranteed deterministic). Run 5 rollouts to absorb variance. Track per-eval score
   variance in `eval_summary.json` as a diagnostic.

4. **eval drift detection** (Phase 1d weekly re-run): if new sessions arrive and scores on
   the frozen eval_set.jsonl shift by >0.05, flag it — even with version pinning the
   weak model can shift if OR's slug-to-version map updates. M6 catches version-string
   drift; this catches behavioural drift at the score level.

5. **Script mutation blast radius**: bundles' `scripts/` is self-contained (confirmed in
   watchmen exploration), so cross-bundle damage is impossible — but the promote step
   should diff `scripts/` against the original and require user confirmation if deletions
   are proposed.

6. **Anonymizer recall**: a 100%-recall regex-based anonymizer is impossible. Plan:
   ship a baseline regex set; the leak scanner (K3) is the safety net at output time. If
   `leak_log.md` shows recurring leaks, extend the anonymizer rules.

7. **Subprocess startup cost** (K10): spawning a Python subprocess per rollout adds ~200ms
   overhead. At 900 rollouts/iter × 200ms = 3 min/iter overhead. Acceptable. If it
   becomes the bottleneck, switch to a long-lived rollout worker process consuming a
   queue.

---

## Implementation phases (sequenced) [v2]

| Phase | Module | What | Est. |
|---|---|---|---|
| 0 | cli.py, providers.py, doctor | Package scaffold, OR ping, OR version-pinning (M6), raw-log-dir (K5), W1 doctor, daycare init | 1 day |
| 1 | corpus.py, anonymize.py | `parse_transcript()` (W2), W3 exists-guard, W6 project_dir mapper, anonymize regex set (M7) | 1.5 days |
| 2 | eval_builder.py | Session pull → type classification → rubric gen → calibration → split → anonymization → round-trip gate (M2) | 2 days |
| 3 | verifier.py, runner.py | LLM judge scoring loop, subprocess-isolated rollouts (K10), aggregate stats (W9), path isolation, smoke-3 (K6) | 1.5 days |
| 4 | anchor.py | Phase 2 baseline scoring | 0.5 day |
| 5 | selector.py | Traffic×error + analyst-priority ranking | 0.5 day |
| 6 | mutator.py, leak_scanner.py | Sentinel-block parser/validator/applier (K1), shebang insurance (K7), leak scanner (K3) | 1 day |
| 7 | evolve.py | Weakness analysis (+ length-quartile M3, invocation telemetry K2) + multi-turn agent K=6 (W7) + candidate scoring + accept/discard + history.jsonl (K4) + watchdog (K9) | 2.5 days |
| 8 | controls.py + finalize.py | Baseline A/B (renamed, M4) + optional reproducibility run + summary + promote logic (W4) | 1 day |
| 9 | daemon.py | launchd/systemd units, cadence loop | 1 day |
| 10 | tests/ + integration | End-to-end on ctf/gpu-pod-sniping with small eval set | 1.5 days |

**Total**: ~13.5 days focused work.

**First milestone**: `daycare eval-build ctf` producing a valid `eval_set.jsonl` with ≥60
surviving evals — this validates the entire Phase 1 pipeline on real data before any
evolution compute is spent.

**Second milestone**: `daycare run ctf --max-iters 2 --rollouts 2` end-to-end run (cheap)
with all guards engaged (anonymizer, leak scanner, sentinel mutator, subprocess runner,
SIGALRM watchdog). Verifies the loop closes before spending real compute.

---

## Directory output layout [v2]

```
~/.watchmen/daycare/
├── runs/
│   └── <project>-<UTC>/
│       ├── run.json
│       ├── doctor.json
│       ├── selector_log.md
│       ├── eval_extraction_log.md
│       ├── eval_set.jsonl              ← contains anonymized_* fields per row
│       ├── metrics.json                ← {iters: [{iter, holdout_score, fitness, tokens, by_length_quartile}]}
│       ├── history.jsonl               ← append-only per-attempt outcomes (K4)
│       ├── leak_log.md                 ← only if leak_policy=warn
│       ├── scratchpad.md
│       ├── iter_0/
│       │   ├── bundle/
│       │   │   ├── SKILL.md
│       │   │   └── scripts/
│       │   ├── eval_summary.json
│       │   └── sampling.json           ← includes model_version_pin
│       ├── iter_<N>/
│       │   ├── weakness_report.md      ← clusters + length-quartile table + invocations_render + unused_render
│       │   ├── candidates/
│       │   │   └── c{0..5}/
│       │   │       ├── proposer_transcript.jsonl   ← multi-turn Agent invocation log (W7)
│       │   │       ├── bundle/SKILL.md
│       │   │       ├── bundle/scripts/
│       │   │       └── eval_summary.json
│       │   ├── mutation_log.md
│       │   ├── bundle/                 ← winner (if promoted)
│       │   └── eval_summary.json
│       ├── held_out_log/               ← NOT proposer-readable (path isolation)
│       │   └── iter_<N>/c{0..5}/<eval_id>_<rollout>.json
│       ├── raw_llm/                    ← only if OPENROUTER_RAW_LOG_DIR set (K5)
│       │   └── <timestamp>_<call_id>.json
│       ├── baseline_a_empty/
│       ├── baseline_b_fewshot/
│       ├── reproducibility/            ← only if --reproducibility
│       │   ├── run.json
│       │   └── (full subtree of a second-seed run)
│       ├── best_iter.json
│       └── optimized/<slug>/
│           ├── SKILL.md
│           └── scripts/
└── daemon/
    ├── schedule.json
    └── logs/
```

---

## Skill injection format [v2 — final audit addition]

Resolved from Open Question #2. Evidence from corpus.db transcripts: Claude Code injects skills
as a system-prompt prefix, no XML wrapper:

```
Base directory for this skill: <SKILL_SLUG_PLACEHOLDER>

<SKILL.md body — frontmatter STRIPPED>

--- File: scripts/<name>
<file contents, truncated to 4000 chars with "...[truncated]" marker if over>
```

Daycare implements this in `daycare/src/daycare/injection.py::build_skill_system_prompt(bundle_dir)`.
The exact byte template's SHA256 is written to `sampling.json.injection_format_hash` at iter_0.
Any deviation in subsequent iters aborts the run (variant of M6 version-pinning, applied to
the injection substrate). Frontmatter is stripped because Claude Code strips it; scripts are
inlined because OR has no filesystem. The base-dir line is kept but path is placeholdered
(`<SKILL_SLUG_PLACEHOLDER>`) per M7.

---

## Phase 1 judge prompt templates [v2 — final audit addition]

All four Phase 1 judge calls are stateless single-shot LLM calls (NOT agent loops). Judge
temperature = 0. Request `response_format: json_object` via OR where supported.

### 1b: classification prompt

```
SYSTEM: Classify the assistant's turn into exactly one of:
  script_gen   — assistant wrote substantive shell/Python/JS code (≥3 lines of actual code)
  skill_invoke — a named skill was invoked AND assistant_text references the skill outcome
  procedural_qa — assistant explained a procedure, concept, or made a decision (text only)
  discard      — confirmation, single-word, live-infra-dependent, or no substantive content

Output JSON only: {"type": "...", "reason": "≤20 words"}

USER: {
  "user_turn": "<verbatim>",
  "assistant_text": "<verbatim, first 1000 chars>",
  "tool_calls": [{"tool": "<name>", "input_preview": "<first 200 chars of input>"}],
  "skill_name_in_session": "<slug or null>"
}
```

Hard rule: `skill_invoke` requires `skill_name_in_session != null`. `script_gen` requires
at least one Bash/Edit/Write tool_call with ≥3 lines of code content in input_preview.
On parse failure: retry once; if still fails, classify as `discard`.

### 1c: rubric generation prompt

```
SYSTEM: Write a scoring rubric for a weak LLM called <WEAK_MODEL_NAME>.
Rules:
- Forbidden: criteria requiring "matches reference exactly", "same approach as reference",
  live internet access, filesystem access, or domain knowledge beyond what a 30B model knows.
- Required: concrete observable properties of the CANDIDATE TEXT ALONE (no reference comparison).
- Format: "Score 0.0–1.0. Award 1.0 if: [criterion]. Award 0.5 if: [partial]. Award 0.0 if: [failure]."
- Max 150 words.

USER: type=<type>, anonymized_prompt=<prompt>, anonymized_reference=<reference first 500 chars>
```

### 1d: calibration scoring prompt (also used in Phase 3c for rollouts)

```
SYSTEM: Score the candidate response 0.0–1.0 using the rubric below.
Output JSON: {"score": <float 0-1>, "reasoning": "≤60 words"}
Do NOT compare to any reference. Judge only what the candidate text says.

USER: prompt: <anonymized_prompt>
rubric: <rubric>
candidate: <candidate_completion, first 2000 chars>
```

On JSON parse failure: do NOT fall back to 0.0. Retry once. If still fails, log
`judge_parse_error` to `history.jsonl` and exclude the eval from this rollout's aggregate
(not scored 0 — that would bias calibration).

### 1a: acceptance signal

Regex-first, LLM only on ambiguous cases.

**Regex `rejected`** (applied to next_user_turn, case-insensitive):
- Starts with: `no |wrong |actually |that's not |try again |re-?do |not right`
- Contains: `"<quoted ≥15-char substring of assistant_response>" followed by correction verb`

**Regex `accepted`**:
- References prior output via anaphora: `\b(it|that|the script|the code|this|the approach)\b`
  followed by a follow-up verb or question mark.

**LLM fallback** (if neither regex fires):
```
SYSTEM: Did the user accept or reject the assistant's prior response?
Output JSON: {"accepted": true|false, "confidence": 0.0–1.0}
USER: assistant_response=<first 300 chars>, next_user_turn=<verbatim>
```
Drop the eval candidate if `confidence < 0.6`.

### 1a: theme-boost scoring

```
SYSTEM: Score 0.0–1.0 how closely the user's prompt matches any theme in the analyst notes.
Output JSON: {"relevance": <float>}
USER: prompt=<anonymized_prompt>
analyst_themes=<_running.md "Skill candidates" section, first 800 chars>
```

Apply as: `theme_boost_multiplier = 1.0 + 0.5 × relevance`. Multiply into candidate priority
before the 3Q-filter.

---

## Proposer agent tool inventory [v2 — final audit addition]

Each of the K=6 candidate agents (Phase 3b, W7) is a `watchmen.Agent` invocation with these
tools. `max_iter=8` per agent (not the default 24 — cost guard). `max_cost_usd` per agent =
`run.json.estimated_cost_usd / (K × max_iters × 4)`.

| Tool | Description | Access scope |
|---|---|---|
| `read_weakness_report()` | Returns `weakness_report.md` in full | Read-only |
| `list_parent_bundle_files()` | Enumerate target skill's parent bundle | Read-only |
| `read_parent_bundle_file(path)` | Read SKILL.md or any `scripts/*` in target skill | Read-only |
| `read_peer_skill(slug, path)` | Read SKILL.md or scripts from OTHER skills in same project bundle | Read-only; blocked outside `bundles/<project>/skills/` |
| `read_mutation_log()` | Returns prior-iter `mutation_log.md` (anonymized) | Read-only |
| `read_history_aggregates()` | Returns `history.jsonl` rolled up to per-iter outcome counts (NOT raw rows) | Read-only |
| `validate_sentinel_patch(patch_text)` | Dry-run `mutator.parse_and_validate()` — returns errors without applying | Lets agent self-correct before terminal |
| `lint_script(path, content)` | Runs `py_compile`/`bash -n` on a candidate script body | Returns pass/fail + error message |
| `count_skill_tokens(content)` | Returns `tiktoken cl100k_base` count for an SKILL.md draft | Lets agent stay under 2500 cap |
| `finish_candidate(patch_text, target_cluster, reasoning)` | **Terminal tool.** `patch_text` = full sentinel-block payload; `reasoning` ≤200 words | Exits agent loop; mutator/leak-scanner/scorer take over |

**No tool exposes**: `held_out_log/`, holdout rows from `eval_set.jsonl`, or per-row completions.
Path isolation enforced in the tool handlers themselves.

**Target-cluster assignment across K=6** (deterministic from `run.json.seed`):

Weakness clusters are ranked by severity. Round-robin assign top-3 clusters:
- c0, c3 → cluster_1 (most severe)
- c1, c4 → cluster_2
- c2, c5 → cluster_3

If fewer than 3 clusters, duplicate highest. Temperature schedule per slot:
`[0.3, 0.6, 0.9, 0.3, 0.6, 0.9]` — each cluster gets both a conservative and an
exploratory candidate. Pass `target_cluster_name` verbatim in the final user turn.

---

## Production failure modes and defenses [v2 — final audit addition]

Five failure modes that would silently kill a 12-iter ctf run. All are spec-level fixes
(not implementation details):

| # | What fails | When | Spec fix |
|---|---|---|---|
| 1 | **Calibration uses 1 rollout; iter_0 uses 5.** Evals that pass calibration (0 < score < 0.9) score 0 deterministically at iter_0 due to rollout variance, making the holdout floor artificially low and promoting skills over nothing. | iter_0 baseline | Run Phase 1d calibration with `n_rollouts=3` (mean over 3, not 1). Match to anchor sampling. |
| 2 | **Stratified split empties one type in holdout.** E.g. 61 evals: 58 `script_gen`, 2 `skill_invoke`, 1 `procedural_qa`. Holdout gets 0 `skill_invoke`. `by_type` mean divides by zero → fitness NaN → all candidates rejected silently. | Phase 1e | After split, assert `n_holdout_per_type ≥ 3 for every present type` else abort with `insufficient_stratification`. |
| 3 | **OR 20MB/hr cap hits mid-iter.** Candidate gets partial score over 40/150 evals before 429; the partial score is accepted as complete, biasing the winner selection. | Phase 3c, iter ≥ 2 | In `verifier.aggregate`, require `successful_evals / n_holdout ≥ 0.90` else mark candidate `eval_error` in `history.jsonl` and exclude from winner selection (NOT scored as 0.0). |
| 4 | **All K=6 fail smoke-3 for 3+ consecutive iters.** Run burns its 12-iter budget on a cluster the proposer can't crack (e.g. binary-exploit reasoning), logging "no improvement" silently every iter. | Phase 3c, iter ≥ 2 | Add `stall_detector`: after 3 consecutive iters where ALL candidates are `smoke_failed` or `rejected_by_fitness`, abort with `status: stalled` and proceed immediately to Phase 4. |
| 5 | **Anonymizer misses external CTF repo slugs.** `ctf` sessions reference GitHub repos not in `projects.json`. Anonymizer regex misses them; leak scanner only knows `projects.json` slugs; proposer ingests raw paths, emits them in patches. | Phase 1e + Phase 3b.5 | Extend `leak_scanner` to also flag any 12-char n-gram that appears in ≥2 different `prompt` fields in `eval_set.jsonl` (cross-eval recurrence = identifier signature). Add to reject set regardless of `projects.json`. |

Additional spec-level clarifications from this audit:
- `parse_transcript()` must also scan `tool_call.input` fields for identifiers (MCP tool args
  contain paths and slugs the anonymizer won't see in text fields alone).
- `watchmen.Agent` default `max_iter=24` is too high for K=6 × 12 iters. Pin `max_iter=8`
  per candidate agent in `evolve.py` and set an explicit `max_cost_usd` ceiling per agent.
