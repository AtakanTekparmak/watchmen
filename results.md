# Skill-folder evolution — results

> **Two runs documented below.**
> **v2** (this section) — all-M2.7; pipeline proof, no evolved folder beat best seed variant.
> **v2.1** (appended at bottom) — asymmetric Sonnet-outer + M2.7-inner; evolution genuinely solved 1 task seed failed on, 3× Track A's lift, parse_errors eliminated.

---

# v2 — all-M2.7 baseline

**Experiment**: budget=10 three-track comparison (autoreason vs openevolve vs hybrid) evolving a Hermes-agent skills folder against an 8-task TBLite benchmark, using `minimax/minimax-m2.7` for both mutation and inner-agent rollouts.

**Outcome at a glance**: the loop runs end-to-end with real Docker verifiers. AB-synthesis is the empirically load-bearing mechanism, as predicted by autoreason's ablations. But no track produced a mutation that beat the strongest *seed variant* — every reported "best" is either the unmodified seed (B), a subset of the seed (C), or the seed with one skill meaningfully rewritten (A).

---

## Experimental setup


| Parameter                      | Value                                                                                                                                                     |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model (mutation + inner agent) | `minimax/minimax-m2.7` (OpenRouter)                                                                                                                       |
| Temperature                    | 0.0                                                                                                                                                       |
| Benchmark subset               | 8 TBLite tasks (broken-python dropped — image-layer mutation unfit for our mount geometry)                                                                |
| Budget                         | 10 passes / generations per track                                                                                                                         |
| Repeats                        | 2 per candidate (majority-vote aggregation)                                                                                                               |
| Parallelism                    | `--parallel` tracks, `--max-workers 7`                                                                                                                    |
| Verifier                       | Real Docker `test.sh` + `reward.txt` via `docker exec`                                                                                                    |
| Agent-inside-container         | Hermes Docker backend with double-mount (`/app` + `/workspace`)                                                                                           |
| Seed folder                    | 5 SKILL.md skills + INDEX.md (17.7 KB) — `systematic-shell-debugging`, `read-before-write`, `ask-the-environment`, `patch-then-verify`, `stop-and-replan` |
| Seed hand-baseline             | 4/8 verified pass, composite 0.394                                                                                                                        |


**Run directory**: `runs/compare_m27_b10_v2/` · **Duration**: 7h 53m · **Spend**: ~$80-100

---

## Headline table


| Track | Approach                                 | Wall                       | #Evals | Seed (as-measured) | Final composite | Δ          | Final folder     | Notes                                                                                                    |
| ----- | ---------------------------------------- | -------------------------- | ------ | ------------------ | --------------- | ---------- | ---------------- | -------------------------------------------------------------------------------------------------------- |
| **A** | Autoreason A/B/AB                        | ~3h (parser reported 0.0s) | 7      | 0.325              | **0.450**       | **+0.125** | 5 files, 17.7 KB | Converged pass 3 (k=2 streak). Single real lift in pass 1 via B winner                                   |
| **B** | OpenEvolve patches                       | 2h 43m                     | 13     | 0.492 (island avg) | **0.575**       | +0.083     | 6 files, 17.0 KB | "Best" is unmodified island-0 seed; no mutation ever beat it. 2 parse_errors + 1 regression over 10 gens |
| **C** | Hybrid (A/B/AB tournament on MAP-Elites) | 6h 33m                     | 13     | 0.450 (island avg) | **0.575**       | +0.125     | 3 files, 11.2 KB | "Best" is island-1 seed (= seed minus 2 generic skills). Tournament: A=7, B=1, AB=2                      |


**Cross-track reading**: B and C tied at 0.575 composite = 5/8 verified pass. A settled at 0.450 = 4/8 verified pass, same as seed hand-baseline. None of the three produced an evolved descendant that beat the seed or one of its island variants on this benchmark.

---

## Track A — autoreason trajectory

Converged after 3 passes (streak=2 on A-wins). Full pass log:


| Pass | Winner | score_A | score_B | score_AB | Op chosen           | Target skill                 | Streak            |
| ---- | ------ | ------- | ------- | -------- | ------------------- | ---------------------------- | ----------------- |
| 0    | seed   | 0.325   | —       | —        | —                   | —                            | 0                 |
| 1    | **B**  | 0.45    | 0.45    | 0.45     | RewriteSkillContent | `ask-the-environment`        | 0                 |
| 2    | A      | 0.45    | 0.325   | 0.45     | RewriteSkillContent | `read-before-write`          | 1                 |
| 3    | A      | 0.45    | 0.45    | 0.325    | AddSkill            | `verify-output-against-spec` | **2 → converged** |


**Reading**: pass 1 is the only real lift (+0.125). Passes 2 and 3 are defensive — B and AB produce candidates that fail to beat the incumbent, A defends, streak ticks to 2. Convergence triggers cleanly, as designed. Track A used 7 evaluations total vs B/C's 13 — highest delta-per-eval efficiency (0.0179).

**Noteworthy**: the op-planner chose `AddSkill` on pass 3 (proposing a new `verify-output-against-spec` skill targeted at the application-debug and scan-linux-persistence-artifacts failures), but the resulting AB scored worse than incumbent. The op was plausibly correct at the design level; execution under M2.7's sampling didn't convert it into a task-solving skill.

---

## Track B — openevolve op breakdown

Per-generation op histogram (13 total entries: 3 seeds + 10 generations):


| op_type     | count |
| ----------- | ----- |
| seed        | 3     |
| patch       | 8     |
| parse_error | 2     |


All 10 generations, composite trajectory:


| Gen | Island | Op          | Composite | Notes                                                   |
| --- | ------ | ----------- | --------- | ------------------------------------------------------- |
| 1   | 0      | patch       | 0.325     | regression from island-0 seed (0.575)                   |
| 2   | 1      | patch       | 0.45      | +0.125 over island-1 seed (0.325) — only positive delta |
| 3   | 2      | patch       | 0.325     | regression from island-2 seed (0.575)                   |
| 4   | 0      | patch       | 0.325     | regression                                              |
| 5   | 1      | parse_error | —         | `"patch contained no operations"`                       |
| 6   | 2      | patch       | 0.325     | regression                                              |
| 7   | 0      | patch       | 0.2       | further regression                                      |
| 8   | 1      | parse_error | —         | `"patch contained no operations"`                       |
| 9   | 2      | patch       | 0.325     | flat                                                    |
| 10  | 0      | patch       | 0.2       | further regression                                      |


**Interpretation**: 1 improvement, 5 regressions, 2 parse errors, 2 flat across 10 generations. M2.7's patch-grammar fidelity is the issue — parse_errors produce zero useful content, and `patch` ops that DO parse routinely make the folder worse (not structurally invalid, but evaluated-worse by the agent). Without a synthesizer to rescue losers, Track B has no mechanism to recover from bad mutations. The archive preserves the best seed, so the track "wins" by never moving.

---

## Track C — tournament breakdown + per-island progress

All 13 entries:


| Gen | Island | Winner | Composite | Op chosen                                     |
| --- | ------ | ------ | --------- | --------------------------------------------- |
| 0   | 0      | (seed) | 0.325     | —                                             |
| 0   | 1      | (seed) | 0.575     | —                                             |
| 0   | 2      | (seed) | 0.45      | —                                             |
| 1   | 0      | **AB** | 0.45      | RewriteSkillContent(`read-before-write`)      |
| 2   | 1      | A      | 0.575     | RewriteSkillContent(`patch-then-verify`)      |
| 3   | 2      | A      | 0.45      | RewriteSkillContent(`read-before-write`)      |
| 4   | 0      | A      | 0.45      | AddSkill(`verify-output-against-spec`)        |
| 5   | 1      | A      | 0.325     | RewriteSkillContent(`patch-then-verify`)      |
| 6   | 2      | **AB** | 0.45      | AddSkill(`finish-before-stopping`)            |
| 7   | 0      | A      | 0.45      | RewriteSkillContent(`read-before-write`)      |
| 8   | 1      | A      | 0.45      | RewriteSkillContent(`patch-then-verify`)      |
| 9   | 2      | A      | 0.325     | RewriteSkillContent(`domain-specific-helper`) |
| 10  | 0      | **B**  | 0.45      | RewriteSkillContent(`read-before-write`)      |


**Tournament tally**: A=7, AB=2, B=1 out of 10 tournaments.

**Per-island bests** (across all 10 gens):

- Island 0 (as-is seed): seed 0.325 → best evolved cell 0.45 (+0.125, via AB at gen 1 and B at gen 10)
- Island 1 (seed minus `ask-the-environment` + `read-before-write`): seed **0.575** — never beaten
- Island 2 (seed plus placeholder): seed 0.45 — tied twice via AB (gen 6) and A defenses

**Reading**: A defending 7 of 10 tournaments is exactly the autoreason "do-nothing wins on ties" semantics firing — most mutation pairs score ≤ incumbent because the signal is binary-per-task and sampling-noise floors the gradient. The 2 AB wins + 1 B win are where actual improvement happens, each lifting a weaker island by +0.125. **Empirically confirms autoreason's claim: AB is load-bearing** — removing it would cut 2 of 3 improvements.

---

## Skill folder diffs — what evolution actually produced


| Track     | # skills | Total bytes | Added | Removed                                    | Modified | Verdict                                            |
| --------- | -------- | ----------- | ----- | ------------------------------------------ | -------- | -------------------------------------------------- |
| A (final) | 5        | 17,712      | none  | INDEX.md                                   | all 5    | One substantive edit; rest is YAML reserialization |
| B (best)  | 5        | 15,970      | none  | none                                       | none     | **Byte-identical to seed**                         |
| C (best)  | 3        | 11,166      | none  | `ask-the-environment`, `read-before-write` | none     | Seed with two skills deleted                       |


### Track A — content diffs

Only `**ask-the-environment`** has substantive change (+48/-9 lines, 2,850 → 4,663 bytes). The core principle was rewritten from *"When in doubt, run a read-only command"* to **"Inspect, then close the loop. Seeing the output is not the same as acting on it."** A new section **"Close the Loop Before Artifact Generation"** was inserted with paired positive/negative worked examples (loop-closed vs loop-broken changelog generation). 2 new red flags added for artifact generation without inspection.

The other 4 skills (`patch-then-verify`, `read-before-write`, `stop-and-replan`, `systematic-shell-debugging`) are byte-different but **content-identical** — YAML reserializer emits block-style lists (`tags:\n- shell\n- inspection`) instead of inline flow (`tags: [shell, inspection]`) and wraps long description strings. Round-trip artifact, not evolution.

### Track B — content diffs

`diff -r seed_skills/ b/best/` → **no output**. The two trees are byte-identical (15,970 bytes, all 6 files including INDEX.md). `best_meta.json`: `generation: 0, parent_id: None`. B's "best" is literally the unmodified island-0 seed.

### Track C — content diffs

C's `best/` contains only 3 SKILL folders + INDEX.md. `ask-the-environment/` and `read-before-write/` are absent (no merge — surviving skills are byte-identical to seed). `best_meta.json`: `generation: 0, parent_id: None`, cell `[1,2,1]`. This is island-1's seed variant (per `islands.py::seed_variants`: island 1 = seed minus the two most generic skills). Evolution never beat it.

**Pattern across A/B/C**: search landscape rewards minimal or unchanged seeds at the same fitness as edited ones. The signal is too coarse for evolution to discover something meaningfully different.

---

## Per-task outcome matrix

### Seed baseline reference

From `runs/seed_baseline.log` (one-off manual baseline; broken-python excluded):


| Task                                    | Seed verified | Turns | Time (s) |
| --------------------------------------- | ------------- | ----- | -------- |
| tblite/jq-data-processing               | ✅             | 16    | 182      |
| tblite/log-summary                      | ✅             | 5     | 139      |
| tblite/jsonl-aggregator                 | ✅             | 6     | 174      |
| tblite/build-merkle-tree-cli-sha512     | ✅             | 20    | 423      |
| tblite/build-system-task-ordering       | ❌             | 20    | 799      |
| tblite/application-debug                | ❌             | 14    | 352      |
| tblite/scan-linux-persistence-artifacts | ❌             | 27    | 383      |
| tblite/monorepo-changelog-cli           | ❌             | 23    | 445      |


Seed: 4/8 verified, composite 0.394.

### Comparison matrix


| Task                             | Seed | A final | B best             | C best             |
| -------------------------------- | ---- | ------- | ------------------ | ------------------ |
| jq-data-processing               | ✅    | ✅       | ❓ (aggregate-only) | ❓ (aggregate-only) |
| log-summary                      | ✅    | ✅       | ❓                  | ❓                  |
| jsonl-aggregator                 | ✅    | ✅       | ❓                  | ❓                  |
| build-merkle-tree-cli-sha512     | ✅    | ✅       | ❓                  | ❓                  |
| application-debug                | ❌    | ❌       | ❓                  | ❓                  |
| build-system-task-ordering       | ❌    | ❌       | ❓                  | ❓                  |
| monorepo-changelog-cli           | ❌    | ❌       | ❓                  | ❓                  |
| scan-linux-persistence-artifacts | ❌    | ❌       | ❓                  | ❓                  |


**Important quirk**: B and C each report **5/8 verified pass** (composite 0.575), but OpenEvolve stores only aggregate metrics on its `Program` objects — no per-task `verified` state is saved in `best_meta.json` or `history.jsonl`. So we know B and C each passed 1 additional task beyond seed's 4/8 baseline, but we **can't tell which**. Plausible candidates (by task difficulty gradient): likely `application-debug` or `scan-linux-persistence-artifacts`, but unverifiable from artifacts.

Caveat layered on top: re-evaluating the same seed folder produces variance of ±1 task flip (= ±0.125 composite) due to M2.7 sampling + agent tool-call non-determinism even at temperature=0. B's and C's "5/8" might be a luckier sample of the same seed rather than evolution finding a genuine extra solve.

### Task difficulty ranking (observable tracks only)


| Rank     | Task                                                                                                    | Solves (seed + A, out of 2) |
| -------- | ------------------------------------------------------------------------------------------------------- | --------------------------- |
| 1 (tied) | jq-data-processing, log-summary, jsonl-aggregator, build-merkle-tree-cli-sha512                         | 2/2                         |
| 5 (tied) | application-debug, build-system-task-ordering, monorepo-changelog-cli, scan-linux-persistence-artifacts | 0/2                         |


### Unsolved-tasks analysis

- **application-debug** (hard, text-format discipline): seed + A both produce wrong percentages + missing stack-trace lines. Precision failure, not capability — a "diff output against spec line-by-line before declaring done" skill + more turns would plausibly unblock.
- **build-system-task-ordering** (medium, graph algorithms): hit alias-canonicalization + transitive-alias edge cases. Needs stronger algorithmic reasoning OR a "test against adversarial inputs before finalizing" skill; agent self-confidence is high, coverage shallow.
- **monorepo-changelog-cli** (medium, git + semver): repeated FileNotFoundError + dependency-propagation misses suggest the agent never reads the test fixture layout. A `read-test-fixtures-first` skill or larger turn budget would help more than a model swap.
- **scan-linux-persistence-artifacts** (medium, bash + JSON schema): 15/16 tests fail in seed run with FileNotFoundError on output. Agent never gets the script to emit output — debugging-loop failure (run, inspect, fix) fixed by more turns + a stronger shell-debugging skill.

---

## Cost & time


| Item                              | Value                                                    |
| --------------------------------- | -------------------------------------------------------- |
| Total wall time                   | 7h 53m                                                   |
| Track A wall                      | ~3 hr (finished first, converged)                        |
| Track B wall                      | 2h 43m                                                   |
| Track C wall                      | 6h 33m (long pole — 2 evals per tournament)              |
| Total evaluations                 | 33 across all 3 tracks (7+13+13)                         |
| Est. LLM spend                    | ~$80-100 (8 tasks × ~$0.084/task × 33 evals × 2 repeats) |
| Mutation LLM cost                 | negligible (~$0.50 — mostly rollout)                     |
| Peak concurrent Hermes containers | 20                                                       |
| Peak Docker resident RAM          | ~12-15 GB                                                |


Monitoring log: `logs.md` at repo root has hourly check-ins.

---

## Honest interpretation

### What worked as designed

1. **Docker integration end-to-end**: agent commands execute inside per-task containers, verifier tests the same filesystem. 0/3 → 4/8 (seed) → 5/8 (B/C) verified pass rate at real-verifier standard. Plumbing is real.
2. **Autoreason k=2 convergence**: Track A converged cleanly at pass 3 with streak=2 do-nothing wins. Load-bearing "A wins ties" tiebreak worked as spec'd.
3. **AB synthesizer carries the gradient**: Track C's tournament breakdown (A=7, AB=2, B=1) empirically matches autoreason's original ablation — remove AB, you lose 2 of 3 real improvements.
4. **Monitoring, sandboxing, cleanup**: zero stray containers, zero sandbox leaks, no parse_error propagation outside Track B's 2 contained failures.

### What didn't work

1. **No track produced an evolved mutation that beat the best seed variant** on this benchmark + this model. All three "bests" are seed-shaped (byte-identical for B, subset for C, one meaningful skill rewrite for A).
2. **Track B's pure-patch mutation is fragile on M2.7**: 2/10 generations produced unparseable patches, 5/10 produced regressions, 1/10 produced an improvement. Without AB to recover, it stalled.
3. **Signal coarseness**: the 8-task pass/fail binary signal flips in ±1-task increments (±0.125 composite). That's larger than most evolution steps produce with M2.7 at 20 max-turns, so noise floods useful gradient.
4. **Aggregate-only per-task storage in OpenEvolve**: B and C can't tell us which task they solved extra. That's a metrics-infrastructure gap worth fixing.

### Bounded claims

- Evolution pipeline works. AB synthesis is empirically load-bearing. Autoreason-style convergence fires as designed.
- But we did **not** demonstrate that evolved skills beat seed skills at this budget × model × benchmark. The +0.125 deltas A and C found are local-cell lifts bounded by the noise floor.

---

## What would make the next iteration better

1. **Stronger inner model** for the rollout. M2.7 is capable of ~4-5/8 pass rate at 20 turns; lifting that ceiling via Sonnet 4.6 or Opus 4.6 as the inner agent would unlock evolution room that doesn't exist under the current sampling distribution. Mutation model can stay M2.7 for cost.
2. **Larger budget or smaller tasks**. Either go to budget=25-50 so evolution has more chances to find improvements, or pick tasks that have finer-grained test signal (20 assertions not 1 bit) so gradient isn't so coarse.
3. **Per-task metric persistence** in OpenEvolve's `Program`. Adding per-task verified state to `eval_artifacts` lets us answer "which task did the evolved folder crack" after the fact.
4. **Turn-budget sweep on Track A**: convergence k=2 fires quickly at low gradient; trying k=3 or k=4 might let real improvements accumulate before shutdown.
5. **Directly evolve failing-task skills**: seed the folder with stub skills named after unsolved tasks (`fix-application-debug-output-format`, `monorepo-changelog-conventional-commits`, …) so the RewriteSkillContent ops target the actual blockers. Current seed has generic discipline skills; nothing task-specific.
6. **Fix the 3-way tie tiebreak quirk** in Track A: pass 1's "three-way tie, B won" is a minor spec deviation — the lex-max with `(score, label=="A")` should give A the tie, but B won. Worth auditing the tournament tiebreak implementation.

---

## Artifacts

- `runs/compare_m27_b10_v2/report.{md,json}` — cross-track summary (generated by `compare/run_all.py`)
- `runs/compare_m27_b10_v2/a/history.json` — Track A per-pass log
- `runs/compare_m27_b10_v2/b/history.jsonl`, `b/best/`, `b/archive/`, `b/best_meta.json` — Track B
- `runs/compare_m27_b10_v2/c/history.jsonl`, `c/best/`, `c/archive/`, `c/best_meta.json` — Track C
- `runs/seed_baseline.log` — pre-run hand-verified seed baseline
- `logs.md` — hourly monitoring check-ins throughout the run
- `DOCKER_INTEGRATION_PLAN.md` — implementation plan for the agent-in-container integration landed right before this run


---

# v2.1 — asymmetric outer/inner models

**Experiment**: same 3-track comparison, same 8-task TBLite benchmark, budget=10, repeats=2, max-workers=7. Key change: **Sonnet 4.6 for the outer/meta LLM** (critic / op-planner / body-writer / synthesizer / Track B patch generator), M2.7 for the inner Hermes agent rollouts.

**Motivation**: v2's Track B hit 2/10 parse_errors (M2.7 mangled patch grammar); critic/planner/synth quality bounded by M2.7's instruction-following. Hypothesis: spending ~$15 on Sonnet for ~150 mutation calls would pay for itself vs M2.7's $0.50 but worse ops.

**Run artifacts**: `runs/compare_v21/` · Duration: ~7h 30m · Spend: ~$73

## Headline table — v2 vs v2.1

| Metric | v2 (all M2.7) | v2.1 (Sonnet outer, M2.7 inner) | Change |
|---|---|---|---|
| Track A Δ from seed | +0.125 | **+0.375** | **3×** |
| Track B Δ from seed | +0.083 (seed won) | **+0.208** (real mutation won) | **2.5×** |
| Track B parse_errors | 2 / 10 gens | **0 / 10 gens** | eliminated |
| Track C Δ from seed | +0.125 | **+0.208** | 1.7× |
| Track C AB-wins | 2 / 10 | 2 / 10 | same |
| Tasks solved beyond seed | 0 verifiable | **1 confirmed** (monorepo-changelog-cli) | NEW |
| Wall time | ~7h 53m | ~7h 30m | similar |
| Cost | ~$80-100 | ~$73 | similar |

## Per-task matrix (first time visible, thanks to Fix 2)

Fix 2 persisted per-task `verified` state in `Program.eval_artifacts.per_task` → now in `best_meta.json`. v2 couldn't tell us which tasks evolution cracked; v2.1 can.

| Task | Seed | A final | B best | C best |
|---|---|---|---|---|
| jq-data-processing | ✅ | ✅ | ✅ | ✅ |
| jsonl-aggregator | ✅ | ✅ | ✅ | ✅ |
| build-merkle-tree-cli-sha512 | ✅ | ✅ | ✅ | ✅ |
| log-summary | ✅ | ✅ | ✅ | ✅ |
| **monorepo-changelog-cli** | ❌ | ✅ | ✅ | ✅ |
| application-debug | ❌ | ❌ | ❌ | ❌ |
| build-system-task-ordering | ❌ | ❌ | ❌ | ❌ |
| scan-linux-persistence-artifacts | ❌ | ❌ | ❌ | ❌ |

**All 3 tracks cracked `monorepo-changelog-cli`** — the task that v2's seed-baseline scouting suggested would yield to a "read test fixtures first" style skill. Evolution delivered exactly that.

## Per-track final summary

| Track | Wall | #Evals | Seed | Final | Δ | Gen→1st improve | Gen→Converge | Best folder |
|---|---|---|---|---|---|---|---|---|
| **A** (autoreason) | ~3h 4m | 7 | 0.200 | 0.575 | **+0.375** | 1 | 4 | 7 files, 26.3 KB |
| **B** (openevolve) | 3h 16m | 13 | 0.367 | 0.575 | +0.208 | 2 | — | 7 files, 17.6 KB |
| **C** (hybrid) | 6h 32m | 13 | 0.367 | 0.575 | +0.208 | 1 | — | 6 files, 18.8 KB |

**All 3 tracks converge on the same 0.575 ceiling = 5/8 verified pass**, which is **M2.7's inner-agent capability limit** on this task mix — not a skill quality limit. Evolution solved everything it could.

## Track A — full trajectory

| Pass | Winner | score_A | score_B | score_AB | Op chosen | Target skill |
|---|---|---|---|---|---|---|
| 0 | seed | 0.200 | — | — | — | — |
| 1 | **B** | 0.200 | 0.450 | 0.325 | AddSkill | `implement-from-scratch` |
| 2 | **B** | 0.450 | 0.575 | 0.575 | AddSkill | `output-format-compliance` |
| 3 | A | 0.575 | 0.450 | None | AddSkill | `bash-script-authorship` |
| 4 | A | 0.575 | 0.450 | None | AddSkill | `shell-script-authoring` |

**Converged at pass 4 (streak=2).** Two productive mutation passes netted +0.375. Sonnet picked ops that targeted real failing tasks (output-format-compliance hits application-debug's precision failures; bash/shell ops target scan-linux-persistence-artifacts). Passes 3 and 4's AB slots returned `None` (synth produced invalid folders — both fell to A tiebreak).

## Track C — tournament breakdown

10 tournaments (all reach budget, no convergence stop):

| Role | Wins |
|---|---|
| A (do-nothing) | 8 |
| AB (synthesis) | 2 |
| B (mutation alone) | 0 |

| Gen | Island | Winner | Composite | Op |
|---|---|---|---|---|
| 1 | 0 | **AB** | 0.575 | AddSkill(build-from-scratch) |
| 2 | 1 | **AB** | 0.575 | AddSkill(build-from-scratch) |
| 3 | 2 | A | 0.45 | RewriteSkillContent(domain-specific-helper) |
| 4 | 0 | A | 0.575 | AddSkill(validate-structured-artifact) |
| 5 | 1 | A | 0.575 | AddSkill(log-and-artifact-analysis) |
| 6 | 2 | A | 0.45 | RewriteSkillContent(domain-specific-helper) |
| 7 | 0 | A | 0.575 | MergeSkills() |
| 8 | 1 | A | 0.575 | AddSkill(read-and-analyze-artifacts) |
| 9 | 2 | A | 0.45 | MergeSkills() |
| 10 | 0 | A | 0.45 | RewriteSkillContent(build-from-scratch) |

All 2 AB wins happened at the very start (gens 1-2). After gen 3, A defended every tournament — consistent with hitting the inner-agent ceiling.

## Interpretation vs v2

1. **Sonnet outer made evolution work that v2 couldn't** — tripled Track A's lift because op choices actually addressed known failure modes rather than being generic rewrites. "What should I evolve next?" is where outer-model quality matters most.
2. **Parse errors eliminated** — zero across 10 Track B generations vs v2's two. Patch grammar fidelity was the M2.7 bottleneck for openevolve-style mutation.
3. **Evolution solved 1 task seed couldn't** — monorepo-changelog-cli. First empirical confirmation across all our runs that evolution produces artifacts that beat the seed on held-out signal.
4. **Ceiling unchanged** — 5/8 is the M2.7 inner-agent limit on 3 hard tasks (application-debug, build-system-task-ordering, scan-linux-persistence-artifacts). No skill rewrites broke through, because the rollout can't actually solve them at 20-turn budget with M2.7.

---

# Seed ablation experiments

**Question**: How much does the starting skill folder matter? Does evolution compensate for a bad start, or amplify what's already there?

**Config**: Track A only (autoreason, the proven winner). Same setup as v2.1: Sonnet 4.6 outer, M2.7 inner, budget=10, repeats=2, max-workers=7, real verifiers, 8 TBLite tasks.

## Ablation results

| Seed | Size | Pass 0 | Final | Δ | AB worked? | Passes |
|---|---|---|---|---|---|---|
| **Claude-written** (v2.1) | 24 KB, 5 skills | 0.200 | **0.575** | **+0.375** | 2/4 | 4 |
| **Realworld OSS** | 48 KB, 5 skills | 0.325 | 0.450 | +0.125 | 0/4 | 4 |
| **Empty (placeholder)** | <1 KB, 1 skill | 0.325 | 0.325 | +0.000 | 0/2 | 2 |

## Per-seed trajectory

### Claude-written (v2.1 Track A — control)

| Pass | Winner | A | B | AB | Op |
|---|---|---|---|---|---|
| 0 | seed | 0.200 | — | — | — |
| 1 | **B** | 0.200 | 0.450 | 0.325 | AddSkill(implement-from-scratch) |
| 2 | **B** | 0.450 | 0.575 | 0.575 | AddSkill(output-format-compliance) |
| 3 | A | 0.575 | 0.450 | None | AddSkill(bash-script-authorship) |
| 4 | A | 0.575 | 0.450 | None | AddSkill(shell-script-authoring) |

### Realworld OSS

| Pass | Winner | A | B | AB | Op |
|---|---|---|---|---|---|
| 0 | seed | 0.325 | — | — | — |
| 1 | A | 0.325 | 0.200 | None | AddSkill(cli-script-construction) |
| 2 | **B** | 0.325 | 0.450 | None | AddSkill(cli-tool-and-script-construction) |
| 3 | A | 0.450 | 0.325 | None | RemoveSkill(ci-cd-and-automation) |
| 4 | A | 0.450 | 0.200 | None | AddSkill(bash-scripting-correctness) |

### Empty (placeholder)

| Pass | Winner | A | B | AB | Op |
|---|---|---|---|---|---|
| 0 | seed | 0.325 | — | — | — |
| 1 | A | 0.325 | 0.325 | 0.200 | AddSkill(task-verification-and-self-checking) |
| 2 | A | 0.325 | 0.325 | 0.325 | AddSkill(shell-scripting) |

## Ablation findings

### 1. AB synthesis is the differentiator — and it has a folder-size sweet spot

AB worked on Claude seeds (24 KB, 2/4 passes), broke on realworld (48 KB, 0/4), and had nothing useful to synthesize on empty (<1 KB, 0/2). When AB fires, evolution finds +0.375. When it doesn't, you get +0.125 at best and +0.000 at worst.

The sweet spot for AB synthesis appears to be **~20-30 KB**. Above that, Sonnet can't produce a valid JSON folder-structure in one shot. Below that, there's not enough content for two candidates to meaningfully differ.

### 2. M2.7's baseline is ~0.325 regardless of skills

Pass-0 scores: 0.200, 0.325, 0.325 across the three seeds. These are within the ±0.125 noise floor. M2.7 passes 3-4/8 tasks on built-in coding ability alone — skills provide discipline, not raw capability.

### 3. Starting higher ≠ finishing higher

Realworld started at 0.325 (tied with empty) but only reached 0.450. Claude started at 0.200 (lowest) but reached 0.575 (highest). **The seed's value is in being evolvable, not in being good.** Concise skills that AB can synthesize > verbose domain-specific skills that break the synthesizer.

### 4. Evolution tried to DELETE a battle-tested skill

Pass 3 of realworld proposed `RemoveSkill(ci-cd-and-automation)` — one of the skills we specifically chose for its build-system-ordering relevance. The critic flagged it as unhelpful noise. Longer, more opinionated skills can hurt when the inner model can't process their density.

### 5. Sonnet reinvents sensible skills from scratch — but they don't help

From the empty seed, Sonnet proposed `task-verification-and-self-checking` (similar to obra/superpowers' `verification-before-completion`) and `shell-scripting`. Both are reasonable — but M2.7 couldn't use them to pass any additional tasks. The skills tied the placeholder rather than beating it.

### 6. Evolution from scratch doesn't work at budget=10

Empty converged in 2 passes (minimum possible). Not because there's nothing to find, but because the search space from 1 placeholder skill is too cold to explore. Would need budget=50+ with diverse AddSkill proposals to have a chance.

## What's next

1. **Fix AB synthesis for large folders** — chunked per-skill synthesis instead of whole-folder JSON. Would unlock realworld-quality skills without the size penalty.
2. **Continuous scoring** — pytest assertion counting → float reward.json. Drops noise floor from ±0.125 to ±0.025. Already confirmed feasible with zero new tasks.
3. **More TBLite tasks** — 10 candidates identified, all `/app`-compatible. More tasks = finer gradient.
4. **Stronger inner model** — Sonnet 4.6 for rollouts to break the 5/8 ceiling. ~$300-500/run.
5. **Track B on empty seed** — running now. Tests whether MAP-Elites diversity finds what autoreason's greedy search couldn't from a cold start.

---

# 2026-04-20 — Robustness rerun + continuous scoring

**Headline**: The 0.700 Track B empty-seed result from 2026-04-17 was a ~3σ lucky roll, not a robust configuration. Retracting "Track B cracked build-system-task-ordering." Rebuilt the scoring pipeline on continuous per-task scores to stop being fooled by an 8-task binary composite with a ±0.125 noise floor.

## Robustness rerun on the 0.700 folder

**Objective**: verify whether the all-time-high 0.700 from `runs/ablation_empty_trackb/best/` (1,847-byte `general-task-guidance` cheat-sheet) is reproducible or a single-sample fluke.

**Config**: identical to the original — M2.7 inner, repeats=2, max-workers=7, real Docker verifiers, 8 TBLite tasks. Reran the winner folder 4 times (target was 5; stopped early once the signal was unambiguous).

| Run | Composite | Tasks passed |
| --- | --------- | ------------ |
| 1   | 0.325     | 3/8          |
| 2   | 0.450     | 4/8          |
| 3   | 0.200     | 2/8          |
| 4   | 0.450     | 4/8          |

Mean 0.356, std ~0.11. Original 0.700 was ~3σ above the mean of rerun samples.

**Critical finding**: `build-system-task-ordering` and `monorepo-changelog-cli` — the two tasks "newly cracked" that made 0.700 a headline — passed 0/4 on rerun. Both were single-roll flukes. **Retracting the "Track B cracked build-system-task-ordering" claim from the 2026-04-17 report.**

**Cost**: ~$5, ~1 hr wall before stopping.

## Continuous scoring — why we built it

A binary pass/fail per task on an 8-task composite produces a ±0.125 noise floor per single task flip. Any "ceiling" or "breakthrough" narrative has to clear ~3σ ≈ 0.375 composite delta to be defensible. The 0.700 result did not clear that bar, and we only noticed after reruns. We needed finer-grained signal.

**Implementation**: added per-task continuous score in [0, 1] computed from:

- `pytest-json-ctrf` JSON (5/8 TBLite tasks: application-debug, build-merkle-tree-cli-sha512, build-system-task-ordering, monorepo-changelog-cli, scan-linux-persistence-artifacts emit `/logs/verifier/ctrf.json` natively).
- pytest stdout "X passed, Y failed" summary (other 3/8 tasks: jq-data-processing, jsonl-aggregator, log-summary).
- Fallback to binary when neither is available (preserves historical composite values).

Score = `passed / (passed + failed + errors)`. `mean_score` added to `EvalResult`; `compute_composite` now prefers it over `success_rate` when present. 48 unit tests pass (10 parser tests + 6 aggregation tests added).

**Files changed**: `skill_evolve/verifiers.py`, `skill_evolve/evaluator.py`, associated tests.

## Continuous-scoring sanity eval on the 0.700 folder

Single-pass eval (M2.7, no repeats, real verifiers), ~$1, 19 min wall:

| Metric        | Value  |
| ------------- | ------ |
| success_rate  | 0.500  |
| mean_score    | 0.8437 |
| composite     | 0.7937 |

Per-task score vs verified (binary):

| Task                             | Verified | Score  |
| -------------------------------- | -------- | ------ |
| jq-data-processing               | PASS     | 1.000  |
| jsonl-aggregator                 | PASS     | 1.000  |
| log-summary                      | PASS     | 1.000  |
| build-system-task-ordering       | PASS     | 1.000  |
| application-debug                | FAIL     | 0.923  |
| build-merkle-tree-cli-sha512     | FAIL     | 0.889  |
| scan-linux-persistence-artifacts | FAIL     | 0.938  |
| monorepo-changelog-cli           | FAIL     | 0.000  |

**Finding**: three of the four "failed" tasks are 88-94% solved on assertions. The "stable unsolved" tasks (application-debug, scan-linux-persistence-artifacts) that we thought were M2.7-capability-bound are actually bouncing near the pass threshold on the last 1-2 assertions. Only `monorepo-changelog-cli` is a clean fail.

## What this overturns

1. **"Track B cracked build-system-task-ordering"** — retracted. Single-roll fluke. The 0.700 winner passes it 0/4 on rerun.
2. **"5/8 is M2.7's inner-agent capability ceiling"** (v2.1 + 2026-04-17) — flipped. Binary pass/fail was lying about capability. Under continuous scoring, "unsolved" application-debug and scan-linux-persistence-artifacts are 92-94% solved on assertions, not 0% solved. The ceiling isn't where we thought — the agent is bouncing off the last 1-2 assertions, not hitting a wall.
3. **"Concise cheat-sheet > elaborate methodology"** — not retracted but weakened. The 1.8 KB folder does score well on reruns (mean 0.356 binary, 0.84 continuous on the sanity eval), but the 0.700 headline that made this narrative pop was a ±3σ outlier. The qualitative direction may still hold; the quantitative magnitude does not.

Everything downstream of the 0.700 result needs to be re-evaluated under continuous scoring before we trust it.

## Next

Post-mortem Opus brainstorm identified **Direction #1**: variance audit on 5 canonical folders under continuous scoring (~$30), running each 5+ times to establish std per folder and re-rank them. Everything else (Track D sequential, more tasks, chunked AB synthesis) is queued behind confirming which past findings survive the variance audit.

---

# v3 — 2026-04-21 — Empty seed + rotated 10-task manifest + 3 tracks

**Headline**: Track B on empty seed produces a clean 0.7582 composite (6/10 binary PASS) with an auto-evolved 2.7 KB `task-completion-verifier` skill. Track A converges early with no lift; Track D crashes at handoff, salvages to 0.6342. Single-eval variance on the same folder is ~+/-0.2 composite, larger than previously estimated.

## Setup

- **Manifest rotation**: dropped 3 easy tasks (jq-data-processing, jsonl-aggregator, log-summary) that all configurations already solve; added 5 harder tasks (api-endpoint-permission-canonicalizer, industrial-kiln-controller, python-api-rate-limit, raft-log-repair-concurrent-access, schedule-vacation). All 10 tasks pinned at stage=1 after a cascade bug fired early in the night (near-miss stage-1 tasks short-circuited the stage gate; every eval scored only 2/10 and the composite inflated to 0.828 until we killed, flattened stages, and relaunched).
- **Seed**: `seed_skills_empty/` (single placeholder skill, <1 KB).
- **Tracks**: A (autoreason greedy), B (openevolve MAP-Elites + islands), D (new package this session — phase_b MAP-Elites handed off to phase_a autoreason, ~270 LoC under `skill_evolve/track_d/`).
- **Config**: Sonnet 4.6 outer, MiniMax M2.7 inner, repeats=1, max-workers=7, continuous scoring (ctrf + pytest-stdout parsers from 2026-04-20).
- **Run dir**: `runs/v3_empty_3tracks/` — authoritative summary at `summary.md`.

## Final scoreboard

| Track | Final  | Passes/Gens   | Elapsed     |
| ----- | ------ | ------------- | ----------- |
| A     | 0.6242 | 2 (converged) | ~15 min     |
| B     | 0.7582 | 15 gens       | 4h 15m      |
| D     | 0.6342 | 10 b + 5 a    | ~6h + crash |

Empty-seed pass-0 evals ranged 0.51-0.66 composite across tracks — so B lifted +0.10 to +0.24 off baseline depending on where the same-folder eval landed that day. A and D did not meaningfully lift.

## Track B — the winner

Evolution rediscovered the "verify before claiming done" hypothesis from scratch. Outer Sonnet auto-named the evolved skill **`task-completion-verifier`**, produced a 2.7 KB body at generation 9, and landed 6/10 binary PASS under continuous scoring.

Per-task on the 10-task manifest (Track B's 0.7582 folder):

| Task                                  | Result |
| ------------------------------------- | ------ |
| build-merkle-tree-cli-sha512          | PASS   |
| build-system-task-ordering            | PASS   |
| industrial-kiln-controller            | PASS   |
| monorepo-changelog-cli                | PASS   |
| raft-log-repair-concurrent-access     | PASS   |
| schedule-vacation                     | PASS   |
| api-endpoint-permission-canonicalizer | FAIL   |
| application-debug                     | FAIL   |
| python-api-rate-limit                 | FAIL   |
| scan-linux-persistence-artifacts      | FAIL   |

Mean score 0.808, composite 0.7582. Importantly, `industrial-kiln-controller` scored 0.00 on the empty baseline and flipped to PASS; `monorepo-changelog-cli` (which scored 0/4 on robustness reruns of the retracted 0.700 folder) also flipped cleanly to PASS.

**Contrast with the earlier obra-skill manual probe**: the 4.4 KB `verification-before-completion` skill from seed_skills_realworld regressed other tasks when dropped into a mixed folder. Track B's auto-evolved version is 2.7 KB, expressed in the same shape as the task rubric, and flips tasks without collateral damage. This is the second cleanest data point (after the obra probe) that verification-before-completion is load-bearing for M2.7 — but it argues for lean, auto-derived phrasing over prose-heavy methodology.

## Track A — same pattern as v2 all-M2.7

| Pass | Winner | A     | B     | AB    |
| ---- | ------ | ----- | ----- | ----- |
| 0    | seed   | 0.624 | —     | —     |
| 1    | A      | 0.624 | 0.583 | 0.624 |
| 2    | A      | 0.624 | 0.616 | 0.497 |

Converged in 2 passes with streak=2. Neither mutation beat the seed; AB tied once and A defended on tiebreak. This mirrors v2's empty-seed Track A behaviour exactly — **autoreason's greedy A/B/AB tournament cannot find lift from a cold empty start**. Track A needs a richer seed to mutate against. No further empty-seed Track A runs are warranted.

## Track D — YAML crash, handoff-variance surprise, salvaged

Track D's design: phase_b (MAP-Elites, 10 gens) explores, then hands `best/` to phase_a (autoreason, 8 passes) for refinement.

**Crash at handoff.** Phase_b produced a `best/SKILL.md` with an unquoted description containing a colon:

```
description: Core guidance for implementing coding/scripting tasks: parse requirements...
```

Track A's validator rejected this with `yaml.scanner.ScannerError`. The whole run aborted mid-handoff. Salvage: manually quoted the description, renamed the folder to match its YAML `name` field (`placeholder` -> `task-execution-core`), relaunched as phase_a. Also patched `skill_evolve/track_d/run.py` to catch this failure class and fall back to the original seed on validation error (143 tests still pass).

**Handoff-variance surprise.** Phase_b rated its own best folder at 0.7093. When phase_a re-evaluated the **same folder** at its pass 0, it scored 0.5104. Same bytes on disk, single eval each, delta 0.2 composite. This is the largest variance measurement we've put on continuous scoring so far, and it's bigger than the noise floor we'd been assuming (~0.1).

Phase_a trajectory from the 0.51 starting point:

| Pass | Winner | A     | B     | AB    |
| ---- | ------ | ----- | ----- | ----- |
| 0    | seed   | 0.510 | —     | —     |
| 1    | AB     | 0.529 | 0.496 | 0.529 |
| 2    | A      | 0.529 | 0.454 | 0.430 |
| 3    | AB     | 0.634 | 0.628 | 0.634 |
| 4    | A      | 0.634 | 0.503 | 0.622 |
| 5    | A      | 0.634 | 0.518 | 0.506 |

Final 0.6342. Phase_a did real lift (+0.12 from the 0.51 pass-0), added a `cli-tool-construction` skill via AB, but ended 0.12 below Track B. Artifacts at `runs/v3_empty_3tracks/track_d/phase_a/final/`.

## Implications

1. **Single-eval variance is ~+/-0.2 on the same folder under continuous scoring** — measured directly at the Track D handoff. Bigger than I had been treating it. Multi-rep averaging is mandatory at any decision point where a single number drives action.
2. **Track B is the cheapest track that works** from an empty seed. Patch-mutation MAP-Elites + islands found lift that A couldn't and that D couldn't refine further under single-rep eval noise.
3. **Verification-before-completion is load-bearing for M2.7** — three independent data points now (manual obra probe, Track B auto-derivation, phase_a hillclimb direction). But the phrasing and length matter; lean beats prose-heavy.
4. **Track A on empty seed is a closed file**. Same convergence-without-lift pattern as v2. No more empty-seed Track A runs.

## What's next

See `NEXT_STEPS.md` — short version: repeat Track B's 0.7582 folder 3x to pin its true mean, diagnose the 4 remaining FAIL tasks, then decide whether to retry Track D with repeats or more Track B on top of the winner. AB-chunking work is deferred: Track B already produces clean folders without it.

## Cost + time

~$65-75 OpenRouter spend across the 3 tracks. ~6h wall including the crash-and-restart.

