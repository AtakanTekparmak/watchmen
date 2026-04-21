# Next steps — synthesis across 4 Opus critiques

Rewritten 2026-04-21 after v3 (empty seed + rotated 10-task manifest + 3 tracks). The prior AB-chunking phase plan is superseded: v3 Track B (patch-mutation MAP-Elites) produced a clean 2.7 KB `task-completion-verifier` at 0.7582 composite without any chunked AB synthesis. AB synthesis is no longer the bottleneck we should spend budget on.

## What v3 established

Track B on an empty seed, under continuous scoring, with MAP-Elites + islands, produced a coherent evolved skill (`task-completion-verifier`, 2.7 KB, gen 9) that lands 6/10 binary PASS at composite 0.7582 — the cleanest evolution result we've measured. Track A converged without lift (same cold-start pattern as v2). Track D crashed at handoff on a YAML quoting bug, salvaged to 0.6342, and exposed a continuous-scoring variance of ~+/-0.2 composite on the SAME folder re-evaluated (phase_b 0.7093 -> phase_a pass-0 0.5104).

The variance finding is the single most actionable result: it means every one-shot number in this project is +/-0.2, bigger than most of the deltas we've been reporting. Multi-rep averaging at decision points is now mandatory.

## Top priorities, ranked

### 1. Rerun Track B's 0.7582 folder 3x under continuous scoring (~$10, 1.5 hr)

Re-evaluate `runs/v3_empty_3tracks/track_b/best/` three times, single-pass each, continuous scoring. Compute mean and std. **Answers**: is 0.7582 the real mean or a lucky high draw on an uncharacterized distribution? Given the Track D handoff showed +/-0.2 on identical bytes, a 3-rep mean is the minimum bar before we build any further claim on top of this folder.

**Gate**: If mean < 0.65, retract "Track B cracked the task-completion-verifier pattern" and treat it like the 0.700 retraction.

### 2. Diagnose the 4 remaining FAIL tasks on Track B's winner (~$0, 30 min)

Read the failing pytest assertions from `runs/v3_empty_3tracks/track_b/...` per-task artifacts for api-endpoint-permission-canonicalizer, application-debug, python-api-rate-limit, scan-linux-persistence-artifacts. Classify each as (a) stopped-short on last 1-2 assertions, (b) genuine logic bug, or (c) impossible at M2.7 capability. Same playbook as the 2026-04-20 stopped-short diagnosis. **Answers**: where the next lift has to come from. Free, blocking nothing.

### 3. Track D with repeats>=2 at handoff, OR seeded with Track B's 0.7582 folder (~$40-60, ~4-6 hr)

Two variants to consider:
- **D-repeats**: rerun phase_b -> phase_a but average phase_a's pass-0 eval over >=2 repeats so the handoff sits on a stable estimate, not a single-draw 0.5104. Tests whether Track D's architecture is sound once variance is tamed.
- **D-seeded**: replace phase_b with Track B's 0.7582 folder as input; run phase_a autoreason on top. Tests whether AB synthesis can refine a known-good folder rather than a phase_b-explored one.

Gate after priority 1 finishes — if Track B's mean is solidly high, D-seeded is the more interesting variant.

### 4. Fresh Track B run from the 0.7582 folder as seed, not empty (~$20, 4 hr)

Does a second pass of MAP-Elites on a good starting point lift further, or is 0.75-0.80 the M2.7 ceiling on the 10-task manifest regardless of skill quality? **Answers**: whether Track B is bottlenecked by cold-start exploration (in which case warm-start compounds) or by inner-model capability (in which case warm-start plateaus). Deferred behind priorities 1-3 since it bets budget on the 0.7582 number being real.

### 5. (deferred) AB synthesis chunking probes

The full 450 LoC chunked-AB plan from the prior NEXT_STEPS.md is deferred indefinitely. v3 showed evolution can produce clean, short, targeted folders without chunking. Revisit only if some later experiment produces a specific case where whole-folder synthesis breaks and chunking is the cheapest fix.

## What NOT to do

- **No more empty-seed Track A runs.** Pattern is confirmed across v2 and v3: autoreason greedy tournament cannot find lift from a 1-skill empty folder. Closed file.
- **No more single-eval comparisons.** The +/-0.2 variance means a single number can swing any story. Every decision-gating evaluation needs >=2 repeats.
- **No more Track C runs.** Hybrid A/B/AB tournament on MAP-Elites: already known not to earn its compute in v2 and v2.1, and v3 did not run it. Don't restart it.
- **No inner-model swap to Sonnet.** Cost premium 10-20x; stopped-short diagnosis and Track B's win both suggest M2.7 is not the binding constraint at this budget.

## Budget

~$335 spent across v2 / v2.1 / ablations / 2026-04-20 robustness + continuous-scoring / v3. ~$165 left of the initial $500.

```
Priority  Cost    Wall     Gate
1         $10     1.5 hr   Kills 3 + 4 if mean < 0.65
2         $0      30 min   None (run alongside 1)
3         $40-60  4-6 hr   Conditional on 1
4         $20     4 hr     Conditional on 1 + 3
5         (defer) —        —
```

Upfront committed through priority 2: ~$10. Conditional through priorities 3+4: ~$60-80. Reserve ~$75-95 for whatever priorities 1-2 surface next.

## Next concrete action

Priority 1 — launch 3 evals on `runs/v3_empty_3tracks/track_b/best/` in parallel under continuous scoring. Priority 2 can run alongside on the same artifacts (no new compute needed for the diagnosis). Everything downstream gates on the mean.
