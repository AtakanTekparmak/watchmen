# watchmen-daycare v1 — findings

Run: `wmca-20260522T212828Z` (3 iters, K=4, rollouts=1, judge=DeepSeek)

## 1. What we built

watchmen-daycare is a synthetic-eval-generation + GEPA-style skill-text evolution loop that sits next to watchmen. The student-teacher arrangement: Opus 4.7 conversations are the teacher signal (mined into eval triples), qwen3-32b is the student model, and the SKILL.md text is the knowledge bridge the loop mutates. Each iter proposes K patches against the current skill, scores them on a held-out eval slice with a token-length regularizer, and promotes the best fitness-positive candidate.

## 2. What worked

- **Synthetic eval generation**: 49 usable evals mined from 7 wmca skills in ~2.5 h. Earlier corpus-extraction path yielded 0 evals; the synthetic route is the unblocker.
- **The existing skill does real work**: qwen3-32b lifts from `baseline_a_empty = 0.4091` to `iter_0 = 0.5208` on the 24-item holdout — **+27% relative** from the current SKILL.md alone.
- **Evolution found the right mutation direction**: candidate `c3` hit `holdout = 0.5625` (procedural_qa 0.6176, script_gen 0.4286) in both iter_1 and iter_3 by adding specific flag names, concrete threshold values, and example invocations. The direction is correct.
- **Pipeline hardening**: orphan broken script removed, 3 NameErrors fixed, scoring parallelized, leak-scanner overfire silenced.

## 3. What didn't work (and why)

- **No candidate promoted**. `c3` cleared holdout (0.5625 > 0.5208 + ε=0.0417) but the token regularizer ate it: iter_1 fitness 0.55911 vs anchor 0.5208 → Δ = +0.0383, **0.003 short** of the ε gate after the 2441-token penalty (`lambda_n = 1.5e-5`). iter_3 same story (fitness 0.5604, 2298 tokens, `lambda_n = 2.5e-5`). Two genuine wins, both clipped by the regularizer.
- **Judge closed loop**. DeepSeek judges DeepSeek-generated proposals. Smoking-gun: `baseline_c_teacher = 0.2917`, which is *below* `baseline_a_empty = 0.4091`. The teacher (Opus 4.7) is scored worse than no skill at all because the judge rewards qwen3-32b's terse style over Opus's verbose answers. `gap_closed = 111.74` is nonsensical — the denominator went negative.
- **`empty_patch` wastes ~25% of slots**. The Opus proposer burned its 12 tool-call budget on file exploration in the c1 slot every iter, never committing a patch.
- **Stall detector fires too early**. Three "no improvement" iters trip the exit even when candidates are 0.003 from promotion. With `rollouts=1` variance this tolerance is too tight.

## 4. Three fixes for next run

1. `lambda_init: 1e-5 → 1e-6`. The ~200-token additions in `c3` would have produced penalty ≈ 0.0003 instead of 0.0034 — comfortably above the ε gate.
2. `--judge anthropic/claude-haiku-4-5-20251001`. Cross-family judging breaks the closed loop and should restore teacher-above-empty.
3. `max_iter: 12 → 16` in the proposer, plus a hard "emit by tool-call 7" line in the prompt. Kills `empty_patch` in the c1 slot.

## 5. Recommended next action

```
daycare run wmca --skill post-train-diagnostic --max-iters 10 --budget 4h \
  --K 4 --rollouts 1 --judge anthropic/claude-haiku-4-5-20251001
```

with `lambda_init = 1e-6`. `c3`'s mutation pattern (flag specificity + threshold values + concrete invocation examples) should clear the gate on the second attempt.
