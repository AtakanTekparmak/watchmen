# watchmen-daycare — next steps (2026-05-23)

## Immediate: re-run wmca with fixed settings (~1h)

The three bugs from the first run (lambda too high, DeepSeek closed-loop judge, empty_patch in c1) are fixed and committed. Reuse the existing eval set — no need to regenerate. Run:

```bash
cd ~/Desktop/work/kai-skills/watchmen-fukara/daycare
OPENROUTER_API_KEY=$(grep OPENROUTER ~/Desktop/work/kai-skills/.env | cut -d= -f2) \
  .venv/bin/daycare run wmca \
  --skill post-train-diagnostic \
  --max-iters 10 \
  --budget 4h \
  --K 4 \
  --rollouts 1 \
  --weak-model qwen/qwen3-32b \
  --proposer anthropic/claude-opus-4 \
  --judge anthropic/claude-haiku-4-5-20251001 \
  --teacher anthropic/claude-opus-4 \
  --max-workers 4 \
  --eval-set ~/.watchmen/daycare/runs/wmca-20260522T133612Z/eval_set.jsonl \
  --leak-policy warn \
  --yes \
  2>&1 | tee /tmp/daycare-wmca-rerun.log
```

Expected: c3-type mutation (flag specificity + threshold values + concrete invocation examples) should now be promoted with `lambda_init=1e-6`. The ~200-token penalty drops from ~0.0034 to ~0.0003, comfortably above the ε gate. If promoted:

```bash
daycare promote wmca post-train-diagnostic
```

## Short-term: first ctf run (~2-4h)

ctf has **35,052 pre-1d candidate triples** vs wmca's 10,254 — much richer corpus, more headroom for evolution to find lift. Steps:

1. **Build the eval set**:
   ```bash
   daycare eval-build ctf --synthetic --n-per-skill 12 \
     --proposer anthropic/claude-opus-4 --max-workers 4
   ```
2. **Check which skill the selector picks**. It scores by traffic × error-rate, so the likely target is `gpu-pod-sniping` or `training-watchdog` (both high-traffic, both have observable failure modes in the corpus).
3. **Run evolution** with the same settings as the wmca re-run above but `--skill <selected>` and `--eval-set <new path>`.

## Medium-term: robustness improvements

- **Increase stall threshold from 3 to 5**. Three "no improvement" iters is too tight with `rollouts=1` variance — c3 sat right at the threshold in 2/3 iters and the loop bailed before the regularizer fix had a chance to land it.
- **OR increase rollouts from 1 to 3** for more reliable per-candidate scoring. Slower (~3× judge calls per candidate) but lower variance, so the stall detector means what it says.
- **Seed the next wmca run from c3**. The candidate bundle already exists at `iter_1/candidates/c3/bundle/` — using it as the starting bundle skips re-discovering the same mutation direction.
- **Phase 4 baseline C should use Haiku as judge** (not Opus) after stall, so the teacher-ceiling number is comparable to the evolution-loop scoring instead of being judged by a different model.

## Longer-term

- `daycare daemon install` — autonomous nightly runs, once the first promotion is confirmed end-to-end.
- Contribute daycare back to `firstbatchxyz/watchmen` upstream as a PR (standalone Python package, 61 passing tests, full CLI — should slot in cleanly).
- Run the **pi** project: 9 skills, 2,125 triples. Smallest corpus but the cleanest task types (commander maintenance, dashboard audit, etc.) — good sanity check that the pipeline generalises off wmca.
