#!/usr/bin/env bash
# One-shot setup for the SkillsBench evaluation pipeline (Phase A).
#
# Idempotent: safe to re-run. Performs:
#   1. Initialize/update the SkillsBench git submodule.
#   2. Expand skillsbench manifest entries (idempotent; required after
#      a fresh clone — manifest.json ships with 0 skillsbench rows).
#   3. uv sync (installs all project deps including benchflow).
#   4. uv pip install benchflow (belt-and-suspenders pin in case it slipped).
#   5. bench tasks init (only if `bench` is on PATH).
#   6. D-2a auth precheck — exit 99 if ANTHROPIC_API_KEY is missing.
#   7. OpenRouter precheck — exit 99 if OPENROUTER_API_KEY is missing
#      (Phase E's outer LLM falls back to SyntheticLLM otherwise).
#   8. Generate skill_evolve/skillsbench/subset_20.json if it's missing
#      AND scripts/select_subset_20.py exists (created in Group C).

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# B7: ensure scripts are executable after a fresh rsync/clone — Git
# preserves the bit but rsync defaults can strip it.
chmod +x scripts/*.sh 2>/dev/null || true

echo ">>> [1/8] git submodule update --init (top-level only; --recursive skipped — upstream has broken nested submodule simpo-code-reproduction/SimPO/alignment-handbook)"
git submodule update --init skill_evolve/benchmark/vendor/skillsbench

echo ">>> [2/8] Expanding skillsbench manifest entries (idempotent)"
uv run python scripts/expand_skillsbench_manifest.py 2>&1 | tail -5

echo ">>> [3/8] uv sync"
uv sync

echo ">>> [4/8] uv pip install 'benchflow>=0.3.4,<0.4'"
uv pip install 'benchflow>=0.3.4,<0.4'

echo ">>> [5/8] verify bench CLI is callable via uv run"
if uv run bench --help >/dev/null 2>&1; then
  echo "OK: 'bench' callable via 'uv run bench' (located at $(uv run --quiet which bench 2>/dev/null || echo '.venv/bin/bench'))."
else
  echo "ERROR: 'uv run bench --help' failed. Check benchflow install."
  exit 1
fi

echo ">>> [6/8] D-2a auth precheck (ANTHROPIC_API_KEY)"
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ERROR: ANTHROPIC_API_KEY is not set in your environment."
  echo "       Export it (export ANTHROPIC_API_KEY=sk-ant-...) and re-run."
  echo "       Phase A's smoke uses 'bench eval create -m claude-haiku-4-5 -a claude-code',"
  echo "       which dispatches direct to api.anthropic.com from this host."
  exit 99
fi
echo "OK: ANTHROPIC_API_KEY present (prefix=${ANTHROPIC_API_KEY:0:10}...)."

echo ">>> [7/8] OpenRouter precheck (OPENROUTER_API_KEY)"
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set."
  echo "       Phase E's outer LLM (moonshotai/kimi-k2.6) routes via OpenRouter."
  echo "       Without it, the run silently falls back to SyntheticLLM (random mutations)."
  exit 99
fi
echo "OK: OPENROUTER_API_KEY present (prefix=${OPENROUTER_API_KEY:0:10}...)."

echo ">>> [8/8] subset_20.json generation (if missing)"
SUBSET_JSON="$REPO_ROOT/skill_evolve/skillsbench/subset_20.json"
SUBSET_SCRIPT="$REPO_ROOT/scripts/select_subset_20.py"
if [[ -f "$SUBSET_JSON" ]]; then
  echo "OK: $SUBSET_JSON already exists, skipping."
elif [[ -f "$SUBSET_SCRIPT" ]]; then
  echo ">>> running scripts/select_subset_20.py"
  uv run python "$SUBSET_SCRIPT"
else
  echo "NOTE: scripts/select_subset_20.py not present yet (Group C creates it)."
  echo "      Skipping subset generation; re-run setup after Group C lands."
fi

echo ">>> Setup complete."
