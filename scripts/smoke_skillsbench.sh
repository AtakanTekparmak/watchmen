#!/usr/bin/env bash
# Phase A smoke driver: runs ONE SkillsBench task end-to-end via the official
# bench CLI, in two conditions (no-skills, with-skills), capturing JSON to
# runs/smoke/.
#
# Smoke task: dialogue-parser. Picked because:
#   - difficulty=easy, agent.timeout_sec=900s (15min cap)
#   - lightweight environment image (python:3.12-slim + graphviz)
#   - deterministic pytest verifier (tests/test_outputs.py via tests/test.sh)
#   - bundled environment/skills/dialogue_graph/ — the with-skills variant
#     actually exercises skill mounting end-to-end
#
# This script makes PAID 'bench eval create' calls. Inspect output before
# advancing to Phase D.

set -euo pipefail

echo "WARNING: This script makes paid bench eval create calls. Press CTRL-C in 5s to abort..."
sleep 5

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TASK_DIR="$REPO_ROOT/skill_evolve/benchmark/vendor/skillsbench/tasks/dialogue-parser"
MODEL="claude-haiku-4-5"
SKILLS_DIR="$REPO_ROOT/skill_evolve/benchmark/vendor/skillsbench/tasks/dialogue-parser/environment/skills"

SCENES_DIR="$REPO_ROOT/skill_evolve/skillsbench/scenes"
NO_SKILLS_TMPL="$SCENES_DIR/smoke.yaml.tmpl"
WITH_SKILLS_TMPL="$SCENES_DIR/baseline_with_skills.yaml.tmpl"

NO_SKILLS_YAML="/tmp/smoke_no_skills.yaml"
WITH_SKILLS_YAML="/tmp/smoke_with_skills.yaml"

# benchflow's Job._get_task_dirs (job.py:355) expects tasks_dir to be a parent
# directory whose children have task.toml. Stage a single-task wrapper that
# symlinks the real task dir under a fresh parent so YAMLs can self-contain
# without scanning the whole tasks/ tree.
WRAPPER_ROOT="/tmp/smoke_skillsbench_tasks"
rm -rf "$WRAPPER_ROOT"
mkdir -p "$WRAPPER_ROOT"
ln -s "$TASK_DIR" "$WRAPPER_ROOT/dialogue-parser"

mkdir -p runs/smoke

echo ">>> Materializing $NO_SKILLS_YAML from $NO_SKILLS_TMPL"
sed \
  -e "s|<task_dir>|${WRAPPER_ROOT}|g" \
  -e "s|<model>|${MODEL}|g" \
  "$NO_SKILLS_TMPL" > "$NO_SKILLS_YAML"

echo ">>> Materializing $WITH_SKILLS_YAML from $WITH_SKILLS_TMPL"
sed \
  -e "s|<task_dir>|${WRAPPER_ROOT}|g" \
  -e "s|<model>|${MODEL}|g" \
  -e "s|<skills_dir>|${SKILLS_DIR}|g" \
  "$WITH_SKILLS_TMPL" > "$WITH_SKILLS_YAML"

# SG-1: ``-a claude-code`` requires the in-process ``register_claude_code``
# side-effect import before the bench CLI resolves ``-a`` against
# ``benchflow.agents.registry.AGENTS``. Invoke bench through ``python -c``
# (mirroring ``skill_evolve/agents/bench_cli.py:_BENCH_SHIM_CODE``) so the
# registration runs in the same interpreter as ``benchflow.cli.main``.
BENCH_SHIM='import sys; import skill_evolve.agents.register_claude_code; from benchflow.cli.main import app; sys.exit(app())'

echo ">>> Running NO-SKILLS smoke"
uv run python -c "$BENCH_SHIM" eval create \
  -f "$NO_SKILLS_YAML" \
  -t "$TASK_DIR" \
  -a claude-code \
  -m "$MODEL" \
  > runs/smoke/no_skills.json

echo ">>> Running WITH-SKILLS smoke"
uv run python -c "$BENCH_SHIM" eval create \
  -f "$WITH_SKILLS_YAML" \
  -t "$TASK_DIR" \
  -a claude-code \
  -m "$MODEL" \
  > runs/smoke/with_skills.json

echo ">>> Smoke complete. Output:"
echo "    runs/smoke/no_skills.json"
echo "    runs/smoke/with_skills.json"
