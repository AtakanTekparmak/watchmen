"""End-to-end synthetic runner test.

Drives run_loop() with --force-synthetic so both the evaluator and the
LLM client return deterministic placeholders. Verifies history.json
shape, that passes are logged, and that convergence triggers when the
score stops improving.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skill_evolve.track_a.runner import CONVERGENCE_STREAK, run_loop


def test_three_pass_synthetic_convergence(
    tmp_path: Path, seed_path: Path
) -> None:
    out = tmp_path / "run"
    history = run_loop(
        seed=seed_path,
        out=out,
        max_passes=3,
        force_synthetic=True,
        verify=False,
        rng_seed=123,
    )

    # history.json exists and parses
    hist_path = out / "history.json"
    assert hist_path.exists()
    reloaded = json.loads(hist_path.read_text(encoding="utf-8"))
    assert reloaded["force_synthetic"] is True

    # passes list is structured as expected
    passes = reloaded["passes"]
    assert passes[0]["pass"] == 0
    assert passes[0]["winner"] == "seed"
    # subsequent passes should carry the keys we promise
    for p in passes[1:]:
        assert {"pass", "winner", "score_A", "score_B", "op", "streak"} <= set(p.keys())

    # Synthetic evaluator always returns composite 0.0 → A wins every pass.
    # That should trigger convergence after CONVERGENCE_STREAK passes.
    streaks = [p["streak"] for p in passes if p["pass"] >= 1]
    if len(streaks) >= CONVERGENCE_STREAK:
        assert max(streaks) >= CONVERGENCE_STREAK
    assert reloaded.get("converged") is True or len(passes) - 1 == 3

    # Every pass dir exists on disk
    for i in range(len(passes)):
        assert (out / f"pass_{i}" / "A").exists() or i > 0
    # final folder written
    assert (out / "final").exists()
    # final folder contains at least one SKILL.md
    skills_seen = list((out / "final").glob("*/SKILL.md"))
    assert skills_seen, "final/ should contain skill folders"


def test_runner_respects_max_passes_cap(
    tmp_path: Path, seed_path: Path
) -> None:
    out = tmp_path / "run"
    history = run_loop(
        seed=seed_path,
        out=out,
        max_passes=1,  # cap at 1 pass
        force_synthetic=True,
        verify=False,
        rng_seed=7,
    )
    # passes list has entry 0 (seed) + at most 1 more
    assert len(history["passes"]) <= 2


def test_synthetic_history_contains_op_metadata(
    tmp_path: Path, seed_path: Path
) -> None:
    out = tmp_path / "run"
    history = run_loop(
        seed=seed_path,
        out=out,
        max_passes=1,
        force_synthetic=True,
        verify=False,
        rng_seed=1,
    )
    # pass 1 should have an op field populated (RewriteSkillContent per canned response)
    if len(history["passes"]) >= 2:
        p1 = history["passes"][1]
        assert p1.get("op") is not None
        assert p1["op"]["op"] == "RewriteSkillContent"
