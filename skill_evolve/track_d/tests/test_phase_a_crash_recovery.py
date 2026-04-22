"""Track D — phase_a crash is caught and persisted, not propagated.

The original v3 crash on 2026-04-21 was at the handoff (phase_b best had
broken YAML). The follow-up risk is that ``run_track_a`` crashes
*mid-run* on a late mutation — e.g. Sonnet emits SKILL.md with an
unquoted colon at pass 3 of 8. Before this fix, that would take down
the whole Track D run; now it should be caught and the run_meta.json
should record the crash instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from skill_evolve.track_d import run as _td


def test_phase_a_exception_is_caught_and_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    seed = tmp_path / "seed"
    (seed / "placeholder").mkdir(parents=True)
    (seed / "placeholder" / "SKILL.md").write_text(
        "---\nname: placeholder\ndescription: p\n---\n", encoding="utf-8",
    )

    # Make phase_b a no-op that just writes a `best/` matching the seed so
    # phase_a's seed-validation path passes; we're testing mid-run crash.
    def fake_run_evolution(*, seed_path, out_dir, config, evaluator, llm):
        best = out_dir / "best"
        best.mkdir(parents=True)
        (best / "placeholder").mkdir(parents=True)
        (best / "placeholder" / "SKILL.md").write_text(
            "---\nname: placeholder\ndescription: p\n---\n", encoding="utf-8",
        )
        # RunResult shape expected by _td.run_sequential.
        class _R:
            best = None
            history = []
            output_dir = out_dir
        return _R()

    monkeypatch.setattr(_td, "run_evolution", fake_run_evolution)

    def boom(**kwargs):
        raise RuntimeError("simulated phase_a mid-run crash")

    monkeypatch.setattr(_td, "run_track_a", boom)

    out = tmp_path / "out"
    summary = _td.run_sequential(
        seed=seed,
        out=out,
        b_generations=1,
        a_max_passes=2,
        outer_model="x",
        inner_model="y",
        max_workers=1,
        repeats=1,
        sources=None,
        force_synthetic=True,
        verify=False,
        num_islands=1,
        migration_interval=5,
        rng_seed=0,
    )

    # Run must return a summary, not raise.
    assert summary["phase_a_crashed"].startswith("RuntimeError:")
    assert "simulated phase_a mid-run crash" in summary["phase_a_crashed"]

    # run_meta.json on disk matches.
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["phase_a_crashed"].startswith("RuntimeError:")

    # Crash detail should also be persisted under phase_a/crash.json.
    crash = json.loads((out / "phase_a" / "crash.json").read_text())
    assert crash["crashed"] is True
    assert "simulated" in crash["crash_detail"]
