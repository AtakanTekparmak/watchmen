"""Mocked tests for :mod:`skill_evolve.skillsbench.baseline`.

All tests run without a real ``bench`` CLI and without API calls. The
single seam under test is ``BenchCliBackend.run_task`` — patched to
return canned :class:`TrajectoryResult` instances. Filesystem fixtures
synthesize a tiny vendor dir with a single task.toml so the
hydration / validation logic exercises the real loader.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

import pytest

from skill_evolve.agents.base import TrajectoryResult
from skill_evolve.skillsbench import baseline


_TOML_BODY = """\
version = "1.0"

[metadata]
author_name = "Test"
difficulty = "medium"
category = "engineering"
tags = ["python"]

[verifier]
timeout_sec = 600.0

[agent]
timeout_sec = 600.0
"""


def _make_vendor(
    tmp_path: Path, task_names: List[str], with_skills: bool = True
) -> Path:
    """Synthesize a vendor dir with N task.toml shells.

    If ``with_skills`` is True, also create an ``environment/skills/``
    subdir so the with-skills condition resolves to a concrete path
    rather than falling back to ``None``.
    """
    vendor = tmp_path / "vendor"
    for name in task_names:
        td = vendor / name
        td.mkdir(parents=True, exist_ok=True)
        (td / "task.toml").write_text(_TOML_BODY, encoding="utf-8")
        (td / "instruction.md").write_text("do the thing", encoding="utf-8")
        env = td / "environment"
        env.mkdir(exist_ok=True)
        if with_skills:
            (env / "skills").mkdir(exist_ok=True)
            (env / "skills" / "README.md").write_text("seed", encoding="utf-8")
        (td / "tests").mkdir(exist_ok=True)
    return vendor


def _make_task_list(tmp_path: Path, ids: List[str]) -> Path:
    p = tmp_path / "task_list.json"
    p.write_text(json.dumps(ids), encoding="utf-8")
    return p


def _good_trajectory(task_id: str, success: bool = True) -> TrajectoryResult:
    return TrajectoryResult(
        task_id=task_id,
        success=success,
        tool_calls=5,
        elapsed_s=1.0,
        last_msg="" if success else "fail",
        notes="",
        verified=success,
        verifier_status="passed" if success else "failed",
        verifier_detail="",
        score=1.0 if success else 0.0,
        cost_usd=None,
    )


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")


# ---------------------------------------------------------------------------
# Validation / pre-flight tests
# ---------------------------------------------------------------------------


def test_missing_api_key_exits_99(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    vendor = _make_vendor(tmp_path, ["foo"])
    task_list = _make_task_list(tmp_path, ["skillsbench/foo"])
    rc = baseline.main(
        [
            "--task-list",
            str(task_list),
            "--vendor-dir",
            str(vendor),
            "--out",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    assert rc == 99
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err


def test_bad_task_list_missing_file(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    rc = baseline.main(
        [
            "--task-list",
            str(tmp_path / "does_not_exist.json"),
            "--vendor-dir",
            str(tmp_path / "vendor"),
            "--dry-run",
        ]
    )
    assert rc == 1
    assert "does not exist" in capsys.readouterr().err


def test_bad_task_list_malformed_json(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    p = tmp_path / "bad.json"
    p.write_text("not json at all", encoding="utf-8")
    rc = baseline.main(
        [
            "--task-list",
            str(p),
            "--vendor-dir",
            str(tmp_path / "vendor"),
            "--dry-run",
        ]
    )
    assert rc == 1
    assert "valid JSON" in capsys.readouterr().err


def test_bad_task_list_non_array(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    p = tmp_path / "obj.json"
    p.write_text(json.dumps({"tasks": ["foo"]}), encoding="utf-8")
    rc = baseline.main(
        [
            "--task-list",
            str(p),
            "--vendor-dir",
            str(tmp_path / "vendor"),
            "--dry-run",
        ]
    )
    assert rc == 1
    assert "JSON array of strings" in capsys.readouterr().err


def test_missing_task_dir_clear_error(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    vendor = _make_vendor(tmp_path, ["foo"])
    task_list = _make_task_list(tmp_path, ["skillsbench/does-not-exist"])
    rc = baseline.main(
        [
            "--task-list",
            str(task_list),
            "--vendor-dir",
            str(vendor),
            "--dry-run",
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "missing or lack task.toml" in err
    assert "does-not-exist" in err


def test_unsupported_agent_backend_raises() -> None:
    with pytest.raises(NotImplementedError):
        baseline.main(
            [
                "--agent-backend",
                "hermes",
                "--dry-run",
            ]
        )


# ---------------------------------------------------------------------------
# Dry-run + cost estimate tests
# ---------------------------------------------------------------------------


def test_dry_run_prints_plan_and_exits_0(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    vendor = _make_vendor(tmp_path, ["a", "b"])
    task_list = _make_task_list(tmp_path, ["skillsbench/a", "skillsbench/b"])
    out_dir = tmp_path / "out"

    with mock.patch.object(baseline.BenchCliBackend, "run_task") as run_mock:
        rc = baseline.main(
            [
                "--task-list",
                str(task_list),
                "--vendor-dir",
                str(vendor),
                "--out",
                str(out_dir),
                "--trials",
                "3",
                "--conditions",
                "with-skills,no-skills",
                "--dry-run",
            ]
        )
    assert rc == 0
    # No backend calls in dry-run.
    run_mock.assert_not_called()
    out = capsys.readouterr().out
    assert "SkillsBench baseline plan" in out
    # n_trials = 2 tasks × 2 conditions × 3 trials = 12
    assert "= 12" in out
    # est_cost = 12 × 0.05 = 0.60
    assert "$0.60" in out
    # Dry-run doesn't materialize the out dir.
    assert not out_dir.exists()


def test_cost_cap_rejection_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    vendor = _make_vendor(tmp_path, ["a", "b"])
    task_list = _make_task_list(tmp_path, ["skillsbench/a", "skillsbench/b"])
    rc = baseline.main(
        [
            "--task-list",
            str(task_list),
            "--vendor-dir",
            str(vendor),
            "--trials",
            "5",
            "--conditions",
            "with-skills,no-skills",
            "--max-budget-usd",
            "0.50",  # 2*2*5*0.05 = $1.00 > $0.50
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc != 0
    err = capsys.readouterr().err
    assert "exceeds budget cap" in err
    assert "n_trials = 2 tasks" in err


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------


def test_resume_skips_completed_rows(tmp_path: Path) -> None:
    vendor = _make_vendor(tmp_path, ["a"])
    task_list = _make_task_list(tmp_path, ["skillsbench/a"])
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-populate results.jsonl with 5 already-complete rows
    # (task=a, cond=with-skills, trials=0..4).
    results_path = out_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for trial in range(5):
            tr = _good_trajectory("skillsbench/a", success=True)
            row = {
                "task_id": "skillsbench/a",
                "condition": "with-skills",
                "trial": trial,
                "trajectory_result": dataclasses.asdict(tr),
                "ts": "2026-04-29T00:00:00",
                "wall_s": 0.1,
            }
            f.write(json.dumps(row) + "\n")

    captured: List[Any] = []

    def _fake_run_task(self: Any, task: Any, **kwargs: Any) -> TrajectoryResult:
        captured.append((task.task_id, kwargs.get("skills_dir")))
        return _good_trajectory(task.task_id, success=True)

    with mock.patch.object(baseline.BenchCliBackend, "run_task", _fake_run_task):
        rc = baseline.main(
            [
                "--task-list",
                str(task_list),
                "--vendor-dir",
                str(vendor),
                "--out",
                str(out_dir),
                "--trials",
                "5",
                "--conditions",
                "with-skills,no-skills",
                "--resume",
                "--concurrency",
                "1",
            ]
        )

    assert rc == 0
    # We pre-filled 5 with-skills trials; 5 no-skills trials should run.
    assert len(captured) == 5
    # And all newly-run trials should be no-skills (skills_dir=None).
    assert all(sd is None for (_tid, sd) in captured)


# ---------------------------------------------------------------------------
# Budget guard mid-run
# ---------------------------------------------------------------------------


def test_budget_guard_halts_mid_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Mid-run guard halts new submissions when cumulative estimate >= cap.

    The plan calls for 4 trials at $0.05 each = $0.20 pre-flight, but
    we set a budget cap of $0.10 so only ~2 trials should complete
    before the guard fires. Pre-flight would normally reject this, so
    we monkeypatch ``_estimate_cost`` to under-report and force
    execution into the mid-run halt path — that's exactly the
    real-world situation we're guarding against (estimate diverges
    from actual cumulative).
    """
    vendor = _make_vendor(tmp_path, ["a"])
    task_list = _make_task_list(tmp_path, ["skillsbench/a"])
    out_dir = tmp_path / "out"

    submitted: List[str] = []

    def _fake_run_task(self: Any, task: Any, **kwargs: Any) -> TrajectoryResult:
        submitted.append(task.task_id)
        return _good_trajectory(task.task_id, success=True)

    # Force pre-flight to under-report so we get into the actual
    # execution path; mid-run guard then halts after 2 trials.
    with mock.patch.object(baseline, "_estimate_cost", lambda _n: 0.0):
        with mock.patch.object(baseline.BenchCliBackend, "run_task", _fake_run_task):
            with caplog.at_level("WARNING"):
                rc = baseline.main(
                    [
                        "--task-list",
                        str(task_list),
                        "--vendor-dir",
                        str(vendor),
                        "--out",
                        str(out_dir),
                        "--trials",
                        "4",
                        "--conditions",
                        "with-skills",
                        "--max-budget-usd",
                        "0.10",
                        "--concurrency",
                        "1",
                    ]
                )
    assert rc == 0
    # 2 trials complete before cumulative ($0.10) hits the cap; the
    # 3rd submission is blocked by the guard.
    assert len(submitted) == 2
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("BUDGET CAP REACHED" in m for m in msgs), (
        f"expected BUDGET CAP REACHED log; got: {msgs}"
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregation_lift_and_pass_rates(tmp_path: Path) -> None:
    vendor = _make_vendor(tmp_path, ["a", "b"])
    task_list = _make_task_list(tmp_path, ["skillsbench/a", "skillsbench/b"])
    out_dir = tmp_path / "out"

    # Canned per-trial results:
    # task a: with-skills = 4/5 success, no-skills = 1/5 success
    # task b: with-skills = 2/5 success, no-skills = 0/5 success
    plan = {
        ("skillsbench/a", "with-skills"): [True, True, True, True, False],
        ("skillsbench/a", "no-skills"): [True, False, False, False, False],
        ("skillsbench/b", "with-skills"): [True, True, False, False, False],
        ("skillsbench/b", "no-skills"): [False, False, False, False, False],
    }
    counters: Dict[tuple, int] = {k: 0 for k in plan}

    def _fake_run_task(self: Any, task: Any, **kwargs: Any) -> TrajectoryResult:
        sd = kwargs.get("skills_dir")
        cond = "with-skills" if sd is not None else "no-skills"
        key = (task.task_id, cond)
        idx = counters[key]
        counters[key] += 1
        success = plan[key][idx]
        return _good_trajectory(task.task_id, success=success)

    with mock.patch.object(baseline.BenchCliBackend, "run_task", _fake_run_task):
        rc = baseline.main(
            [
                "--task-list",
                str(task_list),
                "--vendor-dir",
                str(vendor),
                "--out",
                str(out_dir),
                "--trials",
                "5",
                "--conditions",
                "with-skills,no-skills",
                "--max-budget-usd",
                "100.0",
                "--concurrency",
                "1",
            ]
        )
    assert rc == 0

    summary = json.loads((out_dir / "summary.json").read_text())
    sbs = summary["skillsbench_specific"]

    # Per-task pass rates: a/with = 0.8, a/no = 0.2, b/with = 0.4, b/no = 0
    # Macro: with = (0.8 + 0.4)/2 = 0.6 ; no = (0.2 + 0.0)/2 = 0.1
    assert sbs["with_skills_pass_rate"] == pytest.approx(0.6, abs=1e-6)
    assert sbs["no_skills_pass_rate"] == pytest.approx(0.1, abs=1e-6)
    # Lift = 50pp
    assert sbs["lift_pp"] == pytest.approx(50.0, abs=1e-6)
    # Continuous lift mirrors pp/100 because score is binary here.
    assert sbs["lift_continuous"] == pytest.approx(0.5, abs=1e-6)
    # Leaderboard deltas: with-skills = 60% - 27.7% = 32.3pp ; no = 10 - 11 = -1
    assert sbs["delta_from_leaderboard_with_skills"] == pytest.approx(
        60.0 - 27.7, abs=1e-6
    )
    assert sbs["delta_from_leaderboard_no_skills"] == pytest.approx(
        10.0 - 11.0, abs=1e-6
    )

    # Required EvalResult-shaped keys (R-13).
    for k in (
        "composite",
        "success_rate",
        "tool_calls_per_success",
        "n_tasks",
        "mean_score",
        "scored_task_count",
        "per_task",
        "failures",
    ):
        assert k in summary, f"summary missing required key {k}"
    assert summary["n_tasks"] == 2

    # summary.md exists and includes condition rows.
    md = (out_dir / "summary.md").read_text()
    assert "with-skills" in md
    assert "no-skills" in md
    assert "Lift" in md


def test_console_script_entrypoint_imports() -> None:
    """The pyproject.toml console-script target must resolve."""
    from skill_evolve.skillsbench.baseline import main as imported_main

    assert callable(imported_main)
