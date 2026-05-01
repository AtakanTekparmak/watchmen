"""Regression test for the inner ``evaluate()`` agent-backend dispatch.

Phase E bug 7: ``skill_evolve.evaluator.evaluate(..., agent_backend=...)``
accepted and validated the kwarg but the per-task loop hardcoded a
direct call to ``_run_one_task`` (Hermes/Docker subprocess). Result:
``agent_backend="bench-cli"`` was dead code at the inner loop —
SkillsBench tasks were silently routed through Hermes, which then
crashed on the missing ``docker_image`` field in the SkillsBench
``success_check_payload`` shape.

This module locks in the dispatch wiring with no real LLM / Docker /
bench CLI calls:

  * ``agent_backend="bench-cli"`` invokes
    :meth:`BenchCliBackend.run_task` at least once and never invokes
    :meth:`HermesBackend.run_task`.
  * ``agent_backend="hermes"`` (and the ``None`` default) invoke
    :meth:`HermesBackend.run_task` and never invoke
    :meth:`BenchCliBackend.run_task`.
  * The :class:`TrajectoryResult` returned by the backend is
    projected onto :class:`TaskOutcome` via ``to_task_outcome()`` and
    surfaces in :attr:`EvalResult.per_task` with the expected
    ``task_id``/``success`` values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from skill_evolve.agents.base import TrajectoryResult
from skill_evolve.evaluator import evaluate


def _make_skillsbench_task(
    task_id: str = "skillsbench/citation-check",
) -> Dict[str, Any]:
    """A minimal hydrated-task dict matching the SkillsBench loader's shape."""
    return {
        "task_id": task_id,
        "source": "skillsbench",
        "prompt": "irrelevant — backend is mocked",
        "success_check_kind": "skillsbench_test_sh",
        "success_check_payload": {
            "task_dir": "/tmp/fake/task_dir",
            "environment_dir": "/tmp/fake/task_dir/environment",
            "tests_dir": "/tmp/fake/task_dir/tests",
            "skills_dir": None,
            "domain": "misc",
            "timeout_sec": 60,
        },
        "timeout_s": 60,
        "stage": 1,
        "skill_relevance": "",
        "extra": {"dataset_task_name": task_id.split("/", 1)[-1]},
    }


def _make_tblite_task(task_id: str = "tblite/foo-bar") -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "source": "tblite",
        "prompt": "irrelevant — backend is mocked",
        "success_check_kind": "tblite_run",
        "success_check_payload": {"docker_image": "tblite:latest"},
        "timeout_s": 60,
        "stage": 1,
        "skill_relevance": "",
    }


def _canned_trajectory(task_id: str, *, success: bool = True) -> TrajectoryResult:
    return TrajectoryResult(
        task_id=task_id,
        success=success,
        tool_calls=3,
        elapsed_s=1.5,
        skills_invoked=["mocked"],
        last_msg="mocked done",
        raw_completed=True,
        notes="",
        verified=success,
        verifier_status="passed" if success else "failed",
        verifier_detail="",
        score=1.0 if success else 0.0,
        repeats_detail=[],
        cost_usd=None,
    )


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    """A trivial skills-folder shape that satisfies ``evaluate``'s sanity checks."""
    d = tmp_path / "skills"
    placeholder = d / "placeholder"
    placeholder.mkdir(parents=True)
    (placeholder / "SKILL.md").write_text(
        "---\nname: placeholder\ndescription: placeholder\n---\n\nbody\n",
        encoding="utf-8",
    )
    return d


@pytest.fixture
def env_with_live_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the live-keys gate to pass so ``evaluate`` doesn't synth out."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-used")


@pytest.fixture
def fake_run_agent_py(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Make ``RUN_AGENT_PY.exists()`` truthy without needing a real fork."""
    fake_path = tmp_path / "fake_run_agent.py"
    fake_path.write_text("# placeholder", encoding="utf-8")
    monkeypatch.setattr("skill_evolve.evaluator.RUN_AGENT_PY", fake_path)


def _patch_load_subset(
    monkeypatch: pytest.MonkeyPatch, tasks: List[Dict[str, Any]]
) -> None:
    monkeypatch.setattr(
        "skill_evolve.evaluator.load_subset",
        lambda *a, **kw: list(tasks),
    )


# ---------------------------------------------------------------------------
# Dispatch invariants
# ---------------------------------------------------------------------------


def test_bench_cli_path_invokes_bench_cli_backend_only(
    skills_dir: Path,
    env_with_live_keys: None,
    fake_run_agent_py: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``agent_backend="bench-cli"`` must dispatch through
    :meth:`BenchCliBackend.run_task` and never touch HermesBackend."""
    task = _make_skillsbench_task()
    _patch_load_subset(monkeypatch, [task])

    bench_calls: List[Any] = []
    hermes_calls: List[Any] = []

    def _bench_run_task(self: Any, *args: Any, **kwargs: Any) -> TrajectoryResult:
        bench_calls.append((args, kwargs))
        return _canned_trajectory(task["task_id"], success=True)

    def _hermes_run_task(self: Any, *args: Any, **kwargs: Any) -> TrajectoryResult:
        hermes_calls.append((args, kwargs))
        return _canned_trajectory(task["task_id"], success=True)

    monkeypatch.setattr(
        "skill_evolve.agents.bench_cli.BenchCliBackend.run_task",
        _bench_run_task,
    )
    monkeypatch.setattr(
        "skill_evolve.agents.hermes.HermesBackend.run_task",
        _hermes_run_task,
    )

    res = evaluate(
        skills_dir,
        sources=["skillsbench"],
        task_ids=["skillsbench/citation-check"],
        agent_backend="bench-cli",
        verify=False,
        cascade=False,
        repeats=1,
    )

    assert len(bench_calls) >= 1, "BenchCliBackend.run_task was never invoked"
    assert hermes_calls == [], (
        f"HermesBackend.run_task must NOT be invoked when agent_backend=bench-cli; "
        f"got {len(hermes_calls)} calls"
    )
    # Trajectory result projects cleanly onto per_task.
    assert res.per_task, "expected at least one per_task entry"
    assert res.per_task[0]["task_id"] == task["task_id"]
    assert res.per_task[0]["success"] is True


def test_hermes_explicit_path_invokes_hermes_backend_only(
    skills_dir: Path,
    env_with_live_keys: None,
    fake_run_agent_py: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit ``agent_backend="hermes"`` must dispatch through
    :meth:`HermesBackend.run_task` and never touch BenchCliBackend."""
    task = _make_tblite_task()
    _patch_load_subset(monkeypatch, [task])

    bench_calls: List[Any] = []
    hermes_calls: List[Any] = []

    def _bench_run_task(self: Any, *args: Any, **kwargs: Any) -> TrajectoryResult:
        bench_calls.append((args, kwargs))
        return _canned_trajectory(task["task_id"], success=True)

    def _hermes_run_task(self: Any, *args: Any, **kwargs: Any) -> TrajectoryResult:
        hermes_calls.append((args, kwargs))
        return _canned_trajectory(task["task_id"], success=True)

    monkeypatch.setattr(
        "skill_evolve.agents.bench_cli.BenchCliBackend.run_task",
        _bench_run_task,
    )
    monkeypatch.setattr(
        "skill_evolve.agents.hermes.HermesBackend.run_task",
        _hermes_run_task,
    )

    res = evaluate(
        skills_dir,
        sources=["tblite"],
        agent_backend="hermes",
        verify=False,
        cascade=False,
        repeats=1,
    )

    assert len(hermes_calls) >= 1, "HermesBackend.run_task was never invoked"
    assert bench_calls == [], (
        f"BenchCliBackend.run_task must NOT be invoked when agent_backend=hermes; "
        f"got {len(bench_calls)} calls"
    )
    assert res.per_task, "expected at least one per_task entry"
    assert res.per_task[0]["task_id"] == task["task_id"]


def test_default_agent_backend_is_hermes(
    skills_dir: Path,
    env_with_live_keys: None,
    fake_run_agent_py: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``agent_backend=None`` must behave like ``"hermes"`` —
    HermesBackend is invoked, BenchCliBackend is not."""
    task = _make_tblite_task()
    _patch_load_subset(monkeypatch, [task])

    bench_calls: List[Any] = []
    hermes_calls: List[Any] = []

    def _bench_run_task(self: Any, *args: Any, **kwargs: Any) -> TrajectoryResult:
        bench_calls.append((args, kwargs))
        return _canned_trajectory(task["task_id"], success=True)

    def _hermes_run_task(self: Any, *args: Any, **kwargs: Any) -> TrajectoryResult:
        hermes_calls.append((args, kwargs))
        return _canned_trajectory(task["task_id"], success=True)

    monkeypatch.setattr(
        "skill_evolve.agents.bench_cli.BenchCliBackend.run_task",
        _bench_run_task,
    )
    monkeypatch.setattr(
        "skill_evolve.agents.hermes.HermesBackend.run_task",
        _hermes_run_task,
    )

    evaluate(
        skills_dir,
        sources=["tblite"],
        agent_backend=None,
        verify=False,
        cascade=False,
        repeats=1,
    )

    assert len(hermes_calls) >= 1
    assert bench_calls == []


def test_unknown_agent_backend_raises(skills_dir: Path) -> None:
    """An unknown ``agent_backend`` is rejected at validation."""
    with pytest.raises(ValueError, match="unknown agent_backend"):
        evaluate(skills_dir, agent_backend="openai")


def test_dispatch_log_emits_backend_name_and_task_id(
    skills_dir: Path,
    env_with_live_keys: None,
    fake_run_agent_py: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The dispatch helper logs which backend ran which task — this is
    the load-bearing breadcrumb the audit looks for in run logs."""
    task = _make_skillsbench_task("skillsbench/citation-check")
    _patch_load_subset(monkeypatch, [task])

    monkeypatch.setattr(
        "skill_evolve.agents.bench_cli.BenchCliBackend.run_task",
        lambda self, *a, **k: _canned_trajectory(task["task_id"], success=True),
    )
    monkeypatch.setattr(
        "skill_evolve.agents.hermes.HermesBackend.run_task",
        lambda self, *a, **k: _canned_trajectory(task["task_id"], success=True),
    )

    with caplog.at_level("INFO", logger="skill_evolve.evaluator"):
        evaluate(
            skills_dir,
            sources=["skillsbench"],
            task_ids=["skillsbench/citation-check"],
            agent_backend="bench-cli",
            verify=False,
            cascade=False,
            repeats=1,
        )

    msgs = [r.getMessage() for r in caplog.records]
    assert any(
        "BenchCliBackend" in m and "skillsbench/citation-check" in m for m in msgs
    ), f"expected dispatch log line for BenchCliBackend; got {msgs!r}"


# ---------------------------------------------------------------------------
# Adapter sanity
# ---------------------------------------------------------------------------


def test_trajectory_to_task_outcome_round_trips(
    skills_dir: Path,
    env_with_live_keys: None,
    fake_run_agent_py: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All field values from the canned trajectory must surface on the
    resulting :class:`TaskOutcome` (via ``to_task_outcome()`` projection)."""
    task = _make_skillsbench_task("skillsbench/foo")
    _patch_load_subset(monkeypatch, [task])

    canned = TrajectoryResult(
        task_id=task["task_id"],
        success=False,
        tool_calls=7,
        elapsed_s=4.2,
        skills_invoked=["alpha", "beta"],
        last_msg="boom",
        raw_completed=False,
        notes="canned",
        verified=False,
        verifier_status="failed",
        verifier_detail="pytest tail here",
        score=0.25,
        repeats_detail=[],
        cost_usd=0.01,
    )

    monkeypatch.setattr(
        "skill_evolve.agents.bench_cli.BenchCliBackend.run_task",
        lambda self, *a, **k: canned,
    )
    monkeypatch.setattr(
        "skill_evolve.agents.hermes.HermesBackend.run_task",
        lambda self, *a, **k: canned,
    )

    res = evaluate(
        skills_dir,
        sources=["skillsbench"],
        agent_backend="bench-cli",
        verify=False,
        cascade=False,
        repeats=1,
    )

    pt = res.per_task[0]
    assert pt["task_id"] == task["task_id"]
    assert pt["success"] is False
    assert pt["tool_calls"] == 7
    assert pt["skills_invoked"] == ["alpha", "beta"]
    assert pt["verified"] is False
    assert pt["verifier_status"] == "failed"
    assert pt["score"] == 0.25
    # to_task_outcome drops cost_usd; not present on TaskOutcome.
    assert "cost_usd" not in pt


# ---------------------------------------------------------------------------
# Bug 14 regression — task_id slash/underscore mismatch.
# ---------------------------------------------------------------------------


def test_result_to_trajectory_preserves_manifest_task_id() -> None:
    """Bug 14: ``_result_to_trajectory`` MUST preserve the manifest-canonical
    ``task_id`` (slash form) and never adopt bench's flattened ``task_name``
    field (underscore form).

    Why this matters: the outer aggregator in
    ``skill_evolve.evaluator._run_batch`` buckets ``TaskOutcome`` by
    ``o.task_id`` and looks up by manifest ``task_id``. If the
    trajectory carries the bench-flattened name (``skillsbench_citation-check``)
    while the manifest carries the slash form
    (``skillsbench/citation-check``), the lookup misses every bucket,
    every program scores 0.0, and evolution runs blind.
    """
    from pathlib import Path

    from skill_evolve.agents.bench_cli import _result_to_trajectory

    manifest_task_id = "skillsbench/citation-check"
    bench_flattened = "skillsbench_citation-check"

    # bench writes ``task_name`` = underscore-flattened symlink basename.
    fake_result_json = {
        "task_name": bench_flattened,
        "rewards": {"reward": 1.0},
        "error": None,
        "verifier_error": None,
        "timing": {"total": 12.5},
        "n_tool_calls": 4,
        "n_prompts": 1,
        "trajectory_source": "test",
        "partial_trajectory": False,
    }

    traj = _result_to_trajectory(
        task_id=manifest_task_id,
        result_json=fake_result_json,
        trial_dir=Path("/tmp/does-not-exist"),
        anonymize_map=None,
        budget_usd=None,
    )

    assert traj.task_id == manifest_task_id, (
        f"_result_to_trajectory must preserve manifest task_id; got "
        f"{traj.task_id!r}, expected {manifest_task_id!r} "
        f"(NOT bench's flattened {bench_flattened!r})"
    )
    # Sanity: payload still parses correctly.
    assert traj.success is True
    assert traj.tool_calls == 4


def test_bench_cli_outcome_buckets_under_manifest_task_id(
    skills_dir: Path,
    env_with_live_keys: None,
    fake_run_agent_py: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end Bug 14 regression: when the bench backend returns a
    trajectory tagged with the manifest task_id, the outer aggregator's
    bucket-by-task_id lookup hits and the per_task slot is populated
    (not silently dropped → composite=0.0).
    """
    manifest_task_id = "skillsbench/citation-check"
    task = _make_skillsbench_task(manifest_task_id)
    _patch_load_subset(monkeypatch, [task])

    monkeypatch.setattr(
        "skill_evolve.agents.bench_cli.BenchCliBackend.run_task",
        lambda self, *a, **k: _canned_trajectory(manifest_task_id, success=True),
    )
    monkeypatch.setattr(
        "skill_evolve.agents.hermes.HermesBackend.run_task",
        lambda self, *a, **k: _canned_trajectory(manifest_task_id, success=True),
    )

    res = evaluate(
        skills_dir,
        sources=["skillsbench"],
        task_ids=[manifest_task_id],
        agent_backend="bench-cli",
        verify=False,
        cascade=False,
        repeats=1,
    )

    # The crux: per_task is populated and the aggregator did NOT drop
    # the bucket. Pre-fix this test would see len(res.per_task)==0.
    assert len(res.per_task) == 1, (
        "Bug 14 regression: outer aggregator dropped the bench-cli "
        "outcome because the trajectory's task_id didn't match the "
        f"manifest form. Got per_task={res.per_task!r}"
    )
    assert res.per_task[0]["task_id"] == manifest_task_id
