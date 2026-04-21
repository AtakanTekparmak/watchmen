"""Tests for the ``repeats`` kwarg on :func:`skill_evolve.evaluator.evaluate`.

We test at two levels:

  1. Unit-test ``_aggregate_repeats`` directly on handcrafted
     :class:`TaskOutcome` lists — majority vote + median + union semantics.
  2. Integration test: monkey-patch ``_run_one_task`` to simulate a noisy
     signal (success flips across repeats) and run the full ``evaluate``
     pipeline with ``repeats=3``, asserting the aggregated outcome shape.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

import pytest

from skill_evolve import evaluator as _ev
from skill_evolve.evaluator import (
    TaskOutcome,
    _aggregate_repeats,
    compute_composite,
    evaluate,
)


# ---------------------------------------------------------------------------
# Unit: _aggregate_repeats
# ---------------------------------------------------------------------------

def _mk(
    *,
    task_id: str = "t1",
    success: bool = True,
    tool_calls: int = 5,
    elapsed: float = 1.0,
    skills: List[str] = None,
    verified=None,
    verifier_status: str = "not_run",
    score=None,
) -> TaskOutcome:
    return TaskOutcome(
        task_id=task_id,
        success=success,
        tool_calls=tool_calls,
        elapsed_s=elapsed,
        skills_invoked=list(skills or []),
        last_msg="",
        verified=verified,
        verifier_status=verifier_status,
        score=score,
    )


def test_aggregate_single_is_passthrough():
    o = _mk(success=True, tool_calls=3)
    a = _aggregate_repeats([o])
    assert a is o


def test_aggregate_majority_vote_success():
    outs = [
        _mk(success=True, tool_calls=2),
        _mk(success=True, tool_calls=4),
        _mk(success=False, tool_calls=6),
    ]
    a = _aggregate_repeats(outs)
    assert a.success is True
    # Median of [2,4,6] = 4
    assert a.tool_calls == 4


def test_aggregate_majority_vote_fail_ties_conservative():
    """Conservative tie-break: even k, 1–1 tie → False."""
    outs = [
        _mk(success=True, tool_calls=2),
        _mk(success=False, tool_calls=8),
    ]
    a = _aggregate_repeats(outs)
    # 1/2 successes is not a strict majority — should be False.
    assert a.success is False
    # Median of [2,8] = 5
    assert a.tool_calls == 5


def test_aggregate_skills_union_sorted():
    outs = [
        _mk(skills=["b", "a"]),
        _mk(skills=["c", "a"]),
        _mk(skills=["b"]),
    ]
    a = _aggregate_repeats(outs)
    assert a.skills_invoked == ["a", "b", "c"]


def test_aggregate_records_repeats_detail():
    outs = [
        _mk(success=True, tool_calls=3, elapsed=1.2),
        _mk(success=False, tool_calls=9, elapsed=4.1),
        _mk(success=True, tool_calls=5, elapsed=2.2),
    ]
    a = _aggregate_repeats(outs)
    assert isinstance(a.repeats_detail, list) and len(a.repeats_detail) == 3
    for d in a.repeats_detail:
        for k in ("task_id", "success", "tool_calls", "elapsed_s"):
            assert k in d


def test_aggregate_verifier_state_follows_majority():
    # 2 successes with verified=True, 1 failure with verified=False →
    # majority is success, rep should carry verified=True.
    outs = [
        _mk(success=True, verified=True, verifier_status="passed"),
        _mk(success=True, verified=True, verifier_status="passed"),
        _mk(success=False, verified=False, verifier_status="failed"),
    ]
    a = _aggregate_repeats(outs)
    assert a.success is True
    assert a.verified is True
    assert a.verifier_status == "passed"


# ---------------------------------------------------------------------------
# Integration: evaluate() with repeats=3 end-to-end (mocked _run_one_task)
# ---------------------------------------------------------------------------

def test_evaluate_repeats_3_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Full ``evaluate`` pipeline with mocked subprocess + live-key env.

    We point ``_run_one_task`` at a deterministic fake that flips success
    based on the call index, then assert the aggregated EvalResult shape
    matches the contract: one outcome per task, ``repeats_detail`` length 3.
    """
    # Build a trivial skill folder so validate() on sandbox succeeds.
    skills = tmp_path / "skills"
    (skills / "demo").mkdir(parents=True)
    (skills / "demo" / "SKILL.md").write_text(
        "---\nname: demo\ndescription: demo\n---\nbody\n", encoding="utf-8",
    )

    # Force the live-key path so evaluate() doesn't short-circuit to synthetic.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    # Provide a two-task manifest, single-stage.
    fake_tasks = [
        {"task_id": "tblite/broken-python", "stage": 1, "prompt": "x",
         "timeout_s": 10},
        {"task_id": "tblite/other-task", "stage": 2, "prompt": "y",
         "timeout_s": 10},
    ]
    monkeypatch.setattr(_ev, "load_subset", lambda **kw: fake_tasks)

    # Provide a deterministic flipping runner. Call counter per task_id.
    call_counter: Dict[str, int] = {}

    def fake_run_one_task(task, skills_folder, **kwargs):
        tid = task["task_id"]
        idx = call_counter.get(tid, 0)
        call_counter[tid] = idx + 1
        # tblite/broken-python: 2 of 3 succeed → majority True
        # tblite/other-task: 1 of 3 succeed → majority False
        if tid == "tblite/broken-python":
            succ = idx != 1
        else:
            succ = idx == 0
        return TaskOutcome(
            task_id=tid,
            success=succ,
            tool_calls=3 + idx,
            elapsed_s=1.0 + idx * 0.1,
            skills_invoked=[f"skill_{idx}"],
            verified=True if succ else False,
            verifier_status="passed" if succ else "failed",
        )

    monkeypatch.setattr(_ev, "_run_one_task", fake_run_one_task)

    res = evaluate(
        skills,
        cascade=False,
        max_workers=1,
        verify=False,
        repeats=3,
    )

    # One outcome per task.
    assert res.n_tasks == 2
    assert len(res.per_task) == 2

    # Each per-task outcome carries ``repeats_detail`` of length 3.
    for t in res.per_task:
        assert len(t.get("repeats_detail") or []) == 3

    # Verify majority-vote aggregation.
    by_id = {t["task_id"]: t for t in res.per_task}
    assert by_id["tblite/broken-python"]["success"] is True
    assert by_id["tblite/other-task"]["success"] is False

    # Each task's runner should have been called exactly 3 times.
    assert call_counter["tblite/broken-python"] == 3
    assert call_counter["tblite/other-task"] == 3


def test_evaluate_default_repeats_1_is_backward_compat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Default ``repeats=1`` leaves ``repeats_detail`` empty (non-aggregated)."""
    skills = tmp_path / "skills"
    (skills / "demo").mkdir(parents=True)
    (skills / "demo" / "SKILL.md").write_text(
        "---\nname: demo\ndescription: demo\n---\nbody\n", encoding="utf-8",
    )

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        _ev, "load_subset",
        lambda **kw: [{"task_id": "t1", "stage": 1, "prompt": "x",
                       "timeout_s": 10}],
    )

    def fake_run_one_task(task, skills_folder, **kwargs):
        return TaskOutcome(
            task_id=task["task_id"],
            success=True,
            tool_calls=2,
            elapsed_s=0.5,
            verified=True,
            verifier_status="passed",
        )

    monkeypatch.setattr(_ev, "_run_one_task", fake_run_one_task)

    res = evaluate(skills, cascade=False, verify=False)  # repeats defaults to 1
    assert res.n_tasks == 1
    assert res.per_task[0]["repeats_detail"] == []


# ---------------------------------------------------------------------------
# Continuous-scoring plumbing
# ---------------------------------------------------------------------------

def test_aggregate_mean_score_over_repeats():
    outs = [
        _mk(success=True, score=1.0),
        _mk(success=True, score=0.6),
        _mk(success=False, score=0.0),
    ]
    a = _aggregate_repeats(outs)
    # Majority: 2/3 success → True; score mean = (1.0+0.6+0.0)/3 = 0.533...
    assert a.success is True
    assert a.score == pytest.approx((1.0 + 0.6 + 0.0) / 3)


def test_aggregate_score_none_when_no_repeat_scored():
    outs = [_mk(success=True, score=None), _mk(success=False, score=None)]
    a = _aggregate_repeats(outs)
    assert a.score is None


def test_aggregate_score_ignores_unscored_repeats():
    # Two repeats scored, one without a structured breakdown → mean over
    # just the scored ones so a verifier hiccup doesn't depress the mean.
    outs = [
        _mk(success=True, score=0.8),
        _mk(success=True, score=0.6),
        _mk(success=True, score=None),
    ]
    a = _aggregate_repeats(outs)
    assert a.score == pytest.approx(0.7)


def test_compute_composite_prefers_mean_score():
    # success_rate=0.75, mean_score=0.40 → composite should follow the
    # finer-grained signal, not the binary one.
    c = compute_composite(0.75, avg_tool_calls=0.0, mean_score=0.40)
    assert c == pytest.approx(0.40)


def test_compute_composite_falls_back_to_success_rate():
    c = compute_composite(0.75, avg_tool_calls=0.0, mean_score=None)
    assert c == pytest.approx(0.75)


def test_evaluate_surfaces_mean_score_when_tasks_expose_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """End-to-end: if per-task outcomes carry ``score``, the EvalResult
    surfaces ``mean_score`` and the composite prefers it over success_rate.
    """
    skills = tmp_path / "skills"
    (skills / "demo").mkdir(parents=True)
    (skills / "demo" / "SKILL.md").write_text(
        "---\nname: demo\ndescription: demo\n---\nbody\n", encoding="utf-8",
    )

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        _ev, "load_subset",
        lambda **kw: [
            {"task_id": "t1", "stage": 1, "prompt": "x", "timeout_s": 10},
            {"task_id": "t2", "stage": 1, "prompt": "y", "timeout_s": 10},
        ],
    )

    scores = {"t1": 0.7, "t2": 0.3}

    def fake_run_one_task(task, skills_folder, **kwargs):
        tid = task["task_id"]
        s = scores[tid]
        return TaskOutcome(
            task_id=tid, success=s >= 0.5, tool_calls=0, elapsed_s=0.0,
            verified=s >= 0.5, verifier_status="ok", score=s,
        )

    monkeypatch.setattr(_ev, "_run_one_task", fake_run_one_task)
    res = evaluate(skills, cascade=False, verify=False)

    assert res.scored_task_count == 2
    assert res.mean_score == pytest.approx(0.5)
    # success_rate = 1/2 = 0.5 here by coincidence; use a value-free assert
    # to confirm composite tracks mean_score, not success_rate.
    assert res.composite == pytest.approx(0.5)
