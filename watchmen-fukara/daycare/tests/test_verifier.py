"""Unit tests for daycare.verifier (Stream 6.4)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from daycare.runner import RunResult
from daycare.verifier import (
    PartialScoringError,
    aggregate_rollouts,
    score_bundle,
)


def _ok(score: float) -> RunResult:
    return RunResult(score=score, completion="ok", completion_len_tokens=1, error=None)


def _err(msg: str = "boom") -> RunResult:
    return RunResult(score=0.0, completion="", completion_len_tokens=0, error=msg)


# ─── aggregate_rollouts (W9) ──────────────────────────────────────────────


def test_aggregate_all_success():
    results = [_ok(0.4), _ok(0.6), _ok(1.0), _ok(0.5), _ok(0.5)]
    assert aggregate_rollouts(results) == pytest.approx(0.6)


def test_aggregate_partial_failure_mean_of_survivors():
    """3 errors out of 5 → mean of the remaining 2 valid scores."""
    results = [_ok(0.4), _err(), _err(), _ok(0.8), _err()]
    # mean of [0.4, 0.8] == 0.6
    assert aggregate_rollouts(results) == pytest.approx(0.6)


def test_aggregate_full_failure_returns_zero():
    """All 5 errored → 0.0 (W9 step 2)."""
    results = [_err(), _err(), _err(), _err(), _err()]
    assert aggregate_rollouts(results) == 0.0


def test_aggregate_empty_returns_zero():
    assert aggregate_rollouts([]) == 0.0


def test_aggregate_handles_none_entries():
    """None entries are treated as full failures (parse-fail upstream)."""
    results = [_ok(0.5), None, _ok(0.7)]
    assert aggregate_rollouts(results) == pytest.approx(0.6)


# ─── PartialScoringError gating in score_bundle (FM#3) ────────────────────


def _make_rows(n: int, split: str = "holdout") -> list[dict]:
    """Build n synthetic eval rows with the minimal fields score_bundle reads."""
    return [
        {
            "id": f"e{i}",
            "split": split,
            "type": "script_gen",
            "prompt": f"prompt {i}",
            "anonymized_prompt": f"prompt {i}",
            "rubric": "score 0-1",
            "accepted": True,
            "baseline_completion_len_tokens": 100,
        }
        for i in range(n)
    ]


def test_score_bundle_raises_partial_scoring_when_too_many_errors(tmp_path):
    """≥10% of evals erroring → PartialScoringError (not silently 0.0)."""
    rows = _make_rows(10)

    # Mock _score_one_eval so 5/10 evals return None (full eval-level error).
    # The 90% threshold: successful_evals / n_total must be ≥ 0.90.
    # With 5/10 successful (= 0.5), score_bundle must raise.
    call_count = {"n": 0}

    def fake_score_one(row, bundle_dir, skill_prompt, model, judge_model, api_key, rollouts, seed):
        call_count["n"] += 1
        # First half succeed, second half return None.
        if call_count["n"] <= 5:
            return row.get("id", ""), 0.7, [_ok(0.7)] * rollouts
        return row.get("id", ""), None, [_err()] * rollouts

    with patch("daycare.verifier._score_one_eval", side_effect=fake_score_one):
        with pytest.raises(PartialScoringError) as excinfo:
            score_bundle(
                eval_rows=rows,
                split="holdout",
                bundle_dir=tmp_path,
                model="weak",
                judge_model="judge",
                api_key="dummy",
                rollouts=1,
                max_workers=1,
            )
    assert excinfo.value.total == 10
    assert excinfo.value.successful < 9  # below the 90% bar


def test_score_bundle_succeeds_when_above_threshold(tmp_path):
    """9/10 successful → 90% exactly → score_bundle returns normally."""
    rows = _make_rows(10)
    n_calls = {"n": 0}

    def fake_score_one(row, bundle_dir, skill_prompt, model, judge_model, api_key, rollouts, seed):
        n_calls["n"] += 1
        if n_calls["n"] == 1:
            # First eval errors out.
            return row.get("id", ""), None, [_err()]
        return row.get("id", ""), 0.5, [_ok(0.5)]

    with patch("daycare.verifier._score_one_eval", side_effect=fake_score_one):
        summary = score_bundle(
            eval_rows=rows,
            split="holdout",
            bundle_dir=tmp_path,
            model="weak",
            judge_model="judge",
            api_key="dummy",
            rollouts=1,
            max_workers=1,
        )
    assert summary.n_holdout == 10
    assert summary.successful_evals == 9
    # Holdout = mean of the 9 surviving 0.5s = 0.5.
    assert summary.holdout_score == pytest.approx(0.5)


def test_score_bundle_empty_rows_returns_zero(tmp_path):
    """No rows → return a zeroed EvalSummary (don't crash)."""
    summary = score_bundle(
        eval_rows=[],
        split="holdout",
        bundle_dir=tmp_path,
        model="weak",
        judge_model="judge",
        api_key="dummy",
        rollouts=1,
        max_workers=1,
    )
    assert summary.n_holdout == 0
    assert summary.holdout_score == 0.0
    assert summary.successful_evals == 0


def test_partial_scoring_error_fields():
    """The exception carries successful/total counts for caller logging."""
    err = PartialScoringError("msg", successful=3, total=10)
    assert err.successful == 3
    assert err.total == 10
    assert "msg" in str(err)
