"""Tests for ``shared.reflection.partition_trajectories`` (Group H)."""

from __future__ import annotations

from skill_evolve.shared.reflection import (
    Partition,
    partition_trajectories,
)


def _items(*scores: float) -> list[dict]:
    return [{"task_id": f"t{i}", "score": s} for i, s in enumerate(scores)]


def test_mixed_scores_split_at_threshold() -> None:
    items = _items(0.1, 0.4, 0.6, 0.9)
    p = partition_trajectories(items, threshold=0.5)
    assert isinstance(p, Partition)
    assert p.threshold == 0.5
    assert len(p.failure_items) == 2
    assert len(p.success_items) == 2
    assert [i["task_id"] for i in p.failure_items] == ["t0", "t1"]
    assert [i["task_id"] for i in p.success_items] == ["t2", "t3"]


def test_boundary_threshold_goes_to_success() -> None:
    # ``>=`` is success per plan section 7k.
    items = _items(0.5, 0.5, 0.49999)
    p = partition_trajectories(items, threshold=0.5)
    assert len(p.success_items) == 2
    assert len(p.failure_items) == 1


def test_all_pass_yields_empty_failures() -> None:
    items = _items(0.8, 0.9, 1.0)
    p = partition_trajectories(items, threshold=0.5)
    assert p.failure_items == []
    assert len(p.success_items) == 3


def test_all_fail_yields_empty_successes() -> None:
    items = _items(0.0, 0.1, 0.2)
    p = partition_trajectories(items, threshold=0.5)
    assert p.success_items == []
    assert len(p.failure_items) == 3


def test_empty_input_yields_both_empty() -> None:
    p = partition_trajectories([], threshold=0.5)
    assert p.failure_items == []
    assert p.success_items == []


def test_accepts_eval_result_like_object() -> None:
    class _Stub:
        per_task = [{"task_id": "a", "score": 0.2}, {"task_id": "b", "score": 0.7}]

    p = partition_trajectories(_Stub(), threshold=0.5)
    assert [i["task_id"] for i in p.failure_items] == ["a"]
    assert [i["task_id"] for i in p.success_items] == ["b"]


def test_fallback_score_extraction_keys() -> None:
    # ``score`` missing -> tries composite, then mean_score, then
    # success_rate, then ``success`` boolean.
    items = [
        {"task_id": "a", "composite": 0.7},
        {"task_id": "b", "mean_score": 0.4},
        {"task_id": "c", "success_rate": 1.0},
        {"task_id": "d", "success": False},
        {"task_id": "e"},  # no score keys at all -> 0.0
    ]
    p = partition_trajectories(items, threshold=0.5)
    success_ids = {i["task_id"] for i in p.success_items}
    failure_ids = {i["task_id"] for i in p.failure_items}
    assert success_ids == {"a", "c"}
    assert failure_ids == {"b", "d", "e"}
