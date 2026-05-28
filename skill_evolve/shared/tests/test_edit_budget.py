"""Group F unit tests — bounded edit-budget L_t schedule.

Covers ``parse_schedule`` / ``compute_lt`` / ``clip_ops`` per plan F.7.
"""

from __future__ import annotations

import math

import pytest

from skill_evolve.shared.edit_budget import (
    ScheduleSpec,
    clip_ops,
    compute_lt,
    parse_schedule,
)
from skill_evolve.shared.patch_parser import AddFile, EditFile


# ─── parse_schedule ────────────────────────────────────────────────────────


def test_parse_schedule_constant() -> None:
    spec = parse_schedule("constant:5")
    assert spec == ScheduleSpec(kind="constant", start=5, end=5)


def test_parse_schedule_linear() -> None:
    spec = parse_schedule("linear:8->2")
    assert spec == ScheduleSpec(kind="linear", start=8, end=2)


def test_parse_schedule_cosine() -> None:
    spec = parse_schedule("cosine:8->2")
    assert spec == ScheduleSpec(kind="cosine", start=8, end=2)


def test_parse_schedule_cosine_ascending() -> None:
    spec = parse_schedule("cosine:2->8")
    assert spec == ScheduleSpec(kind="cosine", start=2, end=8)


@pytest.mark.parametrize(
    "spec",
    [
        "",
        "garbage",
        "constant",
        "constant:",
        "constant:1.5",
        "linear:8",
        "linear:8-2",
        "linear:8->",
        "cosine:->2",
        "expon:8->2",
        "constant:5->5",
        "linear:a->b",
    ],
)
def test_parse_schedule_malformed_raises(spec: str) -> None:
    with pytest.raises(ValueError, match="malformed schedule spec"):
        parse_schedule(spec)


# ─── compute_lt ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("k", [0, 6, 11])
def test_compute_lt_constant_invariant(k: int) -> None:
    assert compute_lt("constant:5", k, 12) == 5


def test_compute_lt_linear_endpoints() -> None:
    assert compute_lt("linear:8->2", 0, 12) == 8
    assert compute_lt("linear:8->2", 12, 12) == 2


def test_compute_lt_linear_midpoint_in_range() -> None:
    mid = compute_lt("linear:8->2", 6, 12)
    assert 2 <= mid <= 8
    # linear midpoint: 8 + (2 - 8) * 0.5 = 5
    assert mid == 5


def test_compute_lt_cosine_endpoints() -> None:
    assert compute_lt("cosine:8->2", 0, 12) == 8
    assert compute_lt("cosine:8->2", 12, 12) == 2


def test_compute_lt_cosine_midpoint_approx_five() -> None:
    # cosine: M + 0.5 * (N - M) * (1 + cos(pi * 0.5)) = 2 + 0.5 * 6 * 1 = 5
    mid = compute_lt("cosine:8->2", 6, 12)
    assert mid == 5


def test_compute_lt_cosine_degenerate_collapses_to_constant() -> None:
    # N == M -> always N regardless of iter_n.
    assert compute_lt("cosine:5->5", 3, 12) == 5
    assert compute_lt("cosine:5->5", 0, 12) == 5
    assert compute_lt("cosine:5->5", 12, 12) == 5


def test_compute_lt_linear_degenerate_collapses_to_constant() -> None:
    assert compute_lt("linear:4->4", 7, 12) == 4


def test_compute_lt_clamps_to_one_min() -> None:
    # cosine:1->0 hits 0 at the endpoint; result must clamp up to 1.
    assert compute_lt("cosine:1->0", 12, 12) == 1
    # constant:0 also clamps to 1.
    assert compute_lt("constant:0", 0, 12) == 1


def test_compute_lt_accepts_pre_parsed_spec() -> None:
    spec = parse_schedule("cosine:8->2")
    assert compute_lt(spec, 0, 12) == 8
    assert compute_lt(spec, 12, 12) == 2


def test_compute_lt_handles_zero_max_iters() -> None:
    # Guards against ZeroDivisionError via ``max(1, max_iters)``.
    # With max_iters == 0 we treat it as 1 so iter_n=0 -> t=0 -> N.
    assert compute_lt("cosine:8->2", 0, 0) == 8


def test_compute_lt_cosine_matches_paper_formula() -> None:
    """Spot-check the paper formula against a hand-computed value."""
    # cosine:10->2 at iter_n=3, max_iters=12 -> t=0.25
    # raw = 2 + 0.5 * 8 * (1 + cos(pi * 0.25))
    #     = 2 + 4 * (1 + 0.7071...) ~= 8.828
    # rounded -> 9
    t = 3 / 12
    raw = 2 + 0.5 * 8 * (1 + math.cos(math.pi * t))
    assert compute_lt("cosine:10->2", 3, 12) == int(round(raw))


# ─── clip_ops ──────────────────────────────────────────────────────────────


def _ops(n: int) -> list:
    """Build ``n`` distinct FileOp instances for ordering assertions."""
    return [
        AddFile(op="ADD_FILE", path=f"a/{i}.py", content=f"# op {i}\n")
        for i in range(n)
    ]


def test_clip_ops_truncates_over_budget() -> None:
    ops = _ops(5)
    clipped = clip_ops(ops, 2)
    assert len(clipped) == 2
    assert clipped == ops[:2]


def test_clip_ops_length_stable_under_budget() -> None:
    ops = _ops(2)
    clipped = clip_ops(ops, 5)
    # Length-stable AND returns identity (no copy).
    assert clipped is ops


def test_clip_ops_length_stable_at_exact_budget() -> None:
    ops = _ops(3)
    clipped = clip_ops(ops, 3)
    assert clipped is ops


def test_clip_ops_preserves_proposer_order() -> None:
    # Mix kinds — emit order is the priority. The ADD>EDIT>DELETE>REWRITE
    # ranking is reserved as a tie-break heuristic for future ranking-LLM
    # paths; the deterministic clip_ops MUST NOT re-sort.
    a = EditFile(op="EDIT_FILE", path="SKILL.md", content="z\n")
    b = AddFile(op="ADD_FILE", path="scripts/a.py", content="x\n")
    c = AddFile(op="ADD_FILE", path="scripts/b.py", content="y\n")
    ops = [a, b, c]
    clipped = clip_ops(ops, 2)
    assert clipped == [a, b]  # emit order preserved — EDIT_FILE first.


def test_clip_ops_zero_budget_returns_empty() -> None:
    assert clip_ops(_ops(3), 0) == []


def test_clip_ops_rejects_negative_budget() -> None:
    with pytest.raises(ValueError):
        clip_ops(_ops(3), -1)


def test_clip_ops_handles_empty_input() -> None:
    assert clip_ops([], 5) == []
