"""Unit tests for ``_tiebreak_winner``.

These tests pin the contract that Track A's greedy accept step implements
**A > B > AB** priority on equal composite scores ("do-nothing is
first-class" per the design spec).

Context — resolution of the v2 budget=10 run discrepancy
--------------------------------------------------------
An earlier run log (``runs/compare_m27_b10_v2/a/history.json``) showed
pass 1 with ``winner=B, A=0.45, B=0.45, AB=0.45``, which at a glance
contradicts the "A wins 3-way ties" rule. On inspection the code is
**working as designed** — the three equal ``0.45`` values are an
artifact of the history-record layout, not the tiebreak inputs:

  * pass 0's ``eval_A_summary.composite`` was ``0.325``
    (success_rate=0.375, verified_count=3).
  * ``_tiebreak_winner`` was therefore called with
    ``(score_a=0.325, score_b=0.45, score_ab=0.45)``.
    B legitimately wins because ``0.45 > 0.325``.
  * The pass-1 record logs the **post-update** ``score_A`` (i.e. the new
    incumbent, which is the winner's score — B's ``0.45``), so the three
    printed values collapse to the same number even though B's victory
    was strict, not a tiebreak.

No code change was required; see the ``_tiebreak_winner`` docstring for
the precision caveat. ``test_near_tie_respects_exact_order`` below pins
the non-epsilon semantics so this can never be reinterpreted as a bug.
"""

from __future__ import annotations

import pytest

from skill_evolve.track_a.runner import _tiebreak_winner


# ---------------------------------------------------------------------------
# True 3-way ties — A must win.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("x", [0.0, 0.5, 0.45, 0.999, 1.0, -0.25])
def test_a_wins_three_way_tie(x: float) -> None:
    """A is first-class: any perfect 3-way tie resolves to ``A``."""
    assert _tiebreak_winner(x, x, x) == "A"


# ---------------------------------------------------------------------------
# Unambiguous winners (no tie).
# ---------------------------------------------------------------------------


def test_b_wins_when_higher() -> None:
    assert _tiebreak_winner(0.3, 0.5, 0.4) == "B"


def test_ab_wins_when_higher() -> None:
    assert _tiebreak_winner(0.3, 0.3, 0.5) == "AB"


def test_a_wins_when_higher() -> None:
    assert _tiebreak_winner(0.7, 0.5, 0.4) == "A"


# ---------------------------------------------------------------------------
# 2-way ties.
# ---------------------------------------------------------------------------


def test_b_wins_vs_ab_when_equal_above_a() -> None:
    """B beats AB on ties (priority 1 > 0) when both exceed A."""
    assert _tiebreak_winner(0.3, 0.5, 0.5) == "B"


def test_a_wins_vs_b_when_equal() -> None:
    """A beats B on a 2-way tie (priority 2 > 1), AB below both."""
    assert _tiebreak_winner(0.5, 0.5, 0.3) == "A"


def test_a_wins_vs_ab_when_equal() -> None:
    """A beats AB on a 2-way tie (priority 2 > 0), B below both."""
    assert _tiebreak_winner(0.5, 0.3, 0.5) == "A"


# ---------------------------------------------------------------------------
# Precision / non-epsilon semantics — the real v2 run explanation.
# ---------------------------------------------------------------------------


def test_near_tie_respects_exact_order() -> None:
    """Differences smaller than any rounding threshold still break ties.

    The tiebreak uses raw float comparison with no epsilon window.
    A score difference of ``1e-10`` is enough to flip the outcome —
    priority only applies on *exact* equality.
    """
    eps = 1e-10
    # B edges A by 1e-10 → B wins outright (priority never consulted).
    assert _tiebreak_winner(0.45, 0.45 + eps, 0.45) == "B"
    # AB edges B by 1e-10, both above A → AB wins outright.
    assert _tiebreak_winner(0.45, 0.45 + eps, 0.45 + 2 * eps) == "AB"
    # A edges B by 1e-10 → A wins outright (would also win on tie).
    assert _tiebreak_winner(0.45 + eps, 0.45, 0.45) == "A"


def test_history_json_scenario_from_v2_run() -> None:
    """Regression pin for the v2 budget=10 pass-1 observation.

    history.json rounded all three scores to ``0.45``, but the real
    inputs were ``(0.325, 0.45, 0.45)`` because pass-0 A had
    ``composite=0.325``. The logged equal-looking triple is a display
    artifact of writing ``score_A`` *after* the incumbent was replaced.
    """
    assert _tiebreak_winner(0.325, 0.45, 0.45) == "B"


# ---------------------------------------------------------------------------
# Negative-infinity sentinel (used when B or AB could not be produced).
# ---------------------------------------------------------------------------


def test_neg_inf_never_wins() -> None:
    import math

    assert _tiebreak_winner(0.0, -math.inf, -math.inf) == "A"
    assert _tiebreak_winner(-math.inf, 0.1, -math.inf) == "B"
    assert _tiebreak_winner(-math.inf, -math.inf, 0.1) == "AB"
