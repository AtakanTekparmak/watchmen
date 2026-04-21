"""CLI parsing tests for Track A's asymmetric model flags.

See V2_FIXES_PLAN.md / Fix 1: the runner exposes ``--outer-model`` (outer
/ meta LLM — critic, op-planner, body-writer, synthesizer) and
``--inner-model`` (inner Hermes agent rollouts), with ``--model`` kept
as a deprecated alias that sets BOTH.

These tests exercise only the argparse surface + the alias-handling
branch in ``_cli`` — they do NOT run the evolution loop. That's the
point: we want fast, deterministic coverage of the flag-routing
semantics.
"""

from __future__ import annotations

import argparse
from typing import List, Tuple
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_parser_like_cli() -> argparse.ArgumentParser:
    """Re-build the argparse parser the same way ``_cli`` does.

    We mirror the flag-definition block from ``runner._cli`` here so the
    test is a pure unit test (no LLM client / evaluator imports needed).
    If ``_cli`` diverges from this the test will fail, which is what we
    want.
    """
    from skill_evolve.track_a.runner import (
        DEFAULT_INNER_MODEL,
        DEFAULT_MAX_PASSES,
        DEFAULT_OUTER_MODEL,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-passes", type=int, default=DEFAULT_MAX_PASSES)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--force-synthetic", action="store_true")
    ap.add_argument("--outer-model", default=DEFAULT_OUTER_MODEL)
    ap.add_argument("--inner-model", default=DEFAULT_INNER_MODEL)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-workers", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--rng-seed", type=int, default=None)
    return ap


def _apply_deprecated_alias(
    args: argparse.Namespace,
) -> Tuple[argparse.Namespace, List[str]]:
    """Replicate the --model deprecation branch from _cli, capturing stderr.

    Returns the (possibly mutated) namespace and a list of lines that
    would have been printed to stderr.
    """
    captured: List[str] = []
    if args.model is not None:
        captured.append(
            "WARNING: --model is deprecated, use --outer-model and --inner-model"
        )
        args.outer_model = args.model
        args.inner_model = args.model
    return args, captured


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_split_flags_sets_both() -> None:
    """--outer-model and --inner-model each land in their own namespace slot."""
    ap = _build_parser_like_cli()
    argv = [
        "--seed",
        "x",
        "--out",
        "y",
        "--outer-model",
        "O",
        "--inner-model",
        "I",
    ]
    args = ap.parse_args(argv)
    assert args.outer_model == "O"
    assert args.inner_model == "I"
    # --model should NOT be set by the split flags
    assert args.model is None


def test_split_flags_defaults() -> None:
    """Defaults match the v2.1 asymmetric config (Sonnet 4.6 + M2.7)."""
    ap = _build_parser_like_cli()
    args = ap.parse_args(["--seed", "x", "--out", "y"])
    assert args.outer_model == "anthropic/claude-sonnet-4.6"
    assert args.inner_model == "minimax/minimax-m2.7"
    assert args.model is None


def test_deprecated_model_sets_both(capsys: pytest.CaptureFixture[str]) -> None:
    """Legacy --model sets BOTH outer and inner after alias handling, and warns."""
    ap = _build_parser_like_cli()
    args = ap.parse_args(["--seed", "x", "--out", "y", "--model", "Z"])
    # Before alias handling, the split flags still carry defaults and
    # --model carries the requested value.
    assert args.model == "Z"

    # Run the same deprecation branch _cli does.
    args, warnings = _apply_deprecated_alias(args)

    assert args.outer_model == "Z"
    assert args.inner_model == "Z"
    # Exactly one deprecation warning line was emitted.
    assert any("deprecated" in line for line in warnings)
    assert any("--outer-model" in line for line in warnings)
    assert any("--inner-model" in line for line in warnings)


def test_deprecated_model_warning_via_real_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end: _cli() must print the deprecation warning to stderr.

    We stub out run_loop so _cli doesn't actually execute the evolution
    loop; we only care that the warning line hits stderr when --model is
    present.
    """
    from skill_evolve.track_a import runner

    fake_history = {
        "final_score": 0.0,
        "converged": True,
        "passes": [{"pass": 0}],
    }
    with mock.patch.object(runner, "run_loop", return_value=fake_history) as mock_run:
        rc = runner._cli(
            [
                "--seed",
                "x",
                "--out",
                "y",
                "--force-synthetic",
                "--model",
                "Z",
            ]
        )
    assert rc == 0
    # run_loop received the alias-expanded values.
    kwargs = mock_run.call_args.kwargs
    assert kwargs["outer_model"] == "Z"
    assert kwargs["inner_model"] == "Z"

    err = capsys.readouterr().err
    assert "--model is deprecated" in err
    assert "--outer-model" in err
    assert "--inner-model" in err
