"""Phase E evolution runner — wraps ``skill_evolve.track_b.run`` with
SkillsBench-shaped defaults.

Per ``plans/plan_0.md`` Group E.5, this module is a preset wrapper that
forces ``--task-source skillsbench``, ``--agent-backend bench-cli``,
and ``--anonymize-tasks``, then dispatches to the standard Track B
runner. It also handles the auto-bootstrap: if ``--task-list`` is not
provided and the Phase D baseline summary exists, run
``scripts/select_hot_12.py`` to materialize ``hot_12.json`` first.

Cost guard: pre-flight estimate at the plan-default scale (12 hot
tasks x 20 generations x 3 islands x 2 repeats) is ~$72 at $0.05/trial.
``--max-budget-usd`` is plumbed through to the inner runner.

The wrapper exec-vs-call decision: we drive ``track_b.run.main()``
in-process via its ``main(argv)`` entry point so a Ctrl-C reaches both
processes cleanly. (The ``subprocess`` alternative has Python re-entry
overhead and complicates signal handling.)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Exit codes
_EXIT_OK = 0
_EXIT_ERROR = 1
_EXIT_NO_API_KEY = 99

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BASELINE_SUMMARY = _REPO_ROOT / "runs" / "skillsbench_baseline_v2" / "summary.json"
# Bug 17: hot_12.json lives next to the Phase D baseline summary, not under
# the (yet-to-exist) evolve_v1 dir. The old path forced auto-bootstrap on
# every cold start.
_DEFAULT_HOT_12 = _REPO_ROOT / "runs" / "skillsbench_baseline_v2" / "hot_12.json"
_DEFAULT_OUT = _REPO_ROOT / "runs" / "skillsbench_evolve_v1"
_DEFAULT_SEED = _REPO_ROOT / "seed_skills_empty"
_DEFAULT_SUBSET_FOR_HOT12 = (
    _REPO_ROOT / "skill_evolve" / "skillsbench" / "subset_17.json"
)
_SELECT_HOT_12_SCRIPT = _REPO_ROOT / "scripts" / "select_hot_12.py"

# Cost model: bench does not emit per-call cost; conservative fixed
# multiplier matching ``baseline.COST_PER_TRIAL_USD``.
_COST_PER_TRIAL_USD = 0.05


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m skill_evolve.skillsbench.evolve",
        description=(
            "Phase E evolution runner — drives skill_evolve.track_b.run "
            "with SkillsBench presets (task-source=skillsbench, "
            "agent-backend=bench-cli, anonymize-tasks=on)."
        ),
    )
    p.add_argument(
        "--task-list",
        type=Path,
        default=None,
        help=(
            f"Path to a JSON array of task IDs to evolve over. "
            f"Defaults to {_DEFAULT_HOT_12.relative_to(_REPO_ROOT)} when "
            f"that file exists; otherwise auto-runs select_hot_12.py "
            f"against the baseline summary if available."
        ),
    )
    p.add_argument(
        "--seed",
        type=Path,
        default=_DEFAULT_SEED,
        help="Seed skill folder (default: %(default)s)",
    )
    p.add_argument(
        "--num-generations",
        type=int,
        default=20,
        help="Outer-LLM generations per island (default: %(default)s)",
    )
    p.add_argument(
        "--num-islands",
        type=int,
        default=3,
        help="MAP-Elites islands (default: %(default)s)",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Inner-trial repeats per task (default: %(default)s)",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Parallel inner workers (default: %(default)s)",
    )
    p.add_argument(
        "--max-budget-usd",
        type=float,
        default=80.0,
        help="Hard ceiling on cumulative estimated spend (default: %(default)s)",
    )
    p.add_argument(
        "--max-wall-min",
        type=int,
        default=1440,
        help=(
            "Hard wall-clock cap in minutes (default: %(default)s = 24h). "
            "Installs a SIGALRM watchdog at evolve start; if the cap is "
            "reached the controller process logs 'WALLCLOCK CAP REACHED' "
            "and exits with code 2."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help="Output directory (default: %(default)s)",
    )
    p.add_argument(
        "--leak-policy",
        choices=("warn", "zero", "raise"),
        default="zero",
        help="Anti-leakage scanner policy (default: %(default)s)",
    )
    p.add_argument(
        "--outer-model",
        type=str,
        default=None,
        help="Override outer LLM model (default: track_b.run default)",
    )
    p.add_argument(
        "--inner-model",
        type=str,
        default="claude-haiku-4-5",
        help=(
            "Inner agent model slug. Default: %(default)s. Belt-and-"
            "suspenders for Bug 12: track_b.run also requires this "
            "explicitly under --task-source=skillsbench. The slug "
            "matches the SkillsBench leaderboard 'Haiku 4.5' entry — "
            "no @DATE suffix, no vertex_ai/ prefix."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved track_b.run argv + cost estimate without dispatching",
    )
    # Bug 15: explicit synthetic-mode flag so cold smoke runs don't trip
    # the new ANTHROPIC_API_KEY + OPENROUTER_API_KEY pre-flight (and so
    # the flag is plumbed through to track_b.run for SyntheticLLM use).
    p.add_argument(
        "--force-synthetic",
        action="store_true",
        help=(
            "Skip API-key checks and pass --force-synthetic through to "
            "track_b.run (offline smoke mode; no Anthropic/OpenRouter calls)."
        ),
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def _install_wallclock_watchdog(seconds: int) -> None:
    """Install a SIGALRM-based wall-clock cap.

    When ``seconds`` elapses, the SIGALRM handler prints a single
    ``WALLCLOCK CAP REACHED`` line to stderr and calls ``sys.exit(2)``
    so any registered atexit handlers (history flush, manifest write)
    still fire. Posix-only; no-ops on platforms without ``SIGALRM``
    (e.g. Windows) so unit tests on those hosts don't crash.
    """
    if seconds <= 0:
        return
    if not hasattr(signal, "SIGALRM"):
        logger.warning(
            "wallclock watchdog requested but SIGALRM unavailable on this "
            "platform; skipping"
        )
        return

    def _handler(_signum: int, _frame: object) -> None:
        print(
            f"WALLCLOCK CAP REACHED ({seconds}s); exiting with code 2",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(2)

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    logger.info("wallclock watchdog armed: %ds", seconds)


def _ensure_api_key(*, force_synthetic: bool = False) -> Optional[str]:
    """Return None when both required keys are set; else an error message.

    Two distinct keys gate live runs:
      * ``ANTHROPIC_API_KEY`` — the inner ``claude-code`` agent driven
        by ``bench eval create`` calls Anthropic directly.
      * ``OPENROUTER_API_KEY`` — the OUTER patch-generating LLM
        (``OpenRouterLLM``) calls OpenRouter for the configured slug
        (e.g. ``moonshotai/kimi-k2.6``). Bug 15: previously unchecked,
        which silently fell through to ``SyntheticLLM`` (random
        mutations). The runner would burn its full inner-trial budget
        for nothing.

    ``--force-synthetic`` skips both checks (synthetic mode bypasses
    every API call by design).
    """
    if force_synthetic:
        return None
    missing = []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY (inner claude-code agent)")
    if not os.environ.get("OPENROUTER_API_KEY"):
        missing.append("OPENROUTER_API_KEY (outer patch LLM)")
    if not missing:
        return None
    return (
        "Required API key(s) missing: "
        + ", ".join(missing)
        + ". Phase E evolution needs both — set them in the environment "
        "or pass --force-synthetic for an offline smoke run."
    )


def _resolve_task_list(args: argparse.Namespace) -> Optional[Path]:
    """Materialize ``hot_12.json`` if missing + auto-bootstrap is possible.

    Order of precedence:
      1. ``--task-list`` explicit value -> use it (must exist).
      2. ``_DEFAULT_HOT_12`` exists -> use it.
      3. ``_BASELINE_SUMMARY`` exists -> auto-run select_hot_12.py
         against ``subset_17.json`` to produce ``_DEFAULT_HOT_12``.
      4. Else -> None (caller errors out).
    """
    if args.task_list is not None:
        path = Path(args.task_list)
        if not path.exists():
            print(
                f"ERROR: --task-list {path} does not exist",
                file=sys.stderr,
            )
            return None
        return path
    if _DEFAULT_HOT_12.exists():
        return _DEFAULT_HOT_12
    if _BASELINE_SUMMARY.exists() and _DEFAULT_SUBSET_FOR_HOT12.exists():
        logger.info("auto-bootstrap: hot_12.json missing; running select_hot_12.py")
        _DEFAULT_HOT_12.parent.mkdir(parents=True, exist_ok=True)
        rc = subprocess.run(
            [
                sys.executable,
                str(_SELECT_HOT_12_SCRIPT),
                str(_BASELINE_SUMMARY),
                str(_DEFAULT_SUBSET_FOR_HOT12),
                "-o",
                str(_DEFAULT_HOT_12),
            ],
            check=False,
        ).returncode
        if rc != 0 or not _DEFAULT_HOT_12.exists():
            print(
                f"ERROR: select_hot_12.py failed (rc={rc})",
                file=sys.stderr,
            )
            return None
        return _DEFAULT_HOT_12
    print(
        "ERROR: --task-list not provided, and neither "
        f"{_DEFAULT_HOT_12.relative_to(_REPO_ROOT)} nor "
        f"{_BASELINE_SUMMARY.relative_to(_REPO_ROOT)} exist for "
        "auto-bootstrap.",
        file=sys.stderr,
    )
    return None


def _estimate_cost(args: argparse.Namespace, n_tasks: int) -> float:
    """Conservative pre-flight cost estimate.

    trials = n_tasks x num_generations x num_islands x repeats
    cost   = trials x COST_PER_TRIAL_USD
    """
    trials = n_tasks * args.num_generations * args.num_islands * args.repeats
    return trials * _COST_PER_TRIAL_USD


def _build_track_b_argv(
    args: argparse.Namespace,
    task_list: Path,
) -> List[str]:
    """Construct the equivalent ``python -m skill_evolve.track_b.run`` argv."""
    argv: List[str] = [
        "--task-source",
        "skillsbench",
        "--agent-backend",
        "bench-cli",
        "--anonymize-tasks",
        "--task-list",
        str(task_list),
        "--num-generations",
        str(args.num_generations),
        "--num-islands",
        str(args.num_islands),
        "--repeats",
        str(args.repeats),
        "--max-workers",
        str(args.max_workers),
        "--seed",
        str(args.seed),
        "--out",
        str(args.out),
        "--leak-policy",
        args.leak_policy,
    ]
    if args.outer_model:
        argv.extend(["--outer-model", args.outer_model])
    # Audit 5 Bug 5: always pass --inner-model to track_b (default is
    # claude-haiku-4-5 here, never None). Belt-and-suspenders for
    # Bug 12's track_b-side requirement.
    argv.extend(["--inner-model", args.inner_model])
    # Bug 15: propagate synthetic mode end-to-end.
    if getattr(args, "force_synthetic", False):
        argv.append("--force-synthetic")
    if args.verbose:
        argv.append("--verbose")
    return argv


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    err = _ensure_api_key(force_synthetic=args.force_synthetic)
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        return _EXIT_NO_API_KEY

    # Wall-clock guard: SIGALRM-based mid-run cap. Bench CLI does not
    # emit per-call cost so the budget guard is pre-flight only — the
    # wallclock cap is the only mid-run kill switch.
    _install_wallclock_watchdog(args.max_wall_min * 60)

    task_list = _resolve_task_list(args)
    if task_list is None:
        return _EXIT_ERROR

    try:
        n_tasks = len(json.loads(task_list.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"ERROR: failed to read task list {task_list}: {exc}",
            file=sys.stderr,
        )
        return _EXIT_ERROR

    estimated_cost = _estimate_cost(args, n_tasks)
    print(
        f"[COST ESTIMATE] {n_tasks} tasks x {args.num_generations} gens x "
        f"{args.num_islands} islands x {args.repeats} repeats = "
        f"{n_tasks * args.num_generations * args.num_islands * args.repeats} "
        f"trials x ${_COST_PER_TRIAL_USD:.2f} = ${estimated_cost:.2f}",
        file=sys.stderr,
    )
    if estimated_cost > args.max_budget_usd:
        print(
            f"ERROR: estimated ${estimated_cost:.2f} > "
            f"--max-budget-usd ${args.max_budget_usd:.2f}; aborting.",
            file=sys.stderr,
        )
        return _EXIT_ERROR

    track_b_argv = _build_track_b_argv(args, task_list)
    if args.dry_run:
        cmd_str = f"{sys.executable} -m skill_evolve.track_b.run " + " ".join(
            track_b_argv
        )
        print(f"[DRY RUN] would invoke:\n  {cmd_str}")
        return _EXIT_OK

    # Dispatch in-process so signals reach the Track B runner cleanly.
    from skill_evolve.track_b.run import main as track_b_main

    return track_b_main(track_b_argv)


if __name__ == "__main__":
    raise SystemExit(main())
