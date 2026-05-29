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

# kai-skills patch (Group B, 2026-05-27): canonical task-set resolution.
# Plan section 7i locks the resolution table:
#   hot_5      -> runs/skillsbench_baseline_v2/hot_5.json
#   subset_17  -> skill_evolve/skillsbench/subset_17.json
# Any other name is treated as a path passed through verbatim.
_CANONICAL_HOT_5 = _REPO_ROOT / "runs" / "skillsbench_baseline_v2" / "hot_5.json"
_CANONICAL_SUBSET_17 = _DEFAULT_SUBSET_FOR_HOT12  # same physical path

_TASK_SET_RESOLUTION: dict = {
    "hot_5": _CANONICAL_HOT_5,
    "subset_17": _CANONICAL_SUBSET_17,
}


def _resolve_task_set(name_or_path: Optional[str]) -> Optional[Path]:
    """Resolve a task-set name to a Path per plan section 7i.

    Known names route to the canonical table; everything else is
    treated as a filesystem path and returned verbatim. ``None`` →
    ``None`` so callers can keep their existing default logic.
    """
    if name_or_path is None:
        return None
    if name_or_path in _TASK_SET_RESOLUTION:
        return _TASK_SET_RESOLUTION[name_or_path]
    return Path(name_or_path)


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
        "--agent",
        type=str,
        default="claude-code",
        choices=("claude-code", "gemini"),
        help=(
            "Inner agent harness. Default: %(default)s (Anthropic CLI). "
            "Use 'gemini' for Google's Gemini CLI (requires GEMINI_API_KEY "
            "or GOOGLE_API_KEY in env, escapes the Anthropic 20MB/hr cap)."
        ),
    )
    # kai-skills patch (Group B, 2026-05-27): canonical preset +
    # task-set name resolution + eval-source forwarding. ``--canonical``
    # injects the consolidated v0 config from plan section 4. The four
    # eval-source flags (--eval-source / --eval-set / --judge-model /
    # --validation-task-list) are forwarded to track_b.run.
    p.add_argument(
        "--canonical",
        action="store_true",
        help=(
            "Inject the consolidated v0 config: inner=qwen3.6-27b, "
            "proposer=deepseek-v4-pro, eval-source=skillsbench, "
            "task-set=hot_5, validation=subset_17, patch=sentinel-blocks, "
            "smoke-test on, max-iters=12, budget=6h/$50."
        ),
    )
    p.add_argument(
        "--task-set",
        type=str,
        default=None,
        help=(
            "Canonical task-set name (hot_5, subset_17) or path to a "
            "task-id JSON file. Names route to the locked table; "
            "anything else is treated as a path. Overrides --task-list "
            "when both are provided."
        ),
    )
    p.add_argument(
        "--eval-source",
        choices=("skillsbench", "behavioral"),
        default=None,
        help=(
            "Forwarded to track_b.run. Default skillsbench (preserves "
            "the existing path). ``behavioral`` requires --eval-set."
        ),
    )
    p.add_argument(
        "--eval-set",
        type=Path,
        default=None,
        help="Forwarded to track_b.run (daycare-format eval_set.jsonl).",
    )
    p.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Forwarded to track_b.run (OpenRouter slug for behavioral judge).",
    )
    p.add_argument(
        "--validation-task-list",
        type=Path,
        default=None,
        help=(
            "Forwarded to track_b.run. Held-out task list scored on "
            "each accepted winner; recorded as validation_score."
        ),
    )
    p.add_argument(
        "--patch-format",
        choices=("json-ops", "sentinel-blocks"),
        default=None,
        help="Forwarded to track_b.run (default json-ops; --canonical → sentinel-blocks).",
    )
    p.add_argument(
        "--max-iters",
        type=int,
        default=None,
        help="Forwarded to track_b.run --num-generations (alias).",
    )
    p.add_argument(
        "--budget-hours",
        type=float,
        default=None,
        help="Wall-clock cap (hours); maps to --max-wall-min in track_b.run.",
    )
    p.add_argument(
        "--budget-usd",
        type=float,
        default=None,
        help="Aliases --max-budget-usd for clarity in the canonical preset.",
    )
    p.add_argument(
        "--smoke-test",
        dest="smoke_test",
        action="store_true",
        default=None,
        help="Forward --smoke-test to track_b.run (default in --canonical).",
    )
    p.add_argument(
        "--no-smoke-test",
        dest="smoke_test",
        action="store_false",
        help="Forward --no-smoke-test to track_b.run.",
    )
    # kai-skills patch (Group F, 2026-05-28): forward the bounded
    # edit-budget L_t schedule to track_b.run. Default is None here (not
    # ``cosine:8->2``) so ``_apply_canonical_defaults`` can detect a
    # user-supplied override. The track_b.run side defaults to
    # ``cosine:8->2`` directly when this wrapper omits the flag.
    p.add_argument(
        "--edit-budget",
        type=str,
        default=None,
        help=(
            "Forward --edit-budget L_t scheduler to track_b.run. Spec: "
            "``constant:N``, ``linear:N->M``, ``cosine:N->M``. "
            "``--canonical`` injects ``cosine:8->2``."
        ),
    )
    # kai-skills patch (Group E, 2026-05-28): forward the strict-gate +
    # rejected-buffer flags to track_b.run. Defaults are ``None`` so the
    # canonical preset can detect a user-supplied override; track_b.run
    # carries its own defaults when this wrapper omits the flag.
    p.add_argument(
        "--validation-gate",
        choices=("strict", "record", "relaxed"),
        default=None,
        help=(
            "Forward --validation-gate to track_b.run. ``--canonical`` "
            "injects ``strict``."
        ),
    )
    p.add_argument(
        "--rejected-buffer-size",
        type=int,
        default=None,
        help="Forward --rejected-buffer-size to track_b.run (default 10).",
    )
    p.add_argument(
        "--max-proposer-prompt-tokens",
        type=int,
        default=None,
        help=("Forward --max-proposer-prompt-tokens to track_b.run (default 90000)."),
    )
    # kai-skills patch (Group H, 2026-05-28): forward partition-reflection
    # flags. Defaults are ``None`` so the canonical preset can detect a
    # user-supplied override (per H.6 back-compat note: user-passed
    # ``--reflection-mode single`` must survive --canonical).
    p.add_argument(
        "--reflection-mode",
        choices=("single", "partition"),
        default=None,
        help=(
            "Forward --reflection-mode to track_b.run. ``--canonical`` "
            "injects ``partition`` (paper default)."
        ),
    )
    p.add_argument(
        "--reflection-batch-size",
        type=int,
        default=None,
        help=("Forward --reflection-batch-size to track_b.run (default 8)."),
    )
    p.add_argument(
        "--reflection-success-threshold",
        type=float,
        default=None,
        help=("Forward --reflection-success-threshold to track_b.run (default 0.5)."),
    )
    # kai-skills patch (Group G, 2026-05-28; plan §7l + §7n): forward
    # slow-update consolidator + meta-skill audit log flags. Defaults
    # are ``None`` so the canonical preset can detect a user-supplied
    # override; track_b.run carries its own defaults when this wrapper
    # omits the flag. G is the LAST second-pass group to append in
    # §7n's Round 2 order.
    p.add_argument(
        "--slow-update-every",
        type=int,
        default=None,
        help=(
            "Forward --slow-update-every K to track_b.run (default 4). "
            "Consolidator fires every K iters."
        ),
    )
    p.add_argument(
        "--meta-skill-path",
        type=Path,
        default=None,
        help="Forward --meta-skill-path to track_b.run.",
    )
    p.add_argument(
        "--consolidator-model",
        type=str,
        default=None,
        help=(
            "Forward --consolidator-model to track_b.run. Default: mirror "
            "--outer-model."
        ),
    )
    p.add_argument(
        "--meta-skill-max-iters",
        type=int,
        default=None,
        help="Forward --meta-skill-max-iters to track_b.run (default 20).",
    )
    p.add_argument(
        "--persistent-failure-window",
        type=int,
        default=None,
        help=("Forward --persistent-failure-window to track_b.run (default 3)."),
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
    # HARD-RULE pre-flight escape hatch. The live probe + flag-schema +
    # routability checks run BEFORE the multi-hour loop to catch the
    # dead-key / version-drift failure shapes that silently zeroed the
    # 2026-05-28 smoke. --skip-preflight bypasses them (also implied by
    # --force-synthetic / --dry-run, which never reach the live loop).
    p.add_argument(
        "--skip-preflight",
        action="store_true",
        help=(
            "Skip the live pre-flight checks (outer-key probe, bench flag "
            "schema, inner-model routability). Use only when you've already "
            "validated the environment this session."
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


def _ensure_api_key(
    *, force_synthetic: bool = False, agent: str = "claude-code"
) -> Optional[str]:
    """Return None when required keys are set; else an error message.

    Required keys depend on the agent harness:
      * ``claude-code``: needs ``ANTHROPIC_API_KEY`` (inner) +
        ``OPENROUTER_API_KEY`` (outer mutator).
      * ``gemini``: needs ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY`` (inner) +
        ``OPENROUTER_API_KEY`` (outer mutator). Escapes Anthropic 20MB/hr cap.

    ``--force-synthetic`` skips both checks (synthetic mode bypasses
    every API call by design).
    """
    if force_synthetic:
        return None
    missing = []
    if agent == "claude-code":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            missing.append("ANTHROPIC_API_KEY (inner claude-code agent)")
    elif agent == "gemini":
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            missing.append("GEMINI_API_KEY or GOOGLE_API_KEY (inner gemini agent)")
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


class PreflightAbort(RuntimeError):
    """Raised by the pre-flight when a hard failure is detected.

    Carries the loud-and-early diagnostic for the dead-key / version-drift
    failure shapes that silently zeroed the 2026-05-28 smoke. ``main()``
    catches this and exits before the multi-hour loop ever starts.
    """


def _probe_outer_proposer_key(model: str) -> None:
    """5-token live probe of the OUTER proposer (OpenRouter).

    HARD RULE: a dead/rate-limited outer key must surface in ~$0.01 BEFORE
    a multi-hour run, not as a wall of silent ``SyntheticLLM`` fallbacks
    masquerading as "evolution found nothing". Reuses the same OpenRouter
    base_url + OPENROUTER_API_KEY as ``OpenRouterLLM`` (no hand-rolled httpx).

    * 401/403 -> raise PreflightAbort ("key rejected").
    * 429     -> raise PreflightAbort ("rate-limited").
    * network/transport error -> warn-and-continue (don't kill a run for a
      transient DNS blip; the loop has its own retries).
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        # _ensure_api_key already reports this; nothing live to probe.
        return
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        logger.warning(
            "openai SDK not importable; skipping live outer-key probe "
            "(install openai to enable the HARD-RULE pre-flight)."
        )
        return

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    try:
        client.chat.completions.create(
            model=model,
            max_tokens=5,
            messages=[{"role": "user", "content": "ping"}],
        )
    except Exception as exc:  # noqa: BLE001 — classify by status, see below
        status = getattr(exc, "status_code", None)
        if status is None:
            status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            raise PreflightAbort(
                f"outer proposer key rejected ({status}) for model "
                f"{model!r} — aborting before the run burns budget against "
                "a dead key (HARD RULE)."
            ) from exc
        if status == 429:
            raise PreflightAbort(
                f"outer proposer rate-limited (429) for model {model!r} — "
                "aborting; back off or rotate the OpenRouter key before "
                "launching."
            ) from exc
        # Network/transport/unknown error: warn-and-continue.
        logger.warning(
            "outer-key live probe hit a non-auth error (model=%s): %s. "
            "Continuing — the evolution loop has its own retries — but the "
            "outer key is UNVERIFIED.",
            model,
            exc,
        )


def _assert_bench_flag_schema() -> None:
    """Assert ``bench eval create`` still accepts the long flags we emit.

    Catches a recurrence of the exact regression we just fixed: benchflow
    version drift flipping the accepted flag schema and silently zeroing
    every candidate at arg-parse. Invokes ``bench eval create --help`` via
    the SAME shim/argv path the backend uses, then asserts the four long
    flags are present. Best-effort: if bench isn't importable/installed we
    warn rather than hard-fail (the run may target a different backend).
    """
    try:
        from skill_evolve.agents.bench_cli import (
            _BENCH_SHIM_CODE,  # noqa: PLC0415 — local import keeps cold paths cheap
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "could not import bench_cli shim to verify flag schema: %s. "
            "Skipping flag-compatibility pre-flight.",
            exc,
        )
        return

    argv = [sys.executable, "-c", _BENCH_SHIM_CODE, "eval", "create", "--help"]
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning(
            "bench eval create --help did not run (%s); skipping "
            "flag-compatibility pre-flight.",
            exc,
        )
        return

    help_text = (proc.stdout or "") + (proc.stderr or "")
    required = ("--config", "--tasks-dir", "--agent", "--model")
    missing = [flag for flag in required if flag not in help_text]
    if missing:
        raise PreflightAbort(
            "bench CLI flag schema mismatch (benchflow version drift?) — "
            f"`bench eval create --help` is missing {missing}. The backend "
            "argv emits these long flags; a mismatch silently zeroes every "
            "candidate at arg-parse. Aborting."
        )


def _check_inner_model_routability(*, agent: str, inner_model: str) -> None:
    """Loud warning when the inner model may be unroutable via claude-code.

    The qwen-via-claude-code trap: ``claude-code`` dispatches to
    api.anthropic.com unless a provider-routing shim is wired (Layer 2).
    A non-Anthropic inner slug with no BENCHFLOW_PROVIDER_BASE_URL set will
    fail to route. Warning (not hard-fail) — Layer 2 owns the shim.
    """
    if agent != "claude-code":
        return
    slug = (inner_model or "").lower()
    is_anthropic_native = slug.startswith("claude") or slug.startswith("anthropic")
    has_routing = bool(os.environ.get("BENCHFLOW_PROVIDER_BASE_URL"))
    if not is_anthropic_native and not has_routing:
        logger.warning(
            "INNER-MODEL ROUTABILITY: agent=claude-code but --inner-model=%r "
            "is not an Anthropic-native slug and BENCHFLOW_PROVIDER_BASE_URL "
            "is unset. claude-code dispatches to api.anthropic.com by default, "
            "so this model is likely UNROUTABLE (the qwen-via-claude-code "
            "trap). Set BENCHFLOW_PROVIDER_BASE_URL or wait for the Layer-2 "
            "routing shim.",
            inner_model,
        )


def _run_preflight(args: argparse.Namespace) -> None:
    """HARD-RULE pre-flight: catch dead keys / version drift loud-and-early.

    Runs BEFORE the evolution loop. Skipped under --force-synthetic /
    --dry-run (never reach the live loop) and --skip-preflight. Raises
    PreflightAbort on a hard failure; degrades to warnings otherwise so a
    transient blip never kills a run for the wrong reason.
    """
    if args.force_synthetic or args.dry_run or args.skip_preflight:
        return
    # Resolve the effective outer slug the same way track_b.run does: when
    # --outer-model is omitted the runner falls back to DEFAULT_OUTER_MODEL.
    outer_model = args.outer_model
    if not outer_model:
        from skill_evolve.track_b.run import DEFAULT_OUTER_MODEL

        outer_model = DEFAULT_OUTER_MODEL
    _probe_outer_proposer_key(outer_model)
    _assert_bench_flag_schema()
    _check_inner_model_routability(agent=args.agent, inner_model=args.inner_model)


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


def _apply_canonical_defaults(args: argparse.Namespace) -> None:
    """Inject the consolidated v0 config when ``--canonical`` is set.

    Plan section 4 (Group B.8) canonical args (only fill what the user
    didn't override on the CLI — argparse defaults are detectable via
    ``None`` sentinels on the new options).
    """
    if not getattr(args, "canonical", False):
        return
    if args.inner_model in (None, "claude-haiku-4-5"):
        # Override unless the user picked something else explicitly. The
        # parser's literal default is "claude-haiku-4-5"; we can't tell
        # whether the user set it intentionally, so we only swap when
        # the canonical preset wants a different value.
        args.inner_model = "qwen/qwen3.6-27b"
    if args.outer_model is None:
        args.outer_model = "deepseek/deepseek-v4-pro"
    if args.eval_source is None:
        args.eval_source = "skillsbench"
    if args.task_set is None:
        args.task_set = "hot_5"
    if args.validation_task_list is None:
        args.validation_task_list = _CANONICAL_SUBSET_17
    if args.patch_format is None:
        args.patch_format = "sentinel-blocks"
    if args.smoke_test is None:
        args.smoke_test = True
    if args.max_iters is None:
        args.max_iters = 12
    if args.budget_hours is None:
        args.budget_hours = 6.0
    if args.budget_usd is None:
        args.budget_usd = 50.0
    # kai-skills patch (Group F, 2026-05-28): canonical preset injects
    # the paper-default cosine taper from 8 down to 2 ops per iter.
    # F is FIRST in the §7n append order; E/G/H append their canonical
    # defaults AFTER this block.
    if args.edit_budget is None:
        args.edit_budget = "cosine:8->2"
    # kai-skills patch (Group E, 2026-05-28): canonical preset enables
    # the strict validation gate + 10-entry rejected ring + 90k-token
    # proposer prompt cap (per plan §4b context budget). E appends
    # AFTER F per §7n's Round 1 append order.
    if args.validation_gate is None:
        args.validation_gate = "strict"
    if args.rejected_buffer_size is None:
        args.rejected_buffer_size = 10
    if args.max_proposer_prompt_tokens is None:
        args.max_proposer_prompt_tokens = 90000
    # kai-skills patch (Group H, 2026-05-28): canonical preset enables
    # success/failure minibatch partition reflection per plan section 7m.
    # H is SECOND in §7n Round 2 append order — these lines go AFTER E's
    # block and BEFORE G's (G rebases later). Per H.6: respect explicit
    # user override (only fill when the user hasn't pinned it).
    if args.reflection_mode is None:
        args.reflection_mode = "partition"
    if args.reflection_batch_size is None:
        args.reflection_batch_size = 8
    if args.reflection_success_threshold is None:
        args.reflection_success_threshold = 0.5
    # kai-skills patch (Group G, 2026-05-28; plan §7l + §7n): canonical
    # preset enables the slow-update consolidator + meta-skill audit log
    # at paper defaults — fire every 4 iters, bound the log to 20 tail
    # entries, count a task as "persistent failure" after 3 consecutive
    # failed iters. G appends LAST in §7n's Round 2 append order (H
    # first; G rebases after).
    if getattr(args, "slow_update_every", None) is None:
        args.slow_update_every = 4
    if getattr(args, "meta_skill_max_iters", None) is None:
        args.meta_skill_max_iters = 20
    if getattr(args, "persistent_failure_window", None) is None:
        args.persistent_failure_window = 3


def _build_track_b_argv(
    args: argparse.Namespace,
    task_list: Path,
) -> List[str]:
    """Construct the equivalent ``python -m skill_evolve.track_b.run`` argv."""
    # Honor --max-iters as an alias for --num-generations under
    # --canonical (or whenever the user passes it explicitly).
    num_generations = (
        args.max_iters if args.max_iters is not None else args.num_generations
    )
    argv: List[str] = [
        "--task-source",
        "skillsbench",
        "--agent-backend",
        "bench-cli",
        "--anonymize-tasks",
        "--task-list",
        str(task_list),
        "--num-generations",
        str(num_generations),
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
    argv.extend(["--agent", args.agent])
    # Bug 15: propagate synthetic mode end-to-end.
    if getattr(args, "force_synthetic", False):
        argv.append("--force-synthetic")
    # kai-skills patch (Group B, 2026-05-27): forward eval-source flags.
    if args.eval_source:
        argv.extend(["--eval-source", args.eval_source])
    if args.eval_set:
        argv.extend(["--eval-set", str(args.eval_set)])
    if args.judge_model:
        argv.extend(["--judge-model", args.judge_model])
    if args.validation_task_list is not None:
        argv.extend(["--validation-task-list", str(args.validation_task_list)])
    # kai-skills patch (Group F, 2026-05-28): forward --edit-budget to
    # track_b.run when the user (or --canonical) supplied a value. Omitting
    # the flag lets track_b.run apply its own default (``cosine:8->2``).
    if args.edit_budget is not None:
        argv.extend(["--edit-budget", args.edit_budget])
    # kai-skills patch (Group E, 2026-05-28): forward strict-gate +
    # rejected-buffer flags. ``None`` means "let track_b.run pick its own
    # default" (preserves back-compat for non-canonical invocations).
    if args.validation_gate is not None:
        argv.extend(["--validation-gate", args.validation_gate])
    if args.rejected_buffer_size is not None:
        argv.extend(["--rejected-buffer-size", str(args.rejected_buffer_size)])
    if args.max_proposer_prompt_tokens is not None:
        argv.extend(
            [
                "--max-proposer-prompt-tokens",
                str(args.max_proposer_prompt_tokens),
            ]
        )
    # kai-skills patch (Group H, 2026-05-28): forward partition-reflection
    # flags. ``None`` means "let track_b.run pick its own default"
    # (preserves back-compat for non-canonical invocations).
    if args.reflection_mode is not None:
        argv.extend(["--reflection-mode", args.reflection_mode])
    if args.reflection_batch_size is not None:
        argv.extend(["--reflection-batch-size", str(args.reflection_batch_size)])
    if args.reflection_success_threshold is not None:
        argv.extend(
            [
                "--reflection-success-threshold",
                str(args.reflection_success_threshold),
            ]
        )
    # kai-skills patch (Group G, 2026-05-28; plan §7l + §7n): forward
    # slow-update consolidator + meta-skill flags. ``None`` lets
    # track_b.run apply its own defaults.
    if getattr(args, "slow_update_every", None) is not None:
        argv.extend(["--slow-update-every", str(args.slow_update_every)])
    if getattr(args, "meta_skill_path", None) is not None:
        argv.extend(["--meta-skill-path", str(args.meta_skill_path)])
    if getattr(args, "consolidator_model", None) is not None:
        argv.extend(["--consolidator-model", args.consolidator_model])
    if getattr(args, "meta_skill_max_iters", None) is not None:
        argv.extend(["--meta-skill-max-iters", str(args.meta_skill_max_iters)])
    if getattr(args, "persistent_failure_window", None) is not None:
        argv.extend(
            [
                "--persistent-failure-window",
                str(args.persistent_failure_window),
            ]
        )
    if args.verbose:
        argv.append("--verbose")
    return argv


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # kai-skills patch (Group B, 2026-05-27): inject the canonical preset
    # BEFORE any other resolution so --task-set / --validation-task-list /
    # budget caps pick up the defaults.
    _apply_canonical_defaults(args)
    # Resolve --task-set name → path; overrides --task-list when set.
    if args.task_set is not None:
        resolved = _resolve_task_set(args.task_set)
        if resolved is not None:
            args.task_list = resolved
    # Resolve --validation-task-list canonical names too (so users can
    # pass --validation-task-list subset_17 directly).
    if args.validation_task_list is not None:
        s_name = str(args.validation_task_list)
        if s_name in _TASK_SET_RESOLUTION:
            args.validation_task_list = _TASK_SET_RESOLUTION[s_name]
    # Map --budget-hours onto --max-wall-min when explicitly set.
    if args.budget_hours is not None:
        args.max_wall_min = int(round(args.budget_hours * 60))
    if args.budget_usd is not None:
        args.max_budget_usd = float(args.budget_usd)

    err = _ensure_api_key(force_synthetic=args.force_synthetic, agent=args.agent)
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        return _EXIT_NO_API_KEY

    # HARD-RULE pre-flight: live outer-key probe + bench flag schema +
    # inner-model routability, run BEFORE the multi-hour loop. Catches the
    # dead-key / version-drift failure shapes (which silently zeroed the
    # 2026-05-28 smoke) in ~$0.01 instead of mid-run. Auto-skipped under
    # --force-synthetic / --dry-run / --skip-preflight.
    try:
        _run_preflight(args)
    except PreflightAbort as exc:
        print(f"ERROR: pre-flight aborted: {exc}", file=sys.stderr)
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
