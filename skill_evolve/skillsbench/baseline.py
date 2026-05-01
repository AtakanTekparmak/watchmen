"""Phase D baseline runner for SkillsBench.

Runs the 17-task viable subset (or any other task list) across one or
both of ``with-skills`` / ``no-skills`` conditions through
:class:`skill_evolve.agents.bench_cli.BenchCliBackend`, checkpointing
every trial to ``results.jsonl`` and emitting a ``summary.json`` /
``summary.md`` shaped to mirror :class:`skill_evolve.evaluator.EvalResult`.

See ``plans/plan_0.md`` Group D / §7 for the launch-shape contract:
``5 trials × 20 tasks × 2 conditions = 200 trials``.

Anonymization is intentionally **off** at baseline (``anonymize_map=None``)
— per the plan, anonymization is an evolution-loop concern. The
``BenchCliBackend.run_task`` API still accepts the kwarg as a no-op for
forward-compat / paranoia.

Cost reporting: bench does not currently emit per-call USD spend, so
this runner uses the conservative ``$0.05/trial`` Haiku constant per
the plan. The summary explicitly flags this as an estimate; verify
true spend via the Anthropic billing dashboard.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import logging
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from skill_evolve.agents.base import TrajectoryResult
from skill_evolve.agents.bench_cli import BenchCliBackend
from skill_evolve.benchmark.load import Task
from skill_evolve.benchmark.skillsbench_loader import hydrate_one

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Per the plan: SkillsBench Haiku 4.5 leaderboard reference numbers used
# for the ±5pp delta sanity log. NEVER used as a hard gate (D-9 dropped).
LEADERBOARD_HAIKU45_WITH_SKILLS = 27.7
LEADERBOARD_HAIKU45_NO_SKILLS = 11.0

# Cost-per-trial assumption (Haiku 4.5 estimate). bench does not emit
# real cost so this is a fixed multiplier; surfaced explicitly in the
# summary alongside a verification disclaimer.
COST_PER_TRIAL_USD = 0.05

# Allowed condition names.
_ALLOWED_CONDITIONS = ("with-skills", "no-skills")

# Allowed agent backends. Phase D ships only ``bench-cli``; Phase E may
# add ``hermes`` (rejected here with a clear error).
_ALLOWED_BACKENDS = ("bench-cli",)

_DEFAULT_TASK_LIST = Path(__file__).resolve().parent / "subset_17.json"
_DEFAULT_VENDOR_DIR = (
    Path(__file__).resolve().parent.parent
    / "benchmark"
    / "vendor"
    / "skillsbench"
    / "tasks"
)


# Exit codes
_EXIT_OK = 0
_EXIT_ERROR = 1
_EXIT_KEYBOARD_INTERRUPT = 130
_EXIT_BUDGET = 2
_EXIT_NO_API_KEY = 99


# ---------------------------------------------------------------------------
# Argparse / config
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="skill-evolve-skillsbench-baseline",
        description=(
            "Phase D baseline runner: trials × tasks × conditions through "
            "BenchCliBackend, with rolling JSONL checkpointing and "
            "per-condition aggregation."
        ),
    )
    p.add_argument(
        "--task-list",
        type=Path,
        default=_DEFAULT_TASK_LIST,
        help=(
            "JSON array of task IDs (default: %(default)s — the 17 "
            "viable Phase F tasks; the 3 env-broken ones in subset_20 "
            "hang the bench-CLI run)"
        ),
    )
    p.add_argument(
        "--vendor-dir",
        type=Path,
        default=_DEFAULT_VENDOR_DIR,
        help="Vendored SkillsBench tasks directory (default: %(default)s)",
    )
    p.add_argument(
        "--model",
        type=str,
        default="claude-haiku-4-5",
        help="Inner model identifier (default: %(default)s)",
    )
    p.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Trials per (task, condition) pair (default: %(default)s)",
    )
    p.add_argument(
        "--conditions",
        type=str,
        default="with-skills,no-skills",
        help=(
            "Comma-separated conditions: one or both of "
            "'with-skills', 'no-skills' (default: %(default)s)"
        ),
    )
    p.add_argument(
        "--agent-backend",
        type=str,
        default="bench-cli",
        help="Agent backend; only 'bench-cli' supported in Phase D (default: %(default)s)",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("RATE_PROBE_K") or 1),
        help=(
            "Parallel task workers (default: env RATE_PROBE_K or 1). "
            "Anthropic org rate limit (20M prompt bytes/hour) saturates "
            "at concurrency >1 for SkillsBench-sized tasks; Phase D ran "
            "at 4 and lost 67%% of trials to 429s. Raise only after "
            "upgrading API tier."
        ),
    )
    p.add_argument(
        "--max-budget-usd",
        type=float,
        default=15.0,
        help="Hard cap on cumulative estimated spend (default: %(default)s)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output dir (default: runs/skillsbench_baseline_<ts>/)",
    )
    p.add_argument(
        "--task-timeout-s",
        type=int,
        default=1800,
        help="Per-trial wall-clock timeout in seconds (default: %(default)s)",
    )
    p.add_argument(
        "--skills-dir",
        type=Path,
        default=None,
        help=(
            "Override skills_dir for the with-skills condition (e.g. point "
            "at an evolved bundle for Phase F.1). When unset, baseline "
            "uses each task's bundled environment/skills/."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan + cost estimate without executing",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip (task, condition, trial) tuples already present in results.jsonl",
    )
    return p


@dataclasses.dataclass
class _RunConfig:
    """Resolved runtime configuration."""

    task_list_path: Path
    vendor_dir: Path
    model: str
    trials: int
    conditions: List[str]
    agent_backend: str
    concurrency: int
    max_budget_usd: float
    out_dir: Path
    task_timeout_s: int
    skills_dir_override: Optional[Path]
    dry_run: bool
    resume: bool
    argv: List[str]
    started_at: str

    def to_manifest(self, task_ids: List[str]) -> Dict[str, Any]:
        return {
            "argv": self.argv,
            "model": self.model,
            "trials": self.trials,
            "conditions": list(self.conditions),
            "agent_backend": self.agent_backend,
            "concurrency": self.concurrency,
            "max_budget_usd": self.max_budget_usd,
            "task_timeout_s": self.task_timeout_s,
            "skills_dir_override": (
                str(self.skills_dir_override) if self.skills_dir_override else None
            ),
            "task_list_path": str(self.task_list_path),
            "vendor_dir": str(self.vendor_dir),
            "out_dir": str(self.out_dir),
            "started_at": self.started_at,
            "task_ids": list(task_ids),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task_dir_for_id(task_id: str, vendor_dir: Path) -> Path:
    """Resolve a task ID like ``skillsbench/foo`` to ``<vendor>/foo``."""
    if task_id.startswith("skillsbench/"):
        rel = task_id[len("skillsbench/") :]
    else:
        rel = task_id
    return vendor_dir / rel


def _load_task_list(path: Path) -> List[str]:
    """Load and validate the task-list JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"--task-list does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--task-list is not valid JSON: {path} ({exc})") from exc
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError(f"--task-list must be a JSON array of strings: {path}")
    return data


def _validate_and_hydrate(task_ids: List[str], vendor_dir: Path) -> List[Task]:
    """Resolve each task ID to a hydrated :class:`Task` and validate dirs."""
    tasks: List[Task] = []
    missing: List[str] = []
    for tid in task_ids:
        td = _task_dir_for_id(tid, vendor_dir)
        if not td.is_dir() or not (td / "task.toml").is_file():
            missing.append(f"{tid} -> {td}")
            continue
        task = hydrate_one(td)
        # Force the task_id to match the manifest entry (may include
        # the ``skillsbench/`` prefix even when the dir doesn't).
        task.task_id = tid if tid.startswith("skillsbench/") else f"skillsbench/{tid}"
        tasks.append(task)
    if missing:
        raise FileNotFoundError(
            "Task directories missing or lack task.toml:\n  " + "\n  ".join(missing)
        )
    return tasks


def _validate_conditions(raw: str) -> List[str]:
    """Parse and validate the ``--conditions`` flag."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("--conditions must be non-empty")
    bad = [p for p in parts if p not in _ALLOWED_CONDITIONS]
    if bad:
        raise ValueError(
            f"--conditions contains unknown values: {bad}. "
            f"Allowed: {list(_ALLOWED_CONDITIONS)}"
        )
    return parts


def _resolve_out_dir(out: Optional[Path]) -> Path:
    if out is not None:
        return out
    ts = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    return Path("runs") / f"skillsbench_baseline_{ts}"


def _atomic_append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    """Append a JSON row + flush + fsync for crash-safe checkpointing."""
    line = json.dumps(row, default=str) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            # Some filesystems (e.g. tmpfs in CI) don't support fsync;
            # the flush above is enough for our crash-recovery needs.
            pass


def _read_existing_results(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("malformed jsonl row in %s; skipping", path)
    return rows


def _row_key(row: Dict[str, Any]) -> Tuple[str, str, int]:
    return (
        str(row.get("task_id")),
        str(row.get("condition")),
        int(row.get("trial", -1)),
    )


def _is_complete(row: Dict[str, Any]) -> bool:
    """A row counts as 'complete' if it has a trajectory_result with success/error info."""
    tr = row.get("trajectory_result")
    if not isinstance(tr, dict):
        return False
    # Any row with a verifier_status that isn't the "not_run" placeholder
    # counts as completed (success/failure/timeout/agent_error etc.).
    return bool(tr) and "verifier_status" in tr


# ---------------------------------------------------------------------------
# Pre-flight cost estimate
# ---------------------------------------------------------------------------


def _estimate_cost(n_trials: int) -> float:
    return n_trials * COST_PER_TRIAL_USD


def _print_plan(
    cfg: _RunConfig,
    tasks: List[Task],
    n_trials: int,
    est_cost: float,
) -> None:
    n_tasks = len(tasks)
    n_conditions = len(cfg.conditions)
    est_wall_min = (n_trials * 5.0 / max(1, cfg.concurrency)) / 60.0
    print("=== SkillsBench baseline plan ===")
    print(f"  task_list:   {cfg.task_list_path}")
    print(f"  vendor_dir:  {cfg.vendor_dir}")
    print(f"  out_dir:     {cfg.out_dir}")
    print(f"  model:       {cfg.model}")
    print(f"  conditions:  {cfg.conditions}")
    print(f"  trials:      {cfg.trials}")
    print(f"  concurrency: {cfg.concurrency}")
    print(f"  task_timeout_s: {cfg.task_timeout_s}")
    print(f"  n_tasks:     {n_tasks}")
    print(f"  n_trials:    {n_tasks} × {n_conditions} × {cfg.trials} = {n_trials}")
    print(f"  est_cost:    ${est_cost:.2f} @ ${COST_PER_TRIAL_USD:.2f}/trial")
    print(f"  est_wall:    ~{est_wall_min:.1f} min @ concurrency {cfg.concurrency}")
    print(f"  budget_cap:  ${cfg.max_budget_usd:.2f}")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _per_task_stats(
    rows: List[Dict[str, Any]],
    task_ids: List[str],
    conditions: List[str],
) -> List[Dict[str, Any]]:
    """Build per-task aggregate rows for the summary."""
    out: List[Dict[str, Any]] = []
    for tid in task_ids:
        per_cond: Dict[str, Dict[str, Any]] = {}
        for cond in conditions:
            scores: List[float] = []
            successes = 0
            n = 0
            tool_calls: List[int] = []
            for r in rows:
                if r.get("task_id") != tid or r.get("condition") != cond:
                    continue
                tr = r.get("trajectory_result") or {}
                n += 1
                score = tr.get("score")
                if score is None:
                    score = 1.0 if tr.get("success") else 0.0
                try:
                    score = float(score)
                except (TypeError, ValueError):
                    score = 0.0
                scores.append(score)
                if score >= 1.0:
                    successes += 1
                try:
                    tool_calls.append(int(tr.get("tool_calls") or 0))
                except (TypeError, ValueError):
                    pass
            per_cond[cond] = {
                "n": n,
                "pass_rate": (successes / n) if n else 0.0,
                "score_mean": (sum(scores) / n) if n else 0.0,
                "score_std": (statistics.pstdev(scores) if len(scores) > 1 else 0.0),
                "tool_calls_mean": (
                    sum(tool_calls) / len(tool_calls) if tool_calls else 0.0
                ),
            }
        out.append({"task_id": tid, "by_condition": per_cond})
    return out


def _aggregate(
    rows: List[Dict[str, Any]],
    task_ids: List[str],
    conditions: List[str],
) -> Dict[str, Any]:
    """Collapse per-trial rows into the summary dict."""
    per_task = _per_task_stats(rows, task_ids, conditions)

    # Per-condition aggregates (mean over tasks of per-task pass rate /
    # score; matches the SkillsBench leaderboard's macro-averaging.)
    per_condition: Dict[str, Dict[str, Any]] = {}
    for cond in conditions:
        pass_rates = [pt["by_condition"][cond]["pass_rate"] for pt in per_task]
        score_means = [pt["by_condition"][cond]["score_mean"] for pt in per_task]
        ns = [pt["by_condition"][cond]["n"] for pt in per_task]
        per_condition[cond] = {
            "pass_rate": (sum(pass_rates) / len(pass_rates)) if pass_rates else 0.0,
            "score_mean": (sum(score_means) / len(score_means)) if score_means else 0.0,
            "n_tasks": len(pass_rates),
            "n_trials_completed": sum(ns),
        }

    # Headline: with-skills − no-skills. When only one condition was
    # requested, the lift fields stay None.
    lift_pp: Optional[float] = None
    lift_continuous: Optional[float] = None
    if "with-skills" in conditions and "no-skills" in conditions:
        lift_pp = (
            per_condition["with-skills"]["pass_rate"]
            - per_condition["no-skills"]["pass_rate"]
        ) * 100.0
        lift_continuous = (
            per_condition["with-skills"]["score_mean"]
            - per_condition["no-skills"]["score_mean"]
        )

    # EvalResult-shaped composite: success_rate is "with-skills"
    # if available, else the first listed condition.
    headline_cond = "with-skills" if "with-skills" in conditions else conditions[0]
    headline = per_condition[headline_cond]
    success_rate = headline["pass_rate"]
    mean_score = headline["score_mean"]
    # tool_calls_per_success across the headline condition's rows
    headline_tool_calls = []
    headline_successes = 0
    for r in rows:
        if r.get("condition") != headline_cond:
            continue
        tr = r.get("trajectory_result") or {}
        if tr.get("success"):
            headline_successes += 1
            try:
                headline_tool_calls.append(int(tr.get("tool_calls") or 0))
            except (TypeError, ValueError):
                pass
    avg_tool_calls_per_success = (
        sum(headline_tool_calls) / headline_successes if headline_successes else 0.0
    )
    # composite mirrors evaluator.compute_composite shape
    overhead = max(0.0, min(1.0, avg_tool_calls_per_success / 10.0))
    composite = mean_score - 0.05 * overhead

    # failures: per-task rows where headline condition had pass_rate < 1.0
    failures: List[Dict[str, str]] = []
    for pt in per_task:
        h = pt["by_condition"].get(headline_cond, {})
        if h.get("n") and h.get("pass_rate", 0.0) < 1.0:
            failures.append(
                {
                    "task_id": pt["task_id"],
                    "pass_rate": f"{h['pass_rate']:.3f}",
                    "score_mean": f"{h.get('score_mean', 0.0):.3f}",
                    "n": str(h.get("n", 0)),
                }
            )

    # SkillsBench-specific block: lift + leaderboard deltas.
    skillsbench_specific: Dict[str, Any] = {
        "with_skills_pass_rate": (
            per_condition["with-skills"]["pass_rate"]
            if "with-skills" in per_condition
            else None
        ),
        "no_skills_pass_rate": (
            per_condition["no-skills"]["pass_rate"]
            if "no-skills" in per_condition
            else None
        ),
        "lift_pp": lift_pp,
        "lift_continuous": lift_continuous,
        "leaderboard_haiku45_with_skills": LEADERBOARD_HAIKU45_WITH_SKILLS,
        "leaderboard_haiku45_no_skills": LEADERBOARD_HAIKU45_NO_SKILLS,
        "delta_from_leaderboard_with_skills": (
            (per_condition["with-skills"]["pass_rate"] * 100.0)
            - LEADERBOARD_HAIKU45_WITH_SKILLS
            if "with-skills" in per_condition
            else None
        ),
        "delta_from_leaderboard_no_skills": (
            (per_condition["no-skills"]["pass_rate"] * 100.0)
            - LEADERBOARD_HAIKU45_NO_SKILLS
            if "no-skills" in per_condition
            else None
        ),
    }

    n_completed = sum(1 for r in rows if _is_complete(r))
    cost_estimate = n_completed * COST_PER_TRIAL_USD

    return {
        # ---- EvalResult-shaped headline keys (R-13) ----
        "success_rate": success_rate,
        "tool_calls_per_success": avg_tool_calls_per_success,
        "composite": composite,
        "n_tasks": len(task_ids),
        "mean_score": mean_score,
        "scored_task_count": sum(
            1
            for pt in per_task
            if pt["by_condition"].get(headline_cond, {}).get("n", 0) > 0
        ),
        "per_task": per_task,
        "failures": failures,
        # ---- runner-level aux ----
        "per_condition": per_condition,
        "n_trials_completed": n_completed,
        "cost_estimate_usd": cost_estimate,
        "cost_disclaimer": (
            f"estimate is fixed at ${COST_PER_TRIAL_USD:.2f}/trial; real cost may "
            "differ — verify via Anthropic billing dashboard."
        ),
        "skillsbench_specific": skillsbench_specific,
    }


def _format_summary_md(summary: Dict[str, Any], conditions: List[str]) -> str:
    """Narrow Slack-readable markdown."""
    lines: List[str] = []
    lines.append("# SkillsBench baseline summary")
    lines.append("")
    pc = summary["per_condition"]
    lines.append("| condition | pass_rate | score_mean | n_tasks | n_trials |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for cond in conditions:
        c = pc.get(cond, {})
        lines.append(
            f"| {cond} | {c.get('pass_rate', 0.0):.3f} | "
            f"{c.get('score_mean', 0.0):.3f} | {c.get('n_tasks', 0)} | "
            f"{c.get('n_trials_completed', 0)} |"
        )
    lines.append("")
    sbs = summary["skillsbench_specific"]
    if sbs.get("lift_pp") is not None:
        lines.append(f"**Lift (pp):** {sbs['lift_pp']:+.2f}")
        lines.append(f"**Lift (continuous):** {sbs['lift_continuous']:+.4f}")
        lines.append("")
    if sbs.get("delta_from_leaderboard_with_skills") is not None:
        lines.append(
            f"**Δ vs Haiku 4.5 with-skills leaderboard "
            f"({LEADERBOARD_HAIKU45_WITH_SKILLS:.1f}%):** "
            f"{sbs['delta_from_leaderboard_with_skills']:+.2f} pp"
        )
    if sbs.get("delta_from_leaderboard_no_skills") is not None:
        lines.append(
            f"**Δ vs Haiku 4.5 no-skills leaderboard "
            f"({LEADERBOARD_HAIKU45_NO_SKILLS:.1f}%):** "
            f"{sbs['delta_from_leaderboard_no_skills']:+.2f} pp"
        )
    lines.append("")
    lines.append(
        f"**Cost estimate:** ${summary['cost_estimate_usd']:.2f} "
        f"({summary['n_trials_completed']} trials × ${COST_PER_TRIAL_USD:.2f})"
    )
    lines.append(f"_{summary['cost_disclaimer']}_")
    lines.append("")
    lines.append("## Per-task")
    lines.append("")
    header = "| task_id |"
    sep = "| --- |"
    for cond in conditions:
        header += f" {cond} (pass) | {cond} (score) | n |"
        sep += " ---: | ---: | ---: |"
    lines.append(header)
    lines.append(sep)
    for pt in summary["per_task"]:
        row = f"| {pt['task_id']} |"
        for cond in conditions:
            c = pt["by_condition"].get(cond, {})
            row += (
                f" {c.get('pass_rate', 0.0):.2f} | "
                f"{c.get('score_mean', 0.0):.3f} | "
                f"{c.get('n', 0)} |"
            )
        lines.append(row)
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Trial submission / execution
# ---------------------------------------------------------------------------


def _skills_dir_for(
    task: Task, condition: str, override: Optional[Path]
) -> Optional[Path]:
    """Resolve the skills_dir to mount for a (task, condition) pair."""
    if condition == "no-skills":
        return None
    if override is not None:
        return override
    payload = task.success_check_payload or {}
    sd = payload.get("skills_dir")
    if not sd:
        # Fall back to <task_dir>/environment/skills if populated;
        # the loader sets this to None if the dir isn't present.
        td = payload.get("task_dir")
        if td:
            cand = Path(td) / "environment" / "skills"
            if cand.is_dir():
                return cand
        return None
    return Path(sd)


def _run_one_trial(
    backend: BenchCliBackend,
    task: Task,
    condition: str,
    trial: int,
    cfg: _RunConfig,
    *,
    runner_workdir: Path,
) -> Tuple[str, str, int, TrajectoryResult, float]:
    """Worker fn: dispatch a single trial through the backend.

    Returns (task_id, condition, trial, TrajectoryResult, wall_s).
    """
    skills_dir = _skills_dir_for(task, condition, cfg.skills_dir_override)
    safe_id = task.task_id.replace("/", "_")
    workdir = runner_workdir / "jobs" / f"{safe_id}__{condition}__t{trial}"
    workdir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    try:
        tr = backend.run_task(
            task,
            skills_dir=skills_dir,
            model=cfg.model,
            timeout_s=cfg.task_timeout_s,
            budget_usd=cfg.max_budget_usd,
            anonymize_map=None,  # baseline: no anonymization
            workdir=workdir,
        )
    except TypeError:
        # Forward-compat: backends may not yet accept ``workdir`` kwarg.
        tr = backend.run_task(
            task,
            skills_dir=skills_dir,
            model=cfg.model,
            timeout_s=cfg.task_timeout_s,
            budget_usd=cfg.max_budget_usd,
            anonymize_map=None,
        )
    wall_s = time.monotonic() - t0
    return (task.task_id, condition, trial, tr, wall_s)


_RATE_LIMIT_COOLDOWN_THRESHOLD = 5
_RATE_LIMIT_COOLDOWN_SLEEP_S = 60


def _is_rate_limited_trajectory(tr: TrajectoryResult) -> bool:
    """True if a trial returned the ``rate_limited_max_retries`` shape."""
    return "rate_limited_max_retries" in (tr.notes or "")


def _execute(
    cfg: _RunConfig,
    tasks: List[Task],
    backend: BenchCliBackend,
    *,
    work_items: List[Tuple[Task, str, int]],
    results_jsonl: Path,
) -> Tuple[int, bool]:
    """Submit work items via a thread pool, checkpointing each result.

    Returns ``(n_completed, budget_capped)``.
    """
    completed = 0
    budget_capped = False
    soft_warned = False
    keyboard_interrupt = False
    consecutive_rate_limited = 0  # Layer C: cooldown counter

    if not work_items:
        logger.info("nothing to submit (already complete or empty plan)")
        return (0, False)

    n_total = len(work_items)
    logger.info(
        "submitting %d trials @ concurrency=%d (cap=$%.2f)",
        n_total,
        cfg.concurrency,
        cfg.max_budget_usd,
    )

    def _drain_one(
        completed_n: int,
        in_flight: Dict[concurrent.futures.Future, Tuple[Task, str, int]],
    ) -> Tuple[int, bool, int]:
        """Wait for ONE future to finish, log + checkpoint it.

        Returns ``(new_completed, soft_crossed, rate_limited_streak)``.
        The ``rate_limited_streak`` value is the *closure-local*
        consecutive-429 counter after applying the drained future(s).
        """
        nonlocal consecutive_rate_limited
        done, _pending = concurrent.futures.wait(
            in_flight.keys(),
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        soft_crossed = False
        for fut in done:
            task_local, cond_local, trial_local = in_flight.pop(fut)
            try:
                task_id, _cond, _trial, tr, wall_s = fut.result()
            except Exception as exc:  # pragma: no cover — defensive
                logger.exception(
                    "trial crashed: %s/%s/%d -> %s",
                    task_local.task_id,
                    cond_local,
                    trial_local,
                    exc,
                )
                tr = TrajectoryResult(
                    task_id=task_local.task_id,
                    success=False,
                    tool_calls=0,
                    elapsed_s=0.0,
                    last_msg=str(exc)[:500],
                    notes="runner_exception",
                    verified=None,
                    verifier_status="runner_error",
                    cost_usd=None,
                )
                task_id = task_local.task_id
                wall_s = 0.0

            row = {
                "task_id": task_id,
                "condition": cond_local,
                "trial": trial_local,
                "trajectory_result": dataclasses.asdict(tr),
                "ts": datetime.now().isoformat(timespec="seconds"),
                "wall_s": wall_s,
            }
            _atomic_append_jsonl(results_jsonl, row)
            completed_n += 1

            # Layer C: track consecutive rate-limited trials so the
            # submit loop can introduce a cooldown if pressure persists.
            if _is_rate_limited_trajectory(tr):
                consecutive_rate_limited += 1
            else:
                consecutive_rate_limited = 0

            est = completed_n * COST_PER_TRIAL_USD
            if est >= 0.8 * cfg.max_budget_usd:
                soft_crossed = True
        return (completed_n, soft_crossed, consecutive_rate_limited)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, cfg.concurrency)
    ) as pool:
        in_flight: Dict[concurrent.futures.Future, Tuple[Task, str, int]] = {}
        try:
            for task, cond, trial in work_items:
                # Drain completed futures so ``completed`` is up to date
                # for the budget guard. Bound in-flight to ``concurrency``.
                while len(in_flight) >= max(1, cfg.concurrency):
                    completed, soft_crossed, streak = _drain_one(completed, in_flight)
                    if soft_crossed and not soft_warned:
                        logger.warning(
                            "soft budget warning: cumulative est=$%.2f >= 80%% of "
                            "$%.2f cap",
                            completed * COST_PER_TRIAL_USD,
                            cfg.max_budget_usd,
                        )
                        soft_warned = True
                    # Layer C: if many trials in a row hit the org rate
                    # limit, pause briefly before submitting more.
                    if streak >= _RATE_LIMIT_COOLDOWN_THRESHOLD:
                        logger.warning(
                            "rate-limit pressure: %d consecutive trials hit 429; "
                            "cooling down for %ds before submitting more",
                            streak,
                            _RATE_LIMIT_COOLDOWN_SLEEP_S,
                        )
                        time.sleep(_RATE_LIMIT_COOLDOWN_SLEEP_S)
                        consecutive_rate_limited = 0

                # Pre-submission budget guard.
                est_so_far = completed * COST_PER_TRIAL_USD
                if est_so_far >= cfg.max_budget_usd:
                    logger.warning(
                        "BUDGET CAP REACHED ($%.2f >= $%.2f); halting new submissions",
                        est_so_far,
                        cfg.max_budget_usd,
                    )
                    budget_capped = True
                    break
                if not soft_warned and est_so_far >= 0.8 * cfg.max_budget_usd:
                    logger.warning(
                        "soft budget warning: cumulative est=$%.2f >= 80%% of $%.2f cap",
                        est_so_far,
                        cfg.max_budget_usd,
                    )
                    soft_warned = True

                fut = pool.submit(
                    _run_one_trial,
                    backend,
                    task,
                    cond,
                    trial,
                    cfg,
                    runner_workdir=cfg.out_dir,
                )
                in_flight[fut] = (task, cond, trial)

            # Drain remaining in-flight.
            while in_flight:
                completed, soft_crossed, _streak = _drain_one(completed, in_flight)
                if soft_crossed and not soft_warned:
                    logger.warning(
                        "soft budget warning: cumulative est=$%.2f >= 80%% of $%.2f cap",
                        completed * COST_PER_TRIAL_USD,
                        cfg.max_budget_usd,
                    )
                    soft_warned = True

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt — cancelling pending and draining")
            keyboard_interrupt = True
            for fut in list(in_flight.keys()):
                if not fut.running() and not fut.done():
                    fut.cancel()
            # Drain the rest synchronously (in-flight will finish).
            for fut in concurrent.futures.as_completed(
                [f for f in in_flight.keys() if not f.done()]
            ):
                try:
                    task_id, cond, trial, tr, wall_s = fut.result()
                except Exception:
                    continue
                row = {
                    "task_id": task_id,
                    "condition": cond,
                    "trial": trial,
                    "trajectory_result": dataclasses.asdict(tr),
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "wall_s": wall_s,
                    "interrupted": True,
                }
                _atomic_append_jsonl(results_jsonl, row)
                completed += 1

    if keyboard_interrupt:
        # Re-raise so main() can return 130.
        raise KeyboardInterrupt()
    return (completed, budget_capped)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ---- Validation up front ----
    if args.agent_backend not in _ALLOWED_BACKENDS:
        raise NotImplementedError(
            f"--agent-backend {args.agent_backend!r} not supported in Phase D. "
            f"Allowed: {list(_ALLOWED_BACKENDS)}"
        )

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ERROR: ANTHROPIC_API_KEY is not set. The bench CLI's claude-code "
            "agent requires it to call the Anthropic API.",
            file=sys.stderr,
        )
        return _EXIT_NO_API_KEY

    try:
        conditions = _validate_conditions(args.conditions)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return _EXIT_ERROR

    if "with-skills" in conditions and "no-skills" in conditions:
        logger.info("running BOTH conditions — this is the headline lift case")

    try:
        task_ids = _load_task_list(args.task_list)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return _EXIT_ERROR

    try:
        tasks = _validate_and_hydrate(task_ids, args.vendor_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return _EXIT_ERROR

    # ---- Resolve config + plan ----
    out_dir = _resolve_out_dir(args.out)
    cfg = _RunConfig(
        task_list_path=args.task_list,
        vendor_dir=args.vendor_dir,
        model=args.model,
        trials=args.trials,
        conditions=conditions,
        agent_backend=args.agent_backend,
        concurrency=args.concurrency,
        max_budget_usd=args.max_budget_usd,
        out_dir=out_dir,
        task_timeout_s=args.task_timeout_s,
        skills_dir_override=args.skills_dir,
        dry_run=args.dry_run,
        resume=args.resume,
        argv=list(sys.argv[1:] if argv is None else argv),
        started_at=datetime.now().isoformat(timespec="seconds"),
    )

    n_trials = len(tasks) * len(conditions) * cfg.trials
    est_cost = _estimate_cost(n_trials)

    if est_cost > cfg.max_budget_usd:
        print(
            f"ERROR: pre-flight cost estimate exceeds budget cap.\n"
            f"  n_trials = {len(tasks)} tasks × {len(conditions)} conditions × "
            f"{cfg.trials} trials = {n_trials}\n"
            f"  est_cost = {n_trials} × ${COST_PER_TRIAL_USD:.2f} = ${est_cost:.2f}\n"
            f"  budget   = ${cfg.max_budget_usd:.2f}\n"
            f"Raise --max-budget-usd or reduce --trials/--task-list.",
            file=sys.stderr,
        )
        return _EXIT_BUDGET

    if cfg.dry_run:
        _print_plan(cfg, tasks, n_trials, est_cost)
        return _EXIT_OK

    # ---- Output dir setup ----
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scenes").mkdir(parents=True, exist_ok=True)
    (out_dir / "jobs").mkdir(parents=True, exist_ok=True)
    results_jsonl = out_dir / "results.jsonl"
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(cfg.to_manifest([t.task_id for t in tasks]), indent=2),
        encoding="utf-8",
    )
    _print_plan(cfg, tasks, n_trials, est_cost)

    # ---- Resume support ----
    completed_keys: set = set()
    if cfg.resume and results_jsonl.exists():
        existing = _read_existing_results(results_jsonl)
        for r in existing:
            if _is_complete(r):
                completed_keys.add(_row_key(r))
        logger.info(
            "resume: found %d completed (task,cond,trial) tuples in %s",
            len(completed_keys),
            results_jsonl,
        )

    # ---- Build work-item list ----
    work_items: List[Tuple[Task, str, int]] = []
    fresh = 0
    skipped = 0
    for task in tasks:
        for cond in conditions:
            for trial in range(cfg.trials):
                key = (task.task_id, cond, trial)
                if key in completed_keys:
                    skipped += 1
                    continue
                work_items.append((task, cond, trial))
                fresh += 1
    logger.info("work plan: %d fresh, %d skipped (resumed)", fresh, skipped)

    # ---- Execute ----
    backend = BenchCliBackend()
    interrupted = False
    budget_capped = False
    try:
        _n_completed, budget_capped = _execute(
            cfg,
            tasks,
            backend,
            work_items=work_items,
            results_jsonl=results_jsonl,
        )
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("interrupt drained — proceeding to aggregation")

    # ---- Aggregate ----
    rows = _read_existing_results(results_jsonl)
    summary = _aggregate(rows, [t.task_id for t in tasks], conditions)

    # ±5pp leaderboard delta sanity log (NOT a gate; D-9 dropped).
    sbs = summary["skillsbench_specific"]
    for k in (
        "delta_from_leaderboard_with_skills",
        "delta_from_leaderboard_no_skills",
    ):
        v = sbs.get(k)
        if v is not None and abs(v) > 5.0:
            logger.warning(
                "%s = %+.2f pp — verify pipeline before drawing conclusions",
                k,
                v,
            )

    # ---- Emit summary.json + summary.md ----
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "summary.md").write_text(
        _format_summary_md(summary, conditions), encoding="utf-8"
    )

    print(f"\nWrote {out_dir / 'summary.json'}")
    print(f"Wrote {out_dir / 'summary.md'}")
    if budget_capped:
        print("(BUDGET CAP REACHED — ran fewer trials than planned)")

    # TODO: stability check (Group D step 7) — re-run subset twice at
    # 1 trial, compare per-task disagreement. Lives in a separate
    # parity.py / `--mode stability` flag.

    if interrupted:
        return _EXIT_KEYBOARD_INTERRUPT
    return _EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
