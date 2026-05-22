"""Daycare command-line interface (Click).

Commands:
  - ``doctor``       — Phase 0 standalone health check.
  - ``eval-build``   — Phase 1 only: eval extraction + report.
  - ``run``          — Full Phases 0-5 orchestration.
  - ``promote``      — Copy best bundle → bundles/<project>/skills/<slug>/.
  - ``runs``         — List past runs in a table.
  - ``daemon``       — install / uninstall / run subcommands.

CLI surface mirrors spec §"CLI surface [v2]". rich.console.Console is
used for all interactive output; structured artefacts go to disk.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table


console = Console()


# ─── Helpers ──────────────────────────────────────────────────────────────


def _get_watchmen_home() -> Path:
    """``$WATCHMEN_HOME`` if set, otherwise ``~/.watchmen``."""
    env = os.environ.get("WATCHMEN_HOME")
    if env:
        return Path(env)
    return Path.home() / ".watchmen"


def _get_api_key() -> str:
    """Read ``$OPENROUTER_API_KEY``; raise click.ClickException if missing."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise click.ClickException("OPENROUTER_API_KEY env var is required (see daycare doctor).")
    return key


def _load_run_json(run_dir: Path) -> dict:
    """Load ``run_dir/run.json`` or return an empty dict."""
    p = run_dir / "run.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_run_json(run_dir: Path, data: dict) -> None:
    """Persist ``run_dir/run.json`` (creates dirs as needed)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _utc_slug() -> str:
    """ISO-8601 UTC slug safe for filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_projects_json(watchmen_home: Path) -> dict:
    path = watchmen_home / "projects.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _source_repo_for(watchmen_home: Path, project: str) -> str:
    data = _read_projects_json(watchmen_home)
    entry = data.get(project)
    if isinstance(entry, dict):
        sr = entry.get("source_repo")
        if isinstance(sr, str):
            return sr
    return ""


# ─── Click group ──────────────────────────────────────────────────────────


@click.group()
def main() -> None:
    """daycare — student-teacher skill distillation via mutation."""


# ─── doctor ───────────────────────────────────────────────────────────────


@main.command("doctor")
def doctor() -> None:
    """Phase 0 standalone health check."""
    from .corpus import check_skill_name_column
    from .providers import ping_model

    watchmen_home = _get_watchmen_home()
    db_path = watchmen_home / "corpus.db"
    out_path = watchmen_home / "daycare" / "doctor.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "watchmen_home": str(watchmen_home),
        "corpus_db_exists": db_path.exists(),
        "session_count": 0,
        "skill_name_column_ok": False,
        "openrouter_raw_log_dir": os.environ.get("OPENROUTER_RAW_LOG_DIR"),
        "model_pings": {},
    }

    # Session count.
    if db_path.exists():
        import sqlite3

        try:
            conn = sqlite3.connect(str(db_path))
            try:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM sessions")
                row = cur.fetchone()
                report["session_count"] = int(row[0]) if row else 0
            finally:
                conn.close()
        except sqlite3.Error as exc:
            report["session_count_error"] = str(exc)

    report["skill_name_column_ok"] = check_skill_name_column(db_path)

    # Model pings.
    api_key_present = bool(os.environ.get("OPENROUTER_API_KEY"))
    report["api_key_present"] = api_key_present
    models = [
        "qwen/qwen3-32b",
        "deepseek/deepseek-chat-v3-0324",
        "anthropic/claude-opus-4",
    ]
    for m in models:
        if not api_key_present:
            report["model_pings"][m] = {"ok": False, "detail": "no api key"}
            continue
        try:
            report["model_pings"][m] = ping_model(m)
        except Exception as exc:  # noqa: BLE001
            report["model_pings"][m] = {"ok": False, "detail": str(exc)}

    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Pretty-print.
    table = Table(title="daycare doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    table.add_row(
        "corpus.db",
        "ok" if report["corpus_db_exists"] else "FAIL",
        f"{report['session_count']} sessions",
    )
    table.add_row(
        "skill_name column",
        "ok" if report["skill_name_column_ok"] else "WARN",
        "(W1 fallback to JSONL scan if missing)",
    )
    table.add_row(
        "OPENROUTER_RAW_LOG_DIR",
        "set" if report["openrouter_raw_log_dir"] else "unset",
        report["openrouter_raw_log_dir"] or "-",
    )
    table.add_row(
        "OPENROUTER_API_KEY",
        "set" if api_key_present else "MISSING",
        "-",
    )
    for m, res in report["model_pings"].items():
        table.add_row(
            f"ping {m}",
            "ok" if res.get("ok") else "FAIL",
            str(res.get("detail", ""))[:60],
        )
    console.print(table)
    console.print(f"\nWrote {out_path}")


# ─── eval-build ───────────────────────────────────────────────────────────


@main.command("eval-build")
@click.argument("project")
@click.option("--days", default=60, type=int, show_default=True)
@click.option("--seed", default=42, type=int, show_default=True)
@click.option("--weak-model", default="qwen/qwen3-32b", show_default=True)
@click.option("--judge-model", default="deepseek/deepseek-chat-v3-0324", show_default=True)
@click.option("--skill", "skill_slug", default=None, help="override auto-selection")
@click.option(
    "--dry-run",
    is_flag=True,
    help="skip calibration LLM calls; print pre-1d candidate count only",
)
def eval_build(
    project: str,
    days: int,
    seed: int,
    weak_model: str,
    judge_model: str,
    skill_slug: str | None,
    dry_run: bool,
) -> None:
    """Phase 1 only — dump eval_set.jsonl + report."""
    from .corpus import parse_transcript, query_sessions
    from .eval_builder import run_eval_build

    watchmen_home = _get_watchmen_home()
    projects = _read_projects_json(watchmen_home)
    if project not in projects:
        raise click.ClickException(f"unknown project: {project}")

    source_repo = _source_repo_for(watchmen_home, project)
    bundle_dir = watchmen_home / "bundles" / project

    # J2: if no prior runs anywhere AND no explicit project context, sweep
    # all projects via the survey. (We treat the absence of --skill plus
    # no prior runs as "no project explicitly pinned" — caller invoked
    # `daycare eval-build <p>` which IS explicit, so the survey is not
    # triggered here. Survey lives in `run` for spec accuracy.)

    db_path = watchmen_home / "corpus.db"
    run_dir = watchmen_home / "daycare" / "runs" / f"{project}-{_utc_slug()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]RUN_DIR[/bold] = {run_dir}")

    if dry_run:
        # Just count pre-1d candidate triples (no judge calls).
        sessions = query_sessions(db_path, source_repo, days=days)
        candidate_count = 0
        for s in sessions:
            tp = s.get("transcript_path") or ""
            if not tp:
                continue
            p = Path(tp)
            if not p.exists():
                continue
            turns = parse_transcript(p)
            candidate_count += len(turns)
        table = Table(title=f"eval-build dry-run — {project}")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("sessions", str(len(sessions)))
        table.add_row("pre-1d candidate triples", str(candidate_count))
        console.print(table)
        return

    api_key = _get_api_key()

    try:
        train, holdout = run_eval_build(
            db_path=db_path,
            source_repo=source_repo,
            projects_json=watchmen_home / "projects.json",
            bundle_dir=bundle_dir,
            weak_model=weak_model,
            judge_model=judge_model,
            api_key=api_key,
            seed=seed,
            days=days,
            run_dir=run_dir,
        )
    except ValueError as exc:
        raise click.ClickException(f"eval-build aborted: {exc}") from exc

    # Cluster distribution table.
    by_type: dict[str, int] = {}
    for e in train + holdout:
        t = e.get("type") or "unknown"
        by_type[t] = by_type.get(t, 0) + 1

    table = Table(title=f"eval-build — {project}")
    table.add_column("Slice")
    table.add_column("Count", justify="right")
    table.add_row("train", str(len(train)))
    table.add_row("holdout", str(len(holdout)))
    for t, n in by_type.items():
        table.add_row(f"  type={t}", str(n))
    console.print(table)
    console.print(f"\nWrote {run_dir / 'eval_set.jsonl'}")


# ─── run ──────────────────────────────────────────────────────────────────


_BUDGET_RE = re.compile(r"^(\d+)([smhdSMHD]?)$")


def _parse_budget(s: str) -> int:
    """Parse ``8h`` / ``30m`` / ``3600`` → seconds."""
    m = _BUDGET_RE.match(s.strip())
    if not m:
        raise click.BadParameter(f"invalid budget: {s}")
    n = int(m.group(1))
    unit = (m.group(2) or "s").lower()
    return n * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]


def _existing_runs(watchmen_home: Path) -> list[Path]:
    runs_root = watchmen_home / "daycare" / "runs"
    if not runs_root.exists():
        return []
    return [p for p in runs_root.iterdir() if p.is_dir()]


@main.command("run")
@click.argument("project")
@click.option("--skill", "skill_slug", default=None, help="override auto-selection")
@click.option("--budget", default="8h", show_default=True)
@click.option("--max-iters", default=12, type=int, show_default=True)
@click.option("--K", "K", default=6, type=int, show_default=True)
@click.option("--rollouts", default=5, type=int, show_default=True)
@click.option("--seed", default=42, type=int, show_default=True)
@click.option("--weak-model", default="qwen/qwen3-32b", show_default=True)
@click.option("--proposer", default="deepseek/deepseek-chat-v3-0324", show_default=True)
@click.option("--judge", default="deepseek/deepseek-chat-v3-0324", show_default=True)
@click.option(
    "--teacher",
    default="anthropic/claude-opus-4",
    show_default=True,
    help="Baseline C teacher model",
)
@click.option("--max-workers", default=2, type=int, show_default=True)
@click.option("--smoke-3/--no-smoke-3", default=True, show_default=True)
@click.option(
    "--leak-policy",
    type=click.Choice(["zero", "warn"]),
    default="zero",
    show_default=True,
)
@click.option("--anonymize/--no-anonymize", default=True, show_default=True)
@click.option("--reproducibility", is_flag=True, help="Phase 4c — second seed run")
@click.option(
    "--yes",
    "yes_flag",
    is_flag=True,
    help="auto-confirm cost gate (>$50 estimate)",
)
def run(
    project: str,
    skill_slug: str | None,
    budget: str,
    max_iters: int,
    K: int,
    rollouts: int,
    seed: int,
    weak_model: str,
    proposer: str,
    judge: str,
    teacher: str,
    max_workers: int,
    smoke_3: bool,
    leak_policy: str,
    anonymize: bool,
    reproducibility: bool,
    yes_flag: bool,
) -> None:
    """Full Phases 0-5 orchestration."""
    from .anchor import load_best_bundle, run_anchor
    from .anonymize import build_context
    from .controls import run_all_baselines
    from .corpus import check_skill_name_column
    from .eval_builder import run_eval_build
    from .evolve import run_evolution
    from .finalize import write_optimized, write_summary
    from .leak_scanner import build_fingerprint_set
    from .providers import ping_model
    from .selector import rank_skills
    from .watchdog import Watchdog

    watchmen_home = _get_watchmen_home()
    projects = _read_projects_json(watchmen_home)
    if not projects:
        raise click.ClickException(f"no projects.json under {watchmen_home} — run `watchmen init` first")

    # J2: first-run multi-project survey.
    existing = _existing_runs(watchmen_home)
    if not existing and project not in projects:
        console.print(
            "[yellow]J2: no prior runs and project not explicitly selected — "
            "running eval-build survey on all projects.[/yellow]"
        )
        # Sweep all projects, pick the one with the most surviving evals.
        # We invoke eval-build on each rather than re-implementing it here.
        survey_results: dict[str, int] = {}
        for p in projects:
            try:
                from click.testing import CliRunner

                runner = CliRunner()
                runner.invoke(eval_build, [p, "--dry-run"])
                survey_results[p] = 0  # placeholder — real count needs --dry-run output parse
            except Exception as exc:  # noqa: BLE001
                console.print(f"survey {p} failed: {exc}")
        survey_path = watchmen_home / "daycare" / "eval_survey.json"
        survey_path.parent.mkdir(parents=True, exist_ok=True)
        survey_path.write_text(json.dumps(survey_results, indent=2), encoding="utf-8")
        console.print("Survey complete; pick a project and re-run with --project explicit.")
        return

    if project not in projects:
        raise click.ClickException(f"unknown project: {project}")

    api_key = _get_api_key()
    source_repo = _source_repo_for(watchmen_home, project)
    bundle_dir = watchmen_home / "bundles" / project
    db_path = watchmen_home / "corpus.db"

    # Phase 0 — bootstrap, skill selection.
    if skill_slug is None:
        ranks = rank_skills(
            db_path=db_path,
            bundle_dir=bundle_dir,
            source_repo=source_repo,
            running_md_path=watchmen_home / "analyses" / project / "_running.md",
        )
        if not ranks:
            raise click.ClickException(f"no skills under {bundle_dir}/skills/ — nothing to evolve")
        skill_slug = ranks[0].slug
        console.print(f"[bold]Auto-selected skill[/bold]: {skill_slug}")

    # Phase 0 — pre-launch OR pings (abort on 401/403).
    for m in (weak_model, proposer, judge):
        try:
            probe = ping_model(m)
        except Exception as exc:  # noqa: BLE001
            raise click.ClickException(f"ping {m} failed: {exc}") from exc
        if not probe.get("ok"):
            raise click.ClickException(f"ping {m} returned not-ok: {probe}")

    # Phase 0 — sanity on corpus.
    if not check_skill_name_column(db_path):
        console.print(
            "[yellow]warning: tool_calls.skill_name missing or all-NULL — "
            "selector falls back to JSONL scan (W1).[/yellow]"
        )

    # Phase 0 — RUN_DIR + run.json.
    run_dir = watchmen_home / "daycare" / "runs" / f"{project}-{_utc_slug()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    budget_seconds = _parse_budget(budget)
    epsilon_proxy = 0.033  # rough placeholder; recomputed after Phase 1

    # Estimated cost (per W8): K × n_holdout × rollouts × avg_tokens × $/Mtok × max_iters.
    # We don't know n_holdout yet — use 30 as the lower bound from M8.
    est_calls = K * 30 * rollouts * max_iters
    avg_tokens = 1500
    est_cost = est_calls * avg_tokens * 1e-6 * 2.0  # very rough $2/M-tokens proxy

    run_json = {
        "project": project,
        "skill_slug": skill_slug,
        "weak_model": weak_model,
        "proposer_model": proposer,
        "judge_model": judge,
        "teacher_model": teacher,
        "model_pins": {"weak": "unknown", "proposer": "unknown", "judge": "unknown"},
        "start_ts": datetime.now(timezone.utc).isoformat(),
        "budget_seconds": budget_seconds,
        "max_iters": max_iters,
        "max_skill_tokens": 2500,
        "lambda_init": 1e-5,
        "lambda_cap": 0.05,
        "epsilon": epsilon_proxy,
        "K": K,
        "rollouts": rollouts,
        "seed": seed,
        "max_workers": max_workers,
        "smoke_3_enabled": smoke_3,
        "leak_policy": leak_policy,
        "anonymize": anonymize,
        "estimated_cost_usd": est_cost,
        "raw_log_dir": os.environ.get("OPENROUTER_RAW_LOG_DIR"),
        "status": "running",
    }
    _save_run_json(run_dir, run_json)
    console.print(f"[bold]RUN_DIR[/bold] = {run_dir}")
    console.print(f"[bold]estimated_cost_usd[/bold] ≈ ${est_cost:.2f}")

    if est_cost > 50 and not yes_flag:
        raise click.ClickException(f"estimated cost ${est_cost:.2f} > $50 — re-run with --yes to confirm")

    if not anonymize:
        console.print(
            "[red]WARNING: --no-anonymize is unsafe; aborting. "
            "Use --anonymize (default) unless you know what you're doing.[/red]"
        )
        run_json["status"] = "aborted_no_anonymize"
        _save_run_json(run_dir, run_json)
        return

    try:
        # Phase 1 — eval extraction (skip if already done).
        eval_set_path = run_dir / "eval_set.jsonl"
        if eval_set_path.exists():
            console.print("[dim]Phase 1: eval_set.jsonl already exists, skipping.[/dim]")
            # Re-load.
            train: list[dict] = []
            holdout: list[dict] = []
            with eval_set_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    (holdout if row.get("split") == "holdout" else train).append(row)
        else:
            console.print("[bold]Phase 1[/bold] — eval extraction")
            train, holdout = run_eval_build(
                db_path=db_path,
                source_repo=source_repo,
                projects_json=watchmen_home / "projects.json",
                bundle_dir=bundle_dir / "skills" / skill_slug,
                weak_model=weak_model,
                judge_model=judge,
                api_key=api_key,
                seed=seed,
                days=60,
                run_dir=run_dir,
            )

        epsilon = max(0.01, 1.0 / max(1, len(holdout)))
        run_json["epsilon"] = epsilon
        _save_run_json(run_dir, run_json)

        # Phase 2 — anchor.
        console.print("[bold]Phase 2[/bold] — anchor (iter_0)")
        skill_bundle = bundle_dir / "skills" / skill_slug
        anchor_summary = run_anchor(
            run_dir=run_dir,
            bundle_dir=skill_bundle,
            eval_set_path=eval_set_path,
            model=weak_model,
            judge_model=judge,
            api_key=api_key,
            rollouts=rollouts,
            seed=seed,
            temperature=0.7,
            max_workers=max_workers,
        )
        console.print(f"  iter_0 holdout_score = {anchor_summary.holdout_score:.4f}")

        # Phase 3 — evolution loop with watchdog.
        console.print("[bold]Phase 3[/bold] — evolution loop")
        watchdog = Watchdog(budget_seconds=budget_seconds)
        watchdog.start()
        try:
            fingerprints = build_fingerprint_set(
                eval_set_path,
                watchmen_home / "projects.json",
            )
            run_evolution(
                run_dir=run_dir,
                iter_0_summary=anchor_summary,
                train_evals=train,
                holdout_evals=holdout,
                eval_set_path=eval_set_path,
                best_bundle_dir=run_dir / "iter_0" / "bundle",
                model=weak_model,
                judge_model=judge,
                proposer_model=proposer,
                api_key=api_key,
                rollouts=rollouts,
                max_workers=max_workers,
                max_iters=max_iters,
                lambda_init=1e-5,
                lambda_cap=0.05,
                seed=seed,
                fingerprints=fingerprints,
                watchdog=watchdog,
                leak_policy=leak_policy,
                bundles_root=bundle_dir / "skills",
                K=K,
            )
        finally:
            watchdog.cancel()

        best_bundle, best_fitness = load_best_bundle(run_dir)
        console.print(f"  best fitness = {best_fitness:.4f} (bundle={best_bundle})")

        # Phase 4 — baselines.
        console.print("[bold]Phase 4[/bold] — baselines (A/B/C)")
        ctx = build_context(watchmen_home / "projects.json", bundle_dir, source_repo)
        baseline_results = run_all_baselines(
            run_dir=run_dir,
            train_evals=train,
            holdout_evals=holdout,
            best_bundle_dir=best_bundle,
            best_holdout_score=best_fitness,
            model=weak_model,
            teacher_model=teacher,
            judge_model=judge,
            api_key=api_key,
            rollouts=rollouts,
            epsilon=epsilon,
            ctx=ctx,
        )
        a = baseline_results.get("baseline_a")
        c = baseline_results.get("baseline_c")
        console.print(
            f"  baseline_a={getattr(a, 'holdout_score', 0):.4f}  "
            f"baseline_c={getattr(c, 'holdout_score', 0):.4f}  "
            f"gap_closed={getattr(c, 'gap_closed', None)}"
        )

        # Phase 5 — finalize.
        console.print("[bold]Phase 5[/bold] — finalize")
        write_optimized(run_dir, best_bundle, skill_slug)
        write_summary(run_dir, run_dir / "metrics.json", baseline_results)

        promote_blocked = bool(getattr(a, "promote_blocked", False))
        run_json["status"] = "completed"
        run_json["end_ts"] = datetime.now(timezone.utc).isoformat()
        run_json["best_holdout_score"] = best_fitness
        run_json["promote_blocked"] = promote_blocked
        _save_run_json(run_dir, run_json)

        if promote_blocked:
            console.print("[red]J1: promotion BLOCKED — best did not beat baseline A floor.[/red]")
        else:
            console.print(f"[green]Promotion eligible:[/green] `daycare promote {project} {skill_slug}`")
    except Exception as exc:  # noqa: BLE001
        run_json["status"] = "aborted"
        run_json["error"] = str(exc)
        _save_run_json(run_dir, run_json)
        raise


# ─── promote ──────────────────────────────────────────────────────────────


@main.command("promote")
@click.argument("project")
@click.argument("slug")
def promote(project: str, slug: str) -> None:
    """Copy optimized bundle into watchmen home + pin."""
    from .finalize import promote as _promote

    watchmen_home = _get_watchmen_home()

    # Pick the most recent run for this project (by sorted name — the
    # UTC slug is lex-sortable).
    runs_root = watchmen_home / "daycare" / "runs"
    matching = sorted(
        [p for p in runs_root.glob(f"{project}-*") if p.is_dir()],
        reverse=True,
    )
    if not matching:
        raise click.ClickException(f"no runs found for project={project}")
    run_dir = matching[0]

    try:
        _promote(run_dir, slug, project, watchmen_home)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    console.print(f"[green]Promoted[/green] {slug} → {watchmen_home}/bundles/{project}/skills/{slug}/")


# ─── runs ─────────────────────────────────────────────────────────────────


@main.command("runs")
def runs() -> None:
    """Table of past runs with Δ scores."""
    watchmen_home = _get_watchmen_home()
    runs_root = watchmen_home / "daycare" / "runs"
    if not runs_root.exists():
        console.print("(no runs yet)")
        return

    table = Table(title="daycare runs")
    table.add_column("Project")
    table.add_column("Slug")
    table.add_column("Start (UTC)")
    table.add_column("Status")
    table.add_column("iter_0", justify="right")
    table.add_column("best", justify="right")
    table.add_column("gap_closed", justify="right")
    table.add_column("blocked")

    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        rj_path = run_dir / "run.json"
        if not rj_path.exists():
            continue
        try:
            rj = json.loads(rj_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(rj, dict):
            continue

        # iter_0 score.
        iter0_score = "-"
        i0p = run_dir / "iter_0" / "eval_summary.json"
        if i0p.exists():
            try:
                i0 = json.loads(i0p.read_text(encoding="utf-8"))
                if isinstance(i0, dict):
                    v = i0.get("holdout_score")
                    if isinstance(v, (int, float)):
                        iter0_score = f"{v:.4f}"
            except (json.JSONDecodeError, OSError):
                pass

        # Phase 4 numbers.
        gap = "-"
        blocked = "-"
        p4p = run_dir / "phase4_results.json"
        if p4p.exists():
            try:
                p4 = json.loads(p4p.read_text(encoding="utf-8"))
                if isinstance(p4, dict):
                    g = p4.get("gap_closed")
                    if isinstance(g, (int, float)):
                        gap = f"{g:.4f}"
                    blocked = str(p4.get("promote_blocked", "-"))
            except (json.JSONDecodeError, OSError):
                pass

        best = rj.get("best_holdout_score")
        best_str = f"{best:.4f}" if isinstance(best, (int, float)) else "-"

        table.add_row(
            str(rj.get("project", "?")),
            str(rj.get("skill_slug", "?")),
            str(rj.get("start_ts", "?"))[:19],
            str(rj.get("status", "?")),
            iter0_score,
            best_str,
            gap,
            blocked,
        )

    console.print(table)


# ─── daemon ───────────────────────────────────────────────────────────────


@main.group("daemon")
def daemon_group() -> None:
    """Daemon mode (launchd / systemd)."""


@daemon_group.command("install")
def daemon_install() -> None:
    """Install platform-appropriate unit (launchd on macOS, systemd on Linux)."""
    from . import daemon as daemon_mod

    watchmen_home = _get_watchmen_home()
    daycare_bin = shutil.which("daycare") or "daycare"

    if sys.platform == "darwin":
        daemon_mod.install_launchd(watchmen_home, daycare_bin)
        console.print(f"[green]Installed launchd:[/green] {Path.home()}/Library/LaunchAgents/com.daycare.daemon.plist")
    elif sys.platform.startswith("linux"):
        daemon_mod.install_systemd(watchmen_home, daycare_bin)
        console.print("[green]Installed systemd user unit[/green]")
    else:
        raise click.ClickException(f"unsupported platform: {sys.platform}")


@daemon_group.command("uninstall")
def daemon_uninstall() -> None:
    """Remove the installed unit."""
    from . import daemon as daemon_mod

    if sys.platform == "darwin":
        daemon_mod.uninstall_launchd()
        console.print("[green]Uninstalled launchd[/green]")
    elif sys.platform.startswith("linux"):
        daemon_mod.uninstall_systemd()
        console.print("[green]Uninstalled systemd[/green]")
    else:
        raise click.ClickException(f"unsupported platform: {sys.platform}")


@daemon_group.command("run")
def daemon_run() -> None:
    """Run the daemon loop in the foreground (called by launchd/systemd)."""
    from . import daemon as daemon_mod

    watchmen_home = _get_watchmen_home()
    api_key = os.environ.get("OPENROUTER_API_KEY") or ""
    daemon_mod.run_daemon(watchmen_home, api_key)


if __name__ == "__main__":
    main()
