"""Daycare command-line interface (Click).

Commands:
  - ``doctor``       — Phase 0 standalone health check.
  - ``eval-build``   — Phase 1 only: eval extraction + report.
  - ``runs``         — List past runs in a table.

The full orchestration ``run`` / ``promote`` / ``daemon`` subcommands have
moved to ``skill_evolve``; daycare retains only the eval-extraction and
read-only inspection surface.
"""

from __future__ import annotations

import json
import os
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
    # watchmen stores projects.json as a list of {project_key, source_repo} objects
    if isinstance(data, list):
        return {entry["project_key"]: entry for entry in data if "project_key" in entry}
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
    """daycare — eval extraction and inspection for watchmen skill bundles."""


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
        "qwen/qwen3.6-27b",
        "deepseek/deepseek-v4-pro",
        "anthropic/claude-opus-4.7",
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
@click.option("--weak-model", default="qwen/qwen3.6-27b", show_default=True)
@click.option("--judge-model", default="deepseek/deepseek-v4-pro", show_default=True)
@click.option("--skill", "skill_slug", default=None, help="override auto-selection")
@click.option("--max-candidates", default=None, type=int, help="sample N triples before LLM calls (testing)")
@click.option("--max-workers", default=4, type=int, show_default=True, help="parallel LLM call workers")
@click.option(
    "--dry-run",
    is_flag=True,
    help="skip calibration LLM calls; print pre-1d candidate count only",
)
@click.option(
    "--synthetic",
    is_flag=True,
    help="use synthetic Q&A generation (synth_builder) instead of corpus extraction",
)
@click.option(
    "--n-per-skill",
    default=20,
    type=int,
    show_default=True,
    help="(synthetic only) number of questions to generate per skill",
)
@click.option(
    "--proposer",
    default="deepseek/deepseek-chat-v3-0324",
    show_default=True,
    help="(synthetic only) proposer model for question generation",
)
@click.option(
    "--behavioral/--no-behavioral",
    default=None,
    show_default=False,
    help="Extract behavioral action evals from corpus turns (default unless "
    "--synthetic is passed). --no-behavioral falls back to the legacy "
    "classify-by-type path.",
)
def eval_build(
    project: str,
    days: int,
    seed: int,
    weak_model: str,
    judge_model: str,
    skill_slug: str | None,
    max_candidates: int | None,
    max_workers: int,
    dry_run: bool,
    synthetic: bool,
    n_per_skill: int,
    proposer: str,
    behavioral: bool | None,
) -> None:
    """Phase 1 only — dump eval_set.jsonl + report."""
    from .corpus import parse_transcript, query_sessions
    from .eval_builder import run_eval_build

    # Resolve --behavioral default and check mutual exclusion.
    # Note: in eval-build, the --synthetic flag variable is named `synthetic`
    if behavioral is None:
        behavioral = not synthetic  # default True unless --synthetic given
    if behavioral and synthetic:
        raise click.BadParameter("--behavioral and --synthetic are mutually exclusive")

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
    # triggered here.)

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

    if synthetic:
        from .synth_builder import run_synth_eval_build

        try:
            train, holdout = run_synth_eval_build(
                bundle_dir=bundle_dir,
                watchmen_home=watchmen_home,
                project=project,
                weak_model=weak_model,
                proposer_model=proposer,
                judge_model=judge_model,
                api_key=api_key,
                seed=seed,
                n_per_skill=n_per_skill,
                run_dir=run_dir,
                max_workers=max_workers,
            )
        except ValueError as exc:
            raise click.ClickException(f"synth eval-build aborted: {exc}") from exc
    else:
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
                max_candidates=max_candidates,
                max_workers=max_workers,
                behavioral=behavioral,
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


if __name__ == "__main__":
    main()
