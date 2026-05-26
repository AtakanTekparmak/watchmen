"""Daemon mode (launchd / systemd).

Spec §"Daemon mode" cadence:
  - Every 2h: rescan corpus for new sessions, refresh selector ranking.
  - Daily off-peak (03:00 local): full ``daycare run`` on highest-priority
    skill in the most-active project.
  - Weekly: re-run Phase 1d calibration to detect eval drift.

This module only handles unit install/uninstall + the inner loop. The
work that gets executed per-tick is delegated to the existing
``daycare run`` subprocess invocation — the daemon never imports the
evolution code directly so a per-run crash can't take the daemon down.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .corpus import query_sessions


# ─── Constants ────────────────────────────────────────────────────────────


_TICK_SECONDS = 2 * 60 * 60  # 2h base cycle
_DAILY_RUN_HOUR_LOCAL = 3  # 03:00 local
_WEEKLY_CAL_DAY = 0  # Monday for weekly re-calibration

MIN_NEW_SESSIONS_FOR_REBUILD = 5  # default for daemon flag
MIN_RUN_INTERVAL_HOURS = 24  # default for re-evolution gate


# ─── projects.json helpers ────────────────────────────────────────────────


def _read_projects_json(watchmen_home: Path) -> dict:
    """Read projects.json — handles both list-of-dicts and dict variants.

    Mirrors `daycare.cli._read_projects_json` to avoid circular imports.
    """
    path = watchmen_home / "projects.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
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


# ─── New-sessions and eval-set helpers ────────────────────────────────────


def _count_new_sessions(db_path: Path, source_repo: str, since_iso: str | None) -> int:
    """Count sessions for source_repo whose started_at > since_iso.

    Uses query_sessions(days=36500) — NOT days=None (would TypeError).
    """
    rows = query_sessions(db_path, source_repo, days=36500)
    if since_iso is None:
        return len(rows)
    return sum(1 for r in rows if r["started_at"] > since_iso)


def _latest_session_ts(db_path: Path, source_repo: str) -> str | None:
    """Return max(started_at) over matching sessions, or None."""
    rows = query_sessions(db_path, source_repo, days=36500)
    if not rows:
        return None
    return max(r["started_at"] for r in rows)


def _hash_eval_set(eval_set_path: Path) -> str:
    """Return sha256 hex of eval_set.jsonl file contents."""
    h = hashlib.sha256()
    h.update(eval_set_path.read_bytes())
    return h.hexdigest()


def _latest_eval_set_path(watchmen_home: Path, project: str) -> Path | None:
    """Find most-recent completed run for `project` and return its eval_set.jsonl.

    Scans watchmen_home / "daycare" / "runs" for dirs matching f"{project}-*",
    sorts by name (ISO timestamps are lex-sortable), returns the newest one's
    eval_set.jsonl if it exists, else falls through to the next newest.
    Returns None if runs/ doesn't exist or no candidate has eval_set.jsonl.
    """
    runs_root = watchmen_home / "daycare" / "runs"
    if not runs_root.exists():
        return None
    candidates = sorted(
        (d for d in runs_root.iterdir() if d.is_dir() and d.name.startswith(f"{project}-")),
        reverse=True,
    )
    for d in candidates:
        p = d / "eval_set.jsonl"
        if p.exists():
            return p
    return None


def _hours_since(iso: str | None, now: datetime) -> float:
    """Return elapsed hours since iso (ISO8601 string), or inf if None."""
    if iso is None:
        return float("inf")
    try:
        past = datetime.fromisoformat(iso)
        if past.tzinfo is None:
            past = past.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return (now - past).total_seconds() / 3600.0
    except (ValueError, TypeError):
        return float("inf")


# ─── launchd (macOS) ──────────────────────────────────────────────────────


_LAUNCHD_LABEL = "com.daycare.daemon"


def _launchd_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{_LAUNCHD_LABEL}.plist"


def _launchd_logs_dir(watchmen_home: Path) -> Path:
    p = watchmen_home / "daycare" / "daemon" / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def install_launchd(watchmen_home: Path, daycare_bin: str) -> None:
    """Write ``~/Library/LaunchAgents/com.daycare.daemon.plist`` + launchctl load.

    The plist runs ``daycare daemon run`` every 2h (StartInterval=7200).
    Stdout/stderr land in ``watchmen_home/daycare/daemon/logs/``.
    """
    logs = _launchd_logs_dir(watchmen_home)
    plist_path = _launchd_plist_path()
    plist_path.parent.mkdir(parents=True, exist_ok=True)

    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{_LAUNCHD_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{daycare_bin}</string>
        <string>daemon</string>
        <string>run</string>
    </array>
    <key>StartInterval</key>
    <integer>{_TICK_SECONDS}</integer>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{logs / "daemon.stdout.log"}</string>
    <key>StandardErrorPath</key>
    <string>{logs / "daemon.stderr.log"}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>WATCHMEN_HOME</key>
        <string>{watchmen_home}</string>
    </dict>
</dict>
</plist>
"""
    plist_path.write_text(plist, encoding="utf-8")

    if sys.platform == "darwin":
        try:
            subprocess.run(
                ["launchctl", "load", str(plist_path)],
                check=False,
                capture_output=True,
                timeout=10,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            print(f"[daemon] launchctl load failed: {exc}", file=sys.stderr)


def uninstall_launchd() -> None:
    """``launchctl unload`` and remove the plist."""
    plist_path = _launchd_plist_path()
    if sys.platform == "darwin" and plist_path.exists():
        try:
            subprocess.run(
                ["launchctl", "unload", str(plist_path)],
                check=False,
                capture_output=True,
                timeout=10,
            )
        except (subprocess.SubprocessError, OSError):
            pass
    if plist_path.exists():
        try:
            plist_path.unlink()
        except OSError:
            pass


# ─── systemd (Linux) ──────────────────────────────────────────────────────


def _systemd_dir() -> Path:
    return Path.home() / ".config" / "systemd" / "user"


def install_systemd(watchmen_home: Path, daycare_bin: str) -> None:
    """Write ``~/.config/systemd/user/{daycare.service,daycare.timer}``.

    The timer fires every 2h (OnCalendar=*:0/120). Calls
    ``systemctl --user daemon-reload && enable --now daycare.timer``.
    """
    unit_dir = _systemd_dir()
    unit_dir.mkdir(parents=True, exist_ok=True)
    logs = _launchd_logs_dir(watchmen_home)

    service = f"""[Unit]
Description=daycare daemon (skill evolution loop)
After=network.target

[Service]
Type=oneshot
Environment=WATCHMEN_HOME={watchmen_home}
ExecStart={daycare_bin} daemon run
StandardOutput=append:{logs / "daemon.stdout.log"}
StandardError=append:{logs / "daemon.stderr.log"}
"""
    timer = """[Unit]
Description=daycare daemon timer (every 2h)

[Timer]
OnCalendar=*:0/120
Persistent=true
Unit=daycare.service

[Install]
WantedBy=timers.target
"""

    (unit_dir / "daycare.service").write_text(service, encoding="utf-8")
    (unit_dir / "daycare.timer").write_text(timer, encoding="utf-8")

    if sys.platform.startswith("linux"):
        try:
            subprocess.run(
                ["systemctl", "--user", "daemon-reload"],
                check=False,
                capture_output=True,
                timeout=10,
            )
            subprocess.run(
                ["systemctl", "--user", "enable", "--now", "daycare.timer"],
                check=False,
                capture_output=True,
                timeout=10,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            print(f"[daemon] systemctl call failed: {exc}", file=sys.stderr)


def uninstall_systemd() -> None:
    """``systemctl --user disable --now daycare.timer`` + remove units."""
    if sys.platform.startswith("linux"):
        try:
            subprocess.run(
                ["systemctl", "--user", "disable", "--now", "daycare.timer"],
                check=False,
                capture_output=True,
                timeout=10,
            )
        except (subprocess.SubprocessError, OSError):
            pass
    unit_dir = _systemd_dir()
    for name in ("daycare.timer", "daycare.service"):
        p = unit_dir / name
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass


# ─── run_daemon (inner loop) ──────────────────────────────────────────────


_stop = False


def _signal_handler(signum, frame) -> None:  # noqa: ARG001
    global _stop
    _stop = True


def _build_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("daycare.daemon")
    if logger.handlers:
        return logger
    handler = logging.FileHandler(log_dir / "daemon.log")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def _is_daily_run_due(last_daily_run: datetime | None, now: datetime) -> bool:
    """True if the local clock just crossed 03:00 since ``last_daily_run``."""
    if last_daily_run is None:
        return now.hour == _DAILY_RUN_HOUR_LOCAL
    if now.date() == last_daily_run.date():
        return False
    return now.hour == _DAILY_RUN_HOUR_LOCAL


def _pick_top_skill(watchmen_home: Path) -> tuple[str, str] | None:
    """Pick (project, slug) of the highest-priority skill across projects.

    Uses ``daycare.selector.rank_skills`` against each project's bundle
    dir. Falls back to None if no projects have any skills.
    """
    from .selector import rank_skills

    projects_json = watchmen_home / "projects.json"
    if not projects_json.exists():
        return None
    try:
        data = json.loads(projects_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None

    best: tuple[str, str, float] | None = None
    db_path = watchmen_home / "corpus.db"
    for project, meta in data.items():
        if not isinstance(meta, dict):
            continue
        source_repo = meta.get("source_repo") or ""
        bundle_dir = watchmen_home / "bundles" / project
        if not bundle_dir.exists():
            continue
        ranks = rank_skills(
            db_path=db_path,
            bundle_dir=bundle_dir,
            source_repo=source_repo,
        )
        if not ranks:
            continue
        top = ranks[0]
        if best is None or top.priority > best[2]:
            best = (project, top.slug, top.priority)

    if best is None:
        return None
    return best[0], best[1]


def run_daemon(
    watchmen_home: Path,
    api_key: str,
    min_new_sessions: int = MIN_NEW_SESSIONS_FOR_REBUILD,
    min_run_interval_hours: float = MIN_RUN_INTERVAL_HOURS,
) -> None:
    """Daemon inner loop.

    On each 2h tick:
      - Refresh selector ranking (cheap).
      - Per-project flywheel: if ``min_new_sessions`` new corpus sessions
        have arrived for a project, spawn ``daycare eval-build`` and (if
        the eval set changed and ``min_run_interval_hours`` has elapsed
        since the last evolution) spawn ``daycare run``.
      - If a daily run is due (03:00 local), spawn ``daycare run`` as a
        subprocess on the highest-priority skill.
      - On Mondays, log a re-calibration reminder (the actual
        re-calibration runs as part of the next ``daycare run``'s
        Phase 1).

    SIGINT / SIGTERM flip the stop flag for clean shutdown.
    """
    _min_sessions = min_new_sessions
    _min_run_interval = min_run_interval_hours
    log_dir = watchmen_home / "daycare" / "daemon" / "logs"
    logger = _build_logger(log_dir)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Track state across ticks. last_daily_run / last_weekly are persisted
    # under daemon/state.json so the daemon survives restarts cleanly.
    state_path = watchmen_home / "daycare" / "daemon" / "state.json"
    state: dict = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            state = {}
        if not isinstance(state, dict):
            state = {}

    last_daily_run: datetime | None = None
    last_weekly: datetime | None = None
    if isinstance(state.get("last_daily_run"), str):
        try:
            last_daily_run = datetime.fromisoformat(state["last_daily_run"])
        except ValueError:
            last_daily_run = None
    if isinstance(state.get("last_weekly"), str):
        try:
            last_weekly = datetime.fromisoformat(state["last_weekly"])
        except ValueError:
            last_weekly = None

    daycare_bin = shutil.which("daycare") or "daycare"

    logger.info("daemon starting (watchmen_home=%s)", watchmen_home)

    global _stop
    _stop = False

    while not _stop:
        now = datetime.now().astimezone()
        try:
            # Selector refresh (cheap — sqlite + directory scan).
            top = _pick_top_skill(watchmen_home)
            if top is None:
                logger.info("tick: no skills to rank")
            else:
                project, slug = top
                logger.info("tick: top skill is %s/%s", project, slug)

            # Per-project incremental flywheel.
            db_path = watchmen_home / "corpus.db"
            projects_meta = _read_projects_json(watchmen_home)
            state.setdefault("per_project", {})
            for project in projects_meta.keys():
                source_repo = _source_repo_for(watchmen_home, project)
                if not source_repo:
                    continue

                state["per_project"].setdefault(
                    project,
                    {
                        "last_seen_session_ts": None,
                        "last_eval_build_ts": None,
                        "last_eval_set_hash": None,
                        "last_evolution_run_ts": None,
                        "new_sessions_since_build": 0,
                    },
                )

                spawn_env = {
                    **os.environ,
                    "WATCHMEN_HOME": str(watchmen_home),
                    "OPENROUTER_API_KEY": api_key,
                }

                new_count = _count_new_sessions(
                    db_path,
                    source_repo,
                    state["per_project"][project].get("last_seen_session_ts"),
                )
                if new_count >= _min_sessions:
                    logger.info(
                        "project %s has %d new sessions; spawning eval-build",
                        project,
                        new_count,
                    )
                    proc = subprocess.Popen(
                        [daycare_bin, "eval-build", project, "--behavioral"],
                        env=spawn_env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    )
                    proc.wait()  # synchronous
                    state["per_project"][project]["last_eval_build_ts"] = now.isoformat()
                    state["per_project"][project]["last_seen_session_ts"] = _latest_session_ts(
                        db_path,
                        source_repo,
                    )
                    state["per_project"][project]["new_sessions_since_build"] = 0

                    # Re-evolution gate.
                    eval_set_path = _latest_eval_set_path(watchmen_home, project)
                    if eval_set_path is not None and eval_set_path.exists():
                        new_hash = _hash_eval_set(eval_set_path)
                        prev_hash = state["per_project"][project].get("last_eval_set_hash")
                        last_run_iso = state["per_project"][project].get("last_evolution_run_ts")
                        hours_elapsed = _hours_since(last_run_iso, now)
                        if new_hash != prev_hash and hours_elapsed >= _min_run_interval:
                            logger.info(
                                "project %s eval set changed; spawning run",
                                project,
                            )
                            subprocess.Popen(
                                [daycare_bin, "run", project, "--eval-set", str(eval_set_path), "--yes"],
                                env=spawn_env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                            )
                            state["per_project"][project]["last_evolution_run_ts"] = now.isoformat()
                        state["per_project"][project]["last_eval_set_hash"] = new_hash

            # Daily run gate.
            if top is not None and _is_daily_run_due(last_daily_run, now):
                project, slug = top
                logger.info("spawning daycare run %s --skill %s", project, slug)
                env = os.environ.copy()
                env.setdefault("WATCHMEN_HOME", str(watchmen_home))
                if api_key:
                    env.setdefault("OPENROUTER_API_KEY", api_key)
                try:
                    subprocess.Popen(
                        [
                            daycare_bin,
                            "run",
                            project,
                            "--skill",
                            slug,
                            "--yes",
                        ],
                        env=env,
                        stdout=open(log_dir / "run.stdout.log", "ab"),
                        stderr=open(log_dir / "run.stderr.log", "ab"),
                    )
                    last_daily_run = now
                except (subprocess.SubprocessError, OSError) as exc:
                    logger.error("daily run spawn failed: %s", exc)

            # Weekly recalibration reminder.
            if now.weekday() == _WEEKLY_CAL_DAY and (last_weekly is None or (now - last_weekly) > timedelta(days=6)):
                logger.info("weekly re-calibration due — next daycare run will refresh evals")
                last_weekly = now

            # Persist state — preserve per_project across overwrites.
            existing_per_project = state.get("per_project", {})
            state = {
                "last_daily_run": last_daily_run.isoformat() if last_daily_run else None,
                "last_weekly": last_weekly.isoformat() if last_weekly else None,
                "last_tick": now.isoformat(),
            }
            state["per_project"] = existing_per_project
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 — never let one tick kill the daemon
            logger.exception("tick crashed: %s", exc)

        # Sleep until next tick (cooperative wakeup on SIGINT).
        slept = 0
        while slept < _TICK_SECONDS and not _stop:
            time.sleep(min(30, _TICK_SECONDS - slept))
            slept += 30

    logger.info("daemon stopping")
