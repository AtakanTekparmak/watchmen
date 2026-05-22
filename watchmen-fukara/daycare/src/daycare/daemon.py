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

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


# ─── Constants ────────────────────────────────────────────────────────────


_TICK_SECONDS = 2 * 60 * 60  # 2h base cycle
_DAILY_RUN_HOUR_LOCAL = 3  # 03:00 local
_WEEKLY_CAL_DAY = 0  # Monday for weekly re-calibration


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


def run_daemon(watchmen_home: Path, api_key: str) -> None:
    """Daemon inner loop.

    On each 2h tick:
      - Refresh selector ranking (cheap).
      - If a daily run is due (03:00 local), spawn ``daycare run`` as a
        subprocess on the highest-priority skill.
      - On Mondays, log a re-calibration reminder (the actual
        re-calibration runs as part of the next ``daycare run``'s
        Phase 1).

    SIGINT / SIGTERM flip the stop flag for clean shutdown.
    """
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

            # Persist state.
            state = {
                "last_daily_run": last_daily_run.isoformat() if last_daily_run else None,
                "last_weekly": last_weekly.isoformat() if last_weekly else None,
                "last_tick": now.isoformat(),
            }
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
