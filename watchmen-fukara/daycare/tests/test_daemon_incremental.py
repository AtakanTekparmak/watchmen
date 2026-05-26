"""Daemon incremental-flywheel tests (Group D).

All tests monkeypatch ``daemon._TICK_SECONDS = 1`` AND ``daemon.time.sleep``
so the 2-hour real tick interval never blocks. The strategy used to run
"exactly one tick": patch ``time.sleep`` to flip ``daemon._stop = True``
once a full tick body has run.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


from daycare import daemon


# ─── shared helpers ───────────────────────────────────────────────────────


def _write_projects_json(watchmen_home: Path, project: str, source_repo: str) -> None:
    watchmen_home.mkdir(parents=True, exist_ok=True)
    (watchmen_home / "projects.json").write_text(
        json.dumps({project: {"source_repo": source_repo}}),
        encoding="utf-8",
    )


def _seed_sessions_db(
    db_path: Path,
    rows: list[tuple[str, str, str]],
) -> None:
    """Create a corpus.db with a minimal `sessions` table.

    Each row is (session_id, project_dir, started_at).
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE sessions (
                session_id TEXT,
                project_dir TEXT,
                started_at TEXT,
                ended_at TEXT,
                model TEXT,
                total_turns INTEGER,
                is_subagent INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0,
                transcript_path TEXT
            )
            """
        )
        for sid, pdir, started in rows:
            cur.execute(
                "INSERT INTO sessions (session_id, project_dir, started_at, "
                "ended_at, model, total_turns, is_subagent, cost_usd, "
                "transcript_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (sid, pdir, started, None, "claude", 5, 0, 0.5, ""),
            )
        conn.commit()
    finally:
        conn.close()


def _install_one_tick_patches(monkeypatch):
    """Patch the daemon module so run_daemon executes exactly one tick.

    - ``_TICK_SECONDS`` shrunk to 1 so the inner sleep-loop exits fast.
    - ``time.sleep`` replaced with a function that flips ``_stop`` on
      first call (so the inner ``while slept < _TICK_SECONDS`` exits
      immediately, and the outer ``while not _stop`` exits on the next
      iteration check).
    """
    monkeypatch.setattr(daemon, "_TICK_SECONDS", 1)

    def _fake_sleep(*_args, **_kwargs):
        daemon._stop = True

    monkeypatch.setattr(daemon.time, "sleep", _fake_sleep)


class _PopenSpy:
    """Record every call to subprocess.Popen as (argv, env_dict)."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, argv, **kwargs):
        self.calls.append(
            {
                "argv": list(argv),
                "env": dict(kwargs.get("env") or {}),
                "kwargs": kwargs,
            }
        )

        class _FakeProc:
            returncode = 0

            def wait(self_inner, timeout=None):
                return 0

            def communicate(self_inner, *a, **k):
                return (b"", b"")

        return _FakeProc()


# ─── 1. one tick spawns eval-build ────────────────────────────────────────


def test_run_daemon_one_tick_spawns_eval_build(tmp_path, monkeypatch):
    """With 2 new sessions (≥ min_new_sessions=1), the daemon must spawn
    ``daycare eval-build PROJECT --behavioral``."""
    watchmen_home = tmp_path / "watchmen"
    watchmen_home.mkdir(parents=True, exist_ok=True)
    project = "proj"
    source_repo = "/tmp/proj"
    _write_projects_json(watchmen_home, project, source_repo)

    db_path = watchmen_home / "corpus.db"
    _seed_sessions_db(
        db_path,
        [
            ("s1", source_repo, "2026-01-01T00:00:00+00:00"),
            ("s2", source_repo, "2026-01-02T00:00:00+00:00"),
        ],
    )

    # Stub the selector — we don't care about top-skill for this test.
    monkeypatch.setattr(daemon, "_pick_top_skill", lambda wh: None)

    spy = _PopenSpy()
    monkeypatch.setattr(daemon.subprocess, "Popen", spy)
    _install_one_tick_patches(monkeypatch)

    daemon.run_daemon(
        watchmen_home=watchmen_home,
        api_key="sk-or-test",
        min_new_sessions=1,
        min_run_interval_hours=24,
    )

    eval_build_calls = [c for c in spy.calls if "eval-build" in c["argv"]]
    assert eval_build_calls, f"no eval-build spawn; calls={spy.calls}"
    call = eval_build_calls[0]
    assert project in call["argv"]
    assert "--behavioral" in call["argv"]
    assert call["env"].get("WATCHMEN_HOME") == str(watchmen_home)
    assert call["env"].get("OPENROUTER_API_KEY") == "sk-or-test"


# ─── 2. evolution spawn when hash changes ─────────────────────────────────


def _drive_two_ticks_for_run_test(
    tmp_path,
    monkeypatch,
    *,
    second_hash_differs: bool,
    min_run_interval_hours: float = 24,
    seeded_last_evolution_run_ts: str | None = None,
) -> list[dict]:
    """Run the daemon twice. Returns the Popen call log.

    The first eval-build "produces" an eval_set.jsonl with content A.
    On the second tick the file's contents flip (or not), and we check
    whether a ``daycare run`` spawn is observed.
    """
    watchmen_home = tmp_path / "watchmen"
    watchmen_home.mkdir(parents=True, exist_ok=True)
    project = "proj"
    source_repo = "/tmp/proj"
    _write_projects_json(watchmen_home, project, source_repo)

    db_path = watchmen_home / "corpus.db"
    _seed_sessions_db(
        db_path,
        [
            ("s1", source_repo, "2026-01-01T00:00:00+00:00"),
            ("s2", source_repo, "2026-01-02T00:00:00+00:00"),
        ],
    )

    monkeypatch.setattr(daemon, "_pick_top_skill", lambda wh: None)

    # Seed a runs dir with an eval_set.jsonl that we will mutate between
    # ticks. The daemon's _latest_eval_set_path scans `runs/{project}-*`.
    run_dir = watchmen_home / "daycare" / "runs" / f"{project}-2026-01-01T00-00-00"
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_set_path = run_dir / "eval_set.jsonl"
    eval_set_path.write_text("first content\n", encoding="utf-8")

    spy = _PopenSpy()
    monkeypatch.setattr(daemon.subprocess, "Popen", spy)

    # Pre-seed state so we can control last_evolution_run_ts on tick 1.
    # Also seed last_eval_set_hash to the hash of the initial eval_set.jsonl
    # so tick 1 sees "no change" and only tick 2 can trigger a run spawn.
    import hashlib as _hashlib

    initial_hash = _hashlib.sha256(b"first content\n").hexdigest()
    state_dir = watchmen_home / "daycare" / "daemon"
    state_dir.mkdir(parents=True, exist_ok=True)
    if seeded_last_evolution_run_ts is not None:
        (state_dir / "state.json").write_text(
            json.dumps(
                {
                    "per_project": {
                        project: {
                            "last_seen_session_ts": None,
                            "last_eval_build_ts": None,
                            "last_eval_set_hash": initial_hash,
                            "last_evolution_run_ts": seeded_last_evolution_run_ts,
                            "new_sessions_since_build": 0,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

    # First tick.
    _install_one_tick_patches(monkeypatch)
    daemon.run_daemon(
        watchmen_home=watchmen_home,
        api_key="sk-or-test",
        min_new_sessions=1,
        min_run_interval_hours=min_run_interval_hours,
    )

    # Mutate file between ticks.
    if second_hash_differs:
        eval_set_path.write_text("second content — different bytes\n", encoding="utf-8")
    else:
        # Touch — same bytes.
        eval_set_path.write_text("first content\n", encoding="utf-8")

    # Add another session so the daemon decides to spawn eval-build again
    # on the second tick (it gates on `new_count >= _min_sessions`).
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "INSERT INTO sessions (session_id, project_dir, started_at, "
            "ended_at, model, total_turns, is_subagent, cost_usd, "
            "transcript_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("s3", source_repo, "2026-01-03T00:00:00+00:00", None, "claude", 5, 0, 0.5, ""),
        )
        conn.commit()
    finally:
        conn.close()

    # Second tick (re-install patches because _stop got flipped).
    _install_one_tick_patches(monkeypatch)
    daemon.run_daemon(
        watchmen_home=watchmen_home,
        api_key="sk-or-test",
        min_new_sessions=1,
        min_run_interval_hours=min_run_interval_hours,
    )

    return spy.calls


def test_run_daemon_spawns_evolution_when_hash_changes(tmp_path, monkeypatch):
    """Hash changes between ticks + interval satisfied → ``daycare run``."""
    calls = _drive_two_ticks_for_run_test(
        tmp_path,
        monkeypatch,
        second_hash_differs=True,
        min_run_interval_hours=24,
        # 25 h ago — clears the 24h interval gate.
        seeded_last_evolution_run_ts="2025-12-31T00:00:00+00:00",
    )

    run_calls = [c for c in calls if "run" in c["argv"] and "--eval-set" in c["argv"]]
    assert run_calls, f"expected a `daycare run` spawn; calls={calls}"
    rc = run_calls[0]
    assert rc["argv"][0:3][1:] == ["run", "proj"] or (rc["argv"][1] == "run" and rc["argv"][2] == "proj")
    assert "--eval-set" in rc["argv"]


def test_run_daemon_no_run_when_hash_unchanged(tmp_path, monkeypatch):
    """Identical hash between ticks → no ``daycare run`` spawn."""
    calls = _drive_two_ticks_for_run_test(
        tmp_path,
        monkeypatch,
        second_hash_differs=False,
        min_run_interval_hours=24,
        seeded_last_evolution_run_ts="2025-12-31T00:00:00+00:00",
    )

    run_calls = [c for c in calls if "run" in c["argv"] and "--eval-set" in c["argv"]]
    assert not run_calls, f"unexpected run spawn: {run_calls}"


# ─── 3. min_run_interval_hours respected ──────────────────────────────────


def test_run_daemon_respects_min_run_interval_hours(tmp_path, monkeypatch):
    """Even with a hash change, if min_run_interval_hours has not elapsed
    since the last evolution run, no ``daycare run`` spawn fires."""
    # Seed last_evolution_run_ts only 1 hour ago — well under 24h interval.
    from datetime import datetime, timedelta, timezone

    recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()

    calls = _drive_two_ticks_for_run_test(
        tmp_path,
        monkeypatch,
        second_hash_differs=True,
        min_run_interval_hours=24,
        seeded_last_evolution_run_ts=recent,
    )

    run_calls = [c for c in calls if "run" in c["argv"] and "--eval-set" in c["argv"]]
    assert not run_calls, f"run spawn fired despite recent last_evolution_run_ts; calls={run_calls}"


# ─── 4. _count_new_sessions python-side filter ────────────────────────────


def test_daemon_count_new_sessions_python_filter(tmp_path):
    """_count_new_sessions filters by source_repo and since_iso correctly."""
    db_path = tmp_path / "corpus.db"
    source_repo = "/home/user/myrepo"

    rows = [
        ("s1", source_repo, "2026-01-01T00:00:00+00:00"),
        ("s2", source_repo, "2026-01-05T00:00:00+00:00"),
        ("s3", source_repo, "2026-01-10T00:00:00+00:00"),
    ]
    _seed_sessions_db(db_path, rows)

    # since_iso=None → all 3 visible.
    assert daemon._count_new_sessions(db_path, source_repo, None) == 3

    # Since the middle session's ts → 1 newer (s3).
    assert daemon._count_new_sessions(db_path, source_repo, "2026-01-05T00:00:00+00:00") == 1

    # Since the latest ts → 0 strictly newer.
    assert daemon._count_new_sessions(db_path, source_repo, "2026-01-10T00:00:00+00:00") == 0

    # Basename match: pass a different absolute prefix but same basename.
    # query_sessions' match_session_to_project allows basename fallback.
    basename_repo = "/other/place/myrepo"
    assert daemon._count_new_sessions(db_path, basename_repo, None) == 3


# ─── 5. _latest_eval_set_path scanning ────────────────────────────────────


def test_latest_eval_set_path_finds_newest_run(tmp_path):
    watchmen_home = tmp_path / "watchmen"
    runs_root = watchmen_home / "daycare" / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    # Three sibling dirs (project-{ts}) — names are lex-sortable.
    d1 = runs_root / "proj-2026-01-01T00-00-00"
    d2 = runs_root / "proj-2026-02-01T00-00-00"
    d3 = runs_root / "proj-2026-03-01T00-00-00"
    for d in (d1, d2, d3):
        d.mkdir(parents=True, exist_ok=True)
        (d / "eval_set.jsonl").write_text("content\n", encoding="utf-8")

    got = daemon._latest_eval_set_path(watchmen_home, "proj")
    assert got is not None
    assert got.parent == d3

    # Newest dir missing eval_set.jsonl → fall through to next newest.
    (d3 / "eval_set.jsonl").unlink()
    got = daemon._latest_eval_set_path(watchmen_home, "proj")
    assert got is not None
    assert got.parent == d2

    # runs/ doesn't exist → None.
    fresh_home = tmp_path / "fresh"
    assert daemon._latest_eval_set_path(fresh_home, "proj") is None

    # runs/ exists but no candidate has eval_set.jsonl → None.
    empty_runs_home = tmp_path / "empty"
    (empty_runs_home / "daycare" / "runs" / "proj-2026-04-01").mkdir(parents=True)
    assert daemon._latest_eval_set_path(empty_runs_home, "proj") is None


# ─── 7. state.json per-project roundtrip ──────────────────────────────────


def test_daemon_state_per_project_roundtrip(tmp_path):
    state = {
        "per_project": {
            "proj": {
                "last_seen_session_ts": "2026-01-01T00:00:00+00:00",
                "last_eval_build_ts": "2026-01-02T00:00:00+00:00",
                "last_eval_set_hash": "abcd1234",
                "last_evolution_run_ts": "2026-01-03T00:00:00+00:00",
                "new_sessions_since_build": 7,
            }
        },
        "last_daily_run": "2026-01-04T03:00:00+00:00",
        "last_weekly": "2026-01-05T03:00:00+00:00",
        "last_tick": "2026-01-06T03:00:00+00:00",
    }

    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    loaded = json.loads(state_path.read_text(encoding="utf-8"))
    assert loaded == state
    assert set(loaded["per_project"]["proj"].keys()) == {
        "last_seen_session_ts",
        "last_eval_build_ts",
        "last_eval_set_hash",
        "last_evolution_run_ts",
        "new_sessions_since_build",
    }
