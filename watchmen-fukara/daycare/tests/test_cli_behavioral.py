"""Behavioral flag CLI tests for `eval-build` and `run`.

Verifies:
  - --behavioral and --synthetic are mutually exclusive in BOTH commands.
  - The mutual-exclusion error message is byte-identical between the two
    commands.
  - --behavioral defaults to True for `eval-build` (i.e. the implicit
    default routes through the behavioral path).
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from daycare.cli import eval_build, run as run_cmd


_EXPECTED_MSG = "--behavioral and --synthetic are mutually exclusive"


def _write_projects_json(watchmen_home: Path, project: str) -> None:
    watchmen_home.mkdir(parents=True, exist_ok=True)
    payload = [{"project_key": project, "source_repo": f"/tmp/{project}"}]
    (watchmen_home / "projects.json").write_text(json.dumps(payload), encoding="utf-8")


# ─── Mutual exclusion ─────────────────────────────────────────────────────


def test_eval_build_behavioral_and_synthetic_mutually_exclusive(tmp_path, monkeypatch):
    """`eval-build --behavioral --synthetic` must fail with the canonical msg."""
    monkeypatch.setenv("WATCHMEN_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    _write_projects_json(tmp_path, "proj")

    runner = CliRunner()
    res = runner.invoke(eval_build, ["proj", "--behavioral", "--synthetic"])
    assert res.exit_code != 0
    assert _EXPECTED_MSG in res.output


def test_run_behavioral_and_synthetic_mutually_exclusive(tmp_path, monkeypatch):
    """`run --behavioral --synthetic` must fail with the IDENTICAL message."""
    monkeypatch.setenv("WATCHMEN_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    _write_projects_json(tmp_path, "proj")

    runner = CliRunner()
    res = runner.invoke(run_cmd, ["proj", "--behavioral", "--synthetic"])
    assert res.exit_code != 0
    assert _EXPECTED_MSG in res.output


# ─── Default-is-behavioral ────────────────────────────────────────────────


class _SentinelInvocation(Exception):
    """Carries kwargs captured from a monkeypatched run_eval_build call."""

    def __init__(self, kwargs: dict) -> None:
        super().__init__("sentinel")
        self.kwargs = kwargs


def test_eval_build_default_is_behavioral(tmp_path, monkeypatch):
    """Invoking `eval-build PROJECT` (no flags) must pass behavioral=True
    through to run_eval_build.

    We monkeypatch the eval_builder.run_eval_build to raise a sentinel
    carrying its kwargs, then assert the captured ``behavioral`` arg.
    """
    monkeypatch.setenv("WATCHMEN_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    _write_projects_json(tmp_path, "proj")

    def _fake_run_eval_build(**kwargs):
        raise _SentinelInvocation(kwargs)

    monkeypatch.setattr("daycare.eval_builder.run_eval_build", _fake_run_eval_build)

    runner = CliRunner()
    res = runner.invoke(eval_build, ["proj"])
    # The sentinel exception propagates out — exit code is non-zero, but
    # the captured kwargs are what we care about.
    assert res.exception is not None, res.output
    # Find the sentinel anywhere in the chained exception list.
    exc = res.exception
    found: _SentinelInvocation | None = None
    while exc is not None:
        if isinstance(exc, _SentinelInvocation):
            found = exc
            break
        exc = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    assert found is not None, f"sentinel not raised: {res.output}"
    assert found.kwargs.get("behavioral") is True
