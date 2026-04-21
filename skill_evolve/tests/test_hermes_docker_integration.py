"""Integration tests for the Hermes Docker backend wiring.

Covers three concerns from the integration plan:

1. ``test_double_mount_invariance`` (``@pytest.mark.docker``, opt-in)
   Spins up an ``alpine:3`` container with the SAME host dir bound at
   both ``/app`` and ``/workspace`` and proves edits in one target are
   visible at the other. This is the invariant that makes the double-
   mount trick in ``hermes_docker.build_env_patch`` actually work.

2. ``test_env_patch_merged_into_sandbox_env``
   When ``_run_one_task`` runs with ``verify=True`` and docker is available,
   the ``TERMINAL_*`` keys from ``hermes_docker.build_env_patch`` must make
   it into the subprocess env that ``run_agent.py`` is launched with.
   This is the integration contract the plan's whole point hinges on.

3. ``test_no_patch_on_no_verify``
   The inverse: with ``verify=False`` the hermes_docker functions must
   NOT be called at all — the local backend path stays unaffected so
   ``--no-verify`` fast plumbing runs don't need Docker.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from skill_evolve import evaluator, hermes_docker


# ---------------------------------------------------------------------------
# 1. Real-docker: double-mount invariance (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.docker
def test_double_mount_invariance(tmp_path: Path) -> None:
    """Same host dir mounted at /app and /workspace shows file in both views."""
    if shutil.which("docker") is None:
        pytest.skip("docker CLI not present")

    cname = f"skillevolve-itest-{uuid.uuid4().hex[:10]}"
    host = str(tmp_path.resolve())

    # Start a long-lived container with the same host dir double-mounted.
    run = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            cname,
            "-v",
            f"{host}:/app",
            "-v",
            f"{host}:/workspace",
            "alpine:3",
            "sleep",
            "30",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if run.returncode != 0:
        # Don't trust CI docker availability — bail rather than fail hard.
        pytest.skip(f"docker run failed: {run.stderr[-200:]}")

    try:
        # Write at /app, read at /workspace.
        t1 = subprocess.run(
            ["docker", "exec", cname, "sh", "-c", "touch /app/probe.txt"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert t1.returncode == 0, t1.stderr
        r1 = subprocess.run(
            ["docker", "exec", cname, "test", "-f", "/workspace/probe.txt"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert r1.returncode == 0, (
            f"file touched at /app not visible at /workspace: {r1.stderr}"
        )

        # Symmetric: write at /workspace, read at /app.
        t2 = subprocess.run(
            ["docker", "exec", cname, "sh", "-c", "touch /workspace/probe_reverse.txt"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert t2.returncode == 0, t2.stderr
        r2 = subprocess.run(
            ["docker", "exec", cname, "test", "-f", "/app/probe_reverse.txt"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert r2.returncode == 0, (
            f"file touched at /workspace not visible at /app: {r2.stderr}"
        )
    finally:
        subprocess.run(
            ["docker", "rm", "-f", cname],
            capture_output=True,
            check=False,
        )


# ---------------------------------------------------------------------------
# 2 + 3. Evaluator wiring (no real docker required)
# ---------------------------------------------------------------------------


def _fake_task() -> Dict[str, Any]:
    return {
        "task_id": "tblite/fake-task",
        "prompt": "do the thing",
        "timeout_s": 60,
        "stage": 1,
        "success_check_payload": {
            "docker_image": "nousresearch/tblite-fake:latest",
        },
    }


class _FakeVerifyResult:
    passed = None
    status = "not_run"
    detail = ""


def _run_with_mocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    verify: bool,
    docker_available: bool = True,
) -> List[Dict[str, str]]:
    """Drive ``_run_one_task`` with every external dependency mocked.

    Returns the list of env dicts that the patched ``subprocess.run`` was
    called with — tests inspect these to confirm the env patch made it
    into the child process launch.
    """
    # Point sandbox tmp root at an ephemeral dir so we don't pollute the
    # user's ~/.cache/skill_evolve/.
    monkeypatch.setenv("SKILL_EVOLVE_TMP_ROOT", str(tmp_path / "sbx"))

    # Fake skill folder — sandbox refuses to run without a real dir.
    skills_folder = tmp_path / "skills"
    skills_folder.mkdir()

    # Docker backend stubs.
    monkeypatch.setattr(
        hermes_docker,
        "docker_backend_available",
        lambda: docker_available,
    )
    pull_mock = MagicMock()
    monkeypatch.setattr(hermes_docker, "ensure_image_pulled", pull_mock)

    # Spy on build_env_patch so test 3 can assert .call_count.
    orig_build = hermes_docker.build_env_patch
    build_spy = MagicMock(side_effect=orig_build)
    monkeypatch.setattr(hermes_docker, "build_env_patch", build_spy)

    # Skip real staging (needs Docker to pull /app out of the task image).
    monkeypatch.setattr(
        evaluator,
        "stage_workspace",
        lambda task, ws: Path(ws).mkdir(parents=True, exist_ok=True) or {},
    )
    # Short-circuit verify_task so we don't try to talk to a real daemon.
    monkeypatch.setattr(
        evaluator,
        "verify_task",
        lambda task, run_dir: _FakeVerifyResult(),
    )

    # Capture subprocess.run env dicts without actually spawning run_agent.
    captured: List[Dict[str, str]] = []

    def fake_run(cmd, *args, env=None, **kwargs):  # noqa: ARG001
        captured.append(dict(env or {}))
        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = ""
        mock.stderr = ""
        return mock

    monkeypatch.setattr(evaluator.subprocess, "run", fake_run)

    # Stash the spy on the caller so it can assert call_count later.
    _run_with_mocks.last_build_spy = build_spy  # type: ignore[attr-defined]
    _run_with_mocks.last_pull_mock = pull_mock  # type: ignore[attr-defined]

    evaluator._run_one_task(
        _fake_task(),
        skills_folder,
        model="fake/model",
        max_turns=1,
        enabled_toolsets="terminal",
        verify=verify,
    )
    return captured


def test_env_patch_merged_into_sandbox_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """verify=True → TERMINAL_* keys from build_env_patch reach the subprocess env."""
    captured = _run_with_mocks(monkeypatch, tmp_path, verify=True)

    assert len(captured) == 1, "run_agent.py should be spawned exactly once"
    env = captured[0]

    assert env.get("TERMINAL_ENV") == "docker"
    assert env.get("TERMINAL_DOCKER_IMAGE") == "nousresearch/tblite-fake:latest"
    assert env.get("TERMINAL_CWD") == "/app"
    assert env.get("TERMINAL_CONTAINER_PERSISTENT") == "false"

    volumes_raw = env.get("TERMINAL_DOCKER_VOLUMES", "")
    assert ":/app" in volumes_raw
    assert ":/workspace" in volumes_raw

    # build_env_patch was actually invoked (vs. picked up from stale env).
    assert _run_with_mocks.last_build_spy.call_count == 1  # type: ignore[attr-defined]
    assert _run_with_mocks.last_pull_mock.call_count == 1  # type: ignore[attr-defined]


def test_no_patch_on_no_verify(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """verify=False → hermes_docker is never called, env stays patch-free."""
    captured = _run_with_mocks(monkeypatch, tmp_path, verify=False)

    assert len(captured) == 1
    env = captured[0]
    assert "TERMINAL_ENV" not in env
    assert "TERMINAL_DOCKER_IMAGE" not in env
    assert "TERMINAL_DOCKER_VOLUMES" not in env

    # Crucially: build_env_patch and ensure_image_pulled never invoked.
    assert _run_with_mocks.last_build_spy.call_count == 0  # type: ignore[attr-defined]
    assert _run_with_mocks.last_pull_mock.call_count == 0  # type: ignore[attr-defined]


def test_docker_setup_failure_falls_back(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """ensure_image_pulled exception must NOT crash _run_one_task.

    The evaluator should log, continue in local-backend mode, and surface
    the failure reason via TaskOutcome.notes so operators can tell why a
    given rollout wasn't containerized.
    """
    monkeypatch.setenv("SKILL_EVOLVE_TMP_ROOT", str(tmp_path / "sbx"))
    skills_folder = tmp_path / "skills"
    skills_folder.mkdir()

    monkeypatch.setattr(hermes_docker, "docker_backend_available", lambda: True)

    def boom(image, timeout_s=300):  # noqa: ARG001
        raise RuntimeError("simulated pull failure")

    monkeypatch.setattr(hermes_docker, "ensure_image_pulled", boom)
    monkeypatch.setattr(
        evaluator,
        "stage_workspace",
        lambda task, ws: Path(ws).mkdir(parents=True, exist_ok=True) or {},
    )
    monkeypatch.setattr(
        evaluator,
        "verify_task",
        lambda task, run_dir: _FakeVerifyResult(),
    )

    def fake_run(cmd, *args, env=None, **kwargs):  # noqa: ARG001
        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = ""
        mock.stderr = ""
        return mock

    monkeypatch.setattr(evaluator.subprocess, "run", fake_run)

    outcome = evaluator._run_one_task(
        _fake_task(),
        skills_folder,
        model="fake/model",
        max_turns=1,
        enabled_toolsets="terminal",
        verify=True,
    )

    # Evaluator stayed alive and recorded the setup failure in notes.
    assert "docker_setup_failed" in outcome.notes
