"""Unit tests for ``skill_evolve.hermes_docker``.

These tests do NOT require a running Docker daemon — ``subprocess.run`` is
monkeypatched where needed so the image-pull helpers can be exercised on
CI machines without Docker.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from skill_evolve import hermes_docker


# ---------------------------------------------------------------------------
# build_env_patch
# ---------------------------------------------------------------------------


def _synthetic_task(image: str = "foo/bar:latest") -> dict:
    return {
        "task_id": "x",
        "success_check_payload": {"docker_image": image},
    }


def test_build_env_patch_shape() -> None:
    """All six TERMINAL_* keys are present with the expected string values."""
    task = _synthetic_task("foo/bar:latest")
    patch = hermes_docker.build_env_patch(task, Path("/some/ws"))

    assert patch["TERMINAL_ENV"] == "docker"
    assert patch["TERMINAL_DOCKER_IMAGE"] == "foo/bar:latest"
    assert patch["TERMINAL_CWD"] == "/app"
    assert patch["TERMINAL_CONTAINER_PERSISTENT"] == "false"
    # Present + JSON-shaped (content asserted in dedicated tests below).
    assert "TERMINAL_DOCKER_VOLUMES" in patch
    assert "TERMINAL_DOCKER_FORWARD_ENV" in patch
    # All values are strings (subprocess env dict contract).
    for k, v in patch.items():
        assert isinstance(v, str), f"{k}={v!r} is not a str"


def test_build_env_patch_volumes_json() -> None:
    """TERMINAL_DOCKER_VOLUMES decodes to a 2-element list double-mounting ws."""
    task = _synthetic_task()
    patch = hermes_docker.build_env_patch(task, Path("/some/ws"))
    volumes = json.loads(patch["TERMINAL_DOCKER_VOLUMES"])

    assert isinstance(volumes, list)
    assert len(volumes) == 2

    abs_ws = str(Path("/some/ws").resolve())
    # One entry per expected container target.
    assert any(v.endswith(":/app") for v in volumes)
    assert any(v.endswith(":/workspace") for v in volumes)
    # Host side of each spec is the absolute workspace path.
    for spec in volumes:
        host, _, _ = spec.rpartition(":")
        assert host == abs_ws, f"host side of {spec!r} != {abs_ws!r}"

    # And FORWARD_ENV is JSON list containing OPENROUTER_API_KEY.
    fwd = json.loads(patch["TERMINAL_DOCKER_FORWARD_ENV"])
    assert isinstance(fwd, list)
    assert "OPENROUTER_API_KEY" in fwd


def test_build_env_patch_uses_abs_path(tmp_path: Path) -> None:
    """Relative workspace path is resolved to absolute in the volume spec."""
    # Construct a genuinely relative path — cwd-dependent, not pre-resolved.
    rel = Path("some_ws_rel_dir")
    patch = hermes_docker.build_env_patch(_synthetic_task(), rel)
    volumes = json.loads(patch["TERMINAL_DOCKER_VOLUMES"])
    expected_abs = str(rel.resolve())
    for spec in volumes:
        host, _, _ = spec.rpartition(":")
        assert host == expected_abs
        # Sanity: the expanded path is absolute.
        assert Path(host).is_absolute()


def test_build_env_patch_missing_image_raises() -> None:
    """Task without docker_image in payload raises ValueError."""
    task = {"task_id": "x", "success_check_payload": {}}
    with pytest.raises(ValueError, match="docker_image"):
        hermes_docker.build_env_patch(task, Path("/some/ws"))

    # Also the nothing-at-all case.
    task2 = {"task_id": "x"}
    with pytest.raises(ValueError, match="docker_image"):
        hermes_docker.build_env_patch(task2, Path("/some/ws"))


# ---------------------------------------------------------------------------
# ensure_image_pulled
# ---------------------------------------------------------------------------


def test_ensure_image_pulled_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``docker image inspect`` returns 0, we DO NOT call ``docker pull``."""
    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        # Inspect returns 0 — image already cached.
        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = ""
        mock.stderr = ""
        return mock

    monkeypatch.setattr(hermes_docker.subprocess, "run", fake_run)

    result = hermes_docker.ensure_image_pulled("foo:latest")
    assert result is None

    # Exactly one call — the inspect probe. No pull.
    assert len(calls) == 1
    assert calls[0][:3] == ["docker", "image", "inspect"]
    pulled = [c for c in calls if len(c) >= 2 and c[:2] == ["docker", "pull"]]
    assert pulled == [], f"unexpected docker pull call(s): {pulled}"


def test_ensure_image_pulled_missing_then_pulled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing image → inspect returns 1, then pull returns 0, in order."""
    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        mock = MagicMock()
        if cmd[:3] == ["docker", "image", "inspect"]:
            mock.returncode = 1
            mock.stdout = ""
            mock.stderr = "no such image"
        elif cmd[:2] == ["docker", "pull"]:
            mock.returncode = 0
            mock.stdout = "pulled"
            mock.stderr = ""
        else:
            mock.returncode = 0
        return mock

    monkeypatch.setattr(hermes_docker.subprocess, "run", fake_run)

    hermes_docker.ensure_image_pulled("foo:latest")

    # Inspect then pull, in that order.
    assert len(calls) == 2
    assert calls[0][:3] == ["docker", "image", "inspect"]
    assert calls[0][3] == "foo:latest"
    assert calls[1][:2] == ["docker", "pull"]
    assert calls[1][2] == "foo:latest"


def test_ensure_image_pulled_pull_failure_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pull failure surfaces as RuntimeError with stderr tail in the message."""

    def fake_run(cmd, *args, **kwargs):
        mock = MagicMock()
        if cmd[:3] == ["docker", "image", "inspect"]:
            mock.returncode = 1
        elif cmd[:2] == ["docker", "pull"]:
            mock.returncode = 1
            mock.stdout = ""
            mock.stderr = "manifest not found: nope/nope:latest"
        return mock

    monkeypatch.setattr(hermes_docker.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="manifest not found"):
        hermes_docker.ensure_image_pulled("nope/nope:latest")
