"""Tests for ``_resolve_tmp_root`` and its integration with ``make_sandbox``.

Motivation: on macOS, Docker Desktop's default VirtioFS shared-path set does
NOT include ``/tmp`` (nor the macOS ``/var/folders/...`` that
``tempfile.gettempdir()`` resolves to). Bind-mounting those paths into a
container silently yields empty dirs, which breaks our evaluator's workspace
handshake. This module pins the sandbox root to a ``/Users/``-rooted path by
default on Darwin, while preserving Linux behavior and honoring explicit
caller-supplied or env-var overrides.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from skill_evolve.sandbox import _resolve_tmp_root, make_sandbox


def test_resolve_explicit_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``tmp_root`` argument short-circuits all other resolution."""
    # Even with env var set and on Darwin, explicit wins.
    monkeypatch.setenv("SKILL_EVOLVE_TMP_ROOT", "/should/not/be/used")
    monkeypatch.setattr(sys, "platform", "darwin")
    explicit = Path("/custom/root")
    assert _resolve_tmp_root(explicit) == explicit


def test_resolve_env_var_honored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``SKILL_EVOLVE_TMP_ROOT`` overrides the platform default."""
    target = tmp_path / "evroot"
    monkeypatch.setenv("SKILL_EVOLVE_TMP_ROOT", str(target))
    # Force darwin to confirm env var beats the platform default branch.
    monkeypatch.setattr(sys, "platform", "darwin")

    resolved = _resolve_tmp_root(None)
    assert resolved == target
    assert resolved.is_dir(), "helper must create the env-var-specified dir"


def test_resolve_darwin_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """On macOS without overrides, default to ``~/.cache/skill_evolve``."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("SKILL_EVOLVE_TMP_ROOT", raising=False)

    resolved = _resolve_tmp_root(None)
    expected = Path.home() / ".cache" / "skill_evolve"
    assert resolved == expected
    # And specifically NOT a Docker-Desktop-unshared path.
    resolved_str = str(resolved)
    assert not resolved_str.startswith("/tmp"), resolved_str
    assert not resolved_str.startswith("/var/folders"), resolved_str
    assert resolved.is_dir(), "helper must create the darwin default dir"


def test_resolve_linux_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """On non-Darwin platforms without overrides, preserve legacy behavior."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("SKILL_EVOLVE_TMP_ROOT", raising=False)

    resolved = _resolve_tmp_root(None)
    assert resolved == Path(tempfile.gettempdir())


def test_make_sandbox_uses_resolved_root_on_darwin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``make_sandbox`` without an explicit ``tmp_root`` lands under the
    Darwin-default cache dir (and NOT under ``/tmp``)."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("SKILL_EVOLVE_TMP_ROOT", raising=False)

    # Build a minimal throwaway skill folder so make_sandbox's existence
    # check passes without depending on the real seed_skills dir.
    skills_dir = tmp_path / "fake_skills"
    (skills_dir / "demo").mkdir(parents=True)
    (skills_dir / "demo" / "SKILL.md").write_text(
        "---\nname: demo\ndescription: demo\n---\nbody\n",
        encoding="utf-8",
    )

    handle = make_sandbox(skills_dir, run_id="darwinroot")
    try:
        expected_root = Path.home() / ".cache" / "skill_evolve"
        # The handle.home directory is ``<expected_root>/hermes-eval-<rid>``.
        assert handle.home.parent == expected_root
        assert handle.home.parent.exists(), (
            "helper contract: darwin default root must be auto-created"
        )
        home_str = str(handle.home)
        assert not home_str.startswith("/tmp"), home_str
        assert not home_str.startswith("/var/folders"), home_str
    finally:
        shutil.rmtree(handle.home, ignore_errors=True)
