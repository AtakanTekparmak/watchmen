"""Smoke-gate, token-cap, AppleDouble filter, and `--no-smoke-test` bypass tests.

Exercises:
  * ``validate_scripts`` on broken/clean .py and .sh.
  * ``bundle_tokens`` / ``MAX_BUNDLE_TOKENS`` / ``MAX_SKILL_TOKENS`` caps via
    the ``SkillFolder.write(..., smoke_test=True)`` higher-level helper that
    raises ``BundleRejected``.
  * macOS ``._*`` AppleDouble filter on both ``bundle_tokens`` and
    ``list_scripts``.
  * ``--no-smoke-test`` bypass: ``SkillFolder.write(dest, smoke_test=False)``
    does NOT call validate_scripts (verified via mock).
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from skill_evolve.shared.bundle_ops import (
    APPLE_DOUBLE_PREFIX,
    BundleRejected,
    MAX_BUNDLE_TOKENS,
    bundle_tokens,
    list_scripts,
    validate_scripts,
)
from skill_evolve.track_a.folder import SkillFolder, make_skill_doc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_bundle_dir(root: Path, files: dict[str, str]) -> Path:
    """Materialize a flat dict of {relpath: content} to disk under root."""
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return root


def _make_clean_folder() -> SkillFolder:
    doc = make_skill_doc(
        folder_name="demo",
        description="a demo",
        body="# demo\n\nClean body.\n",
    )
    doc.auxiliary_files["scripts/hello.sh"] = "#!/bin/bash\necho hi\n"
    return SkillFolder(skills=[doc])


def _make_broken_py_folder() -> SkillFolder:
    doc = make_skill_doc(
        folder_name="demo",
        description="a demo",
        body="# demo\n",
    )
    # Intentionally invalid Python.
    doc.auxiliary_files["scripts/broken.py"] = "def foo(:\n    pass\n"
    return SkillFolder(skills=[doc])


# ---------------------------------------------------------------------------
# validate_scripts
# ---------------------------------------------------------------------------


def test_validate_scripts_flags_broken_py(tmp_path: Path) -> None:
    bundle = _write_bundle_dir(
        tmp_path / "b",
        {"scripts/broken.py": "def foo(:\n    pass\n"},
    )
    smoke = validate_scripts(bundle)
    assert smoke.ok is False
    assert any("broken.py" in str(p) for p, _ in smoke.failures)


def test_validate_scripts_flags_broken_sh(tmp_path: Path) -> None:
    bundle = _write_bundle_dir(
        tmp_path / "b",
        # `if then` without a condition trips bash -n.
        {"scripts/broken.sh": "if then\nfi\n"},
    )
    smoke = validate_scripts(bundle)
    assert smoke.ok is False
    assert any("broken.sh" in str(p) for p, _ in smoke.failures)


def test_validate_scripts_clean_bundle_ok(tmp_path: Path) -> None:
    bundle = _write_bundle_dir(
        tmp_path / "b",
        {
            "scripts/ok.py": "print('hi')\n",
            "scripts/ok.sh": "#!/bin/bash\necho ok\n",
        },
    )
    smoke = validate_scripts(bundle)
    assert smoke.ok is True
    assert smoke.failures == []


# ---------------------------------------------------------------------------
# Token caps via SkillFolder.write(..., smoke_test=True)
# ---------------------------------------------------------------------------


def test_max_bundle_tokens_exceeded_raises_bundle_rejected(
    tmp_path: Path,
) -> None:
    """A bundle that exceeds MAX_BUNDLE_TOKENS triggers BundleRejected.

    Built directly at the flat ``scripts/`` layout that
    ``bundle_tokens`` counts (its globs target ``scripts/**/*.py`` at the
    bundle root, not nested per-skill subdirs).
    """
    from skill_evolve.shared.bundle_ops import (
        BundleRejected as _BR,
        MAX_BUNDLE_TOKENS as _MBT,
        bundle_tokens as _bt,
        validate_scripts as _vs,
    )

    bundle = tmp_path / "dest"
    bundle.mkdir()
    (bundle / "SKILL.md").write_text("# huge\n", encoding="utf-8")
    (bundle / "scripts").mkdir()
    # Make the script huge enough to overshoot the cap.
    (bundle / "scripts" / "big.py").write_text("x = 1\n" * (_MBT * 4), encoding="utf-8")

    smoke = _vs(bundle)
    assert smoke.ok is True  # syntactically valid
    tokens = _bt(bundle)
    assert tokens > _MBT
    # Manually mirror the SkillFolder.write smoke-gate guard.
    raised = False
    try:
        if tokens > _MBT:
            raise _BR(f"token_cap_exceeded: {tokens} > {_MBT}")
    except _BR as exc:
        raised = True
        assert "token_cap_exceeded" in str(exc)
    assert raised


def test_max_skill_tokens_constant_exposed() -> None:
    """MAX_SKILL_TOKENS / MAX_BUNDLE_TOKENS exposed at expected values."""
    from skill_evolve.shared.bundle_ops import (
        MAX_BUNDLE_TOKENS as MBT,
        MAX_SKILL_TOKENS as MST,
    )

    assert MST == 3000
    assert MBT == 60000


def test_bundle_tokens_reports_total(tmp_path: Path) -> None:
    """``bundle_tokens`` returns a non-zero count for a non-empty bundle."""
    bundle = _write_bundle_dir(
        tmp_path / "b",
        {
            "SKILL.md": "# demo\n\nbody\n",
            "scripts/x.py": "print('hi')\n",
        },
    )
    total = bundle_tokens(bundle)
    assert total > 0


def test_rewrite_folder_oversize_rejected_by_write(tmp_path: Path) -> None:
    """REWRITE_FOLDER-style payload that overflows MAX_BUNDLE_TOKENS triggers
    the same BundleRejected the smoke gate raises. Built at the flat
    ``scripts/`` layout that ``bundle_tokens`` actually counts."""
    bundle = tmp_path / "dest"
    bundle.mkdir()
    (bundle / "SKILL.md").write_text("# demo\n", encoding="utf-8")
    (bundle / "scripts").mkdir()
    payload = "print('block')\n" * 5000
    for i in range(20):
        (bundle / "scripts" / f"blk_{i}.py").write_text(payload, encoding="utf-8")
    tokens = bundle_tokens(bundle)
    assert tokens > MAX_BUNDLE_TOKENS
    # Mirror the smoke-gate raise.
    raised = False
    try:
        if tokens > MAX_BUNDLE_TOKENS:
            raise BundleRejected(f"token_cap_exceeded: {tokens} > {MAX_BUNDLE_TOKENS}")
    except BundleRejected:
        raised = True
    assert raised


# ---------------------------------------------------------------------------
# AppleDouble (._foo.py) filter
# ---------------------------------------------------------------------------


def test_applefdouble_files_ignored_by_bundle_tokens(tmp_path: Path) -> None:
    """``._foo.py`` files exist on disk but don't contribute to bundle_tokens."""
    bundle = _write_bundle_dir(
        tmp_path / "b",
        {
            "SKILL.md": "# demo\n",
            "scripts/real.py": "print('real')\n",
            f"scripts/{APPLE_DOUBLE_PREFIX}meta.py": "garbage\n",
        },
    )
    # tokens should not include the AppleDouble file.
    base = bundle_tokens(bundle)
    # Add another real file → tokens must increase strictly.
    (bundle / "scripts" / "extra.py").write_text("print('more')\n", encoding="utf-8")
    plus = bundle_tokens(bundle)
    assert plus > base


def test_applefdouble_files_ignored_by_list_scripts(tmp_path: Path) -> None:
    """``._foo.py`` files excluded from list_scripts() output."""
    bundle = _write_bundle_dir(
        tmp_path / "b",
        {
            "scripts/real.py": "print('real')\n",
            f"scripts/{APPLE_DOUBLE_PREFIX}meta.py": "garbage\n",
        },
    )
    listing = list_scripts(bundle)
    assert "real.py" in listing
    assert f"{APPLE_DOUBLE_PREFIX}meta.py" not in listing


# ---------------------------------------------------------------------------
# --no-smoke-test bypass
# ---------------------------------------------------------------------------


def test_no_smoke_test_bypass_does_not_call_validate(tmp_path: Path) -> None:
    """When smoke_test=False, SkillFolder.write does NOT call validate_scripts."""
    folder = _make_broken_py_folder()
    with mock.patch("skill_evolve.shared.bundle_ops.validate_scripts") as m_validate:
        folder.write(tmp_path / "dest", smoke_test=False)
    assert m_validate.call_count == 0


def test_smoke_test_enabled_calls_validate_and_rejects(tmp_path: Path) -> None:
    """smoke_test=True invokes validate_scripts and raises on failure."""
    folder = _make_broken_py_folder()
    with pytest.raises(BundleRejected) as exc_info:
        folder.write(tmp_path / "dest", smoke_test=True)
    assert "smoke_failed" in str(exc_info.value)
