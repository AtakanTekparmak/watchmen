"""SkillDoc.auxiliary_files + SkillFolder load/write round-trip.

Phase 0 of the code-bearing skills work (2026-04-22): SkillFolder.load
must pick up files under hermes's 4 recognised subdirs (scripts/,
references/, templates/, assets/), and SkillFolder.write must put them
back with executable bit on scripts/*.sh|*.py.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from skill_evolve.track_a.folder import SkillFolder


def _write_skill(root: Path, name: str) -> Path:
    sdir = root / name
    sdir.mkdir(parents=True)
    (sdir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: d\n---\nbody\n",
        encoding="utf-8",
    )
    return sdir


def test_load_empty_skill_dir_has_no_aux(tmp_path: Path) -> None:
    _write_skill(tmp_path, "solo")
    f = SkillFolder.load(tmp_path)
    assert len(f.skills) == 1
    assert f.skills[0].auxiliary_files == {}


def test_load_picks_up_hermes_subdirs(tmp_path: Path) -> None:
    sdir = _write_skill(tmp_path, "alpha")
    (sdir / "scripts").mkdir()
    (sdir / "scripts" / "hello.sh").write_text("#!/bin/sh\necho hi\n")
    (sdir / "references").mkdir()
    (sdir / "references" / "spec.md").write_text("# spec\n")
    (sdir / "templates").mkdir()
    (sdir / "templates" / "config.yaml").write_text("key: value\n")
    (sdir / "assets").mkdir()
    (sdir / "assets" / "data.json").write_text('{"k": "v"}\n')

    f = SkillFolder.load(tmp_path)
    aux = f.skills[0].auxiliary_files
    assert set(aux) == {
        "scripts/hello.sh",
        "references/spec.md",
        "templates/config.yaml",
        "assets/data.json",
    }
    assert aux["scripts/hello.sh"] == "#!/bin/sh\necho hi\n"


def test_load_ignores_non_recognised_subdirs(tmp_path: Path) -> None:
    # A ``tests/`` or ``draft/`` subdir must NOT be treated as auxiliary —
    # they're not hermes-recognised and shouldn't pollute the artifact.
    sdir = _write_skill(tmp_path, "alpha")
    (sdir / "tests").mkdir()
    (sdir / "tests" / "stuff.py").write_text("# ignored\n")
    (sdir / "draft").mkdir()
    (sdir / "draft" / "notes.md").write_text("# ignored\n")

    f = SkillFolder.load(tmp_path)
    assert f.skills[0].auxiliary_files == {}


def test_load_skips_binary_files(tmp_path: Path) -> None:
    sdir = _write_skill(tmp_path, "alpha")
    (sdir / "assets").mkdir()
    (sdir / "assets" / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")

    f = SkillFolder.load(tmp_path)
    assert f.skills[0].auxiliary_files == {}


def test_load_walks_nested_aux_subdirs(tmp_path: Path) -> None:
    # e.g. scripts/utils/helper.py should be picked up too.
    sdir = _write_skill(tmp_path, "alpha")
    (sdir / "scripts" / "utils").mkdir(parents=True)
    (sdir / "scripts" / "utils" / "helper.py").write_text("def f(): pass\n")

    f = SkillFolder.load(tmp_path)
    assert "scripts/utils/helper.py" in f.skills[0].auxiliary_files


def test_write_materializes_aux_and_sets_exec_bit(tmp_path: Path) -> None:
    src = tmp_path / "src"
    sdir = _write_skill(src, "alpha")
    (sdir / "scripts").mkdir()
    (sdir / "scripts" / "run.sh").write_text("#!/bin/sh\necho ok\n")
    (sdir / "scripts" / "run.py").write_text("#!/usr/bin/env python3\nprint('ok')\n")
    (sdir / "references").mkdir()
    (sdir / "references" / "doc.md").write_text("# doc\n")

    f = SkillFolder.load(src)

    dest = tmp_path / "dest"
    f.write(dest)

    sh_path = dest / "alpha" / "scripts" / "run.sh"
    py_path = dest / "alpha" / "scripts" / "run.py"
    ref_path = dest / "alpha" / "references" / "doc.md"

    assert sh_path.read_text() == "#!/bin/sh\necho ok\n"
    # scripts/*.sh and scripts/*.py must be +x.
    assert sh_path.stat().st_mode & stat.S_IXUSR
    assert py_path.stat().st_mode & stat.S_IXUSR
    # references/*.md must NOT be +x (read-only data).
    assert not (ref_path.stat().st_mode & stat.S_IXUSR)


def test_total_bytes_includes_aux(tmp_path: Path) -> None:
    sdir = _write_skill(tmp_path, "alpha")
    (sdir / "scripts").mkdir()
    (sdir / "scripts" / "s.sh").write_text("a" * 1000)

    f = SkillFolder.load(tmp_path)
    total = f.total_bytes()
    # SKILL.md is ~40 bytes; the aux file is 1000.
    assert total >= 1000
    assert total < 1500  # sanity — not absurdly high


def test_clone_deep_copies_aux(tmp_path: Path) -> None:
    sdir = _write_skill(tmp_path, "alpha")
    (sdir / "scripts").mkdir()
    (sdir / "scripts" / "s.sh").write_text("orig\n")

    f = SkillFolder.load(tmp_path)
    clone = f.skills[0].clone()
    clone.auxiliary_files["scripts/s.sh"] = "mutated\n"

    # Original untouched.
    assert f.skills[0].auxiliary_files["scripts/s.sh"] == "orig\n"
    assert clone.auxiliary_files["scripts/s.sh"] == "mutated\n"


def test_render_summary_lists_aux_paths(tmp_path: Path) -> None:
    sdir = _write_skill(tmp_path, "alpha")
    (sdir / "scripts").mkdir()
    (sdir / "scripts" / "tool.sh").write_text("#!/bin/sh\n")

    f = SkillFolder.load(tmp_path)
    out = f.render_summary()
    assert "auxiliary_files:" in out
    assert "scripts/tool.sh" in out
