"""Group G — deployment strip excludes meta_skill.md (plan §7l).

Coverage:
* ``copy_for_deployment`` excludes ``meta_skill.md`` at bundle root.
* Other files are byte-identical in dst.
* AppleDouble ``._*`` and ``__pycache__/`` and ``*.pyc`` are excluded.
* ``FolderArtifact.write_to(..., deployment=True)`` strips meta_skill.md.
"""

from __future__ import annotations

from pathlib import Path

from skill_evolve.shared.bundle_ops import copy_for_deployment
from skill_evolve.track_b.openevolve_skills.folder_artifact import (
    FolderArtifact,
)


def _make_fixture_bundle(root: Path) -> dict[str, bytes]:
    """Build a fixture bundle with files of mixed shapes; return the
    expected (post-strip) {relpath: bytes} mapping for byte-identity
    assertions.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "SKILL.md").write_bytes(b"# Best Skill\n\nbody\n")
    (root / "meta_skill.md").write_bytes(b"# audit log\n## Iteration 1\n")
    scripts_dir = root / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "foo.py").write_bytes(b"#!/usr/bin/env python3\nprint('hi')\n")
    (scripts_dir / "bar.sh").write_bytes(b"#!/bin/bash\necho hi\n")
    # AppleDouble metadata sibling that must be stripped.
    (scripts_dir / "._foo.py").write_bytes(b"\x00\x05Mac OS X")
    # __pycache__ directory under scripts/ — must be stripped.
    pycache = scripts_dir / "__pycache__"
    pycache.mkdir()
    (pycache / "foo.cpython-312.pyc").write_bytes(b"\x00bytecode")
    # references/
    refs = root / "references"
    refs.mkdir()
    (refs / "notes.md").write_bytes(b"# refs\n")

    return {
        "SKILL.md": b"# Best Skill\n\nbody\n",
        "scripts/foo.py": b"#!/usr/bin/env python3\nprint('hi')\n",
        "scripts/bar.sh": b"#!/bin/bash\necho hi\n",
        "references/notes.md": b"# refs\n",
    }


def test_copy_for_deployment_excludes_meta_skill_and_appledouble(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    expected = _make_fixture_bundle(src)

    copy_for_deployment(src, dst)

    # meta_skill.md absent.
    assert not (dst / "meta_skill.md").exists()
    # AppleDouble + __pycache__ absent.
    assert not (dst / "scripts" / "._foo.py").exists()
    assert not (dst / "scripts" / "__pycache__").exists()

    # All other files byte-identical.
    for rel, payload in expected.items():
        copied = dst / rel
        assert copied.exists(), f"missing in dst: {rel}"
        assert copied.read_bytes() == payload, f"byte mismatch at {rel}"


def test_copy_for_deployment_with_no_meta_skill(tmp_path: Path) -> None:
    """When src has no meta_skill.md, copy_for_deployment is a clean copy."""
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    (src / "SKILL.md").write_bytes(b"# only skill\n")
    (src / "scripts").mkdir()
    (src / "scripts" / "ok.py").write_bytes(b"# ok\n")

    copy_for_deployment(src, dst)

    assert (dst / "SKILL.md").read_bytes() == b"# only skill\n"
    assert (dst / "scripts" / "ok.py").read_bytes() == b"# ok\n"


def test_copy_for_deployment_overwrites_existing_dst(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    (src / "SKILL.md").write_bytes(b"fresh\n")
    # Pre-existing dst with stale content.
    dst.mkdir()
    (dst / "stale.txt").write_bytes(b"old\n")

    copy_for_deployment(src, dst)

    assert (dst / "SKILL.md").read_bytes() == b"fresh\n"
    assert not (dst / "stale.txt").exists()


# ─── FolderArtifact.write_to(deployment=True) strips meta_skill.md ─────


def test_folder_artifact_write_deployment_strips_meta_skill(
    tmp_path: Path,
) -> None:
    artifact = FolderArtifact(
        files={
            "SKILL.md": "# best\n",
            "meta_skill.md": "# audit\n## Iteration 1\n",
            "scripts/foo.py": "#!/usr/bin/env python3\nprint('hi')\n",
        }
    )
    out = tmp_path / "out"
    artifact.write_to(out, deployment=True)
    assert (out / "SKILL.md").exists()
    assert not (out / "meta_skill.md").exists()
    assert (out / "scripts" / "foo.py").exists()


def test_folder_artifact_write_default_keeps_meta_skill(tmp_path: Path) -> None:
    """Default write_to does NOT strip — meta_skill.md must propagate
    across evolution iters so the consolidator can read prior content.
    """
    artifact = FolderArtifact(
        files={
            "SKILL.md": "# best\n",
            "meta_skill.md": "# audit\n## Iteration 1\n",
        }
    )
    out = tmp_path / "out"
    artifact.write_to(out)
    assert (out / "meta_skill.md").exists()
