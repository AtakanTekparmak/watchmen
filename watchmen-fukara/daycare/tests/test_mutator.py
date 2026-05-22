"""Unit tests for daycare.mutator (Stream 6.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from daycare.mutator import (
    hash_bundle,
    parse_sentinel_blocks,
    shebang_insurance,
)


# ─── parse_sentinel_blocks — round-trip on each block type ────────────────


def test_parse_add_file_block():
    text = "<<<ADD_FILE scripts/foo.py>>>\nprint('hello')\n<<<END_FILE>>>\n"
    ops = parse_sentinel_blocks(text)
    assert len(ops) == 1
    op = ops[0]
    assert op.op == "ADD_FILE"
    assert op.path == "scripts/foo.py"
    assert op.content == "print('hello')"


def test_parse_edit_file_block():
    """EDIT_FILE requires the path to exist in parent — pass existing_paths."""
    text = "<<<EDIT_FILE SKILL.md>>>\n---\nname: foo\n---\n# rewritten body\n<<<END_FILE>>>\n"
    ops = parse_sentinel_blocks(text, existing_paths={"SKILL.md"})
    assert len(ops) == 1
    assert ops[0].op == "EDIT_FILE"
    assert ops[0].path == "SKILL.md"
    assert "# rewritten body" in ops[0].content


def test_parse_delete_file_block():
    text = "<<<DELETE_FILE scripts/old.sh>>>\n"
    ops = parse_sentinel_blocks(text, existing_paths={"scripts/old.sh"})
    assert len(ops) == 1
    assert ops[0].op == "DELETE_FILE"
    assert ops[0].path == "scripts/old.sh"
    assert ops[0].content is None


def test_parse_rewrite_folder_block():
    text = (
        "<<<REWRITE_FOLDER scripts>>>\n"
        "--- file: scripts/foo.py\n"
        "print('a')\n"
        "--- file: scripts/bar.sh\n"
        "echo hi\n"
        "<<<END_REWRITE>>>\n"
    )
    ops = parse_sentinel_blocks(text)
    assert len(ops) == 1
    op = ops[0]
    assert op.op == "REWRITE_FOLDER"
    assert op.path == "scripts"
    assert set(op.files.keys()) == {"scripts/foo.py", "scripts/bar.sh"}
    assert op.files["scripts/foo.py"].strip() == "print('a')"
    assert op.files["scripts/bar.sh"].strip() == "echo hi"


# ─── Validation guards ────────────────────────────────────────────────────


def test_traversal_guard_rejects_dotdot():
    """K1 — path containing ``..`` must raise ValueError."""
    text = "<<<ADD_FILE ../escape.py>>>\nprint('owned')\n<<<END_FILE>>>\n"
    with pytest.raises(ValueError, match="path_traversal"):
        parse_sentinel_blocks(text)


def test_traversal_guard_rejects_absolute():
    """K1 — absolute path must raise ValueError."""
    text = "<<<ADD_FILE /etc/passwd>>>\nboom\n<<<END_FILE>>>\n"
    with pytest.raises(ValueError, match="absolute_path"):
        parse_sentinel_blocks(text)


def test_unterminated_add_block_raises():
    """K1 — ADD_FILE without END_FILE must raise."""
    text = "<<<ADD_FILE foo.py>>>\nprint('partial')\n"
    with pytest.raises(ValueError):
        parse_sentinel_blocks(text)


def test_unterminated_rewrite_block_raises():
    text = "<<<REWRITE_FOLDER scripts>>>\n--- file: scripts/foo.py\nprint('partial')\n"
    with pytest.raises(ValueError):
        parse_sentinel_blocks(text)


def test_add_existing_path_rejected():
    text = "<<<ADD_FILE SKILL.md>>>\nduplicate\n<<<END_FILE>>>\n"
    with pytest.raises(ValueError, match="add_existing"):
        parse_sentinel_blocks(text, existing_paths={"SKILL.md"})


def test_edit_missing_path_rejected():
    text = "<<<EDIT_FILE scripts/ghost.py>>>\nx = 1\n<<<END_FILE>>>\n"
    with pytest.raises(ValueError, match="edit_missing"):
        parse_sentinel_blocks(text, existing_paths={"SKILL.md"})


# ─── Shebang insurance (K7) ───────────────────────────────────────────────


def test_shebang_insurance_adds_python(tmp_path):
    p = tmp_path / "foo.py"
    p.write_text("print('hi')\n", encoding="utf-8")
    shebang_insurance(p)
    text = p.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env python3\n")
    assert "print('hi')" in text


def test_shebang_insurance_adds_bash(tmp_path):
    p = tmp_path / "run.sh"
    p.write_text("echo hello\n", encoding="utf-8")
    shebang_insurance(p)
    text = p.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/bash\n")
    assert "echo hello" in text


def test_shebang_insurance_idempotent_python(tmp_path):
    p = tmp_path / "foo.py"
    original = "#!/usr/bin/env python3\nprint('hi')\n"
    p.write_text(original, encoding="utf-8")
    shebang_insurance(p)
    shebang_insurance(p)
    assert p.read_text(encoding="utf-8") == original


def test_shebang_insurance_idempotent_bash(tmp_path):
    p = tmp_path / "go.sh"
    original = "#!/bin/bash\necho hi\n"
    p.write_text(original, encoding="utf-8")
    shebang_insurance(p)
    assert p.read_text(encoding="utf-8") == original


def test_shebang_insurance_ignores_non_script(tmp_path):
    """Non-.py/.sh files are left alone."""
    p = tmp_path / "README.md"
    original = "# hello\n"
    p.write_text(original, encoding="utf-8")
    shebang_insurance(p)
    assert p.read_text(encoding="utf-8") == original


# ─── hash_bundle determinism ──────────────────────────────────────────────


def _seed_bundle(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def test_hash_bundle_deterministic(tmp_path):
    """Same dir hashed twice → same result. Same content in a sibling dir
    → same result (path-relative, no inode dependence)."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    files = {
        "SKILL.md": "# heading\nbody\n",
        "scripts/foo.py": "print('x')\n",
        "scripts/bar.sh": "echo y\n",
    }
    _seed_bundle(a, files)
    _seed_bundle(b, files)

    h_a1 = hash_bundle(a)
    h_a2 = hash_bundle(a)
    h_b = hash_bundle(b)

    assert h_a1 == h_a2
    assert h_a1 == h_b


def test_hash_bundle_changes_on_content_edit(tmp_path):
    a = tmp_path / "a"
    _seed_bundle(a, {"SKILL.md": "# heading\n"})
    h1 = hash_bundle(a)
    (a / "SKILL.md").write_text("# different\n", encoding="utf-8")
    h2 = hash_bundle(a)
    assert h1 != h2


def test_hash_bundle_changes_on_filename(tmp_path):
    """Same content under a different path → different hash (path is hashed)."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    _seed_bundle(a, {"SKILL.md": "hello\n"})
    _seed_bundle(b, {"scripts/SKILL.md": "hello\n"})
    assert hash_bundle(a) != hash_bundle(b)


def test_hash_bundle_missing_dir(tmp_path):
    """Missing dir → empty-hash sentinel (sha256 of nothing)."""
    h = hash_bundle(tmp_path / "does-not-exist")
    # sha256("") == "e3b0c4..."
    assert h == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
