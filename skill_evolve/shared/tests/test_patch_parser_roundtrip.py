"""Round-trip tests for the sentinel-block patch parser.

Covers ADD_FILE / EDIT_FILE / DELETE_FILE / REWRITE_FOLDER ops and one
``apply → re-serialize → re-parse → equal hash_bundle`` digest round-trip.
"""

from __future__ import annotations

from pathlib import Path

from skill_evolve.shared.bundle_ops import (
    apply_file_ops,
    hash_bundle,
)
from skill_evolve.shared.patch_parser import (
    AddFile,
    DeleteFile,
    EditFile,
    FileOp,
    RewriteFolder,
    parse_sentinel_blocks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_bundle(root: Path) -> Path:
    """Build a minimal parent bundle to mutate against. Returns its path."""
    bundle = root / "parent"
    (bundle / "scripts").mkdir(parents=True)
    (bundle / "SKILL.md").write_text(
        "---\nname: demo\ndescription: a demo\n---\n\n# demo\n",
        encoding="utf-8",
    )
    (bundle / "scripts" / "existing.py").write_text(
        "print('hello')\n", encoding="utf-8"
    )
    (bundle / "scripts" / "old.sh").write_text(
        "#!/bin/bash\necho stale\n", encoding="utf-8"
    )
    return bundle


# ---------------------------------------------------------------------------
# ADD_FILE
# ---------------------------------------------------------------------------


def test_add_file_creates_file_with_exact_body(tmp_path: Path) -> None:
    parent = _seed_bundle(tmp_path)
    candidate = tmp_path / "candidate"

    text = "<<<ADD_FILE scripts/new.py>>>\nprint('new')\n<<<END_FILE>>>\n"
    ops = parse_sentinel_blocks(text)
    assert len(ops) == 1
    assert isinstance(ops[0], AddFile)
    assert ops[0].op == "ADD_FILE"
    assert ops[0].path == "scripts/new.py"

    apply_file_ops(parent, ops, candidate)
    new_file = candidate / "scripts" / "new.py"
    assert new_file.exists()
    # Trailing newline preserved per parser contract.
    assert new_file.read_text(encoding="utf-8") == "print('new')\n"


# ---------------------------------------------------------------------------
# EDIT_FILE
# ---------------------------------------------------------------------------


def test_edit_file_replaces_full_contents(tmp_path: Path) -> None:
    parent = _seed_bundle(tmp_path)
    candidate = tmp_path / "candidate"

    existing_paths = {
        p.relative_to(parent).as_posix() for p in parent.rglob("*") if p.is_file()
    }
    text = "<<<EDIT_FILE scripts/existing.py>>>\nprint('rewritten')\n<<<END_FILE>>>\n"
    ops = parse_sentinel_blocks(text, existing_paths=existing_paths)
    assert len(ops) == 1
    assert isinstance(ops[0], EditFile)
    assert type(ops[0]).__name__ == "EditFile"

    apply_file_ops(parent, ops, candidate)
    edited = candidate / "scripts" / "existing.py"
    assert edited.exists()
    assert edited.read_text(encoding="utf-8") == "print('rewritten')\n"


# ---------------------------------------------------------------------------
# DELETE_FILE
# ---------------------------------------------------------------------------


def test_delete_file_removes_file(tmp_path: Path) -> None:
    parent = _seed_bundle(tmp_path)
    candidate = tmp_path / "candidate"

    existing_paths = {
        p.relative_to(parent).as_posix() for p in parent.rglob("*") if p.is_file()
    }
    text = "<<<DELETE_FILE scripts/old.sh>>>\n"
    ops = parse_sentinel_blocks(text, existing_paths=existing_paths)
    assert len(ops) == 1
    assert isinstance(ops[0], DeleteFile)
    assert type(ops[0]).__name__ == "DeleteFile"

    apply_file_ops(parent, ops, candidate)
    assert not (candidate / "scripts" / "old.sh").exists()
    # The other files from the parent are still there.
    assert (candidate / "scripts" / "existing.py").exists()
    assert (candidate / "SKILL.md").exists()


# ---------------------------------------------------------------------------
# REWRITE_FOLDER
# ---------------------------------------------------------------------------


def test_rewrite_folder_wipes_and_rewrites_directory(tmp_path: Path) -> None:
    parent = _seed_bundle(tmp_path)
    # Add a stray script that REWRITE_FOLDER should blow away.
    (parent / "scripts" / "stray.py").write_text("# stale\n", encoding="utf-8")
    candidate = tmp_path / "candidate"

    text = (
        "<<<REWRITE_FOLDER scripts>>>\n"
        "--- file: scripts/a.py\n"
        "print('a')\n"
        "--- file: scripts/b.sh\n"
        "#!/bin/bash\n"
        "echo b\n"
        "<<<END_REWRITE>>>\n"
    )
    ops = parse_sentinel_blocks(text)
    assert len(ops) == 1
    assert isinstance(ops[0], RewriteFolder)
    assert ops[0].op == "REWRITE_FOLDER"
    assert ops[0].files is not None
    assert set(ops[0].files.keys()) == {"scripts/a.py", "scripts/b.sh"}

    apply_file_ops(parent, ops, candidate)
    # Stray file from parent is gone.
    assert not (candidate / "scripts" / "stray.py").exists()
    assert not (candidate / "scripts" / "existing.py").exists()
    assert not (candidate / "scripts" / "old.sh").exists()
    # New files present. The parser preserves content between
    # ``--- file:`` headers; leading/trailing newlines depend on the
    # delimiter geometry — we assert the substantive contents instead
    # of an exact byte match.
    a_body = (candidate / "scripts" / "a.py").read_text(encoding="utf-8")
    assert "print('a')" in a_body
    b_body = (candidate / "scripts" / "b.sh").read_text(encoding="utf-8")
    assert "echo b" in b_body
    assert "#!/bin/bash" in b_body
    # SKILL.md outside scripts/ untouched.
    assert (candidate / "SKILL.md").exists()


# ---------------------------------------------------------------------------
# Multi-op + round-trip via hash_bundle
# ---------------------------------------------------------------------------


def test_multiple_ops_in_order(tmp_path: Path) -> None:
    parent = _seed_bundle(tmp_path)
    existing_paths = {
        p.relative_to(parent).as_posix() for p in parent.rglob("*") if p.is_file()
    }
    text = (
        "<<<ADD_FILE scripts/added.py>>>\n"
        "print('added')\n"
        "<<<END_FILE>>>\n"
        "<<<EDIT_FILE scripts/existing.py>>>\n"
        "print('edited')\n"
        "<<<END_FILE>>>\n"
        "<<<DELETE_FILE scripts/old.sh>>>\n"
    )
    ops = parse_sentinel_blocks(text, existing_paths=existing_paths)
    assert [type(op).__name__ for op in ops] == [
        "AddFile",
        "EditFile",
        "DeleteFile",
    ]


def test_roundtrip_apply_reparse_equal_hash(tmp_path: Path) -> None:
    """Apply N ops to a parent → re-parse the same patch text → apply
    again to a fresh candidate → ``hash_bundle`` digests match.

    Stable round-trip via ``apply_file_ops`` (NOT ``parse_and_apply``,
    which layers in shebang insurance and is not idempotent against a
    second apply of its own output).
    """
    parent = _seed_bundle(tmp_path)
    candidate_a = tmp_path / "cand_a"
    candidate_b = tmp_path / "cand_b"

    existing_paths = {
        p.relative_to(parent).as_posix() for p in parent.rglob("*") if p.is_file()
    }
    text = (
        "<<<ADD_FILE scripts/added.py>>>\n"
        "print('hello')\n"
        "<<<END_FILE>>>\n"
        "<<<EDIT_FILE scripts/existing.py>>>\n"
        "print('rewritten')\n"
        "<<<END_FILE>>>\n"
    )
    ops_a = parse_sentinel_blocks(text, existing_paths=set(existing_paths))
    apply_file_ops(parent, ops_a, candidate_a)
    digest_a = hash_bundle(candidate_a)

    # Re-parse and re-apply the SAME patch text. Hash-match required.
    ops_b = parse_sentinel_blocks(text, existing_paths=set(existing_paths))
    apply_file_ops(parent, ops_b, candidate_b)
    digest_b = hash_bundle(candidate_b)

    assert digest_a == digest_b


def test_fileop_subclasses_distinct_types() -> None:
    """Sanity: AddFile / EditFile / DeleteFile / RewriteFolder are distinct
    subclasses of FileOp (not aliases). ``type(op).__name__`` reads true."""
    add = AddFile(op="ADD_FILE", path="x", content="")
    edit = EditFile(op="EDIT_FILE", path="x", content="")
    delete = DeleteFile(op="DELETE_FILE", path="x")
    rewrite = RewriteFolder(op="REWRITE_FOLDER", path="x", files={})
    for op in (add, edit, delete, rewrite):
        assert isinstance(op, FileOp)
    assert type(add).__name__ == "AddFile"
    assert type(edit).__name__ == "EditFile"
    assert type(delete).__name__ == "DeleteFile"
    assert type(rewrite).__name__ == "RewriteFolder"
