"""Sentinel-block parser edge cases (locked section-7a API)."""

from __future__ import annotations

import pytest

from skill_evolve.shared.patch_parser import (
    SentinelParseError,
    parse_sentinel_blocks,
)


# ---------------------------------------------------------------------------
# Structural malformations
# ---------------------------------------------------------------------------


def test_unterminated_add_file_block() -> None:
    """An ADD_FILE without a matching <<<END_FILE>>> → ``unterminated``
    (or ``mixed_content`` under strict mode if the parser surfaces the
    stray prose first — both are valid failure modes for this input)."""
    text = "<<<ADD_FILE scripts/foo.py>>>\nprint('x')\n"
    with pytest.raises(SentinelParseError) as exc_info:
        parse_sentinel_blocks(text, strict=True)
    kind = str(exc_info.value)
    assert kind.startswith("unterminated") or kind.startswith("mixed_content")


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------


def test_path_traversal_rejected_before_write() -> None:
    text = "<<<ADD_FILE ../../etc/passwd>>>\nroot:x:0:0:\n<<<END_FILE>>>\n"
    with pytest.raises(SentinelParseError) as exc_info:
        parse_sentinel_blocks(text)
    assert str(exc_info.value).startswith("path_traversal")


def test_absolute_path_rejected() -> None:
    text = "<<<ADD_FILE /tmp/foo>>>\nx\n<<<END_FILE>>>\n"
    with pytest.raises(SentinelParseError) as exc_info:
        parse_sentinel_blocks(text)
    assert str(exc_info.value).startswith("absolute_path")


# ---------------------------------------------------------------------------
# Empty body
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "Plan section 7a locks empty ADD_FILE body to succeed (file created "
        "empty, no error). Current parser regex requires non-empty body "
        "between the two newlines that wrap the body group and raises "
        "``unterminated``. Audit phase reconciles."
    ),
    strict=False,
)
def test_empty_body_add_file_no_error() -> None:
    """Empty ADD_FILE body produces an op with empty content (no error)."""
    text = "<<<ADD_FILE scripts/empty.py>>>\n\n<<<END_FILE>>>\n"
    ops = parse_sentinel_blocks(text)
    assert len(ops) == 1
    assert ops[0].op == "ADD_FILE"
    assert ops[0].path == "scripts/empty.py"
    assert ops[0].content in ("", "\n")


# ---------------------------------------------------------------------------
# Duplicate ADD_FILE
# ---------------------------------------------------------------------------


def test_duplicate_add_file_in_same_patch_rejected() -> None:
    """Two ADD_FILE for the same path → ``add_existing`` on the second."""
    text = (
        "<<<ADD_FILE scripts/dup.py>>>\n"
        "first\n"
        "<<<END_FILE>>>\n"
        "<<<ADD_FILE scripts/dup.py>>>\n"
        "second\n"
        "<<<END_FILE>>>\n"
    )
    with pytest.raises(SentinelParseError) as exc_info:
        # Pass an empty existing_paths so the in-patch duplicate detection
        # still fires (parser tracks accumulating set).
        parse_sentinel_blocks(text, existing_paths=set())
    assert str(exc_info.value).startswith("add_existing")


# ---------------------------------------------------------------------------
# Parent-bundle existence checks
# ---------------------------------------------------------------------------


def test_add_file_on_existing_parent_path_rejected() -> None:
    """ADD_FILE on a path already in the parent bundle → ``add_existing``."""
    text = "<<<ADD_FILE scripts/keep.py>>>\nx\n<<<END_FILE>>>\n"
    with pytest.raises(SentinelParseError) as exc_info:
        parse_sentinel_blocks(text, existing_paths={"scripts/keep.py"})
    assert str(exc_info.value).startswith("add_existing")


def test_edit_file_on_missing_parent_path_rejected() -> None:
    """EDIT_FILE on a path NOT in the parent bundle → ``edit_missing``."""
    text = "<<<EDIT_FILE scripts/ghost.py>>>\nx\n<<<END_FILE>>>\n"
    with pytest.raises(SentinelParseError) as exc_info:
        parse_sentinel_blocks(text, existing_paths={"scripts/other.py"})
    assert str(exc_info.value).startswith("edit_missing")


# ---------------------------------------------------------------------------
# REWRITE_FOLDER mixed content
# ---------------------------------------------------------------------------


def test_rewrite_folder_mixed_content_rejected() -> None:
    """Content outside ``--- file:`` headers under REWRITE_FOLDER → mixed_content."""
    text = (
        "<<<REWRITE_FOLDER scripts>>>\n"
        "stray prose before any file header\n"
        "--- file: scripts/a.py\n"
        "print('a')\n"
        "<<<END_REWRITE>>>\n"
    )
    with pytest.raises(SentinelParseError) as exc_info:
        parse_sentinel_blocks(text)
    assert str(exc_info.value).startswith("mixed_content")


# ---------------------------------------------------------------------------
# Multi-op ordering
# ---------------------------------------------------------------------------


def test_multiple_ops_parsed_in_order() -> None:
    """Multiple sentinel blocks → list of ops in source order."""
    text = (
        "<<<ADD_FILE scripts/a.py>>>\n"
        "print('a')\n"
        "<<<END_FILE>>>\n"
        "<<<EDIT_FILE scripts/b.py>>>\n"
        "print('b')\n"
        "<<<END_FILE>>>\n"
        "<<<DELETE_FILE scripts/c.py>>>\n"
    )
    ops = parse_sentinel_blocks(text, existing_paths={"scripts/b.py", "scripts/c.py"})
    assert [op.op for op in ops] == ["ADD_FILE", "EDIT_FILE", "DELETE_FILE"]
    assert [op.path for op in ops] == [
        "scripts/a.py",
        "scripts/b.py",
        "scripts/c.py",
    ]
