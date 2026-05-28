"""Tests for ``shared.reflection.merge_patches`` (Group H)."""

from __future__ import annotations

from skill_evolve.shared.patch_parser import (
    AddFile,
    DeleteFile,
    EditFile,
    FileOp,
)
from skill_evolve.shared.reflection import merge_patches


def _edit(path: str, content: str = "x\n") -> FileOp:
    return EditFile(op="EDIT_FILE", path=path, content=content)


def _add(path: str, content: str = "x\n") -> FileOp:
    return AddFile(op="ADD_FILE", path=path, content=content)


def _delete(path: str) -> FileOp:
    return DeleteFile(op="DELETE_FILE", path=path)


def test_disjoint_paths_concatenate_failures_first() -> None:
    failure = [_edit("a/SKILL.md", "f-body\n")]
    success = [_edit("b/SKILL.md", "s-body\n")]
    merged, stats = merge_patches(failure, success)
    assert [op.path for op in merged] == ["a/SKILL.md", "b/SKILL.md"]
    assert stats["collisions"] == []
    assert stats["failure_count"] == 1
    assert stats["success_count"] == 1
    assert stats["merged_count"] == 2
    assert stats["clipped"] is False


def test_collision_failure_wins_and_recorded() -> None:
    failure = [_edit("a/SKILL.md", "FAILURE-BODY\n")]
    success = [_edit("a/SKILL.md", "SUCCESS-BODY\n")]
    merged, stats = merge_patches(failure, success)
    assert len(merged) == 1
    assert merged[0].content == "FAILURE-BODY\n"
    assert len(stats["collisions"]) == 1
    collision = stats["collisions"][0]
    assert collision["key"] == ("EDIT_FILE", "a/SKILL.md")


def test_different_op_kinds_same_path_do_not_collide() -> None:
    failure = [_edit("a/SKILL.md")]
    success = [_delete("a/SKILL.md")]
    merged, stats = merge_patches(failure, success)
    # Different (op_kind, path) keys -> both survive.
    assert len(merged) == 2
    assert stats["collisions"] == []


def test_lt_clip_after_merge() -> None:
    failure = [_edit(f"f{i}/SKILL.md") for i in range(3)]
    success = [_edit(f"s{i}/SKILL.md") for i in range(3)]
    merged, stats = merge_patches(failure, success, lt=2)
    assert len(merged) == 2
    # Failure-priority + parser order -> first two failure ops survive.
    assert [op.path for op in merged] == ["f0/SKILL.md", "f1/SKILL.md"]
    assert stats["merged_count"] == 2
    assert stats["clipped"] is True


def test_lt_none_disables_clip() -> None:
    failure = [_edit("f1/SKILL.md")]
    success = [_edit("s1/SKILL.md")]
    merged, stats = merge_patches(failure, success, lt=None)
    assert len(merged) == 2
    assert stats["clipped"] is False


def test_empty_failure_returns_success_only() -> None:
    success = [_edit("s1/SKILL.md"), _add("s2/scripts/foo.py")]
    merged, stats = merge_patches([], success)
    assert merged == success
    assert stats["failure_count"] == 0
    assert stats["success_count"] == 2
    assert stats["merged_count"] == 2
    assert stats["collisions"] == []


def test_empty_success_returns_failure_only() -> None:
    failure = [_edit("f1/SKILL.md"), _delete("f2/SKILL.md")]
    merged, stats = merge_patches(failure, [])
    assert merged == failure
    assert stats["failure_count"] == 2
    assert stats["success_count"] == 0
    assert stats["merged_count"] == 2


def test_both_empty_returns_empty() -> None:
    merged, stats = merge_patches([], [])
    assert merged == []
    assert stats["failure_count"] == 0
    assert stats["success_count"] == 0
    assert stats["merged_count"] == 0


def test_lt_clip_with_empty_success() -> None:
    failure = [_edit(f"f{i}/SKILL.md") for i in range(5)]
    merged, stats = merge_patches(failure, [], lt=3)
    assert len(merged) == 3
    assert stats["clipped"] is True


def test_collision_drops_success_then_concatenates_remaining() -> None:
    # Failure side has 1 op; success side has 2 ops one of which
    # collides. Result should have failure[0] + the non-colliding success.
    failure = [_edit("a/SKILL.md", "F\n")]
    success = [
        _edit("a/SKILL.md", "DROPPED\n"),  # collides
        _add("b/SKILL.md", "KEPT\n"),
    ]
    merged, stats = merge_patches(failure, success)
    assert len(merged) == 2
    assert merged[0].content == "F\n"
    assert merged[1].path == "b/SKILL.md"
    assert len(stats["collisions"]) == 1
