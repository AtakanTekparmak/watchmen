"""Tests for the patch parser: tokens → ops → applied folder."""

from __future__ import annotations

import pytest

from skill_evolve.track_b.openevolve_skills.folder_artifact import (
    FolderArtifact,
    FolderArtifactError,
)
from skill_evolve.track_b.openevolve_skills.patch_parser import (
    AddFile,
    DeleteFile,
    EditFile,
    PatchParseError,
    RewriteFolder,
    apply_patch,
    mutate,
    parse_patch,
)


PARENT = FolderArtifact(files={
    "alpha/SKILL.md": "---\nname: alpha\ndescription: A\n---\n\n# alpha\n",
    "beta/SKILL.md": "---\nname: beta\ndescription: B\n---\n\n# beta\n",
})


def test_parse_add_file():
    patch = (
        "<<<ADD_FILE gamma/SKILL.md>>>\n"
        "---\nname: gamma\ndescription: G\n---\n\n# gamma\n"
        "<<<END_FILE>>>\n"
    )
    ops = parse_patch(patch)
    assert len(ops) == 1
    assert isinstance(ops[0], AddFile)
    assert ops[0].path == "gamma/SKILL.md"
    assert "name: gamma" in ops[0].content


def test_parse_edit_and_delete():
    patch = (
        "<<<EDIT_FILE alpha/SKILL.md>>>\nnew body\n<<<END_FILE>>>\n"
        "<<<DELETE_FILE beta/SKILL.md>>>\n"
    )
    ops = parse_patch(patch)
    assert [type(o).__name__ for o in ops] == ["EditFile", "DeleteFile"]
    assert ops[0].path == "alpha/SKILL.md"
    assert ops[1].path == "beta/SKILL.md"


def test_parse_ignores_surrounding_prose():
    patch = (
        "Some thinking here.\n"
        "<<<EDIT_FILE alpha/SKILL.md>>>\nnew body\n<<<END_FILE>>>\n"
        "More thinking.\n"
    )
    ops = parse_patch(patch)
    assert len(ops) == 1 and isinstance(ops[0], EditFile)


def test_parse_rejects_unterminated():
    with pytest.raises(PatchParseError):
        parse_patch("<<<ADD_FILE a/SKILL.md>>>\ncontent without end\n")


def test_parse_rejects_bad_path():
    with pytest.raises(PatchParseError):
        parse_patch("<<<ADD_FILE ../evil>>>\nx\n<<<END_FILE>>>\n")


def test_apply_add_edit_delete():
    patch = (
        "<<<ADD_FILE gamma/SKILL.md>>>\n"
        "---\nname: gamma\ndescription: G\n---\n\n# gamma\n"
        "<<<END_FILE>>>\n"
        "<<<EDIT_FILE alpha/SKILL.md>>>\nrewritten alpha\n<<<END_FILE>>>\n"
        "<<<DELETE_FILE beta/SKILL.md>>>\n"
    )
    child = apply_patch(PARENT, parse_patch(patch))
    assert "gamma/SKILL.md" in child.files
    assert child.files["alpha/SKILL.md"].startswith("rewritten alpha")
    assert "beta/SKILL.md" not in child.files


def test_apply_rejects_add_existing():
    patch = (
        "<<<ADD_FILE alpha/SKILL.md>>>\ncontent\n<<<END_FILE>>>\n"
    )
    with pytest.raises(PatchParseError):
        apply_patch(PARENT, parse_patch(patch))


def test_apply_rejects_edit_missing():
    patch = "<<<EDIT_FILE missing/SKILL.md>>>\nx\n<<<END_FILE>>>\n"
    with pytest.raises(PatchParseError):
        apply_patch(PARENT, parse_patch(patch))


def test_apply_rejects_delete_missing():
    with pytest.raises(PatchParseError):
        apply_patch(PARENT, parse_patch("<<<DELETE_FILE missing/SKILL.md>>>\n"))


def test_rewrite_folder_replaces_all():
    inner = FolderArtifact(files={"only/SKILL.md": "---\nname: only\n---\n\n# only\n"})
    blob = inner.serialize()
    patch = f"<<<REWRITE_FOLDER>>>\n{blob}<<<END_REWRITE>>>\n"
    ops = parse_patch(patch)
    assert len(ops) == 1 and isinstance(ops[0], RewriteFolder)
    child = apply_patch(PARENT, ops)
    assert child == inner


def test_rewrite_must_be_sole_op():
    inner = FolderArtifact(files={"only/SKILL.md": "---\nname: only\n---\n"})
    blob = inner.serialize()
    patch = (
        f"<<<REWRITE_FOLDER>>>\n{blob}<<<END_REWRITE>>>\n"
        "<<<DELETE_FILE alpha/SKILL.md>>>\n"
    )
    with pytest.raises(PatchParseError):
        apply_patch(PARENT, parse_patch(patch))


def test_mutate_end_to_end_validates():
    # Deleting all skills should fail validation (no skills remain).
    patch = (
        "<<<DELETE_FILE alpha/SKILL.md>>>\n"
        "<<<DELETE_FILE beta/SKILL.md>>>\n"
    )
    # Either a size-constraint FolderArtifactError or a PatchParseError
    # about "0 skills" is acceptable — both signal the same problem.
    with pytest.raises((PatchParseError, FolderArtifactError)):
        mutate(PARENT, patch)


def test_mutate_empty_patch_raises():
    with pytest.raises(PatchParseError):
        mutate(PARENT, "this is just prose")
