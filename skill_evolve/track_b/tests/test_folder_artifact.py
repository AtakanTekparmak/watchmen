"""Tests for FolderArtifact — ser/de round-trip, hashing, I/O."""

from __future__ import annotations

from pathlib import Path

import pytest

from skill_evolve.track_b.openevolve_skills.folder_artifact import (
    FolderArtifact,
    FolderArtifactError,
    MAX_FILES,
    MAX_TOTAL_BYTES,
)


SAMPLE = {
    "alpha/SKILL.md": "---\nname: alpha\ndescription: A\n---\n\n# alpha\nbody\n",
    "beta/SKILL.md": "---\nname: beta\ndescription: B\n---\n\n# beta\nbody\n",
    "INDEX.md": "# index\n- alpha\n- beta\n",
}


def test_construction_normalizes_paths():
    art = FolderArtifact(files={"./a/SKILL.md": "x\n"})
    assert "a/SKILL.md" in art.files


def test_reject_absolute_path():
    with pytest.raises(FolderArtifactError):
        FolderArtifact(files={"/abs/SKILL.md": "x"})


def test_reject_parent_traversal():
    with pytest.raises(FolderArtifactError):
        FolderArtifact(files={"../escape/SKILL.md": "x"})


def test_serialize_deserialize_roundtrip():
    art = FolderArtifact(files=dict(SAMPLE))
    blob = art.serialize()
    art2 = FolderArtifact.deserialize(blob)
    assert art == art2
    # And a second round-trip produces identical blob.
    assert art2.serialize() == blob


def test_hash_is_stable_across_insertion_order():
    a = FolderArtifact(files={
        "a/SKILL.md": "x\n",
        "b/SKILL.md": "y\n",
    })
    b = FolderArtifact(files={
        "b/SKILL.md": "y\n",
        "a/SKILL.md": "x\n",
    })
    assert a.stable_hash() == b.stable_hash()
    assert hash(a) == hash(b)


def test_hash_changes_with_content():
    a = FolderArtifact(files={"a/SKILL.md": "x\n"})
    b = FolderArtifact(files={"a/SKILL.md": "y\n"})
    assert a.stable_hash() != b.stable_hash()


def test_write_to_and_from_path_roundtrip(tmp_path: Path):
    art = FolderArtifact(files=dict(SAMPLE))
    target = tmp_path / "out"
    art.write_to(target)
    round = FolderArtifact.from_path(target, include_exts={".md"})
    assert round == art


def test_validate_rejects_too_many_files():
    files = {f"s{i}/SKILL.md": "x\n" for i in range(MAX_FILES + 1)}
    art = FolderArtifact(files=files)
    with pytest.raises(FolderArtifactError):
        art.validate()


def test_validate_rejects_oversize():
    big = "x" * (MAX_TOTAL_BYTES + 100)
    art = FolderArtifact(files={"a/SKILL.md": big})
    with pytest.raises(FolderArtifactError):
        art.validate()


def test_validate_rejects_empty():
    art = FolderArtifact(files={})
    with pytest.raises(FolderArtifactError):
        art.validate()


def test_num_skills_counts_skill_md_dirs():
    art = FolderArtifact(files={
        "a/SKILL.md": "x\n",
        "b/SKILL.md": "y\n",
        "INDEX.md": "z\n",
        "a/helper.md": "q\n",
    })
    assert art.num_skills() == 2
    assert art.skill_names() == ["a", "b"]


def test_deserialize_rejects_malformed():
    blob = "===== FILE: a =====\ncontent\n"  # no END FILE
    with pytest.raises(FolderArtifactError):
        FolderArtifact.deserialize(blob)


# --- YAML frontmatter validation (added 2026-04-22) ----------------------

def test_validate_rejects_unquoted_colon_description():
    # Exact shape that crashed Track D's phase-a handoff on 2026-04-21:
    # description is a bare string containing ':', which YAML interprets
    # as a nested mapping and fails with "mapping values are not allowed
    # here". FolderArtifact.validate() must catch this before the folder
    # reaches the MAP-Elites archive.
    bad = (
        "---\n"
        "name: x\n"
        "description: Core guidance for tasks: parse requirements\n"
        "version: 0.1.0\n"
        "---\n"
        "body\n"
    )
    art = FolderArtifact(files={"x/SKILL.md": bad})
    with pytest.raises(FolderArtifactError, match="YAML frontmatter unparseable"):
        art.validate()


def test_validate_accepts_quoted_colon_description():
    good = (
        "---\n"
        "name: x\n"
        'description: "Core guidance for tasks: parse requirements"\n'
        "version: 0.1.0\n"
        "---\n"
        "body\n"
    )
    art = FolderArtifact(files={"x/SKILL.md": good})
    art.validate()  # no raise


def test_validate_rejects_unclosed_frontmatter_fence():
    bad = "---\nname: x\n# body starts but no closing fence\n"
    art = FolderArtifact(files={"x/SKILL.md": bad})
    with pytest.raises(FolderArtifactError, match="without a closing '---'"):
        art.validate()


def test_validate_allows_skill_md_without_frontmatter():
    # A SKILL.md that doesn't declare a frontmatter block is treated as
    # body-only and allowed through. Track A will reject it at validate()
    # time (missing name/description) but openevolve shouldn't block the
    # patch at the artifact boundary.
    body_only = "# just a body\nno frontmatter here\n"
    art = FolderArtifact(files={"x/SKILL.md": body_only})
    art.validate()


def test_validate_rejects_non_dict_frontmatter():
    # A frontmatter block that parses to a list / scalar isn't a SKILL.md.
    bad = "---\n- list\n- item\n---\nbody\n"
    art = FolderArtifact(files={"x/SKILL.md": bad})
    with pytest.raises(FolderArtifactError, match="expected dict"):
        art.validate()


def test_validate_only_checks_skill_md_files():
    # Non-SKILL.md files (e.g. INDEX.md) can contain whatever YAML-looking
    # text they want.
    art = FolderArtifact(files={
        "x/SKILL.md": "---\nname: x\ndescription: ok\n---\nbody\n",
        "INDEX.md": "random: text: with: colons: everywhere:\n",
    })
    art.validate()  # no raise
