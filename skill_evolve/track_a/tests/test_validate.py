"""Invariant checks for validate() and each operator's post-state."""

from __future__ import annotations

import json

import pytest

from skill_evolve.track_a.folder import SkillFolder, make_skill_doc
from skill_evolve.track_a.llm import LLMClient
from skill_evolve.track_a.ops import (
    apply_add_skill,
    apply_merge_skills,
    apply_remove_skill,
    apply_rename_skill,
    apply_rewrite_content,
    apply_split_skill,
)
from skill_evolve.track_a.validate import MAX_SKILLS, MAX_TOTAL_BYTES, validate, validates


def test_seed_validates(seed_folder: SkillFolder) -> None:
    ok, errs = validate(seed_folder)
    assert ok, f"seed should validate, got {errs}"


def test_empty_folder_invalid() -> None:
    f = SkillFolder(skills=[])
    ok, errs = validate(f)
    assert not ok
    assert any("MIN_SKILLS" in e for e in errs)


def test_duplicate_names_invalid() -> None:
    f = SkillFolder(skills=[
        make_skill_doc("a", description="x", body="# A"),
        make_skill_doc("a", description="y", body="# A2"),
    ])
    ok, errs = validate(f)
    assert not ok


def test_folder_name_mismatch_invalid() -> None:
    doc = make_skill_doc("dir-name", description="x", body="# A")
    doc.frontmatter["name"] = "different-name"
    f = SkillFolder(skills=[doc])
    ok, errs = validate(f)
    assert not ok


def test_missing_description_invalid() -> None:
    doc = make_skill_doc("a", description="", body="# A")
    f = SkillFolder(skills=[doc])
    ok, errs = validate(f)
    assert not ok


def test_oversize_folder_invalid() -> None:
    big = "x" * (MAX_TOTAL_BYTES + 10)
    doc = make_skill_doc("a", description="desc", body=big)
    f = SkillFolder(skills=[doc])
    ok, errs = validate(f)
    assert not ok
    assert any("MAX_TOTAL_BYTES" in e for e in errs)


def test_max_skills_cap() -> None:
    docs = [make_skill_doc(f"s{i}", description=f"d{i}", body=f"# S{i}") for i in range(MAX_SKILLS + 1)]
    f = SkillFolder(skills=docs)
    ok, errs = validate(f)
    assert not ok


# ---------------------------------------------------------------------------
# Per-op post-state validation
# ---------------------------------------------------------------------------

def test_add_skill_validates(seed_folder: SkillFolder, synthetic_client: LLMClient) -> None:
    out = apply_add_skill(
        seed_folder, synthetic_client,
        name="new-skill",
        description="A brand-new skill used only in tests.",
        seed="Covers a capability missing from seed.",
    )
    assert validates(out)
    assert "new-skill" in out.names()
    # seed unchanged (clone semantics)
    assert "new-skill" not in seed_folder.names()


def test_remove_skill_validates(seed_folder: SkillFolder, synthetic_client: LLMClient) -> None:
    out = apply_remove_skill(seed_folder, synthetic_client, name="ask-the-environment")
    assert validates(out)
    assert "ask-the-environment" not in out.names()


def test_rename_skill_validates(seed_folder: SkillFolder, synthetic_client: LLMClient) -> None:
    out = apply_rename_skill(
        seed_folder, synthetic_client,
        old="ask-the-environment",
        new="inspect-the-environment",
    )
    assert validates(out)
    assert "inspect-the-environment" in out.names()
    assert "ask-the-environment" not in out.names()
    # frontmatter name also updated
    doc = out.by_name("inspect-the-environment")
    assert doc is not None
    assert doc.frontmatter["name"] == "inspect-the-environment"


def test_split_skill_validates(seed_folder: SkillFolder, synthetic_client: LLMClient) -> None:
    out = apply_split_skill(
        seed_folder, synthetic_client,
        name="stop-and-replan",
        into=[
            {"name": "halt-on-repeat", "description": "Detect and halt flailing."},
            {"name": "replan-new-angle", "description": "Extract assumption, pick new angle."},
        ],
        rationale="Separate detection from replanning.",
    )
    assert validates(out), validate(out)
    assert "stop-and-replan" not in out.names()
    assert "halt-on-repeat" in out.names()
    assert "replan-new-angle" in out.names()


def test_merge_skills_validates(seed_folder: SkillFolder, synthetic_client: LLMClient) -> None:
    out = apply_merge_skills(
        seed_folder, synthetic_client,
        a="systematic-shell-debugging",
        b="ask-the-environment",
        into="shell-grounding",
        description="Merged: methodical shell debugging with environment grounding.",
        rationale="Both concern shell/env inspection.",
    )
    assert validates(out), validate(out)
    assert "shell-grounding" in out.names()
    assert "systematic-shell-debugging" not in out.names()
    assert "ask-the-environment" not in out.names()


def test_rewrite_content_validates(seed_folder: SkillFolder, synthetic_client: LLMClient) -> None:
    out = apply_rewrite_content(
        seed_folder, synthetic_client,
        name="ask-the-environment",
        critique="description is too generic; cites no concrete tool names.",
        skill_failures="task_123: INVOKED but still failed — env check missing which python",
    )
    assert validates(out)
    doc = out.by_name("ask-the-environment")
    assert doc is not None
    # Body should have been replaced with the canned synthetic body.
    assert doc.body.strip().startswith("# Ask the Environment")
