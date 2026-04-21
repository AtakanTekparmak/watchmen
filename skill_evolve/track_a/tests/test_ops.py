"""Operator round-trip and selection tests."""

from __future__ import annotations

import pytest

from skill_evolve.track_a.folder import SkillFolder
from skill_evolve.track_a.llm import LLMClient
from skill_evolve.track_a.ops import (
    OpRecord,
    _coerce_op_record,
    apply_add_skill,
    apply_remove_skill,
    pick_op,
)


def test_add_then_remove_round_trip(
    seed_folder: SkillFolder, synthetic_client: LLMClient
) -> None:
    """AddSkill then RemoveSkill of same name restores original skill set."""
    original_names = sorted(seed_folder.names())

    added = apply_add_skill(
        seed_folder, synthetic_client,
        name="scratch-skill",
        description="Temporary scratch skill for round-trip test.",
        seed="Used only in tests.",
    )
    assert "scratch-skill" in added.names()
    assert sorted(n for n in added.names() if n != "scratch-skill") == original_names

    removed = apply_remove_skill(added, synthetic_client, name="scratch-skill")
    assert sorted(removed.names()) == original_names


def test_pick_op_synthetic_parses_to_rewrite(
    seed_folder: SkillFolder, synthetic_client: LLMClient
) -> None:
    """Synthetic canned op_planner response parses as RewriteSkillContent."""
    op = pick_op(seed_folder, "dummy critique", client=synthetic_client)
    assert op.op == "RewriteSkillContent"
    assert op.args.get("name") == "ask-the-environment"


def test_coerce_rejects_unknown_skill(seed_folder: SkillFolder) -> None:
    # RemoveSkill on a skill that doesn't exist should fall back to None.
    bad = _coerce_op_record(
        {"op": "RemoveSkill", "name": "does-not-exist"}, seed_folder
    )
    assert bad is None


def test_coerce_rejects_rename_collision(seed_folder: SkillFolder) -> None:
    bad = _coerce_op_record(
        {
            "op": "RenameSkill",
            "old": "ask-the-environment",
            "new": "patch-then-verify",  # already exists
        },
        seed_folder,
    )
    assert bad is None


def test_coerce_accepts_valid_add(seed_folder: SkillFolder) -> None:
    op = _coerce_op_record(
        {
            "op": "AddSkill",
            "name": "brand-new",
            "description": "desc",
            "seed": "rationale",
        },
        seed_folder,
    )
    assert op is not None
    assert op.op == "AddSkill"
    assert op.args["name"] == "brand-new"


def test_coerce_accepts_valid_merge(seed_folder: SkillFolder) -> None:
    op = _coerce_op_record(
        {
            "op": "MergeSkills",
            "a": "ask-the-environment",
            "b": "systematic-shell-debugging",
            "into": "shell-grounding",
            "description": "merged",
            "rationale": "overlap",
        },
        seed_folder,
    )
    assert op is not None
    assert op.op == "MergeSkills"


def test_coerce_rejects_remove_when_only_one_skill() -> None:
    from skill_evolve.track_a.folder import make_skill_doc

    f = SkillFolder(skills=[make_skill_doc("only", description="d", body="# b")])
    bad = _coerce_op_record({"op": "RemoveSkill", "name": "only"}, f)
    assert bad is None
