"""Group G — parser slow-update fence enforcement (plan §7l).

Coverage:
* Fence-overlap EDIT_FILE on SKILL.md → SentinelParseError("slow_update_violation").
* Out-of-fence EDIT_FILE on SKILL.md accepted.
* EDIT_FILE on scripts/* unaffected (fence only applies to SKILL.md).
* parent_skill_md=None preserves back-compat.
"""

from __future__ import annotations

import pytest

from skill_evolve.shared.patch_parser import (
    SentinelParseError,
    parse_sentinel_blocks,
)
from skill_evolve.shared.slow_update import (
    SLOW_UPDATE_END,
    SLOW_UPDATE_START,
    inject_slow_update_field,
)


def _parent_with_fence(body: str = "PROTECTED CONTENT") -> str:
    return inject_slow_update_field("# Best Skill\n\nProse.\n", content=body)


# ─── fence-overlap raises ──────────────────────────────────────────────


def test_edit_file_mutating_fence_content_raises() -> None:
    parent = _parent_with_fence("ORIGINAL")
    # Proposed body keeps the fence markers but mutates the in-fence body.
    mutated = parent.replace("ORIGINAL", "TAMPERED")
    patch = f"<<<EDIT_FILE SKILL.md>>>\n{mutated}<<<END_FILE>>>\n"
    with pytest.raises(SentinelParseError) as excinfo:
        parse_sentinel_blocks(
            patch,
            existing_paths={"SKILL.md"},
            parent_skill_md=parent,
        )
    assert "slow_update_violation" in str(excinfo.value)


def test_edit_file_dropping_fence_raises() -> None:
    parent = _parent_with_fence("ORIGINAL")
    # Proposed body has no fence at all.
    patch = (
        "<<<EDIT_FILE SKILL.md>>>\n"
        "# Best Skill\n\nProse rewritten without the fence.\n"
        "<<<END_FILE>>>\n"
    )
    with pytest.raises(SentinelParseError, match="slow_update_violation"):
        parse_sentinel_blocks(
            patch,
            existing_paths={"SKILL.md"},
            parent_skill_md=parent,
        )


def test_delete_file_on_fenced_skill_md_raises() -> None:
    parent = _parent_with_fence()
    patch = "<<<DELETE_FILE SKILL.md>>>\n"
    with pytest.raises(SentinelParseError, match="slow_update_violation"):
        parse_sentinel_blocks(
            patch,
            existing_paths={"SKILL.md"},
            parent_skill_md=parent,
        )


def test_rewrite_folder_dropping_skill_md_fence_raises() -> None:
    parent = _parent_with_fence("ORIGINAL")
    # REWRITE_FOLDER body that produces a SKILL.md without the fence.
    patch = (
        "<<<REWRITE_FOLDER my_skill>>>\n"
        "--- file: my_skill/SKILL.md\n"
        "# Rewritten with no fence\n"
        "<<<END_REWRITE>>>\n"
    )
    with pytest.raises(SentinelParseError, match="slow_update_violation"):
        parse_sentinel_blocks(
            patch,
            existing_paths={"my_skill/SKILL.md"},
            parent_skill_md=parent,
        )


# ─── out-of-fence edits accepted ───────────────────────────────────────


def test_edit_file_preserving_fence_accepted() -> None:
    parent = _parent_with_fence("PROTECTED")
    # Keep the fence body byte-identical; mutate only the prose ABOVE.
    new_skill = parent.replace("Prose.", "Prose rewritten above fence.")
    patch = f"<<<EDIT_FILE SKILL.md>>>\n{new_skill}<<<END_FILE>>>\n"
    ops = parse_sentinel_blocks(
        patch,
        existing_paths={"SKILL.md"},
        parent_skill_md=parent,
    )
    assert len(ops) == 1
    assert ops[0].op == "EDIT_FILE"
    assert ops[0].path == "SKILL.md"


def test_edit_on_script_file_unaffected() -> None:
    parent = _parent_with_fence()
    # scripts/foo.py is NOT a SKILL.md — fence rules don't apply.
    patch = (
        "<<<EDIT_FILE scripts/foo.py>>>\n"
        "#!/usr/bin/env python3\nprint('hi')\n"
        "<<<END_FILE>>>\n"
    )
    ops = parse_sentinel_blocks(
        patch,
        existing_paths={"SKILL.md", "scripts/foo.py"},
        parent_skill_md=parent,
    )
    assert len(ops) == 1
    assert ops[0].op == "EDIT_FILE"
    assert ops[0].path == "scripts/foo.py"


def test_delete_file_on_script_unaffected() -> None:
    parent = _parent_with_fence()
    patch = "<<<DELETE_FILE scripts/stale.py>>>\n"
    ops = parse_sentinel_blocks(
        patch,
        existing_paths={"SKILL.md", "scripts/stale.py"},
        parent_skill_md=parent,
    )
    assert len(ops) == 1
    assert ops[0].op == "DELETE_FILE"
    assert ops[0].path == "scripts/stale.py"


# ─── back-compat: parent_skill_md=None disables check ──────────────────


def test_parent_skill_md_none_disables_fence_check() -> None:
    # Even an EDIT_FILE that would mutate the fenced region is accepted
    # when the parser isn't told about the parent.
    patch = "<<<EDIT_FILE SKILL.md>>>\n# Anything goes\n<<<END_FILE>>>\n"
    ops = parse_sentinel_blocks(
        patch,
        existing_paths={"SKILL.md"},
        # parent_skill_md NOT passed → back-compat path
    )
    assert len(ops) == 1
    assert ops[0].op == "EDIT_FILE"


def test_parent_without_fence_skips_check() -> None:
    # Parent SKILL.md without a fence — any EDIT is OK.
    parent_no_fence = "# Skill\n\nProse.\n"
    patch = "<<<EDIT_FILE SKILL.md>>>\n# Different\n<<<END_FILE>>>\n"
    ops = parse_sentinel_blocks(
        patch,
        existing_paths={"SKILL.md"},
        parent_skill_md=parent_no_fence,
    )
    assert len(ops) == 1


# Silence linter — markers re-used in fixture construction above.
_ = (SLOW_UPDATE_START, SLOW_UPDATE_END)
