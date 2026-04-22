"""Prompt-sampler regressions for the scripts/ affordance.

Phase 3 taught Track B's outer-mutation prompt that skill folders can
contain executable helpers under `<skill>/scripts/` and that those can
be CRUD'd via the existing ADD_FILE / EDIT_FILE / DELETE_FILE grammar.
These tests lock in the visible surface area of that change so it
doesn't silently regress.
"""

from __future__ import annotations

from skill_evolve.track_b.openevolve_skills.database import Program
from skill_evolve.track_b.openevolve_skills.folder_artifact import (
    FolderArtifact,
    MAX_FILES,
    MAX_TOTAL_BYTES,
)
from skill_evolve.track_b.openevolve_skills.prompt_sampler import (
    SYSTEM_TEMPLATE,
    PromptSampler,
)


# ------------------------------------------------------------------ fixtures

_SEED_SKILL_MD = """\
---
name: toy
description: A toy skill for prompt-sampler tests.
---

# Toy

Body.
"""


def _make_program() -> Program:
    art = FolderArtifact(files={"toy/SKILL.md": _SEED_SKILL_MD})
    return Program(
        id="p0",
        artifact=art,
        metrics={"composite": 0.0, "success_rate": 0.0,
                 "tool_calls_per_success": 0.0},
    )


# ------------------------------------------------------------------ SYSTEM

def test_system_prompt_mentions_scripts_subdir():
    # The outer LLM needs to know scripts/ exists as a first-class
    # affordance, not just a made-up directory the validator would reject.
    assert "scripts/" in SYSTEM_TEMPLATE
    # And the other recognised Hermes subdirs.
    for subdir in ("references/", "templates/", "assets/"):
        assert subdir in SYSTEM_TEMPLATE, f"missing {subdir}"
    # Explicit +x contract so the LLM stops worrying about permissions.
    assert "executable bit" in SYSTEM_TEMPLATE
    # And the task-container convention.
    assert "/app" in SYSTEM_TEMPLATE


# ------------------------------------------------------------------ USER

def test_user_prompt_has_shebanged_script_example():
    sampler = PromptSampler()
    rendered = sampler.build(_make_program())
    user = rendered.user

    # "How to reply" section present.
    assert "## How to reply" in user
    # ADD_FILE for a SKILL.md is shown.
    assert "<<<ADD_FILE my_skill/SKILL.md>>>" in user
    # ADD_FILE for a script with a bash shebang is shown.
    assert "<<<ADD_FILE my_skill/scripts/analyze.sh>>>" in user
    assert "#!/usr/bin/env bash" in user
    assert "set -euo pipefail" in user
    # EDIT_FILE rewriting a script is shown.
    assert "<<<EDIT_FILE my_skill/scripts/analyze.sh>>>" in user
    # DELETE_FILE for a script is shown.
    assert "<<<DELETE_FILE my_skill/scripts/stale_helper.sh>>>" in user
    # Shebang rule spelled out in constraints.
    assert "shebang" in user.lower()
    # Python shebang also called out (for python helpers).
    assert "#!/usr/bin/env python3" in user


def test_user_prompt_reflects_bumped_limits():
    sampler = PromptSampler()
    rendered = sampler.build(_make_program())
    user = rendered.user
    # Constraints block pulls from folder_artifact constants now.
    assert f"<= {MAX_FILES} files" in user
    assert f"<= {MAX_TOTAL_BYTES // 1024} KiB" in user


def test_user_prompt_has_script_vs_prose_guidance():
    # The "when to reach for a script vs edit prose" heuristic from the
    # task brief — locked in so it doesn't vanish in a future tidy-up.
    # Compared with whitespace collapsed so line-wrapping doesn't defeat
    # us.
    sampler = PromptSampler()
    rendered = sampler.build(_make_program())
    flat = " ".join(rendered.user.split())
    assert "Prefer adding/editing a script" in flat
    assert "Prefer editing SKILL.md prose" in flat


def test_prompt_sampler_defaults_track_folder_artifact_limits():
    sampler = PromptSampler()
    assert sampler.max_files == MAX_FILES
    assert sampler.max_total_bytes == MAX_TOTAL_BYTES
