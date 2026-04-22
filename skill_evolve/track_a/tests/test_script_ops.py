"""Script operator unit tests.

Covers AddScript / RewriteScript / RemoveScript + the _coerce_op_record
gating for each. Synthetic LLM only (no OpenRouter calls); behavioral
assertions are on folder state + path / shebang invariants.
"""

from __future__ import annotations

import pytest

from skill_evolve.track_a.folder import SkillFolder, SkillDoc
from skill_evolve.track_a.llm import LLMClient
from skill_evolve.track_a.ops import (
    _coerce_op_record,
    _validate_script_path,
    apply_add_script,
    apply_remove_script,
    apply_rewrite_script,
)


@pytest.fixture
def small_folder() -> SkillFolder:
    return SkillFolder(skills=[
        SkillDoc(
            folder_name="alpha",
            frontmatter={"name": "alpha", "description": "d"},
            body="body\n",
        ),
    ])


@pytest.fixture
def synth_client() -> LLMClient:
    return LLMClient(synthetic=True)


# --- path validator ------------------------------------------------------

@pytest.mark.parametrize("path", [
    "scripts/run.sh",
    "scripts/util/helper.sh",
    "scripts/tool.py",
])
def test_validate_script_path_accepts_valid(path: str) -> None:
    assert _validate_script_path(path) == path


@pytest.mark.parametrize("path", [
    "run.sh",               # no scripts/ prefix
    "references/spec.md",   # wrong subdir
    "scripts/",             # no filename
    "/abs/scripts/x.sh",    # absolute
    "scripts/../etc/x.sh",  # traversal
    "scripts/tool.txt",     # wrong extension
    "scripts/tool",         # no extension
])
def test_validate_script_path_rejects_invalid(path: str) -> None:
    with pytest.raises(ValueError):
        _validate_script_path(path)


# --- apply_add_script ----------------------------------------------------

def test_add_script_creates_with_shebang(
    small_folder: SkillFolder, synth_client: LLMClient,
) -> None:
    out = apply_add_script(
        small_folder, synth_client,
        skill="alpha",
        path="scripts/hello.sh",
        purpose="Write a probe marker",
    )
    content = out.by_name("alpha").auxiliary_files["scripts/hello.sh"]
    assert content.startswith("#!"), "shebang must be present"
    assert content.endswith("\n"), "trailing newline guaranteed"


def test_add_script_rejects_collision(
    small_folder: SkillFolder, synth_client: LLMClient,
) -> None:
    folder = small_folder
    folder.skills[0].auxiliary_files["scripts/hello.sh"] = "#!/bin/sh\n"
    with pytest.raises(ValueError, match="already exists"):
        apply_add_script(
            folder, synth_client,
            skill="alpha", path="scripts/hello.sh", purpose="",
        )


def test_add_script_rejects_missing_skill(
    small_folder: SkillFolder, synth_client: LLMClient,
) -> None:
    with pytest.raises(KeyError, match="not in folder"):
        apply_add_script(
            small_folder, synth_client,
            skill="ghost", path="scripts/x.sh", purpose="",
        )


def test_add_script_rejects_bad_path(
    small_folder: SkillFolder, synth_client: LLMClient,
) -> None:
    with pytest.raises(ValueError):
        apply_add_script(
            small_folder, synth_client,
            skill="alpha", path="references/doc.md", purpose="",
        )


# --- apply_rewrite_script ------------------------------------------------

def test_rewrite_script_updates_content(
    small_folder: SkillFolder, synth_client: LLMClient,
) -> None:
    folder = small_folder
    folder.skills[0].auxiliary_files["scripts/tool.sh"] = (
        "#!/bin/sh\n# old content\n"
    )
    out = apply_rewrite_script(
        folder, synth_client,
        skill="alpha", path="scripts/tool.sh", critique="too short",
    )
    new_content = out.by_name("alpha").auxiliary_files["scripts/tool.sh"]
    assert "old content" not in new_content
    assert new_content.startswith("#!"), "shebang preserved / re-added"


def test_rewrite_script_missing_path_raises(
    small_folder: SkillFolder, synth_client: LLMClient,
) -> None:
    with pytest.raises(KeyError, match="not in skill"):
        apply_rewrite_script(
            small_folder, synth_client,
            skill="alpha", path="scripts/missing.sh", critique="",
        )


# --- apply_remove_script -------------------------------------------------

def test_remove_script_deletes(
    small_folder: SkillFolder, synth_client: LLMClient,
) -> None:
    folder = small_folder
    folder.skills[0].auxiliary_files["scripts/tool.sh"] = "#!/bin/sh\n"
    out = apply_remove_script(
        folder, synth_client, skill="alpha", path="scripts/tool.sh",
    )
    assert "scripts/tool.sh" not in out.by_name("alpha").auxiliary_files


def test_remove_script_missing_path_raises(
    small_folder: SkillFolder, synth_client: LLMClient,
) -> None:
    with pytest.raises(KeyError):
        apply_remove_script(
            small_folder, synth_client,
            skill="alpha", path="scripts/ghost.sh",
        )


# --- _coerce_op_record gating -------------------------------------------

def test_coerce_add_script_rejects_collision(small_folder: SkillFolder) -> None:
    folder = small_folder
    folder.skills[0].auxiliary_files["scripts/hello.sh"] = "#!/bin/sh\n"
    parsed = {
        "op": "AddScript", "skill": "alpha",
        "path": "scripts/hello.sh", "purpose": "x",
    }
    assert _coerce_op_record(parsed, folder) is None


def test_coerce_add_script_accepts_new_path(small_folder: SkillFolder) -> None:
    parsed = {
        "op": "AddScript", "skill": "alpha",
        "path": "scripts/new.sh", "purpose": "x",
    }
    rec = _coerce_op_record(parsed, small_folder)
    assert rec is not None
    assert rec.op == "AddScript"
    assert rec.args["path"] == "scripts/new.sh"


def test_coerce_rewrite_script_rejects_missing(small_folder: SkillFolder) -> None:
    parsed = {
        "op": "RewriteScript", "skill": "alpha",
        "path": "scripts/ghost.sh", "critique_excerpt": "",
    }
    assert _coerce_op_record(parsed, small_folder) is None


def test_coerce_rewrite_script_accepts_existing(small_folder: SkillFolder) -> None:
    small_folder.skills[0].auxiliary_files["scripts/tool.sh"] = "#!/bin/sh\n"
    parsed = {
        "op": "RewriteScript", "skill": "alpha",
        "path": "scripts/tool.sh", "critique_excerpt": "fix it",
    }
    rec = _coerce_op_record(parsed, small_folder)
    assert rec is not None
    assert rec.op == "RewriteScript"


def test_coerce_remove_script_rejects_bad_path(small_folder: SkillFolder) -> None:
    parsed = {
        "op": "RemoveScript", "skill": "alpha",
        "path": "../etc/bad",
    }
    assert _coerce_op_record(parsed, small_folder) is None


def test_coerce_script_op_rejects_unknown_skill(small_folder: SkillFolder) -> None:
    for op in ("AddScript", "RewriteScript", "RemoveScript"):
        parsed = {"op": op, "skill": "ghost", "path": "scripts/x.sh"}
        assert _coerce_op_record(parsed, small_folder) is None
