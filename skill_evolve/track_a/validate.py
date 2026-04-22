"""Validation invariants for a candidate skills folder.

A folder is valid iff:

* Each subfolder contains a ``SKILL.md`` with parseable YAML frontmatter.
* Required fields present: ``name``, ``description``.
* ``name`` in frontmatter matches the folder name.
* No duplicate skill names across the folder.
* Total folder size < 100KB (anti-runaway guard).
* >= 1 skill, <= 20 skills.

:func:`validate` returns ``(ok: bool, errors: list[str])``. Operators call
it after every mutation; if it fails we reject the mutation and fall back
or retry.
"""

from __future__ import annotations

from typing import List, Tuple

from .folder import SkillFolder

MAX_TOTAL_BYTES = 500 * 1024  # 500 KiB (bumped from 100 KiB on 2026-04-22
#                               for code-bearing skills with scripts/
#                               subdirs; realworld SWE scripts are 5-20
#                               KiB each, 5-skill seeds land ~30-50 KiB).
MIN_SKILLS = 1
MAX_SKILLS = 20

REQUIRED_FIELDS = ("name", "description")


def validate(folder: SkillFolder) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    n = len(folder.skills)
    if n < MIN_SKILLS:
        errors.append(f"skill count {n} < MIN_SKILLS={MIN_SKILLS}")
    if n > MAX_SKILLS:
        errors.append(f"skill count {n} > MAX_SKILLS={MAX_SKILLS}")

    seen_names: set[str] = set()
    for s in folder.skills:
        prefix = f"skill '{s.folder_name}'"
        if not s.folder_name or "/" in s.folder_name or s.folder_name.startswith("."):
            errors.append(f"{prefix}: invalid folder name")
        if not isinstance(s.frontmatter, dict) or not s.frontmatter:
            errors.append(f"{prefix}: missing or unparseable YAML frontmatter")
            continue
        for field_name in REQUIRED_FIELDS:
            v = s.frontmatter.get(field_name)
            if v is None or (isinstance(v, str) and not v.strip()):
                errors.append(f"{prefix}: frontmatter missing '{field_name}'")
        declared = s.frontmatter.get("name")
        if isinstance(declared, str) and declared and declared != s.folder_name:
            errors.append(
                f"{prefix}: frontmatter name '{declared}' != folder '{s.folder_name}'"
            )
        if s.folder_name in seen_names:
            errors.append(f"{prefix}: duplicate skill name")
        seen_names.add(s.folder_name)
        # Also guard duplicates of the *declared* name (YAML name) separately —
        # two folders could have different dir names but the same YAML name.
        # The folder-name check above catches the dir-name case; here we
        # ensure the YAML names are also unique.
    seen_yaml: set[str] = set()
    for s in folder.skills:
        yname = s.frontmatter.get("name")
        if isinstance(yname, str) and yname:
            if yname in seen_yaml:
                errors.append(f"duplicate YAML name '{yname}'")
            seen_yaml.add(yname)

    size = folder.total_bytes()
    if size > MAX_TOTAL_BYTES:
        errors.append(f"total size {size}B > MAX_TOTAL_BYTES={MAX_TOTAL_BYTES}B")

    return (len(errors) == 0, errors)


def validates(folder: SkillFolder) -> bool:
    """Convenience: just the boolean."""
    ok, _ = validate(folder)
    return ok
