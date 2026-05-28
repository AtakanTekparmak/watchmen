"""Skill-folder I/O helpers: load, clone, render, introspect.

A "skill folder" is a directory of ``<skill_name>/SKILL.md`` subdirs. Each
``SKILL.md`` starts with a YAML frontmatter block (``---`` fenced) followed
by free-form markdown body.

We keep representations dirt-simple:

* :class:`SkillDoc` — a parsed (frontmatter, body) pair plus the folder name.
* :class:`SkillFolder` — a list of SkillDoc + the filesystem path they came
  from. ``SkillFolder.write(dest)`` materializes back to disk.

The rest of the track_a package manipulates ``SkillFolder`` in-memory, then
writes the winning candidate to ``pass_<N>/{A,B,AB}/`` on disk so the
evaluator can chew on it.
"""

from __future__ import annotations

import copy
import io
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


FRONTMATTER_FENCE = "---"

# Hermes-agent recognises these 4 subdirs inside a skill folder and
# exposes their files to the agent via ``skill_view(name, file_path=...)``.
# Matches hermes-agent/tools/skills_tool.py:981-987. kai-skills mirrors
# the same convention so evolved skills work unchanged inside hermes.
AUX_SUBDIRS: tuple[str, ...] = ("scripts", "references", "templates", "assets")

# File extensions that should be materialized with the executable bit
# set (relative to their parent skill dir). Only applies to files under
# ``scripts/`` — other subdirs (references/templates/assets) are read-only
# data and stay 0644.
EXECUTABLE_EXTS: frozenset[str] = frozenset({".sh", ".py"})


@dataclass
class SkillDoc:
    """A single SKILL.md file, parsed into frontmatter dict + body text.

    ``folder_name`` is the directory that contains the SKILL.md — it must
    equal ``frontmatter['name']`` for the folder to validate.

    ``auxiliary_files`` holds sibling files under ``scripts/``,
    ``references/``, ``templates/``, or ``assets/`` subdirs of the skill
    dir. Keys are POSIX relpaths under the skill root (e.g.
    ``"scripts/hello.sh"``). Values are UTF-8 text contents. We restrict
    to hermes's recognised subdirs so other stray directories don't get
    mistaken for the skill's own code; binary files are not supported
    (they can't round-trip through the text artifact serialization).
    """

    folder_name: str
    frontmatter: Dict[str, Any]
    body: str
    auxiliary_files: Dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        v = self.frontmatter.get("name", "")
        return str(v) if v is not None else ""

    @property
    def description(self) -> str:
        v = self.frontmatter.get("description", "")
        return str(v) if v is not None else ""

    def render(self) -> str:
        """Render back to the on-disk SKILL.md format (YAML frontmatter + body)."""
        buf = io.StringIO()
        buf.write(FRONTMATTER_FENCE)
        buf.write("\n")
        # sort_keys=False keeps the human-authored ordering intact.
        yaml.safe_dump(
            self.frontmatter,
            buf,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
        buf.write(FRONTMATTER_FENCE)
        buf.write("\n\n")
        buf.write(self.body.rstrip())
        buf.write("\n")
        return buf.getvalue()

    def clone(self) -> "SkillDoc":
        return SkillDoc(
            folder_name=self.folder_name,
            frontmatter=copy.deepcopy(self.frontmatter),
            body=self.body,
            auxiliary_files=dict(self.auxiliary_files),
        )


@dataclass
class SkillFolder:
    """A collection of SkillDocs loaded from (or to be written to) a dir."""

    skills: List[SkillDoc] = field(default_factory=list)
    source_path: Optional[Path] = None

    # ------------- loading ----------------------------------------------

    @classmethod
    def load(cls, path: Path) -> "SkillFolder":
        path = Path(path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        skills: List[SkillDoc] = []
        for child in sorted(path.iterdir()):
            if not child.is_dir():
                continue
            skill_md = child / "SKILL.md"
            if not skill_md.exists():
                continue
            doc = _parse_skill_md(
                child.name,
                skill_md.read_text(encoding="utf-8"),
            )
            # Load auxiliary files from the 4 hermes-native subdirs.
            # Non-recognised subdirs are ignored so a ``tests/`` or
            # ``draft/`` dir doesn't land in auxiliary_files.
            for sub in AUX_SUBDIRS:
                sub_dir = child / sub
                if not sub_dir.is_dir():
                    continue
                for aux in sorted(sub_dir.rglob("*")):
                    if not aux.is_file():
                        continue
                    rel = aux.relative_to(child).as_posix()
                    try:
                        content = aux.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        # Skip binaries — text-only artifact invariant.
                        continue
                    doc.auxiliary_files[rel] = content
            skills.append(doc)
        return cls(skills=skills, source_path=path)

    # ------------- mutation ---------------------------------------------

    def clone(self) -> "SkillFolder":
        return SkillFolder(
            skills=[s.clone() for s in self.skills],
            source_path=self.source_path,
        )

    def by_name(self, name: str) -> Optional[SkillDoc]:
        for s in self.skills:
            if s.folder_name == name or s.name == name:
                return s
        return None

    def names(self) -> List[str]:
        return [s.folder_name for s in self.skills]

    def remove(self, name: str) -> bool:
        before = len(self.skills)
        self.skills = [s for s in self.skills if s.folder_name != name]
        return len(self.skills) < before

    def add(self, doc: SkillDoc) -> None:
        if self.by_name(doc.folder_name):
            raise ValueError(f"skill '{doc.folder_name}' already exists")
        self.skills.append(doc)

    def rename(self, old: str, new: str) -> None:
        doc = self.by_name(old)
        if not doc:
            raise KeyError(old)
        if any(s.folder_name == new for s in self.skills if s is not doc):
            raise ValueError(f"target name '{new}' already in use")
        doc.folder_name = new
        doc.frontmatter["name"] = new

    # ------------- persistence ------------------------------------------

    def write(self, dest: Path, *, smoke_test: bool = False) -> Path:
        """Materialize the folder to ``dest``.

        When ``smoke_test`` is True, after writing we run a syntax
        validation pass (``py_compile`` on .py, ``bash -n`` on .sh)
        plus a token-cap check via
        :func:`skill_evolve.shared.bundle_ops.bundle_tokens`. On
        failure we raise :class:`BundleRejected` so the runner pass
        loop can record a "rejected" candidate without an eval call.
        Default is False to preserve existing call-sites' behavior;
        runner.py opts in based on ``--smoke-test``.
        """
        dest = Path(dest).expanduser().resolve()
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        for s in self.skills:
            sdir = dest / s.folder_name
            sdir.mkdir(parents=True)
            (sdir / "SKILL.md").write_text(s.render(), encoding="utf-8")
            for rel, content in s.auxiliary_files.items():
                target = sdir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                # Scripts under scripts/ with a recognised executable
                # extension get +x so the agent can ``bash`` / ``python3``
                # them directly. Other aux files (refs/templates/assets)
                # stay 0644.
                if rel.startswith("scripts/") and target.suffix in EXECUTABLE_EXTS:
                    target.chmod(0o755)
        if smoke_test:
            # Import here to avoid a circular import at module load
            # (shared/bundle_ops imports patch_parser which doesn't
            # touch folder.py — but keeping the import local also keeps
            # the cost out of the non-gated write path).
            from skill_evolve.shared.bundle_ops import (
                BundleRejected,
                MAX_BUNDLE_TOKENS,
                bundle_tokens,
                validate_scripts,
            )

            smoke = validate_scripts(dest)
            if not smoke.ok:
                reasons = "; ".join(f"{rel}: {msg}" for rel, msg in smoke.failures)
                raise BundleRejected(f"smoke_failed: {reasons}")
            tokens = bundle_tokens(dest)
            if tokens > MAX_BUNDLE_TOKENS:
                raise BundleRejected(
                    f"token_cap_exceeded: {tokens} > {MAX_BUNDLE_TOKENS}"
                )
        return dest

    # ------------- introspection ----------------------------------------

    def total_bytes(self) -> int:
        total = 0
        for s in self.skills:
            total += len(s.render().encode("utf-8"))
            for content in s.auxiliary_files.values():
                total += len(content.encode("utf-8"))
        return total

    def render_summary(self, *, body_chars: int = 1200) -> str:
        """Compact human/LLM-readable render of the whole folder.

        Trims each skill body to ``body_chars`` chars so LLM context stays
        bounded. Used in Critic / Synth prompts.
        """
        parts: List[str] = []
        for s in self.skills:
            body = s.body.strip()
            if len(body) > body_chars:
                body = body[:body_chars].rstrip() + "\n...[truncated]"
            aux_line = ""
            if s.auxiliary_files:
                # Show auxiliary paths (not contents) so outer prompts
                # know scripts/refs exist and can choose to mutate them.
                aux_line = (
                    "auxiliary_files: "
                    + ", ".join(sorted(s.auxiliary_files.keys()))
                    + "\n"
                )
            parts.append(
                f"### skill: {s.folder_name}\n"
                f"name: {s.name}\n"
                f"description: {s.description}\n"
                f"{aux_line}\n"
                f"{body}\n"
            )
        if not parts:
            return "(empty skill folder)"
        return "\n---\n".join(parts)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_skill_md(folder_name: str, text: str) -> SkillDoc:
    """Split a SKILL.md into frontmatter dict + body string.

    Accepts standard ``---\\n<yaml>\\n---\\n<body>`` form. If no frontmatter
    is present we return an empty frontmatter dict and the full text as body.
    """
    stripped = text.lstrip("\ufeff")  # drop BOM if any
    if not stripped.startswith(FRONTMATTER_FENCE):
        return SkillDoc(folder_name=folder_name, frontmatter={}, body=stripped)

    # Find closing fence. We scan line-by-line so a literal "---" inside the
    # YAML body can't confuse us (valid YAML frontmatter ends at a line that
    # is *exactly* "---").
    lines = stripped.splitlines()
    if not lines or lines[0].strip() != FRONTMATTER_FENCE:
        return SkillDoc(folder_name=folder_name, frontmatter={}, body=stripped)

    close_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == FRONTMATTER_FENCE:
            close_idx = i
            break
    if close_idx is None:
        # Malformed: opening fence with no close. Treat whole file as body
        # so validators flag it.
        return SkillDoc(folder_name=folder_name, frontmatter={}, body=stripped)

    fm_text = "\n".join(lines[1:close_idx])
    body = "\n".join(lines[close_idx + 1 :]).lstrip("\n")
    try:
        fm = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError:
        fm = {}
    if not isinstance(fm, dict):
        fm = {}
    return SkillDoc(folder_name=folder_name, frontmatter=fm, body=body)


def make_skill_doc(
    folder_name: str,
    *,
    name: Optional[str] = None,
    description: str = "",
    body: str = "",
    extra_frontmatter: Optional[Dict[str, Any]] = None,
) -> SkillDoc:
    """Build a SkillDoc with valid frontmatter. ``name`` defaults to folder_name."""
    fm: Dict[str, Any] = {
        "name": name or folder_name,
        "description": description,
        "version": "0.1.0",
        "author": "Hermes Agent (track_a)",
        "license": "MIT",
    }
    if extra_frontmatter:
        fm.update(extra_frontmatter)
    return SkillDoc(folder_name=folder_name, frontmatter=fm, body=body)
