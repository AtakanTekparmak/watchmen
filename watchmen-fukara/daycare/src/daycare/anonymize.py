"""Anonymization layer (M7 + spec §Anonymization).

Every field the proposer sees runs through ``strip()`` first. The
output-side leak scanner (Phase 3b.5) is the safety net; this is the
first line of defence.

Replacement order matters — UUID first (least context-sensitive), then
absolute paths, then narrow patterns. Slug replacement is whole-word so
substrings like "ctf" don't randomly hit ``<REPO_SLUG>`` inside the word
"ctfsomething".
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


# ─── Regex constants ──────────────────────────────────────────────────────

_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
# Email: simple but covers real-world cases. Avoids matching @ inside URLs.
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
# OpenRouter API keys: "sk-or-" prefix, then base64-ish + hyphens/underscores.
_OR_KEY_RE = re.compile(r"sk-or-[A-Za-z0-9\-_]+")


@dataclass
class AnonymizeContext:
    """Project-specific slug sets used by ``strip()``.

    project_slugs: every GitHub repo basename or project identifier from
        projects.json — replaced with ``<REPO_SLUG>``.
    skill_slugs: every directory name under ``bundles/<project>/skills/``
        — replaced with ``<SKILL_SLUG>``.
    source_repo: the absolute path to the project root — replaced with
        ``<PROJECT_ROOT>`` BEFORE the user-home replacement runs (so the
        more specific match wins).
    user_home: defaults to ``~`` for the current user. Matches in absolute
        paths get rewritten to ``/<USER>``.
    """

    project_slugs: set[str] = field(default_factory=set)
    skill_slugs: set[str] = field(default_factory=set)
    source_repo: str = ""
    user_home: str = field(default_factory=lambda: str(Path.home()))


def _whole_word_pattern(slug: str) -> re.Pattern:
    """Compile a whole-word regex for ``slug`` (escaped).

    Whole-word boundaries (\b) are needed so "ctf" doesn't match inside
    "ctfsomething". Slugs with hyphens still get \b boundaries since
    Python's \b treats hyphen as a non-word char — which is what we want
    here (it lets us match "my-skill" cleanly).
    """
    return re.compile(rf"\b{re.escape(slug)}\b")


def strip(text: str, ctx: AnonymizeContext) -> str:
    """Run all replacements in spec-mandated order. Idempotent.

    Order:
      1. UUIDs → ``<SESSION_ID>``
      2. Absolute paths under user_home → ``/<USER>/...``
      3. source_repo literal → ``<PROJECT_ROOT>``
      4. Emails → ``<EMAIL>``
      5. OR API keys → ``<OR_KEY>``
      6. Each project_slug (whole-word) → ``<REPO_SLUG>``
      7. Each skill_slug (whole-word) → ``<SKILL_SLUG>``
    """
    if not isinstance(text, str) or not text:
        return text

    out = text

    # 1. UUIDs.
    out = _UUID_RE.sub("<SESSION_ID>", out)

    # 2. Absolute home-prefixed paths. We replace the user_home prefix
    # (not the bare username) so /Users/alice/Desktop/work/foo →
    # /<USER>/Desktop/work/foo. Doing this before source_repo would
    # rewrite source_repo to /<USER>/... before step 3 had a chance —
    # so we do source_repo first, then user_home.

    # 3. source_repo literal → <PROJECT_ROOT>. Has to run before user_home
    # so the more-specific project path wins.
    if ctx.source_repo:
        out = out.replace(ctx.source_repo, "<PROJECT_ROOT>")

    # 2 (continued). user_home prefix replacement.
    if ctx.user_home:
        out = out.replace(ctx.user_home, "/<USER>")

    # 4. Emails.
    out = _EMAIL_RE.sub("<EMAIL>", out)

    # 5. OR API keys.
    out = _OR_KEY_RE.sub("<OR_KEY>", out)

    # 6. Project slugs (whole-word).
    for slug in ctx.project_slugs:
        if not slug:
            continue
        out = _whole_word_pattern(slug).sub("<REPO_SLUG>", out)

    # 7. Skill slugs (whole-word).
    for slug in ctx.skill_slugs:
        if not slug:
            continue
        out = _whole_word_pattern(slug).sub("<SKILL_SLUG>", out)

    return out


def _walk_and_strip(value, ctx: AnonymizeContext):
    """Recursively descend a JSON-like value, applying ``strip()`` to every
    string. Lists and dicts are descended; other scalars pass through."""
    if isinstance(value, str):
        return strip(value, ctx)
    if isinstance(value, dict):
        return {k: _walk_and_strip(v, ctx) for k, v in value.items()}
    if isinstance(value, list):
        return [_walk_and_strip(item, ctx) for item in value]
    return value


def tool_call_strip(tool_call: dict, ctx: AnonymizeContext) -> dict:
    """Deep-copy ``tool_call`` and run ``strip()`` on every string value.

    MCP tool args (per the spec's failure-mode #5) often contain paths
    and slugs that the text-field anonymizer never sees — this recursion
    catches them.
    """
    cloned = copy.deepcopy(tool_call)
    return _walk_and_strip(cloned, ctx)


def build_context(projects_json_path: Path, bundle_dir: Path, source_repo: str) -> AnonymizeContext:
    """Construct an AnonymizeContext from on-disk metadata.

    project_slugs comes from projects.json (top-level keys are project
    slugs in watchmen's schema; we also pull any ``source_repo`` basename
    we find so the regex catches both forms).

    skill_slugs comes from the directories under ``bundle_dir/skills/``.
    """
    project_slugs: set[str] = set()
    if projects_json_path.exists():
        try:
            data = json.loads(projects_json_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for key, val in data.items():
                    if isinstance(key, str):
                        project_slugs.add(key)
                    # Also pick up the basename of any source_repo
                    # value so e.g. {"ctf": {"source_repo": "/.../ctf"}}
                    # contributes "ctf" twice (harmless) and any other
                    # repo basenames in the file.
                    if isinstance(val, dict):
                        sr = val.get("source_repo")
                        if isinstance(sr, str) and sr:
                            project_slugs.add(Path(sr).name)
        except (json.JSONDecodeError, OSError):
            # Bad projects.json — proceed with empty set; the leak scanner
            # is the safety net.
            pass

    skill_slugs: set[str] = set()
    skills_dir = bundle_dir / "skills"
    if skills_dir.exists() and skills_dir.is_dir():
        for entry in skills_dir.iterdir():
            if entry.is_dir():
                skill_slugs.add(entry.name)

    return AnonymizeContext(
        project_slugs=project_slugs,
        skill_slugs=skill_slugs,
        source_repo=source_repo,
    )
