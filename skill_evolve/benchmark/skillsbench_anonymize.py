"""SkillsBench-scope task-name anonymization.

Mirror of the in-file functions at
``skill_evolve.track_b.openevolve_skills.evaluator`` (lines 56-156).
Same algorithm — alpha-sort the IDs, assign stable ``task_NNN`` aliases,
redact both fully-qualified (``skillsbench/foo``) and bare (``foo``)
forms in evaluator-visible artifacts. We split into a separate module
so the SkillsBench id-map domain (the 20-task subset) doesn't collide
with the existing tblite/swebench manifest sweep.

The functions take a pre-built ``id_map`` rather than constructing one
themselves; the builder lives at :func:`build_skillsbench_id_map` and
is meant to be called once at controller setup.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

from skill_evolve.benchmark.load import Task
from skill_evolve.track_b.openevolve_skills.folder_artifact import FolderArtifact


def _short_name(task_id: str) -> str:
    """Return the bare segment of a task_id (drops "<source>/" prefix)."""
    return task_id.rsplit("/", 1)[-1]


def build_skillsbench_id_map(task_records: Iterable[Task]) -> Dict[str, str]:
    """Build a stable task_id -> ``task_NNN`` mapping for SkillsBench.

    Accepts an iterable of :class:`Task` records (typically the hydrated
    20-task subset). IDs are alpha-sorted; both the fully-qualified form
    (``skillsbench/forensics-disk-recovery``) and the bare segment
    (``forensics-disk-recovery``) map to the same alias so substring
    sanitization catches either spelling.
    """
    ordered = sorted(
        {(t.task_id if isinstance(t, Task) else t["task_id"]) for t in task_records}
    )
    mapping: Dict[str, str] = {}
    for i, tid in enumerate(ordered, start=1):
        alias = f"task_{i:03d}"
        mapping[tid] = alias
        bare = _short_name(tid)
        mapping.setdefault(bare, alias)
    return mapping


def sanitize_text_skillsbench(
    text: str, id_map: Dict[str, str]
) -> Tuple[str, List[str]]:
    """Replace every SkillsBench task name in ``text`` with its alias.

    Returns ``(sanitized_text, hits)`` — ``hits`` is the deduplicated
    list of original names that matched, in match order. Longer keys
    are matched first so ``skillsbench/foo`` wins over ``foo`` when
    both appear in the mapping.
    """
    if not text or not id_map:
        return text, []
    keys = sorted(id_map.keys(), key=len, reverse=True)
    pattern = re.compile("(" + "|".join(re.escape(k) for k in keys) + r")\b")
    hits: List[str] = []
    seen: set[str] = set()

    def _sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in seen:
            seen.add(name)
            hits.append(name)
        return id_map[name]

    return pattern.sub(_sub, text), hits


def sanitize_artifact_skillsbench(
    artifact: FolderArtifact, id_map: Dict[str, str]
) -> Tuple[FolderArtifact, List[Tuple[str, List[str]]]]:
    """In-place sanitize ``*.md`` files of ``artifact``.

    Returns ``(artifact, [(path, hits), ...])``. Non-md files (scripts,
    templates, etc.) are left untouched — the redaction target is the
    prose layer the router/outer-LLM reads. The artifact itself is
    mutated and also returned for caller convenience.
    """
    detail: List[Tuple[str, List[str]]] = []
    for path in list(artifact.files):
        if not path.endswith(".md"):
            continue
        original = artifact.files[path]
        cleaned, hits = sanitize_text_skillsbench(original, id_map)
        if hits:
            artifact.files[path] = cleaned
            detail.append((path, hits))
    return artifact, detail


def find_leaked_skillsbench_names(
    artifact: FolderArtifact, id_map: Dict[str, str]
) -> List[Tuple[str, str]]:
    """Return ``[(file, name), ...]`` for every verbatim task name found
    in any ``*.md`` file of ``artifact``. Empty list = clean."""
    leaks: List[Tuple[str, str]] = []
    if not id_map:
        return leaks
    keys = sorted(id_map.keys(), key=len, reverse=True)
    pattern = re.compile("(" + "|".join(re.escape(k) for k in keys) + r")\b")
    for path, content in artifact.files.items():
        if not path.endswith(".md"):
            continue
        for m in pattern.finditer(content or ""):
            leaks.append((path, m.group(1)))
    return leaks
