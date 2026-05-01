"""R-12 anti-leakage scanner — Phase E (2026-04-29).

Scans a candidate :class:`FolderArtifact`'s SKILL.md files and any
``*/scripts/*.{sh,py}`` files for verbatim references to:

* Real task IDs (full + bare-segment forms; covered by the existing
  anonymizer's ``find_leaked_*_names``).
* Full task directory names (the bare segment, e.g.
  ``forensics-disk-recovery``).
* File paths from the task's ``environment/`` tree (e.g.
  ``environment/data.csv``, ``logs/syslog``).
* Magic numbers from ``tests/test_outputs.py`` — any literal int/float
  that the test asserts against.
* Exact commands from ``solution/solve.sh`` (full lines, ignoring
  comments + shebangs + blank lines, deduped).

The scanner runs against the bench-CLI exposed surface — the
materialized scene YAML + skill folder that ``BenchCliBackend``
mounts. We model that surface as the candidate's full FolderArtifact
plus, optionally, the materialized scene YAML path passed in from
the runner.

Returns a structured ``LeakScanResult`` so the policy layer
(``warn``/``zero``/``raise``) can decide what to do.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .folder_artifact import FolderArtifact

logger = logging.getLogger(__name__)


# Patterns / heuristics --------------------------------------------------

# ``test_outputs.py`` magic-number extraction. Match int and float
# literals (skip 0/1 which are too noisy, and 2-3 char ints which
# legitimately appear all over). The threshold of >=4 chars is a
# heuristic — the goal is to flag distinctive constants like 1234567.89
# or 0xdeadbeef without flagging "n=5".
_NUMBER_PATTERN = re.compile(
    r"(?<![\w.])(?:0[xX][0-9a-fA-F]+|"
    r"\d+\.\d+(?:[eE][+-]?\d+)?|"
    r"\d{4,})"
)

# A solve.sh command line is "interesting" if it contains a non-trivial
# program invocation. We strip leading/trailing whitespace and skip
# lines that are pure comments/shebangs/empty.
_SHEBANG_OR_COMMENT = re.compile(r"^\s*(#|$)")

# Files we scan inside the candidate artifact. Mirrors the seed
# loader's ``include_exts`` set in controller.py (~L78-L91): a leaked
# task name in a .json config or .yaml schema would otherwise slip
# through the scanner.
_SCANNED_SUFFIXES = (
    ".md",
    ".sh",
    ".py",
    ".jq",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".xml",
)
_SCANNED_PATH_SUBSTRINGS = ("/scripts/",)


@dataclass
class LeakHit:
    """One leaked-string occurrence."""

    file: str  # candidate-relative path
    kind: str  # "task_id" | "task_dir" | "env_path" | "magic_number" | "solve_cmd"
    needle: str  # the leaked literal
    context: str = ""  # short snippet (for the leak_warning blob)


@dataclass
class LeakScanResult:
    """Structured result of one R-12 scan over a candidate."""

    hits: List[LeakHit] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.hits

    def to_artifact_blob(self) -> str:
        """Compact one-blob summary suitable for ``EvaluationResult.artifacts``."""
        if not self.hits:
            return "clean"
        # Sort + dedupe for stable output.
        seen: Set[Tuple[str, str, str]] = set()
        lines: List[str] = []
        for h in self.hits:
            key = (h.file, h.kind, h.needle)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"{h.kind}\t{h.file}\t{h.needle}")
        return "\n".join(lines)


def _candidate_files_to_scan(
    artifact: FolderArtifact,
) -> List[Tuple[str, str]]:
    """Return ``[(path, content), ...]`` for files in the scanner's scope."""
    out: List[Tuple[str, str]] = []
    for path, content in artifact.files.items():
        if path.endswith(_SCANNED_SUFFIXES) or any(
            sub in path for sub in _SCANNED_PATH_SUBSTRINGS
        ):
            out.append((path, content or ""))
    return out


def _extract_env_paths(env_dir: Path) -> List[str]:
    """List relative paths under ``env_dir`` (excl. dotfiles + binaries).

    The strings returned are the full leaf names + any sub-segments
    long enough to be distinctive (>=5 chars, not all digits). The
    scanner uses these to detect verbatim references like
    ``environment/data/syslog`` in a SKILL.md.
    """
    if not env_dir.is_dir():
        return []
    out: List[str] = []
    for p in env_dir.rglob("*"):
        if p.is_dir():
            continue
        if p.name.startswith("."):
            continue
        rel = p.relative_to(env_dir).as_posix()
        # Add full relative + bare leaf, both forms.
        if len(rel) >= 5:
            out.append(rel)
        if len(p.name) >= 5:
            out.append(p.name)
    return sorted(set(out))


def _extract_magic_numbers(test_outputs_path: Path) -> List[str]:
    """Extract literal numbers from a tests/test_outputs.py-style file.

    Returns the numbers as canonical strings (for substring matching).
    Distinct values only; preserve original spelling so ``1234.5`` and
    ``1.2345e3`` are both checked.
    """
    if not test_outputs_path.is_file():
        return []
    try:
        text = test_outputs_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    seen: Set[str] = set()
    for m in _NUMBER_PATTERN.finditer(text):
        seen.add(m.group(0))
    return sorted(seen)


def _extract_solve_commands(solve_sh_path: Path) -> List[str]:
    """Pull command lines from solution/solve.sh.

    Drops comments, shebangs, and blank lines. Each surviving line is
    stripped and deduplicated. We match these as substrings inside
    candidate files so even partial copy-paste of the solution flags.
    """
    if not solve_sh_path.is_file():
        return []
    try:
        text = solve_sh_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    seen: Set[str] = set()
    for line in text.splitlines():
        if _SHEBANG_OR_COMMENT.match(line):
            continue
        stripped = line.strip()
        if len(stripped) < 8:
            # Trivially short lines (echo, fi, done, ls) over-flag.
            continue
        seen.add(stripped)
    return sorted(seen)


def _scan_for_substrings(
    files: Iterable[Tuple[str, str]],
    needles: Iterable[str],
    *,
    kind: str,
) -> List[LeakHit]:
    """Substring-scan ``files`` for any of ``needles``.

    Each (file, needle) pair contributes at most one hit (deduped).
    Empty/short needles are skipped to keep false-positive rate low.
    """
    hits: List[LeakHit] = []
    needle_list = [n for n in needles if n and len(n) >= 4]
    if not needle_list:
        return hits
    for path, content in files:
        if not content:
            continue
        for needle in needle_list:
            if needle in content:
                idx = content.find(needle)
                ctx = content[max(0, idx - 20) : idx + len(needle) + 20]
                hits.append(LeakHit(file=path, kind=kind, needle=needle, context=ctx))
    return hits


def scan_artifact(
    artifact: FolderArtifact,
    *,
    id_map: Optional[Dict[str, str]] = None,
    task_records: Optional[List[Any]] = None,
) -> LeakScanResult:
    """Run all R-12 checks on ``artifact``.

    Args:
        artifact: candidate skills folder.
        id_map: alias map produced by the anonymizer; keys are real
            task IDs (both qualified + bare). Used to detect raw
            task-name leaks.
        task_records: hydrated SkillsBench :class:`Task` records (or
            tblite tasks, but those don't carry environment/ data).
            When provided, the scanner extracts environment paths,
            magic numbers, and solve.sh commands per task.

    Returns:
        :class:`LeakScanResult` with all hits found.
    """
    files = _candidate_files_to_scan(artifact)
    if not files:
        return LeakScanResult()

    hits: List[LeakHit] = []

    # 1. Task IDs (qualified + bare). We only check the bare-segment
    #    side of id_map since fully-qualified strings would also match
    #    on bare. The anonymizer's find_leaked already covered .md;
    #    here we extend to scripts.
    if id_map:
        bare_ids = sorted({k for k in id_map if "/" not in k})
        hits.extend(_scan_for_substrings(files, bare_ids, kind="task_id"))

    # 2-4. Per-task environment / magic numbers / solve.sh.
    if task_records:
        env_paths_pool: Set[str] = set()
        magic_numbers_pool: Set[str] = set()
        solve_cmds_pool: Set[str] = set()
        for rec in task_records:
            payload = _extract_payload(rec)
            env_dir = payload.get("environment_dir")
            tests_dir = payload.get("tests_dir")
            task_dir = payload.get("task_dir")
            if env_dir:
                env_paths_pool.update(_extract_env_paths(Path(env_dir)))
            if tests_dir:
                magic_numbers_pool.update(
                    _extract_magic_numbers(Path(tests_dir) / "test_outputs.py")
                )
            if task_dir:
                solve_cmds_pool.update(
                    _extract_solve_commands(Path(task_dir) / "solution" / "solve.sh")
                )

        hits.extend(
            _scan_for_substrings(files, sorted(env_paths_pool), kind="env_path")
        )
        hits.extend(
            _scan_for_substrings(files, sorted(magic_numbers_pool), kind="magic_number")
        )
        hits.extend(
            _scan_for_substrings(files, sorted(solve_cmds_pool), kind="solve_cmd")
        )

    return LeakScanResult(hits=hits)


def _extract_payload(rec: Any) -> Dict[str, Any]:
    """Pull the success_check_payload off a Task or dict record."""
    payload = getattr(rec, "success_check_payload", None)
    if payload is None and isinstance(rec, dict):
        payload = rec.get("success_check_payload")
    return payload or {}
