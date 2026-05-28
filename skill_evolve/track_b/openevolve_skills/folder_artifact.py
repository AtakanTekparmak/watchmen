"""FolderArtifact — a ``dict[path, content]`` with deterministic ser/de.

The upstream openevolve ``Program.code`` is a single ``str``. Track B
evolves *directories* of skill files, so we replace ``code: str`` with
``artifact: FolderArtifact``. Everything that used to hash, serialize, or
write ``.code`` now routes through this class.

Design:

* ``files`` is a plain ``dict[relpath_str, content_str]``. ``relpath_str``
  is always a POSIX path ("skills/foo/SKILL.md") — never absolute, never
  contains ``..``.
* ``serialize()`` yields a canonical textual form (paths sorted, one
  header per file) so two artifacts with the same contents hash identical
  regardless of insertion order. This is what MAP-Elites dedup keys on.
* ``deserialize(blob)`` round-trips the same canonical form. Round-trip
  stability is a tested invariant.

Constraints (all enforced in :func:`validate`):

* ``len(files) <= MAX_FILES`` — currently 20. Skills folders shouldn't
  explode; an LLM that tries to emit a 200-file folder is almost
  certainly confused.
* Total serialized size ``<= MAX_TOTAL_BYTES`` — currently 100 KiB.
  Prevents prompt/context-window blowups and runaway mutations.
* All paths must be relative to the root. No leading ``/``, no ``..``
  segments, no absolute paths. Enforced by :func:`_check_path`.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Iterator, List

import yaml


# Hard limits — tuned for skill folders. Override only in tests.
MAX_FILES: int = 60  # Bumped from 20 on 2026-04-22 to accommodate
#                      code-bearing skills (SKILL.md + scripts/ + refs
#                      + templates + assets; 5-skill seed ~= 30 files).
MAX_TOTAL_BYTES: int = 500 * 1024  # 500 KiB (bumped from 100 KiB for
#                      the same reason; realworld SWE skill scripts
#                      routinely hit 5-20 KiB each).
# Serialize fence — chosen to be unlikely in any skill source.
_FILE_FENCE = "===== FILE: {path} ====="
_END_FENCE = "===== END FILE ====="


class FolderArtifactError(ValueError):
    """Raised when a FolderArtifact violates size / path constraints."""


_FRONTMATTER_FENCE = "---"


def _is_skill_md(path: str) -> bool:
    """True iff ``path`` refers to a SKILL.md at any folder depth."""
    return path == "SKILL.md" or path.endswith("/SKILL.md")


def _check_skill_md_frontmatter(path: str, content: str) -> None:
    """Raise :class:`FolderArtifactError` if the SKILL.md frontmatter is
    unparseable as YAML.

    Mirrors the extraction rules in :func:`skill_evolve.track_a.folder._parse_skill_md`
    so Track A's consumer and Track B's producer agree on what "valid" means.
    Files without a leading ``---`` fence are allowed through (treated as a
    body-only SKILL.md); only files that declare a frontmatter block but
    produce a YAML error are rejected.
    """
    stripped = content.lstrip("\ufeff")
    if not stripped.startswith(_FRONTMATTER_FENCE):
        return  # No frontmatter block; downstream will handle defaults.
    lines = stripped.splitlines()
    if not lines or lines[0].strip() != _FRONTMATTER_FENCE:
        return
    close_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == _FRONTMATTER_FENCE:
            close_idx = i
            break
    if close_idx is None:
        raise FolderArtifactError(
            f"{path}: opened '---' frontmatter fence without a closing '---'"
        )
    fm_text = "\n".join(lines[1:close_idx])
    try:
        loaded = yaml.safe_load(fm_text)
    except yaml.YAMLError as exc:
        raise FolderArtifactError(
            f"{path}: YAML frontmatter unparseable: {exc}"
        ) from exc
    if loaded is not None and not isinstance(loaded, dict):
        raise FolderArtifactError(
            f"{path}: YAML frontmatter is {type(loaded).__name__}, expected dict"
        )


def _check_path(path: str) -> str:
    """Normalize ``path`` to a clean POSIX relative path or raise."""
    if not isinstance(path, str) or not path:
        raise FolderArtifactError(f"invalid path (empty or non-str): {path!r}")

    # Reject absolute / drive-rooted paths outright.
    if path.startswith(("/", "\\")) or (len(path) >= 2 and path[1] == ":"):
        raise FolderArtifactError(f"absolute path not allowed: {path!r}")

    p = PurePosixPath(path)
    parts = p.parts

    if any(part in ("..", "") for part in parts):
        raise FolderArtifactError(f"path traversal not allowed: {path!r}")
    if p.is_absolute():
        raise FolderArtifactError(f"absolute path not allowed: {path!r}")

    # Re-emit a canonical POSIX path (normalizes e.g. "a/./b" -> "a/b").
    normalized = p.as_posix()
    if normalized.startswith("/") or ".." in normalized.split("/"):
        raise FolderArtifactError(f"unsafe path after normalize: {path!r}")
    return normalized


@dataclass
class FolderArtifact:
    """A canonicalized dict of relative-path → file contents."""

    files: Dict[str, str] = field(default_factory=dict)

    # -------------------- construction / normalization --------------------

    def __post_init__(self) -> None:
        # Canonicalize keys on construction.
        normalized: Dict[str, str] = {}
        for k, v in self.files.items():
            nk = _check_path(k)
            if not isinstance(v, str):
                raise FolderArtifactError(
                    f"file contents must be str; got {type(v).__name__} for {k!r}"
                )
            if nk in normalized:
                raise FolderArtifactError(f"duplicate path after normalize: {nk!r}")
            normalized[nk] = v
        self.files = normalized

    # -------------------- validation --------------------

    def validate(self) -> None:
        """Raise :class:`FolderArtifactError` if any constraint is violated."""
        if not self.files:
            raise FolderArtifactError("artifact has 0 files")
        if len(self.files) > MAX_FILES:
            raise FolderArtifactError(
                f"too many files: {len(self.files)} > {MAX_FILES}"
            )
        total = sum(len(c.encode("utf-8")) for c in self.files.values())
        if total > MAX_TOTAL_BYTES:
            raise FolderArtifactError(f"total size {total} > {MAX_TOTAL_BYTES} bytes")
        # Re-check each path (defense in depth).
        for p in self.files:
            _check_path(p)
        # SKILL.md frontmatter must round-trip through yaml.safe_load. A
        # patch mutation that produces an unquoted description containing a
        # ":" passes every check above but breaks any downstream consumer
        # that strictly parses the frontmatter (Track A, Track D handoff).
        # We catch it at the artifact boundary so broken SKILL.md files
        # cannot land in the MAP-Elites archive in the first place.
        for p, content in self.files.items():
            if _is_skill_md(p):
                _check_skill_md_frontmatter(p, content)

    # -------------------- hashing / equality --------------------

    def canonical_blob(self) -> str:
        """Serialized canonical form used for hashing + persistence."""
        return self.serialize()

    def stable_hash(self) -> str:
        """SHA-256 hex of the canonical blob."""
        return hashlib.sha256(self.canonical_blob().encode("utf-8")).hexdigest()

    def __hash__(self) -> int:  # type: ignore[override]
        # 64-bit truncation of the sha-256 for dict/set usage. Deterministic
        # across processes (doesn't depend on PYTHONHASHSEED).
        return int(self.stable_hash()[:16], 16)

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if not isinstance(other, FolderArtifact):
            return NotImplemented
        return self.files == other.files

    # -------------------- stats --------------------

    def total_bytes(self) -> int:
        return sum(len(c.encode("utf-8")) for c in self.files.values())

    def num_skills(self) -> int:
        """Count of top-level subfolders that contain a SKILL.md."""
        skills = set()
        for p in self.files:
            parts = p.split("/")
            if len(parts) >= 2 and parts[-1] == "SKILL.md":
                skills.add(parts[0])
        return len(skills)

    def skill_names(self) -> List[str]:
        names: List[str] = []
        for p in sorted(self.files):
            parts = p.split("/")
            if len(parts) >= 2 and parts[-1] == "SKILL.md":
                names.append(parts[0])
        return names

    # -------------------- serialization --------------------

    def serialize(self) -> str:
        """Canonical string form. Paths sorted, file-fence delimited.

        Format::

            ===== FILE: path/a =====
            <contents>
            ===== END FILE =====
            ===== FILE: path/b =====
            <contents>
            ===== END FILE =====
        """
        chunks: List[str] = []
        for path in sorted(self.files):
            content = self.files[path]
            chunks.append(_FILE_FENCE.format(path=path))
            # Ensure exactly one trailing newline before the end-fence so
            # the round-trip is stable even for files that already ended
            # in a newline (very common for markdown).
            if content.endswith("\n"):
                chunks.append(content.rstrip("\n"))
            else:
                chunks.append(content)
            chunks.append(_END_FENCE)
        return "\n".join(chunks) + ("\n" if chunks else "")

    @classmethod
    def deserialize(cls, blob: str) -> "FolderArtifact":
        """Inverse of :meth:`serialize`. Raises on malformed input."""
        files: Dict[str, str] = {}
        current_path: str | None = None
        current_lines: List[str] = []

        for line in blob.splitlines():
            if line.startswith("===== FILE: ") and line.endswith(" ====="):
                if current_path is not None:
                    raise FolderArtifactError(
                        f"nested FILE header inside {current_path!r}"
                    )
                current_path = line[len("===== FILE: ") : -len(" =====")]
                current_lines = []
            elif line == _END_FENCE:
                if current_path is None:
                    raise FolderArtifactError("END FILE without open FILE")
                files[current_path] = "\n".join(current_lines) + "\n"
                current_path = None
                current_lines = []
            else:
                if current_path is None:
                    # Content outside any file — tolerate blank lines,
                    # reject real content so we don't silently drop data.
                    if line.strip():
                        raise FolderArtifactError(
                            f"content outside FILE block: {line!r}"
                        )
                else:
                    current_lines.append(line)

        if current_path is not None:
            raise FolderArtifactError(f"unterminated FILE block: {current_path!r}")

        return cls(files=files)

    # -------------------- filesystem I/O --------------------

    def write_to(self, target: Path, *, deployment: bool = False) -> Path:
        """Write all files under ``target``. Creates parent dirs.

        Args:
            target: destination directory (created if missing).
            deployment: kai-skills patch (Group G, 2026-05-28; plan §7l).
                When True, training-only files are stripped from the
                output: ``meta_skill.md`` at the bundle root and
                ``__pycache__/`` entries anywhere in the tree. Used by
                bundle-promotion / validation-eval sites that must not
                leak training-side bookkeeping into the deployed bundle.
                Default False to preserve back-compat behavior for the
                evolution-loop sites that need meta_skill.md to
                propagate across iterations.
        """
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        for relpath, content in self.files.items():
            if deployment:
                # Strip the bundle-root meta_skill.md (paper §7l: never
                # deployed) and any __pycache__ entries that survived
                # the in-memory dict.
                if relpath == "meta_skill.md":
                    continue
                if "__pycache__" in relpath.split("/"):
                    continue
                if Path(relpath).name.startswith("._"):
                    continue
            dst = target / relpath
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(content, encoding="utf-8")
            # Mirror track_a.folder.SkillFolder.write: executable bit on
            # files under any skill's ``scripts/`` subdir with a
            # recognised extension. Path pattern: ``<skill>/scripts/*.sh``.
            parts = relpath.split("/")
            if (
                len(parts) >= 3
                and parts[1] == "scripts"
                and dst.suffix in {".sh", ".py"}
            ):
                dst.chmod(0o755)
        return target

    @classmethod
    def from_path(
        cls, source: Path, *, include_exts: Iterable[str] | None = None
    ) -> "FolderArtifact":
        """Read a directory into a :class:`FolderArtifact`.

        Only text files are loaded. Binary or unreadable files are
        skipped with a silent pass (we will surface the problem via
        validate() if total files → 0). ``include_exts`` optionally
        restricts to a set of extensions (e.g. ``{".md"}``); None means
        "everything we can decode as utf-8".
        """
        source = Path(source).expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)

        files: Dict[str, str] = {}
        for root, _dirs, fnames in os.walk(source):
            for fname in fnames:
                fp = Path(root) / fname
                rel = fp.relative_to(source).as_posix()
                if include_exts is not None:
                    if fp.suffix not in include_exts:
                        continue
                # Skip hidden files / caches.
                if any(part.startswith(".") for part in Path(rel).parts):
                    continue
                try:
                    files[rel] = fp.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
        return cls(files=files)

    # -------------------- iteration helpers --------------------

    def items(self) -> Iterator[tuple[str, str]]:
        for k in sorted(self.files):
            yield k, self.files[k]

    def __len__(self) -> int:
        return len(self.files)

    def __contains__(self, path: str) -> bool:  # type: ignore[override]
        try:
            return _check_path(path) in self.files
        except FolderArtifactError:
            return False
