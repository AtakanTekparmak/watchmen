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


# Hard limits — tuned for skill folders. Override only in tests.
MAX_FILES: int = 20
MAX_TOTAL_BYTES: int = 100 * 1024  # 100 KiB
# Serialize fence — chosen to be unlikely in any skill source.
_FILE_FENCE = "===== FILE: {path} ====="
_END_FENCE = "===== END FILE ====="


class FolderArtifactError(ValueError):
    """Raised when a FolderArtifact violates size / path constraints."""


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
                    f"file contents must be str; got {type(v).__name__} for {k!r}")
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
                f"too many files: {len(self.files)} > {MAX_FILES}")
        total = sum(len(c.encode("utf-8")) for c in self.files.values())
        if total > MAX_TOTAL_BYTES:
            raise FolderArtifactError(
                f"total size {total} > {MAX_TOTAL_BYTES} bytes")
        # Re-check each path (defense in depth).
        for p in self.files:
            _check_path(p)

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
                        f"nested FILE header inside {current_path!r}")
                current_path = line[len("===== FILE: "): -len(" =====")]
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
                            f"content outside FILE block: {line!r}")
                else:
                    current_lines.append(line)

        if current_path is not None:
            raise FolderArtifactError(f"unterminated FILE block: {current_path!r}")

        return cls(files=files)

    # -------------------- filesystem I/O --------------------

    def write_to(self, target: Path) -> Path:
        """Write all files under ``target``. Creates parent dirs."""
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        for relpath, content in self.files.items():
            dst = target / relpath
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(content, encoding="utf-8")
        return target

    @classmethod
    def from_path(cls, source: Path, *,
                  include_exts: Iterable[str] | None = None) -> "FolderArtifact":
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
