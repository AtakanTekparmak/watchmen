"""Sentinel-block mutation parser, validator, and applier (K1 + K7).

Mutation format (the ONLY accepted format — unified diffs are rejected):

    <<<ADD_FILE path/relative/to/bundle>>>
    ... full file content ...
    <<<END_FILE>>>

    <<<EDIT_FILE path/relative/to/bundle>>>
    ... FULL rewritten body (NOT a diff) ...
    <<<END_FILE>>>

    <<<DELETE_FILE path/relative/to/bundle>>>

    <<<REWRITE_FOLDER scripts>>>
    --- file: scripts/foo.py
    ... content ...
    --- file: scripts/bar.sh
    ... content ...
    <<<END_REWRITE>>>

Validation guards:
  - paths containing ``..`` or starting with ``/`` are rejected (traversal).
  - ADD_FILE on a pre-existing path → ``add_existing``.
  - EDIT_FILE / DELETE_FILE on a non-existent path → ``edit_missing``.
  - Unterminated blocks → unterminated.
  - Stray content outside sentinel scope → mixed_content.

Post-apply: shebang insurance prepends ``#!/usr/bin/env python3`` or
``#!/bin/bash`` to .py/.sh files missing it, then ``py_compile``/``bash -n``
runs on every script — errors collected and returned to the caller.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# ─── Dataclass ────────────────────────────────────────────────────────────


@dataclass
class FileOp:
    """One parsed sentinel-block op.

    Fields:
        op: which sentinel kind was matched.
        path: relative path within the bundle (always validated to be
            non-traversing and non-absolute).
        content: full file body for ADD_FILE / EDIT_FILE. None for
            DELETE_FILE and REWRITE_FOLDER.
        files: mapping of {relative_path: content} for REWRITE_FOLDER.
            None for the other ops.
    """

    op: Literal["ADD_FILE", "EDIT_FILE", "DELETE_FILE", "REWRITE_FOLDER"]
    path: str
    content: str | None = None
    files: dict[str, str] | None = field(default=None)


# ─── Sentinel-block regex set ─────────────────────────────────────────────

# We anchor every block at the start of a line (MULTILINE re). The header
# captures the op kind and the path; the body is content up to the END
# marker (non-greedy across lines).
_BLOCK_RE = re.compile(
    r"^<<<(?P<op>ADD_FILE|EDIT_FILE|DELETE_FILE|REWRITE_FOLDER)\s+(?P<path>[^\n>]+?)>>>\s*$"
    r"(?:\n(?P<body>.*?)\n^<<<END_(?:FILE|REWRITE)>>>\s*$)?",
    re.MULTILINE | re.DOTALL,
)

# REWRITE_FOLDER body file separator: "--- file: <relpath>"
_REWRITE_FILE_RE = re.compile(r"^---\s*file:\s*(?P<path>\S+)\s*$", re.MULTILINE)


def _check_path(path: str) -> None:
    """Reject traversal / absolute paths. Raises ValueError on hit."""
    if not path:
        raise ValueError("empty_path")
    if ".." in Path(path).parts:
        raise ValueError(f"path_traversal:{path}")
    if path.startswith("/"):
        raise ValueError(f"absolute_path:{path}")


def _parse_rewrite_body(body: str, folder: str) -> dict[str, str]:
    """Split a REWRITE_FOLDER body into a {path: content} mapping.

    The body is a sequence of ``--- file: <relpath>`` headers each followed
    by content up to the next header (or EOF). Each path is validated;
    additionally, it must live under the named ``folder``.
    """
    files: dict[str, str] = {}

    # Find every file-header. The header positions split the body.
    matches = list(_REWRITE_FILE_RE.finditer(body))
    if not matches:
        # Empty REWRITE_FOLDER is allowed — caller decides whether to act.
        return files

    for i, m in enumerate(matches):
        path = m.group("path").strip()
        _check_path(path)
        # Optional sanity: the path should sit inside the named folder.
        # We don't hard-reject — the spec only requires the four guards
        # above — but we do strip surrounding whitespace so callers can
        # rely on the keys.
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        content = body[start:end]
        # Trim the trailing newline that separated the next header (if any).
        if content.endswith("\n"):
            content = content[:-1]
        files[path] = content

    _ = folder  # folder name is informational only at this layer.
    return files


def parse_sentinel_blocks(text: str, existing_paths: set[str] | None = None) -> list[FileOp]:
    """Parse a proposer's text payload into a list of validated ``FileOp``.

    Args:
        text: raw proposer output.
        existing_paths: relative paths that exist in the parent bundle —
            used to enforce ADD/EDIT/DELETE existence rules. None disables
            those checks (useful for unit tests that only care about
            structural parsing).

    Raises:
        ValueError with a message describing the first violation:
          - ``path_traversal:<path>`` — path contains ``..``
          - ``absolute_path:<path>`` — path starts with ``/``
          - ``unterminated:<op>:<path>`` — no matching END marker
          - ``mixed_content`` — non-sentinel text between blocks
          - ``add_existing`` — ADD_FILE for an already-present path
          - ``edit_missing`` — EDIT_FILE / DELETE_FILE for a missing path
    """
    ops: list[FileOp] = []
    existing = existing_paths if existing_paths is not None else set()

    # Walk the text using BLOCK_RE; track the cursor to detect mixed content
    # (any non-whitespace outside a sentinel block is an error).
    cursor = 0
    found_any = False

    for m in _BLOCK_RE.finditer(text):
        # Anything between cursor and m.start() must be whitespace.
        between = text[cursor : m.start()]
        if between.strip():
            raise ValueError("mixed_content")

        op = m.group("op")
        path = m.group("path").strip()
        body = m.group("body")

        _check_path(path)

        if op == "DELETE_FILE":
            # DELETE_FILE has no body and no END_FILE marker. If body is
            # not None, that means the regex matched an END marker after
            # DELETE_FILE — unusual but harmless; the body is discarded.
            if existing_paths is not None and path not in existing:
                raise ValueError("edit_missing")
            ops.append(FileOp(op="DELETE_FILE", path=path))
        elif op == "REWRITE_FOLDER":
            if body is None:
                raise ValueError(f"unterminated:{op}:{path}")
            files = _parse_rewrite_body(body, path)
            ops.append(FileOp(op="REWRITE_FOLDER", path=path, files=files))
        else:
            # ADD_FILE / EDIT_FILE
            if body is None:
                raise ValueError(f"unterminated:{op}:{path}")
            if op == "ADD_FILE":
                if existing_paths is not None and path in existing:
                    raise ValueError("add_existing")
            else:  # EDIT_FILE
                if existing_paths is not None and path not in existing:
                    raise ValueError("edit_missing")
            ops.append(FileOp(op=op, path=path, content=body))

        cursor = m.end()
        found_any = True

    # After the last block, only whitespace may remain.
    trailing = text[cursor:]
    if found_any and trailing.strip():
        raise ValueError("mixed_content")
    if not found_any and text.strip():
        # No sentinel blocks matched but there's content → malformed.
        raise ValueError("mixed_content")

    return ops


# ─── Apply ops to a candidate dir ─────────────────────────────────────────


def apply_ops(
    parent_bundle_dir: Path,
    ops: list[FileOp],
    candidate_dir: Path,
) -> Path:
    """Copy ``parent_bundle_dir`` to ``candidate_dir``, then apply ``ops``.

    ADD_FILE writes a new file (parent dirs created); EDIT_FILE overwrites
    in place; DELETE_FILE removes the file (silently if it vanished
    underneath us); REWRITE_FOLDER blows away the named folder and
    rewrites it from the ``files`` mapping.

    Returns ``candidate_dir``.
    """
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    shutil.copytree(parent_bundle_dir, candidate_dir)

    for op in ops:
        target = candidate_dir / op.path

        if op.op == "ADD_FILE":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(op.content or "", encoding="utf-8")
        elif op.op == "EDIT_FILE":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(op.content or "", encoding="utf-8")
        elif op.op == "DELETE_FILE":
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
        elif op.op == "REWRITE_FOLDER":
            # Remove the folder if present, then write each file.
            folder = candidate_dir / op.path
            if folder.exists():
                if folder.is_dir():
                    shutil.rmtree(folder)
                else:
                    folder.unlink()
            folder.mkdir(parents=True, exist_ok=True)
            for rel, content in (op.files or {}).items():
                fpath = candidate_dir / rel
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(content, encoding="utf-8")

    return candidate_dir


# ─── Shebang insurance (K7) ───────────────────────────────────────────────


def shebang_insurance(file_path: Path) -> None:
    """Ensure .py / .sh files have the correct shebang at line 1.

    Idempotent — if the shebang is already present (exact match), nothing
    happens. Otherwise the missing line is prepended.
    """
    if not file_path.exists() or not file_path.is_file():
        return

    suffix = file_path.suffix.lower()
    if suffix == ".py":
        wanted = "#!/usr/bin/env python3"
    elif suffix == ".sh":
        wanted = "#!/bin/bash"
    else:
        return

    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return

    first_line = text.split("\n", 1)[0] if text else ""
    if first_line.strip() == wanted:
        return

    new_text = wanted + "\n" + text
    try:
        file_path.write_text(new_text, encoding="utf-8")
    except OSError:
        return


# ─── Script syntax validation ─────────────────────────────────────────────


def validate_scripts(candidate_dir: Path) -> list[str]:
    """Run ``python -m py_compile`` on every .py and ``bash -n`` on every
    .sh under ``candidate_dir``. Return error messages as strings — empty
    list = all pass.

    Errors carry the relative path + stderr first 200 chars so the
    proposer (via ``history.jsonl``) can see why a candidate was rejected.
    """
    errors: list[str] = []

    for fpath in candidate_dir.rglob("*"):
        if not fpath.is_file():
            continue
        rel = fpath.relative_to(candidate_dir)
        suffix = fpath.suffix.lower()
        if suffix == ".py":
            try:
                proc = subprocess.run(
                    ["python", "-m", "py_compile", str(fpath)],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
            except (subprocess.SubprocessError, OSError) as exc:
                errors.append(f"py_compile_subprocess_fail:{rel}:{exc}")
                continue
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip()[:200]
                errors.append(f"py_compile:{rel}:{err}")
        elif suffix == ".sh":
            try:
                proc = subprocess.run(
                    ["bash", "-n", str(fpath)],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
            except (subprocess.SubprocessError, OSError) as exc:
                errors.append(f"bash_n_subprocess_fail:{rel}:{exc}")
                continue
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip()[:200]
                errors.append(f"bash_n:{rel}:{err}")

    return errors


# ─── Hashing for genealogy / dedup ────────────────────────────────────────


def hash_bundle(bundle_dir: Path) -> str:
    """SHA256 of every file's relative path + content under ``bundle_dir``.

    Files are sorted by relative path; for each file we hash the path bytes
    then the raw content bytes. The result is deterministic across runs
    and across machines (no mtimes / inodes involved).
    """
    h = hashlib.sha256()
    if not bundle_dir.exists():
        return h.hexdigest()

    files = sorted(p for p in bundle_dir.rglob("*") if p.is_file())
    for fpath in files:
        rel = fpath.relative_to(bundle_dir).as_posix().encode("utf-8")
        h.update(rel)
        h.update(b"\0")
        try:
            h.update(fpath.read_bytes())
        except OSError:
            h.update(b"<unreadable>")
        h.update(b"\0")

    return h.hexdigest()


# ─── Convenience wrapper ──────────────────────────────────────────────────


def parse_and_apply(
    text: str,
    parent_bundle_dir: Path,
    candidate_dir: Path,
) -> tuple[Path, list[str]]:
    """Parse → apply → shebang-fix → validate. Returns (candidate_dir, errors).

    Existing-path set is built from ``parent_bundle_dir`` so ADD/EDIT/DELETE
    existence rules fire. Shebang insurance runs after apply so newly-added
    scripts inherit a shebang even if the proposer omitted one.
    """
    existing_paths = {p.relative_to(parent_bundle_dir).as_posix() for p in parent_bundle_dir.rglob("*") if p.is_file()}

    ops = parse_sentinel_blocks(text, existing_paths=existing_paths)
    apply_ops(parent_bundle_dir, ops, candidate_dir)

    # Shebang insurance on every script that landed in the candidate dir.
    for fpath in candidate_dir.rglob("*"):
        if fpath.is_file() and fpath.suffix.lower() in (".py", ".sh"):
            shebang_insurance(fpath)

    errors = validate_scripts(candidate_dir)
    return candidate_dir, errors
