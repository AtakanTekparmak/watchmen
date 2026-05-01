"""Patch parser — turns LLM output into file operations on a FolderArtifact.

Two mutation styles are supported, same format whether the LLM emits a
patch or a full rewrite:

1. **Patch-style (default)**:
    A sequence of commands inside fenced blocks. Each command operates
    on exactly one file::

        <<<ADD_FILE path/to/new_skill/SKILL.md>>>
        ---
        name: foo
        ---
        <<<END_FILE>>>

        <<<EDIT_FILE path/to/old_skill/SKILL.md>>>
        <entire new content>
        <<<END_FILE>>>

        <<<DELETE_FILE path/to/stale/SKILL.md>>>

2. **Full rewrite**:
    A single block that produces a whole folder::

        <<<REWRITE_FOLDER>>>
        ===== FILE: path/to/a =====
        ...
        ===== END FILE =====
        <<<END_REWRITE>>>

    (the inner body is :func:`FolderArtifact.deserialize`-compatible).

Defensive: any malformed token stream raises :class:`PatchParseError`
with a descriptive message — we never silently partially-apply a broken
patch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from .folder_artifact import FolderArtifact, FolderArtifactError, _check_path


class PatchParseError(ValueError):
    """Raised on malformed patch-style mutations."""


# ---- operation dataclasses ---------------------------------------------------


@dataclass
class AddFile:
    path: str
    content: str


@dataclass
class EditFile:
    path: str
    content: str


@dataclass
class DeleteFile:
    path: str


@dataclass
class RewriteFolder:
    blob: str


Operation = AddFile | EditFile | DeleteFile | RewriteFolder


# ---- tokenizers --------------------------------------------------------------

_ADD_OPEN = re.compile(r"^<<<ADD_FILE\s+(.+?)>>>\s*$")
_EDIT_OPEN = re.compile(r"^<<<EDIT_FILE\s+(.+?)>>>\s*$")
_DELETE_LINE = re.compile(r"^<<<DELETE_FILE\s+(.+?)>>>\s*$")
_END_FILE = "<<<END_FILE>>>"
_REWRITE_OPEN = "<<<REWRITE_FOLDER>>>"
_REWRITE_CLOSE = "<<<END_REWRITE>>>"


def parse_patch(text: str) -> List[Operation]:
    """Tokenize ``text`` into an ordered list of file operations.

    Unknown lines outside of any block are ignored (so the LLM can write
    free-form prose around its patch). Malformed blocks raise.
    """
    ops: List[Operation] = []
    lines = text.splitlines()
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # --- full-folder rewrite ---
        if line.strip() == _REWRITE_OPEN:
            body, j = _collect_until(
                lines, i + 1, _REWRITE_CLOSE, origin_line=i, origin_token=_REWRITE_OPEN
            )
            ops.append(RewriteFolder(blob="\n".join(body) + "\n"))
            i = j + 1
            continue

        # --- single-line delete ---
        m = _DELETE_LINE.match(line.strip())
        if m:
            path = _validate_path_or_raise(
                m.group(1).strip(), ctx=f"DELETE_FILE at line {i}"
            )
            ops.append(DeleteFile(path=path))
            i += 1
            continue

        # --- add / edit (multi-line body, ends at END_FILE) ---
        m_add = _ADD_OPEN.match(line.strip())
        m_edit = _EDIT_OPEN.match(line.strip())
        if m_add or m_edit:
            path_raw = (m_add or m_edit).group(1).strip()
            path = _validate_path_or_raise(
                path_raw, ctx=f"{'ADD_FILE' if m_add else 'EDIT_FILE'} at line {i}"
            )
            body, j = _collect_until(
                lines, i + 1, _END_FILE, origin_line=i, origin_token=line.strip()
            )
            content = "\n".join(body)
            # Tolerate absence / presence of trailing newline — normalize.
            if not content.endswith("\n"):
                content += "\n"
            op: Operation = (
                AddFile(path=path, content=content)
                if m_add
                else EditFile(path=path, content=content)
            )
            ops.append(op)
            i = j + 1
            continue

        # Unknown free-form line: ignore.
        i += 1

    return ops


def _collect_until(
    lines: List[str], start: int, sentinel: str, *, origin_line: int, origin_token: str
) -> Tuple[List[str], int]:
    """Collect ``lines[start:]`` until a line == ``sentinel``.

    Returns ``(body, index_of_sentinel)``. Raises if not found.
    """
    for k in range(start, len(lines)):
        if lines[k].strip() == sentinel:
            return lines[start:k], k
    raise PatchParseError(
        f"no matching {sentinel!r} for {origin_token!r} opened at line {origin_line}"
    )


def _validate_path_or_raise(raw: str, *, ctx: str) -> str:
    try:
        return _check_path(raw)
    except FolderArtifactError as exc:
        raise PatchParseError(f"{ctx}: invalid path ({exc})") from exc


# ---- application -------------------------------------------------------------


def apply_patch(parent: FolderArtifact, ops: List[Operation]) -> FolderArtifact:
    """Apply an ordered operation list to a parent folder and return new folder.

    Semantics (strict — we want to surface confused LLM output, not
    paper over it):

    * ``ADD_FILE``: path must NOT already exist in the result.
    * ``EDIT_FILE``: path must exist (either in parent or from earlier
       ADD_FILE in this patch).
    * ``DELETE_FILE``: path must exist.
    * ``REWRITE_FOLDER``: replaces the entire folder. Must be the sole
       operation in the patch.
    """
    # Mutable working copy.
    files = dict(parent.files)

    # Full-rewrite short-circuit — allowed only as sole op.
    rewrites = [o for o in ops if isinstance(o, RewriteFolder)]
    if rewrites:
        if len(ops) != 1:
            raise PatchParseError(
                "REWRITE_FOLDER must be the sole operation in a patch"
            )
        return FolderArtifact.deserialize(rewrites[0].blob)

    for op in ops:
        if isinstance(op, AddFile):
            if op.path in files:
                raise PatchParseError(f"ADD_FILE {op.path!r}: path already exists")
            files[op.path] = op.content
        elif isinstance(op, EditFile):
            if op.path not in files:
                raise PatchParseError(f"EDIT_FILE {op.path!r}: path does not exist")
            files[op.path] = op.content
        elif isinstance(op, DeleteFile):
            if op.path not in files:
                raise PatchParseError(f"DELETE_FILE {op.path!r}: path does not exist")
            del files[op.path]
        else:  # pragma: no cover — type system ensures exhaustive
            raise PatchParseError(f"unknown op type: {type(op).__name__}")

    return FolderArtifact(files=files)


def mutate(parent: FolderArtifact, llm_text: str) -> FolderArtifact:
    """End-to-end: parse ``llm_text`` → apply on ``parent`` → validate child.

    Raises :class:`PatchParseError` for bad syntax or bad ops, and
    :class:`FolderArtifactError` if the resulting child violates size /
    count / path constraints. Both are recoverable at the caller: discard
    the mutation, log, move on.
    """
    ops = parse_patch(llm_text)
    if not ops:
        raise PatchParseError("patch contained no operations")
    child = apply_patch(parent, ops)
    child.validate()
    # Must still have ≥1 SKILL.md after the patch — an empty skill folder
    # is useless for the agent.
    if child.num_skills() == 0:
        raise PatchParseError(
            "after applying patch: 0 skills (need ≥1 SKILL.md subfolder)"
        )
    return child
