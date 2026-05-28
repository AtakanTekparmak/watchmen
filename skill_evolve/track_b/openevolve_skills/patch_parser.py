"""Patch parser shim — re-exports the shared sentinel-block parser.

The actual parser body lives at ``skill_evolve.shared.patch_parser`` so
both track_a and track_b consume the same implementation. This shim
keeps the track_b-specific ``apply_patch`` and ``mutate`` helpers that
materialize a :class:`FolderArtifact` (track_b operates on in-memory
folder dicts rather than candidate directories on disk).

Backwards-compat names re-exported via ``*`` so existing track_b
imports (``parse_patch``, ``Operation``, ``PatchParseError``,
``AddFile``/``EditFile``/``DeleteFile``/``RewriteFolder``) keep working.
"""

from __future__ import annotations

from typing import List

from skill_evolve.shared.patch_parser import *  # noqa: F401,F403
from skill_evolve.shared.patch_parser import (
    AddFile,
    DeleteFile,
    EditFile,
    FileOp,
    Operation,
    PatchParseError,
    RewriteFolder,
    SentinelParseError,
    parse_patch,
    parse_sentinel_blocks,
)

from .folder_artifact import FolderArtifact


__all__ = [
    "AddFile",
    "DeleteFile",
    "EditFile",
    "FileOp",
    "Operation",
    "PatchParseError",
    "RewriteFolder",
    "SentinelParseError",
    "apply_patch",
    "mutate",
    "parse_patch",
    "parse_sentinel_blocks",
]


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
        rw = rewrites[0]
        # Two flavors: the legacy track_b shape carries ``blob`` (a
        # FolderArtifact.deserialize-compatible string); the daycare
        # shape carries ``files`` (a dict). Prefer ``blob`` when present.
        if getattr(rw, "blob", None):
            return FolderArtifact.deserialize(rw.blob)
        if getattr(rw, "files", None):
            return FolderArtifact(files=dict(rw.files))
        # An empty REWRITE_FOLDER with neither shape — caller error.
        raise PatchParseError("REWRITE_FOLDER has no body")

    for op in ops:
        if isinstance(op, AddFile):
            if op.path in files:
                raise PatchParseError(f"ADD_FILE {op.path!r}: path already exists")
            files[op.path] = op.content or ""
        elif isinstance(op, EditFile):
            if op.path not in files:
                raise PatchParseError(f"EDIT_FILE {op.path!r}: path does not exist")
            files[op.path] = op.content or ""
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
