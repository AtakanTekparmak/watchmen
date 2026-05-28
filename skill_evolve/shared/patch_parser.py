"""Sentinel-block patch parser — ported from daycare.

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

Backwards-compat: track_b's older parser used a path-less
``<<<REWRITE_FOLDER>>>...<<<END_REWRITE>>>`` block whose body is a
``FolderArtifact.deserialize``-compatible blob. The parser still accepts
that shape and surfaces it as ``RewriteFolder`` with ``blob`` populated
(daycare-style ``files`` left as None). track_b's ``apply_patch`` consumes
``blob`` directly.

Primary exports (daycare names):
    parse_sentinel_blocks, FileOp, SentinelParseError

Back-compat aliases (track_b consumers):
    parse_patch = parse_sentinel_blocks
    Operation = FileOp
    PatchParseError = SentinelParseError

Per-kind names AddFile / EditFile / DeleteFile / RewriteFolder are
*distinct empty subclasses* of FileOp so consumers can do both
``isinstance(op, AddFile)`` discrimination AND
``type(op).__name__ == "AddFile"`` checks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


# ─── Errors ───────────────────────────────────────────────────────────────


class SentinelParseError(ValueError):
    """Raised on malformed sentinel-block mutations.

    The first positional arg is a short ``kind`` string drawn from a small
    fixed set:

        {"unterminated", "path_traversal", "absolute_path",
         "add_existing", "edit_missing", "mixed_content",
         "slow_update_violation"}

    A descriptive suffix may follow (e.g. ``f"path_traversal:{path}"``)
    matching the daycare convention. Tests pin ``str(exc).startswith(kind)``.

    kai-skills patch (Group G, 2026-05-28; plan §7l): the
    ``slow_update_violation`` kind is raised by ``parse_sentinel_blocks``
    when ``parent_skill_md`` is supplied and an EDIT_FILE / DELETE_FILE /
    REWRITE_FOLDER op on SKILL.md overlaps the slow-update fence. Paper
    silently skips; skill_evolve strict-raises to keep the patch contract
    all-or-nothing.
    """


# Back-compat alias — track_b historically imported PatchParseError.
PatchParseError = SentinelParseError


# ─── Dataclass ────────────────────────────────────────────────────────────


@dataclass
class FileOp:
    """One parsed sentinel-block op.

    Fields:
        op: which sentinel kind was matched.
        path: relative path within the bundle (always validated to be
            non-traversing and non-absolute). Empty string for the
            path-less track_b ``REWRITE_FOLDER`` shape.
        content: full file body for ADD_FILE / EDIT_FILE. None for
            DELETE_FILE and REWRITE_FOLDER.
        files: mapping of {relative_path: content} for daycare-style
            REWRITE_FOLDER. None for the other ops.
        blob: serialized FolderArtifact body for the track_b path-less
            ``<<<REWRITE_FOLDER>>>...<<<END_REWRITE>>>`` shape. None
            otherwise.
    """

    op: Literal["ADD_FILE", "EDIT_FILE", "DELETE_FILE", "REWRITE_FOLDER"]
    path: str
    content: str | None = None
    files: dict[str, str] | None = field(default=None)
    blob: str | None = field(default=None)


# Distinct empty subclasses — NOT aliases. Tests assert
# ``type(op).__name__ == "AddFile"`` etc. which requires the parser to
# instantiate the correct subclass at construction time.
@dataclass
class AddFile(FileOp):
    pass


@dataclass
class EditFile(FileOp):
    pass


@dataclass
class DeleteFile(FileOp):
    pass


@dataclass
class RewriteFolder(FileOp):
    pass


# Back-compat alias for the union type that used to live in track_b.
# Every concrete op is a FileOp subclass, so ``isinstance(x, Operation)``
# still discriminates as expected.
Operation = FileOp


# ─── Sentinel-block regex set ─────────────────────────────────────────────

# Daycare-style block: header carries an op + a path; optional body up
# to a matching END marker.
_BLOCK_RE = re.compile(
    r"^<<<(?P<op>ADD_FILE|EDIT_FILE|DELETE_FILE|REWRITE_FOLDER)\s+(?P<path>[^\n>]+?)>>>\s*$"
    r"(?:\n(?P<body>.*?)\n^<<<END_(?:FILE|REWRITE)>>>\s*$)?",
    re.MULTILINE | re.DOTALL,
)

# track_b path-less REWRITE_FOLDER shape: ``<<<REWRITE_FOLDER>>>``
# (NO path), body up to ``<<<END_REWRITE>>>``.
_BARE_REWRITE_RE = re.compile(
    r"^<<<REWRITE_FOLDER>>>\s*$\n(?P<body>.*?)\n?^<<<END_REWRITE>>>\s*$",
    re.MULTILINE | re.DOTALL,
)

# REWRITE_FOLDER body file separator: "--- file: <relpath>"
_REWRITE_FILE_RE = re.compile(r"^---\s*file:\s*(?P<path>\S+)\s*$", re.MULTILINE)


def _check_path(path: str) -> None:
    """Reject traversal / absolute paths. Raises SentinelParseError on hit."""
    if not path:
        raise SentinelParseError("empty_path")
    if ".." in Path(path).parts:
        raise SentinelParseError(f"path_traversal:{path}")
    if path.startswith("/") or path.startswith("\\"):
        raise SentinelParseError(f"absolute_path:{path}")


def _parse_rewrite_body(body: str, folder: str) -> dict[str, str]:
    """Split a REWRITE_FOLDER body into a {path: content} mapping.

    The body is a sequence of ``--- file: <relpath>`` headers each followed
    by content up to the next header (or EOF). Each path is validated;
    additionally, it must live under the named ``folder``.
    """
    files: dict[str, str] = {}

    matches = list(_REWRITE_FILE_RE.finditer(body))
    if not matches:
        # No file headers in the body. If the body has non-whitespace
        # content (e.g. track_b-style blob inside a path-bearing block)
        # we surface ``mixed_content`` so the proposer can correct it.
        if body.strip():
            raise SentinelParseError("mixed_content")
        return files

    # If there is non-whitespace content BEFORE the first ``--- file:``
    # header that's outside any per-file body, that's mixed_content too.
    preamble = body[: matches[0].start()]
    if preamble.strip():
        raise SentinelParseError("mixed_content")

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


def _construct(
    op: str,
    path: str,
    *,
    content: str | None = None,
    files: dict[str, str] | None = None,
    blob: str | None = None,
) -> FileOp:
    """Instantiate the per-kind subclass so ``type(x).__name__`` reads true."""
    if op == "ADD_FILE":
        return AddFile(op="ADD_FILE", path=path, content=content)
    if op == "EDIT_FILE":
        return EditFile(op="EDIT_FILE", path=path, content=content)
    if op == "DELETE_FILE":
        return DeleteFile(op="DELETE_FILE", path=path)
    if op == "REWRITE_FOLDER":
        return RewriteFolder(op="REWRITE_FOLDER", path=path, files=files, blob=blob)
    raise SentinelParseError(f"unknown_op:{op}")


# ─── Slow-update fence helpers (Group G, 2026-05-28) ─────────────────────
#
# Per plan §7l. The fence is paper-faithful: the same HTML-comment
# markers used by SkillOpt's ``skillopt/optimizer/slow_update.py``.
# We import lazily to avoid a circular import — slow_update.py is in
# the same package but doesn't import patch_parser.


def _is_skill_md_path(path: str) -> bool:
    """Return True iff ``path`` refers to a SKILL.md file (bundle root or
    nested under any skill folder).
    """
    # Normalize to forward-slash path components.
    parts = path.replace("\\", "/").split("/")
    return parts[-1] == "SKILL.md"


def _check_slow_update_violation_edit(parent_skill_md: str, new_content: str) -> None:
    """Raise ``slow_update_violation`` if an EDIT_FILE on SKILL.md would
    mutate the protected fence region.

    Rules:
    - If parent has no fence: any new_content is fine (no protected
      region to violate).
    - If parent has a fence: the new_content MUST also contain a
      well-formed fence whose body bytes are byte-identical to the
      parent's body. Anything else is a violation.
    """
    from skill_evolve.shared.slow_update import (
        extract_slow_update_field,
        has_slow_update_field,
    )

    if not has_slow_update_field(parent_skill_md):
        return
    parent_body = extract_slow_update_field(parent_skill_md)
    if not has_slow_update_field(new_content):
        # New content dropped the fence entirely — violation.
        raise SentinelParseError("slow_update_violation:fence_removed_from_SKILL.md")
    try:
        new_body = extract_slow_update_field(new_content)
    except ValueError as exc:
        # Half-fence in new content — likewise a violation.
        raise SentinelParseError(
            f"slow_update_violation:malformed_fence:{exc}"
        ) from None
    if new_body != parent_body:
        raise SentinelParseError(
            "slow_update_violation:in_fence_content_mutated_by_fast_proposer"
        )


def _check_slow_update_violation_delete(parent_skill_md: str, path: str) -> None:
    """Raise ``slow_update_violation`` when DELETE_FILE targets a SKILL.md
    that has an active fence — deletion destroys the protected region.
    """
    from skill_evolve.shared.slow_update import has_slow_update_field

    if has_slow_update_field(parent_skill_md):
        raise SentinelParseError(
            f"slow_update_violation:delete_of_fenced_SKILL.md:{path}"
        )


def _check_slow_update_violation_rewrite(
    parent_skill_md: str, files: dict[str, str]
) -> None:
    """Raise ``slow_update_violation`` if a REWRITE_FOLDER body produces
    a SKILL.md whose fence is missing or whose in-fence bytes differ
    from the parent.

    Only triggered when ``parent_skill_md`` carries a fence — otherwise
    REWRITE_FOLDER is unconstrained on the slow-update axis.
    """
    from skill_evolve.shared.slow_update import (
        extract_slow_update_field,
        has_slow_update_field,
    )

    if not has_slow_update_field(parent_skill_md):
        return
    parent_body = extract_slow_update_field(parent_skill_md)
    for rel, content in files.items():
        if not _is_skill_md_path(rel):
            continue
        if not has_slow_update_field(content):
            raise SentinelParseError(
                f"slow_update_violation:rewrite_dropped_fence:{rel}"
            )
        try:
            new_body = extract_slow_update_field(content)
        except ValueError as exc:
            raise SentinelParseError(
                f"slow_update_violation:rewrite_malformed_fence:{rel}:{exc}"
            ) from None
        if new_body != parent_body:
            raise SentinelParseError(
                f"slow_update_violation:rewrite_mutated_in_fence:{rel}"
            )


def parse_sentinel_blocks(
    text: str,
    existing_paths: Optional[set[str]] = None,
    *,
    strict: bool = False,
    parent_skill_md: Optional[str] = None,
) -> list[FileOp]:
    """Parse a proposer's text payload into a list of validated ``FileOp``.

    Args:
        text: raw proposer output.
        existing_paths: relative paths that exist in the parent bundle —
            used to enforce ADD/EDIT/DELETE existence rules. None disables
            those checks (useful for unit tests that only care about
            structural parsing).
        strict: when True (daycare mode), any non-whitespace text outside
            a sentinel block raises ``mixed_content``. When False
            (track_b mode), free-form prose around sentinel blocks is
            silently ignored. ``parse_and_apply`` always passes
            ``strict=True``.
        parent_skill_md: kai-skills patch (Group G, 2026-05-28; plan §7l).
            When set, EDIT_FILE / DELETE_FILE / REWRITE_FOLDER ops whose
            path resolves to ``SKILL.md`` (the bundle root SKILL.md OR
            any ``<skill>/SKILL.md`` path) AND whose effective edit
            range overlaps the slow-update fence raise
            ``SentinelParseError("slow_update_violation")``. When None
            (back-compat default) the check is skipped — daycare-side
            callers and pre-G tests see identical behavior to the
            original Group A parser.

    Raises:
        SentinelParseError on first violation. The error's first arg is a
        short ``kind`` string (see SentinelParseError docstring).
    """
    ops: list[FileOp] = []
    existing = existing_paths if existing_paths is not None else set()

    # First pass: track_b's path-less ``<<<REWRITE_FOLDER>>>`` shape.
    # If present, capture and excise from the working text so the
    # daycare regex doesn't see stray prose.
    bare_rewrites: list[tuple[int, int, str]] = []
    for m in _BARE_REWRITE_RE.finditer(text):
        body = m.group("body")
        bare_rewrites.append((m.start(), m.end(), body))

    # Map of (start, end) ranges occupied by bare-rewrite blocks so we
    # can detect mixed_content correctly.
    bare_ranges = [(s, e) for s, e, _ in bare_rewrites]

    # Sentinel for both formats: walk text using _BLOCK_RE, track cursor
    # to detect mixed content (non-whitespace outside any sentinel).
    cursor = 0
    found_any = False

    def _is_in_bare(pos: int) -> bool:
        return any(s <= pos < e for s, e in bare_ranges)

    for m in _BLOCK_RE.finditer(text):
        # Skip matches inside a bare-rewrite block (their bodies can
        # syntactically contain content that looks like a header).
        if _is_in_bare(m.start()):
            continue

        # Anything between cursor and m.start() must be whitespace or
        # entirely contained in a bare-rewrite block (strict mode only).
        if strict:
            between = text[cursor : m.start()]
            # Strip out bare-rewrite spans inside the between range.
            clean_between = between
            for s, e, _ in bare_rewrites:
                if cursor <= s and e <= m.start():
                    clean_between = clean_between.replace(text[s:e], "")
            if clean_between.strip():
                raise SentinelParseError("mixed_content")

        op = m.group("op")
        path = m.group("path").strip()
        body = m.group("body")

        _check_path(path)

        if op == "DELETE_FILE":
            if existing_paths is not None and path not in existing:
                raise SentinelParseError("edit_missing")
            # kai-skills patch (Group G, 2026-05-28; plan §7l):
            # DELETE_FILE on a SKILL.md that has a slow-update fence is
            # a fence-overlap by definition — deleting the file would
            # destroy the protected region.
            if parent_skill_md is not None and _is_skill_md_path(path):
                _check_slow_update_violation_delete(parent_skill_md, path)
            ops.append(_construct("DELETE_FILE", path))
        elif op == "REWRITE_FOLDER":
            if body is None:
                raise SentinelParseError(f"unterminated:{op}:{path}")
            files = _parse_rewrite_body(body, path)
            # kai-skills patch (Group G, 2026-05-28; plan §7l):
            # REWRITE_FOLDER replacing a folder whose tree contains a
            # SKILL.md with a slow-update fence must preserve the fence
            # bytes in the new SKILL.md. If the new files mapping
            # contains a SKILL.md whose fenced region differs from the
            # parent's, or omits the fence entirely, it's a violation.
            if parent_skill_md is not None:
                _check_slow_update_violation_rewrite(parent_skill_md, files)
            ops.append(_construct("REWRITE_FOLDER", path, files=files))
        else:
            # ADD_FILE / EDIT_FILE
            if body is None:
                raise SentinelParseError(f"unterminated:{op}:{path}")
            if op == "ADD_FILE":
                # Duplicate ADD_FILE in the same patch is always an
                # error (even without existing_paths) — the second
                # ADD would overwrite the first which is exactly the
                # ambiguity the spec rejects.
                if existing_paths is not None and path in existing:
                    raise SentinelParseError("add_existing")
                if existing_paths is None and path in existing:
                    raise SentinelParseError("add_existing")
                existing.add(path)
            else:  # EDIT_FILE
                if existing_paths is not None and path not in existing:
                    raise SentinelParseError("edit_missing")
            # Tolerate absence / presence of trailing newline — normalize
            # so apply_patch sees content ending in \n (track_b parser
            # contract).
            content = body
            if content and not content.endswith("\n"):
                content += "\n"
            # kai-skills patch (Group G, 2026-05-28; plan §7l): when the
            # EDIT_FILE targets a SKILL.md that has an existing
            # slow-update fence, the new content must preserve the
            # fenced region bytes verbatim. Step-level proposer edits
            # MUST NOT mutate the protected region — that channel is
            # owned by the consolidator (slow) proposer fired every K
            # iters. ADD_FILE is uncovered because ADD_FILE on a SKILL.md
            # that already exists in the parent would have raised
            # add_existing above; a fresh ADD_FILE has no parent fence
            # to violate.
            if (
                op == "EDIT_FILE"
                and parent_skill_md is not None
                and _is_skill_md_path(path)
            ):
                _check_slow_update_violation_edit(parent_skill_md, content)
            ops.append(_construct(op, path, content=content))

        cursor = m.end()
        found_any = True

    # Splice in the path-less REWRITE_FOLDER blocks. They go to the end
    # — apply_patch enforces that a REWRITE_FOLDER is the sole op.
    for s, e, body in bare_rewrites:
        # The body is the inner blob — preserve trailing newline so
        # FolderArtifact.deserialize round-trips.
        blob = body if body.endswith("\n") else body + "\n"
        ops.append(_construct("REWRITE_FOLDER", path="", files=None, blob=blob))
        found_any = True

    # After the last block (in either format), only whitespace may remain
    # in strict mode.
    if strict:
        trailing = text[cursor:]
        # Excise bare-rewrite spans from the trailing region.
        for s, e, _ in bare_rewrites:
            if s >= cursor:
                trailing = trailing.replace(text[s:e], "")
        if found_any and trailing.strip():
            raise SentinelParseError("mixed_content")
        if not found_any and text.strip():
            raise SentinelParseError("mixed_content")

    return ops


# Back-compat alias for callers that imported ``parse_patch``.
parse_patch = parse_sentinel_blocks


__all__ = [
    "FileOp",
    "AddFile",
    "EditFile",
    "DeleteFile",
    "RewriteFolder",
    "Operation",
    "SentinelParseError",
    "PatchParseError",
    "parse_sentinel_blocks",
    "parse_patch",
]
