"""Slow-update protected region for SKILL.md.

Ports SkillOpt's ``skillopt/optimizer/slow_update.py`` fence-marker
machinery: a pair of paper-faithful HTML-comment markers delimit a
protected region inside SKILL.md that the step-level (fast) proposer
cannot edit. A separate consolidator (slow) proposer fires every K
iterations and rewrites the in-fence content from accumulated
cross-iteration evidence.

Paper-divergence (per plan §7l):
    * Fence position is at the END of SKILL.md
      (mirrors ``inject_empty_slow_update_field``'s
      ``return skill.rstrip() + block`` shape).
    * Fence-overlap edits raise ``SentinelParseError("slow_update_violation")``
      from ``shared.patch_parser.parse_sentinel_blocks`` rather than being
      silently skipped (paper does ``report["status"] = "skipped_..."``).
      The strict-raise behavior aligns with skill_evolve's all-or-nothing
      patch contract.

All five functions in this module are pure-string — no filesystem.
"""

from __future__ import annotations


SLOW_UPDATE_START = "<!-- SLOW_UPDATE_START -->"
SLOW_UPDATE_END = "<!-- SLOW_UPDATE_END -->"


def has_slow_update_field(skill_md: str) -> bool:
    """Return True iff both fence markers are present and well-formed.

    "Well-formed" means START appears before END at least once. Repeated
    markers are NOT well-formed and return False (a half-fence case is
    handled by :func:`extract_slow_update_field` which raises).
    """
    start_idx = skill_md.find(SLOW_UPDATE_START)
    end_idx = skill_md.find(SLOW_UPDATE_END)
    if start_idx == -1 or end_idx == -1:
        return False
    if end_idx < start_idx:
        return False
    # Reject duplicate markers — fence must be unique.
    if skill_md.count(SLOW_UPDATE_START) != 1:
        return False
    if skill_md.count(SLOW_UPDATE_END) != 1:
        return False
    return True


def extract_slow_update_field(skill_md: str) -> str:
    """Return the content strictly between the markers (no markers, no
    leading/trailing newlines).

    Raises ``ValueError`` when exactly one marker is present (half-fence).
    Returns an empty string when neither marker is present (no fence yet).
    """
    has_start = SLOW_UPDATE_START in skill_md
    has_end = SLOW_UPDATE_END in skill_md
    if has_start != has_end:
        raise ValueError(
            "slow_update: half-fence detected — exactly one of "
            f"{SLOW_UPDATE_START!r} / {SLOW_UPDATE_END!r} present"
        )
    if not has_start and not has_end:
        return ""
    if skill_md.count(SLOW_UPDATE_START) != 1 or skill_md.count(SLOW_UPDATE_END) != 1:
        raise ValueError(
            "slow_update: duplicate fence markers; expected exactly one of each"
        )
    start_idx = skill_md.find(SLOW_UPDATE_START) + len(SLOW_UPDATE_START)
    end_idx = skill_md.find(SLOW_UPDATE_END)
    if end_idx < start_idx:
        raise ValueError("slow_update: END marker precedes START marker")
    body = skill_md[start_idx:end_idx]
    # Strip ONE leading and ONE trailing newline so the round-trip with
    # ``inject_slow_update_field`` is content-stable.
    if body.startswith("\n"):
        body = body[1:]
    if body.endswith("\n"):
        body = body[:-1]
    return body


def inject_slow_update_field(skill_md: str, content: str = "") -> str:
    """Paper-faithful injection: append the fenced block at the END of
    SKILL.md.

    Shape (per plan §7l): ``skill.rstrip() + "\\n\\n" + START + "\\n" +
    content + "\\n" + END + "\\n"``.

    Idempotent in the sense that calling this on a SKILL.md that already
    contains a fence raises ``ValueError`` — callers that want to update
    an existing fence MUST use :func:`replace_slow_update_field`.
    """
    if SLOW_UPDATE_START in skill_md or SLOW_UPDATE_END in skill_md:
        raise ValueError(
            "slow_update: fence already present; use "
            "replace_slow_update_field to update an existing region"
        )
    block = SLOW_UPDATE_START + "\n" + content + "\n" + SLOW_UPDATE_END + "\n"
    return skill_md.rstrip() + "\n\n" + block


def replace_slow_update_field(skill_md: str, new_content: str) -> str:
    """Replace the fenced region's content with ``new_content``.

    When no fence exists, injects a fresh one at END of file (so
    callers don't have to special-case the first iter). When a half-fence
    exists, raises ``ValueError`` via :func:`extract_slow_update_field`.

    Markers are preserved verbatim; bytes OUTSIDE the fence are
    unchanged.
    """
    if not has_slow_update_field(skill_md):
        # Half-fence raises through extract; clean absence falls through
        # to inject. We call extract first to surface the half-fence
        # error message rather than the inject-side "fence already
        # present" mismatch.
        _ = extract_slow_update_field(skill_md)
        return inject_slow_update_field(skill_md, new_content)
    start_idx = skill_md.find(SLOW_UPDATE_START)
    end_idx = skill_md.find(SLOW_UPDATE_END) + len(SLOW_UPDATE_END)
    block = SLOW_UPDATE_START + "\n" + new_content + "\n" + SLOW_UPDATE_END
    return skill_md[:start_idx] + block + skill_md[end_idx:]


def is_in_slow_update_region(skill_md: str, char_offset: int) -> bool:
    """Return True iff ``char_offset`` falls between the markers.

    Used by the sentinel parser to detect EDIT_FILE / DELETE_FILE /
    REWRITE_FOLDER ops whose effective character range overlaps the
    protected region.

    Boundary semantics (locked): offsets AT the markers themselves
    (i.e. the bytes covered by ``SLOW_UPDATE_START`` or
    ``SLOW_UPDATE_END``) are considered IN the region — touching the
    fence is a violation. Offsets strictly outside the marker spans
    are out.
    """
    if not has_slow_update_field(skill_md):
        return False
    start_idx = skill_md.find(SLOW_UPDATE_START)
    end_idx = skill_md.find(SLOW_UPDATE_END) + len(SLOW_UPDATE_END)
    return start_idx <= char_offset < end_idx


__all__ = [
    "SLOW_UPDATE_START",
    "SLOW_UPDATE_END",
    "has_slow_update_field",
    "extract_slow_update_field",
    "inject_slow_update_field",
    "replace_slow_update_field",
    "is_in_slow_update_region",
]
