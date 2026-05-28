"""Meta-skill audit log — per-iteration markdown rolling history.

Ports the spirit of SkillOpt's ``meta_skill.md`` but with a paper-
divergent on-disk shape (per plan §7l):

    * Paper produces a rolling JSON ``{"reasoning", "meta_skill_content"}``
      that is an optimizer-side coach memo, revised/replaced each epoch.
    * skill_evolve writes a per-iteration markdown audit log with
      ``## Iteration N`` sections (summary + edit-pattern stats +
      persistent failures). Tail-truncated to the last
      ``--meta-skill-max-iters`` entries (default 20).
    * Consumer is the proposer system prompt prepend
      (``{meta_skill_body}`` slot in
      ``SENTINEL_PROPOSER_SYSTEM_PROMPT``).

The audit-log shape maps onto skill_evolve's existing per-iter
write-once artifact discipline. A rolling-rewritten JSON memo would
need separate state-file semantics not present elsewhere in the
codebase.

This module is intentionally pure-python (no markdown library
dependency) — the rendered output is meant to be small (≤20 sections
of ≤500 tokens each) and the format is fixed enough that string
concatenation is the appropriate tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class MetaSkillEntry:
    """One iteration's worth of meta-skill audit data.

    Fields:
        iteration: the iter_n (1-based, matching ``IterationResult.generation``).
        patch_summary: one-line natural-language summary written by the
            consolidator LLM ("added DELETE_FILE op for stale scripts/foo.py;
            train-side passes still failing on subset_17:bug_fix_3").
        lessons: zero or more "what worked / what didn't" bullet points.
        failures_observed: task IDs that have failed in
            ``args.persistent_failure_window`` consecutive iters.
        timestamp: ISO-8601 UTC timestamp, set by the caller.
    """

    iteration: int
    patch_summary: str
    lessons: list[str] = field(default_factory=list)
    failures_observed: list[str] = field(default_factory=list)
    timestamp: str = ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _format_entry(entry: MetaSkillEntry) -> str:
    """Render one entry as a ``## Iteration N`` markdown section."""
    lines: list[str] = [f"## Iteration {entry.iteration}"]
    if entry.timestamp:
        lines.append(f"_timestamp: {entry.timestamp}_")
    lines.append("")
    lines.append(f"**Patch summary:** {entry.patch_summary}")
    lines.append("")
    if entry.lessons:
        lines.append("**Lessons:**")
        for lesson in entry.lessons:
            lines.append(f"- {lesson}")
        lines.append("")
    if entry.failures_observed:
        lines.append("**Persistent failures:**")
        for failure in entry.failures_observed:
            lines.append(f"- {failure}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _parse_entry(section: str) -> MetaSkillEntry | None:
    """Parse a single ``## Iteration N`` section back into an entry.

    Returns None when the section is malformed (best-effort recovery —
    a corrupted meta_skill.md should not crash a run).
    """
    lines = section.strip().splitlines()
    if not lines or not lines[0].startswith("## Iteration"):
        return None
    try:
        iter_n = int(lines[0][len("## Iteration") :].strip())
    except ValueError:
        return None

    timestamp = ""
    patch_summary = ""
    lessons: list[str] = []
    failures: list[str] = []
    section_mode: str | None = None

    for raw in lines[1:]:
        line = raw.rstrip()
        if line.startswith("_timestamp:") and line.endswith("_"):
            timestamp = line[len("_timestamp:") : -1].strip()
            continue
        if line.startswith("**Patch summary:**"):
            patch_summary = line[len("**Patch summary:**") :].strip()
            section_mode = None
            continue
        if line.startswith("**Lessons:**"):
            section_mode = "lessons"
            continue
        if line.startswith("**Persistent failures:**"):
            section_mode = "failures"
            continue
        if not line.strip():
            continue
        if line.startswith("- "):
            item = line[2:].strip()
            if section_mode == "lessons":
                lessons.append(item)
            elif section_mode == "failures":
                failures.append(item)

    return MetaSkillEntry(
        iteration=iter_n,
        patch_summary=patch_summary,
        lessons=lessons,
        failures_observed=failures,
        timestamp=timestamp,
    )


class MetaSkill:
    """In-memory list of :class:`MetaSkillEntry` with markdown round-trip.

    Use :meth:`append` to add an entry, :meth:`render_for_prompt` to
    produce the ``{meta_skill_body}`` slot body, :meth:`to_markdown_file`
    to persist to disk (paper-divergent: NOT JSONL — per plan §7l),
    and :meth:`from_markdown_file` to load a prior file.

    Bounded tail truncation: the in-memory list is capped at
    ``max_entries`` (default 20). When a new entry pushes the count
    over the cap, the oldest entry is evicted FIFO. This matches the
    paper's per-epoch coach-memo bounding even though our artifact
    shape is per-iter sections instead of a rolling document.
    """

    DEFAULT_MAX_ENTRIES = 20

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        if max_entries <= 0:
            raise ValueError(f"max_entries must be > 0; got {max_entries}")
        self._entries: list[MetaSkillEntry] = []
        self._max_entries = int(max_entries)

    # ── construction / persistence ──────────────────────────────────

    @property
    def max_entries(self) -> int:
        return self._max_entries

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> list[MetaSkillEntry]:
        """Return a defensive copy of the entries (oldest first)."""
        return list(self._entries)

    def append(self, entry: MetaSkillEntry) -> None:
        """Append a new entry; truncate the head if past ``max_entries``.

        If the caller passes an entry without a timestamp, we stamp it
        with the current UTC time so on-disk round-trips preserve a
        deterministic order even when the consolidator forgets.
        """
        if not entry.timestamp:
            entry = MetaSkillEntry(
                iteration=entry.iteration,
                patch_summary=entry.patch_summary,
                lessons=list(entry.lessons),
                failures_observed=list(entry.failures_observed),
                timestamp=_now_iso(),
            )
        self._entries.append(entry)
        while len(self._entries) > self._max_entries:
            self._entries.pop(0)

    def render_for_prompt(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> str:
        """Render the meta-skill body for the ``{meta_skill_body}`` slot.

        Tail-truncated to the last ``max_entries`` (default 20) so a
        long-running session does not exceed the proposer prompt's
        context budget. The empty case renders the locked literal
        ``(empty — no consolidated patterns yet)`` so the section does
        not collapse to a bare header (matches Group E's empty-buffer
        convention).
        """
        if max_entries <= 0:
            raise ValueError(f"max_entries must be > 0; got {max_entries}")
        if not self._entries:
            return "(empty — no consolidated patterns yet)"
        tail = self._entries[-max_entries:]
        return "\n".join(_format_entry(e) for e in tail).rstrip() + "\n"

    def to_markdown_file(self, path: Path) -> None:
        """Write the rolling audit log to ``path`` as markdown.

        Paper-divergence (plan §7l): per-iter markdown audit log instead
        of the paper's single rolling JSON ``{"reasoning",
        "meta_skill_content"}`` memo. The consumer is the proposer
        prompt prepend, not an optimizer-side coach.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._entries:
            # An empty audit log is allowed on disk; write a single
            # comment so a subsequent ``from_markdown_file`` round-trips
            # cleanly to an empty list.
            path.write_text(
                "# meta-skill audit log\n\n(empty)\n",
                encoding="utf-8",
            )
            return
        body = "# meta-skill audit log\n\n" + "\n".join(
            _format_entry(e) for e in self._entries
        )
        path.write_text(body.rstrip() + "\n", encoding="utf-8")

    @classmethod
    def from_markdown_file(
        cls,
        path: Path,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> "MetaSkill":
        """Load a prior audit log from disk.

        Missing file → empty MetaSkill. Malformed sections are silently
        skipped (best-effort recovery; a corrupted log should not crash
        a run). The ``max_entries`` bound is applied to the loaded
        result so a previously-larger log gets re-truncated.
        """
        m = cls(max_entries=max_entries)
        path = Path(path)
        if not path.exists():
            return m
        text = path.read_text(encoding="utf-8", errors="replace")
        # Split into ``## Iteration N`` sections — anything before the
        # first such header is treated as a preamble and ignored.
        sections: list[str] = []
        current: list[str] = []
        for line in text.splitlines():
            if line.startswith("## Iteration"):
                if current:
                    sections.append("\n".join(current))
                current = [line]
            elif current:
                current.append(line)
        if current:
            sections.append("\n".join(current))

        for sec in sections:
            entry = _parse_entry(sec)
            if entry is not None:
                m.append(entry)
        return m


__all__ = [
    "MetaSkill",
    "MetaSkillEntry",
]
