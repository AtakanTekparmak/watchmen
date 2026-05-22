"""Skill selection — traffic × error rate × analyst priority.

For each skill slug under ``bundles/<project>/skills/``:
  - traffic_score: count of tool_calls rows with skill_name=slug in last N days
  - error_boost: tool_error_count / tool_use_count across sessions where the
    skill fired (join via session_id)
  - analyst_boost: 1.5× if the slug appears in ``analyses/<project>/_running.md``
    under the "Skill candidates" section
  - priority = traffic_score × (1 + error_boost) × analyst_boost

Cold-start fallback: if every skill has traffic_score == 0 (a brand-new
bundle dir, no corpus traffic yet), rank by SKILL.md mtime descending so
the freshly-edited skill surfaces first.

Phase 0 doctor calls ``rank_skills()`` and writes the ranking table to
``RUN_DIR/selector_log.md`` for human review.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ─── Dataclass ────────────────────────────────────────────────────────────


@dataclass
class SkillRank:
    """Per-skill priority record (one row of the selector table).

    Fields:
        slug: directory name under bundles/<project>/skills/.
        traffic_score: count of tool_calls rows in the window.
        error_boost: tool_error_count / tool_use_count ratio.
        analyst_boost: 1.5 if slug in _running.md "Skill candidates", else 1.0.
        priority: traffic_score × (1 + error_boost) × analyst_boost.
    """

    slug: str
    traffic_score: int
    error_boost: float
    analyst_boost: float
    priority: float


# ─── Analyst-candidate parser ─────────────────────────────────────────────

# Slug-like tokens: kebab-case words ≥3 chars (allows underscores too, since
# some bundles use snake_case). Anchored to word boundaries to avoid picking
# up halves of longer identifiers.
_SLUG_TOKEN_RE = re.compile(r"\b[a-z][a-z0-9]*(?:[-_][a-z0-9]+)+\b")


def get_analyst_candidates(running_md_path: Path) -> set[str]:
    """Extract slug-like tokens from the "Skill candidates" section of
    ``_running.md``.

    The section header is matched case-insensitively; the section ends at
    the next markdown header of equal-or-higher level (``#``, ``##``, ...)
    or at EOF. Any kebab/snake-case token of 3+ chars within is returned.

    Returns an empty set if the file is missing or no section is found.
    """
    if not running_md_path.exists():
        return set()

    try:
        text = running_md_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()

    # Locate the "Skill candidates" section. Markdown headers look like
    # "## Skill candidates" — match any leading-# count followed by the
    # phrase. Case-insensitive.
    header_re = re.compile(r"^(#+)\s*Skill\s*candidates\s*$", re.IGNORECASE | re.MULTILINE)
    match = header_re.search(text)
    if not match:
        return set()

    section_start = match.end()
    header_level = len(match.group(1))

    # Find the next header of equal-or-higher level (fewer-or-equal hashes).
    # Build a regex that matches headers ≤ header_level deep.
    next_header_re = re.compile(rf"^#{{1,{header_level}}}\s+\S", re.MULTILINE)
    rest = text[section_start:]
    next_match = next_header_re.search(rest)
    if next_match:
        section_text = rest[: next_match.start()]
    else:
        section_text = rest

    return set(_SLUG_TOKEN_RE.findall(section_text))


# ─── Per-slug telemetry queries ───────────────────────────────────────────


def _traffic_score(db_path: Path, slug: str, days: int) -> int:
    """Count tool_calls rows with ``skill_name = slug`` in last ``days``.

    Returns 0 if the db is missing, the table/columns are missing, or no
    rows match. The W1 fallback (JSONL scanning) is handled at the caller
    level via ``check_skill_name_column``; here we just trust the column.
    """
    if not db_path.exists():
        return 0

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT COUNT(*)
                FROM tool_calls
                WHERE skill_name = ?
                  AND timestamp >= ?
                """,
                (slug, cutoff),
            )
            row = cur.fetchone()
            return int(row[0]) if row and row[0] is not None else 0
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


def _error_boost(db_path: Path, slug: str) -> float:
    """Compute mean tool_error_count / tool_use_count across sessions
    where ``slug`` fired (joined via session_id on tool_calls).

    Returns 0.0 if no matching sessions, the db is missing, or any of the
    columns are unavailable. Sessions with ``tool_use_count == 0`` are
    skipped (avoids divide-by-zero).
    """
    if not db_path.exists():
        return 0.0

    try:
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT DISTINCT s.tool_error_count, s.tool_use_count
                FROM sessions s
                JOIN tool_calls tc ON tc.session_id = s.session_id
                WHERE tc.skill_name = ?
                """,
                (slug,),
            )
            ratios: list[float] = []
            for err, use in cur.fetchall():
                try:
                    err_i = int(err or 0)
                    use_i = int(use or 0)
                except (TypeError, ValueError):
                    continue
                if use_i <= 0:
                    continue
                ratios.append(err_i / use_i)
            if not ratios:
                return 0.0
            return sum(ratios) / len(ratios)
        finally:
            conn.close()
    except sqlite3.Error:
        return 0.0


# ─── Public ranking entry point ───────────────────────────────────────────


def _selector_log_table(ranks: list[SkillRank]) -> str:
    """Render the ranking as a markdown table — written to selector_log.md."""
    lines = [
        "# Selector ranking",
        "",
        "| Rank | Slug | Traffic | Error boost | Analyst boost | Priority |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(ranks, start=1):
        lines.append(
            f"| {i} | {r.slug} | {r.traffic_score} | {r.error_boost:.4f} | {r.analyst_boost:.2f} | {r.priority:.4f} |"
        )
    if not ranks:
        lines.append("| - | (no skills found) | - | - | - | - |")
    return "\n".join(lines) + "\n"


def rank_skills(
    db_path: Path,
    bundle_dir: Path,
    source_repo: str,
    days: int = 60,
    running_md_path: Path | None = None,
    run_dir: Path | None = None,
) -> list[SkillRank]:
    """Rank every skill under ``bundle_dir/skills/`` by priority.

    Args:
        db_path: path to ``corpus.db``.
        bundle_dir: ``~/.watchmen/bundles/<project>/``.
        source_repo: source repo for project (unused at this layer but
            preserved in the signature for symmetry with corpus.py).
        days: lookback window for traffic_score (default 60).
        running_md_path: optional ``analyses/<project>/_running.md`` —
            slugs in its "Skill candidates" section get the 1.5× boost.
        run_dir: if provided, ``selector_log.md`` is written here.

    Returns:
        List of ``SkillRank`` sorted by priority descending. Cold-start
        fallback: if every traffic_score is 0, the list is re-sorted by
        SKILL.md mtime descending.
    """
    # source_repo is part of the spec'd signature even though this layer
    # doesn't use it directly — leave for callers' symmetry.
    _ = source_repo

    skills_dir = bundle_dir / "skills"
    if not skills_dir.exists() or not skills_dir.is_dir():
        ranks: list[SkillRank] = []
    else:
        slugs = sorted(entry.name for entry in skills_dir.iterdir() if entry.is_dir())

        analyst_set: set[str] = set()
        if running_md_path is not None:
            analyst_set = get_analyst_candidates(running_md_path)

        ranks = []
        for slug in slugs:
            traffic = _traffic_score(db_path, slug, days)
            err_boost = _error_boost(db_path, slug)
            an_boost = 1.5 if slug in analyst_set else 1.0
            priority = traffic * (1 + err_boost) * an_boost
            ranks.append(
                SkillRank(
                    slug=slug,
                    traffic_score=traffic,
                    error_boost=err_boost,
                    analyst_boost=an_boost,
                    priority=priority,
                )
            )

        # Cold-start fallback: if every traffic_score is 0, rank by
        # SKILL.md mtime descending so the most-recently-touched bundle
        # surfaces first. This stops the priority-0 tie from being
        # decided alphabetically.
        if ranks and all(r.traffic_score == 0 for r in ranks):

            def _mtime(slug: str) -> float:
                skill_md = skills_dir / slug / "SKILL.md"
                try:
                    return skill_md.stat().st_mtime
                except OSError:
                    return 0.0

            ranks.sort(key=lambda r: _mtime(r.slug), reverse=True)
        else:
            ranks.sort(key=lambda r: r.priority, reverse=True)

    if run_dir is not None:
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "selector_log.md").write_text(_selector_log_table(ranks), encoding="utf-8")
        except OSError:
            # Writing the log is best-effort; never fail the ranking call.
            pass

    return ranks
