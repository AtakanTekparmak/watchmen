"""Unit tests for daycare.corpus (Stream 6.1)."""

from __future__ import annotations

import sqlite3
from pathlib import Path


from daycare.corpus import (
    Turn,
    check_skill_name_column,
    match_session_to_project,
    parse_transcript,
)


FIXTURE = Path(__file__).parent / "fixtures" / "sample_transcript.jsonl"


# ─── parse_transcript ─────────────────────────────────────────────────────


def test_parse_transcript_turn_count():
    """Synthetic fixture has 5 user + 5 assistant messages → 5 Turn triples."""
    turns = parse_transcript(FIXTURE)
    assert len(turns) == 5
    for t in turns:
        assert isinstance(t, Turn)
        assert t.user_text  # every Turn has a paired user message
        assert t.assistant_text


def test_parse_transcript_tool_calls():
    """Fixture has exactly 2 non-Skill tool_use blocks (Write + Edit) and 1 Skill."""
    turns = parse_transcript(FIXTURE)
    all_tool_calls = [tc for t in turns for tc in t.tool_calls]
    # 2 tool_calls + 1 Skill tool_use = 3 tool_use blocks total
    assert len(all_tool_calls) == 3
    names = sorted(tc["name"] for tc in all_tool_calls)
    assert names == ["Edit", "Skill", "Write"]


def test_parse_transcript_skill_name_extracted():
    """The Skill tool_use carries input.skill='systematic-debugging'."""
    turns = parse_transcript(FIXTURE)
    skill_names = [t.skill_name for t in turns if t.skill_name]
    assert skill_names == ["systematic-debugging"]


def test_parse_transcript_no_truncation():
    """W2 contract: parser does NOT truncate to 600 chars like watchmen's
    read_session_full does. The last assistant turn in the fixture is
    intentionally long — verify it survives intact."""
    turns = parse_transcript(FIXTURE)
    final = turns[-1]
    assert "unit tests should be cheap and numerous" in final.assistant_text
    # Sanity: the long final reply is preserved as a single block.
    assert len(final.assistant_text) > 200


def test_parse_transcript_missing_file_returns_empty():
    """W3 — transcripts can be GC'd by Claude Code. Missing file → []."""
    assert parse_transcript(Path("/tmp/does-not-exist-xyz.jsonl")) == []


# ─── match_session_to_project (W6) ────────────────────────────────────────


def test_match_session_to_project_exact():
    assert match_session_to_project("/Users/me/work/ctf", "/Users/me/work/ctf") is True


def test_match_session_to_project_startswith():
    """Subdirs of the source repo should match (sd-zero/, pedogogical-rl/)."""
    assert match_session_to_project("/Users/me/work/ctf/sd-zero", "/Users/me/work/ctf") is True


def test_match_session_to_project_basename():
    """Basename fallback for users who moved the repo."""
    assert match_session_to_project("/other/place/ctf", "/Users/me/work/ctf") is True


def test_match_session_to_project_no_match():
    assert match_session_to_project("/Users/me/work/wmca", "/Users/me/work/ctf") is False


def test_match_session_to_project_empty_strings():
    """Empty either side → False (defensive)."""
    assert match_session_to_project("", "/foo") is False
    assert match_session_to_project("/foo", "") is False


def test_match_session_to_project_partial_prefix_not_match():
    """startswith must require a trailing slash — otherwise '/foo/bar'
    would falsely match source_repo='/foo/ba'."""
    assert (
        match_session_to_project("/Users/me/work/ctf2", "/Users/me/work/ctf")
        # basename differs ("ctf2" vs "ctf"); no exact match; no slash-prefix
        is False
    )


# ─── check_skill_name_column (W1) ─────────────────────────────────────────


def _make_db(path: Path, *, with_column: bool, with_value: bool) -> None:
    """Build a synthetic tool_calls schema variant for the W1 doctor check."""
    conn = sqlite3.connect(str(path))
    try:
        cur = conn.cursor()
        if with_column:
            cur.execute("CREATE TABLE tool_calls (id INTEGER PRIMARY KEY, session_id TEXT, name TEXT, skill_name TEXT)")
            if with_value:
                cur.execute(
                    "INSERT INTO tool_calls (session_id, name, skill_name) "
                    "VALUES ('s1', 'Skill', 'systematic-debugging')"
                )
            else:
                # Insert a row with skill_name NULL — column exists but no
                # populated rows.
                cur.execute("INSERT INTO tool_calls (session_id, name, skill_name) VALUES ('s1', 'Bash', NULL)")
        else:
            # Old schema — no skill_name column.
            cur.execute("CREATE TABLE tool_calls (id INTEGER PRIMARY KEY, session_id TEXT, name TEXT)")
            cur.execute("INSERT INTO tool_calls (session_id, name) VALUES ('s1', 'Bash')")
        conn.commit()
    finally:
        conn.close()


def test_check_skill_name_column_missing(tmp_path):
    """Old schema (no column) → False."""
    db = tmp_path / "corpus.db"
    _make_db(db, with_column=False, with_value=False)
    assert check_skill_name_column(db) is False


def test_check_skill_name_column_all_null(tmp_path):
    """New schema but all rows NULL → False (re-ingest needed)."""
    db = tmp_path / "corpus.db"
    _make_db(db, with_column=True, with_value=False)
    assert check_skill_name_column(db) is False


def test_check_skill_name_column_populated(tmp_path):
    """Populated column → True (doctor passes)."""
    db = tmp_path / "corpus.db"
    _make_db(db, with_column=True, with_value=True)
    assert check_skill_name_column(db) is True


def test_check_skill_name_column_missing_db(tmp_path):
    """Missing db file → False (don't crash)."""
    assert check_skill_name_column(tmp_path / "nope.db") is False
