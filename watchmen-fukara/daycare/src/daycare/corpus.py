"""Corpus layer — JSONL transcript parsing and corpus.db queries.

Daycare ships its OWN transcript parser instead of using
watchmen.transcript.read_session_full, which truncates content to 600
chars (W2 in the spec). We need full content because the proposer needs
to see what the strong model actually wrote.

Two transcript formats are supported gracefully:
  - Claude Code JSONL (``~/.claude/projects/*/<session_uuid>.jsonl``)
  - pi-agent / claude-agent-acp JSONL (``~/.pi/agent/sessions/...``)

Both share the same envelope shape (type, uuid, parentUuid, timestamp,
message) but differ in subtle content-block details — we try/except
per line so a malformed row never kills the parse.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Turn:
    """One (user_text, assistant_response, tool_calls_in_turn) triple
    reconstructed from a transcript.

    Fields:
        user_text: verbatim user turn (full content, no truncation).
        assistant_text: concatenated text blocks from the assistant reply.
        tool_calls: list of {name, input, ...} dicts (one per tool_use
            block in the assistant's content array).
        skill_name: extracted from tool_use blocks where ``name == "Skill"``
            and ``input["skill"]`` is set; None otherwise.
        timestamp: assistant message's ``timestamp`` field (iso8601 string).
        accepted: implicit acceptance signal computed downstream — None at
            parse time, filled in by eval_builder later.
    """

    user_text: str
    assistant_text: str
    tool_calls: list[dict] = field(default_factory=list)
    skill_name: str | None = None
    timestamp: str = ""
    accepted: bool | None = None


def _coerce_user_content(content) -> str:
    """User content can be a bare string OR a list of blocks. Normalise.

    For block lists, concatenate all ``text`` fields and skip tool_result
    blocks (those don't carry the user's verbal turn). pi-agent format
    uses the same shape but may omit a ``type`` key — we treat any block
    with a string ``text`` field as a user-text contribution.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            # Skip tool_result — that's the previous-turn's tool reply,
            # not the user's verbal turn. The downstream proposer doesn't
            # care about it for eval extraction purposes.
            if btype == "tool_result":
                continue
            txt = block.get("text")
            if isinstance(txt, str):
                parts.append(txt)
        return "\n".join(parts)
    return ""


def _extract_assistant_payload(content) -> tuple[str, list[dict], str | None]:
    """Walk an assistant content block list. Return (text, tool_calls, skill_name).

    Assistant ``content`` is always a list of typed blocks per Claude /
    pi-agent JSONL conventions. We grab every ``text`` block's text and
    every ``tool_use`` block's name/input; the Skill tool_use also yields
    the optional skill_name.
    """
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    skill_name: str | None = None

    if not isinstance(content, list):
        return "", [], None

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            txt = block.get("text")
            if isinstance(txt, str):
                text_parts.append(txt)
        elif btype == "tool_use":
            call = {
                "id": block.get("id"),
                "name": block.get("name"),
                "input": block.get("input", {}),
            }
            tool_calls.append(call)
            # Skill tool_use blocks carry input.skill = "<slug>". This is
            # the load-bearing source of skill_name when corpus.db's
            # skill_name column is missing or all-NULL (W1 fallback).
            if (
                block.get("name") == "Skill"
                and isinstance(block.get("input"), dict)
                and isinstance(block["input"].get("skill"), str)
            ):
                skill_name = block["input"]["skill"]

    return "\n".join(text_parts), tool_calls, skill_name


def parse_transcript(path: Path) -> list[Turn]:
    """Parse a Claude Code or pi-agent JSONL transcript into Turn triples.

    Walks the file linearly (parentUuid order is implicit in JSONL append
    order in practice; we don't bother reconstructing a tree). Each
    user message followed by an assistant message becomes one Turn.

    Returns ``[]`` if the file doesn't exist — W3 guard, transcripts can
    be GC'd by Claude Code. Malformed lines are skipped silently
    (try/except per line). No content truncation.
    """
    if not path.exists():
        return []

    turns: list[Turn] = []
    pending_user: str | None = None

    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    env = json.loads(line)
                except json.JSONDecodeError:
                    # Malformed line — skip, don't kill the parse.
                    continue
                if not isinstance(env, dict):
                    continue

                msg = env.get("message") or {}
                if not isinstance(msg, dict):
                    msg = {}

                env_type = env.get("type") or msg.get("role")

                try:
                    if env_type == "user":
                        # Buffer the user content; pair with the next
                        # assistant message we see.
                        pending_user = _coerce_user_content(msg.get("content"))
                    elif env_type == "assistant":
                        atxt, tool_calls, skill_name = _extract_assistant_payload(msg.get("content"))
                        turns.append(
                            Turn(
                                user_text=pending_user or "",
                                assistant_text=atxt,
                                tool_calls=tool_calls,
                                skill_name=skill_name,
                                timestamp=str(env.get("timestamp", "")),
                                accepted=None,
                            )
                        )
                        # Reset — next user message starts a new pairing.
                        pending_user = None
                except Exception:
                    # Per-line guard: any parse exception inside this
                    # envelope just skips it, never aborts the file.
                    continue
    except OSError:
        # Filesystem error reading the transcript (rare; permission
        # change after Path.exists succeeded). Treat as missing.
        return []

    return turns


def match_session_to_project(project_dir: str, source_repo: str) -> bool:
    """W6 — project_dir → bundle mapping.

    Matches iff:
      - ``project_dir == source_repo``, OR
      - ``project_dir.startswith(source_repo + "/")`` (subdir of repo), OR
      - ``basename(project_dir) == basename(source_repo)`` (basename fallback
        for users who moved the repo).
    """
    if not project_dir or not source_repo:
        return False
    if project_dir == source_repo:
        return True
    if project_dir.startswith(source_repo + "/"):
        return True
    if Path(project_dir).name == Path(source_repo).name:
        return True
    return False


def query_sessions(db_path: Path, source_repo: str, days: int = 60) -> list[dict]:
    """Query non-subagent sessions from the last ``days`` for ``source_repo``.

    Strategy:
      - SELECT all non-subagent sessions in the time window, sorted by
        cost_usd DESC (so the mega-sessions surface first — they're 97%
        of user-turn volume per the v3 corpus-reality findings).
      - Filter in Python via match_session_to_project so we get all three
        W6 match modes (exact / startswith / basename), not just SQL LIKE.

    Returns a list of dicts (column_name → value) for downstream
    transcript reading. Empty list if the db is missing.
    """
    if not db_path.exists():
        return []

    sessions: list[dict] = []
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        # Use a relative cutoff expressed in seconds since the epoch.
        # `started_at` in watchmen's schema is stored as ISO8601 text,
        # so we compare lexically against an ISO cutoff — works because
        # ISO8601 is lex-sortable.
        from datetime import datetime, timedelta, timezone

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        cur.execute(
            """
            SELECT *
            FROM sessions
            WHERE is_subagent = 0
              AND started_at >= ?
            ORDER BY cost_usd DESC
            """,
            (cutoff,),
        )
        for row in cur.fetchall():
            row_dict = dict(row)
            project_dir = row_dict.get("project_dir") or ""
            if match_session_to_project(project_dir, source_repo):
                sessions.append(row_dict)
    finally:
        conn.close()

    return sessions


def check_skill_name_column(db_path: Path) -> bool:
    """W1 — check whether tool_calls.skill_name is populated.

    Returns True iff:
      - the column exists in the tool_calls table, AND
      - at least one row has skill_name IS NOT NULL.

    Phase 0 doctor calls this; if it returns False we either re-ingest
    or fall back to JSONL scanning.
    """
    if not db_path.exists():
        return False

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        # PRAGMA returns (cid, name, type, notnull, dflt_value, pk) rows.
        cur.execute("PRAGMA table_info(tool_calls)")
        cols = {row[1] for row in cur.fetchall()}
        if "skill_name" not in cols:
            return False
        cur.execute("SELECT 1 FROM tool_calls WHERE skill_name IS NOT NULL LIMIT 1")
        return cur.fetchone() is not None
    except sqlite3.Error:
        # Table missing / db schema mismatch — treat as not populated.
        return False
    finally:
        conn.close()
