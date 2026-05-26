"""Phase 1b' — behavioral eval extraction.

Where ``eval_builder.pull_and_classify`` extracts (user_turn, assistant_text)
triples and judges *what the assistant produced as final output*, this
module extracts *what the assistant chose to DO next* — the behavioral
decision point. The reference is the strong model's next move (tool
invocation or terse text), not its full prose.

Why a separate path:
- Corpus-extracted procedural_qa rows train shape, not behavior. A weak
  model needs to learn: "given this conversation state, invoke <tool>
  with <args>", not "explain how to do X in three paragraphs".
- The judge can score *action equivalence* (same tool? same intent?)
  without exact-string compare. That's the behavioral rubric below.

Output dicts match ``eval_builder.pull_and_classify`` shape so the
existing Phase 1e (anonymize/dedup/split) pipeline runs unchanged.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .corpus import Turn, parse_transcript, query_sessions
from .eval_builder import (
    _has_live_infra,
    _has_safety_refusal,
    _judge_call,
    _extract_json_obj,
    _log_discard,
    calibrate_eval,
)


# ─── Decision-point heuristics ─────────────────────────────────────────────


_ERROR_KEYWORDS_RE = re.compile(
    r"\b(error|failed|traceback|exception|wrong|broken)\b",
    re.IGNORECASE,
)

_REASON_THEN_ACT_MARKERS: tuple[str, ...] = (
    "I'll",
    "Let me",
    "First,",
    "Step 1",
)

_TRIVIAL_READ_TOOLS: frozenset[str] = frozenset({"Read", "LS", "Glob", "Grep"})

# Claude Code corpus stores one tool call per turn (not batched). These tool
# names represent substantive behavioral decisions worth learning from.
_ACTION_TOOLS: frozenset[str] = frozenset(
    {
        "Bash",
        "Edit",
        "MultiEdit",
        "Write",
        "Agent",
        "Skill",
        "TaskCreate",
        "TaskUpdate",
        "NotebookEdit",
        "WebFetch",
        "WebSearch",
    }
)


def is_behavioral_decision_point(turn: Turn, next_turn: Turn | None) -> tuple[bool, str]:
    """Return (keep, reason). True only if the turn contains a non-trivial
    behavioral move.

    Claude Code corpus stores ONE tool call per turn (not batched), so
    multi_tool(≥2) and reason_then_act (requires non-empty assistant_text
    on tool-using turns) are structurally impossible. We use action_tool
    as the primary positive signal.

    Keep iff ANY of:
      - skill_invoke: any tool_call with name=="Skill"
      - multi_tool: ≥2 tool_calls (kept for non-CC corpus compatibility)
      - error_recovery: the turn's user_text matches an error keyword
      - action_tool: single tool_call whose name is in _ACTION_TOOLS

    Reject iff:
      - tool_calls empty AND assistant_text < 40 chars
      - assistant_text matches eval_builder._has_safety_refusal
      - turn passes eval_builder._has_live_infra
      - single trivial Read/LS/Glob/Grep with short or no assistant_text
    """
    atxt = turn.assistant_text or ""
    tcs = turn.tool_calls or []

    # Hard rejects first.
    if _has_safety_refusal(turn):
        return False, "rejected_safety"
    if _has_live_infra(turn):
        return False, "rejected_live_infra"
    if not tcs and len(atxt) < 40:
        return False, "rejected_empty_text"
    if len(tcs) == 1 and len(atxt) < 80 and (tcs[0].get("name") in _TRIVIAL_READ_TOOLS):
        return False, "rejected_trivial_read"

    # Accept signals.
    for tc in tcs:
        if tc.get("name") == "Skill":
            return True, "skill_invoke"

    if len(tcs) >= 2:
        return True, "multi_tool"

    if turn.user_text and _ERROR_KEYWORDS_RE.search(turn.user_text):
        return True, "error_recovery"

    # Primary signal for CC corpus: any substantive single-tool action.
    if tcs and tcs[0].get("name") in _ACTION_TOOLS:
        return True, "action_tool"

    # No positive signal — drop as low-signal text.
    return False, "rejected_empty_text"


# ─── Prompt / reference construction ───────────────────────────────────────


def _truncate_input_preview(inp: object, max_chars: int) -> str:
    """Render a tool_call input as a short preview string."""
    if isinstance(inp, dict):
        for k in ("command", "content", "code", "file_path", "path", "prompt"):
            v = inp.get(k)
            if isinstance(v, str):
                return v[:max_chars]
        try:
            return json.dumps(inp, ensure_ascii=False)[:max_chars]
        except Exception:
            return ""
    if isinstance(inp, str):
        return inp[:max_chars]
    return ""


def build_prompt_with_history(turns: list[Turn], idx: int, max_chars: int = 3000) -> str:
    """Walk turns[0:idx] and the user portion of turns[idx], concatenating:

      "user: <text>\\n\\nassistant: <text>\\n\\n[tool: <name>] <input_preview>\\n\\n"

    Truncate from the FRONT (keep most recent context). The last entry is
    the user turn that triggered turns[idx] — the assistant decision is
    the held-back reference.
    """
    chunks: list[str] = []
    for prior in turns[:idx]:
        if prior.user_text:
            chunks.append(f"user: {prior.user_text}\n\n")
        if prior.assistant_text:
            chunks.append(f"assistant: {prior.assistant_text}\n\n")
        for tc in prior.tool_calls or []:
            name = tc.get("name", "")
            preview = _truncate_input_preview(tc.get("input"), 200)
            chunks.append(f"[tool: {name}] {preview}\n\n")

    # Held-back user turn (the one that triggered turns[idx]).
    trigger = turns[idx].user_text if 0 <= idx < len(turns) else ""
    if trigger:
        chunks.append(f"user: {trigger}\n\n")

    full = "".join(chunks)
    if len(full) <= max_chars:
        return full
    # Truncate from the FRONT — keep the most recent context.
    return full[-max_chars:]


def build_action_reference(turn: Turn) -> str:
    """Return a compact action-oriented reference string.

    If turn.tool_calls is non-empty:
        "ACTION: invoke <tool_name>\\nINPUT: <truncated_json_input_300_chars>"
        (one line per tool call, max 3 calls)
        + "\\nTEXT: <first 200 chars of assistant_text if any>"

    Else (text-only decision):
        "ACTION: text_response\\nTEXT: <first 600 chars of assistant_text>"
    """
    atxt = turn.assistant_text or ""
    tcs = turn.tool_calls or []
    if tcs:
        lines: list[str] = []
        for tc in tcs[:3]:
            name = tc.get("name", "")
            inp = tc.get("input")
            if isinstance(inp, (dict, list)):
                try:
                    inp_str = json.dumps(inp, ensure_ascii=False)[:300]
                except Exception:
                    inp_str = ""
            elif isinstance(inp, str):
                inp_str = inp[:300]
            else:
                inp_str = ""
            lines.append(f"ACTION: invoke {name}\nINPUT: {inp_str}")
        body = "\n".join(lines)
        if atxt:
            body += f"\nTEXT: {atxt[:200]}"
        return body
    return f"ACTION: text_response\nTEXT: {atxt[:600]}"


# ─── Behavioral rubric ─────────────────────────────────────────────────────


_BEHAVIORAL_RUBRIC_SYSTEM = (
    "You write rubrics that judge whether a weak LLM ({weak_model_name}) took the\n"
    "SAME behavioral action as a strong model, given the same conversation context.\n\n"
    "The reference encodes the strong model's action as either:\n"
    "  ACTION: invoke <tool>\\n INPUT: <args>\n"
    "  ACTION: text_response\\n TEXT: <body>\n\n"
    "Rules:\n"
    "- Score the CANDIDATE COMPLETION against the reference action.\n"
    "- Do NOT require exact wording. Award based on whether the candidate would\n"
    "  produce a functionally equivalent next step.\n"
    "- For tool-invocation references: 1.0 if candidate invokes the same tool with\n"
    "  semantically equivalent arguments; 0.5 if same tool but different/missing\n"
    "  argument values; 0.0 if wrong tool or refuses to act.\n"
    "- For text-response references: 1.0 if candidate states the same conclusion\n"
    "  or next-step direction; 0.5 if partial; 0.0 if contradicts or irrelevant.\n"
    "- Forbidden: comparing exact strings; demanding identical phrasing.\n"
    '- Format: "Score 0.0–1.0. The correct action is [X]. Award 1.0 if: [criterion].\n'
    '  Award 0.5 if: [partial]. Award 0.0 if: [failure]."\n'
    "- Max 150 words.\n"
    'Output JSON: {{"rubric": "<text>"}}'
)


def generate_behavioral_rubric(
    prompt: str,
    reference: str,
    weak_model_name: str,
    judge_model: str,
    api_key: str,
) -> str:
    """Produce a ≤150-word behavioral rubric. Returns "" on failure."""
    system = _BEHAVIORAL_RUBRIC_SYSTEM.format(weak_model_name=weak_model_name)
    user = json.dumps(
        {
            "anonymized_prompt": prompt[:1500],
            "anonymized_reference": reference[:500],
        },
        ensure_ascii=False,
    )
    raw = _judge_call(system, user, judge_model, api_key, max_tokens=400)
    parsed = _extract_json_obj(raw or "")
    if parsed and isinstance(parsed.get("rubric"), str):
        return parsed["rubric"].strip()
    return (raw or "").strip()[:1200]


# ─── Top-level extraction ──────────────────────────────────────────────────


def extract_behavioral_evals(
    db_path: Path,
    source_repo: str,
    bundle_dir: Path,
    weak_model: str,
    judge_model: str,
    api_key: str,
    seed: int,
    days: int,
    run_dir: Path,
    max_candidates: int | None = None,
    max_workers: int = 4,
) -> list[dict]:
    """Pull sessions → parse turns → pick behavioral decision points →
    generate action rubrics → calibrate → return pre-anonymization eval dicts.

    Each returned dict has the same key set as eval_builder.pull_and_classify
    so downstream Phase 1e (anonymize/dedup/split) works unchanged:
      {id, type="behavioral_action", prompt, reference, rubric,
       baseline_score, baseline_completion_len_tokens=0, accepted=True,
       source_session, source_skill}
    """
    log_path = run_dir / "eval_extraction_log.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_lock = threading.Lock()

    def _log(kind: str, detail: str) -> None:
        with log_lock:
            _log_discard(log_path, kind, detail)

    sessions = query_sessions(db_path, source_repo, days=days)

    # ── Step 1: collect candidate (session, idx, turns, reason) tuples ──────
    candidates: list[tuple[dict, int, list[Turn], str]] = []
    for sess in sessions:
        transcript_path = sess.get("transcript_path") or ""
        if not transcript_path:
            _log("no_transcript_path", str(sess.get("session_id", "")))
            continue
        path = Path(transcript_path)
        if not path.exists():
            _log("transcript_gone", str(sess.get("session_id", "")))
            continue
        turns = parse_transcript(path)
        if not turns:
            _log("empty_transcript", str(sess.get("session_id", "")))
            continue
        for i, turn in enumerate(turns):
            next_turn = turns[i + 1] if (i + 1) < len(turns) else None
            keep, reason = is_behavioral_decision_point(turn, next_turn)
            sid = str(sess.get("session_id", ""))
            if not keep:
                _log(reason, f"session={sid} turn={i}")
                continue
            candidates.append((sess, i, turns, reason))

    if max_candidates and len(candidates) > max_candidates:
        import random as _rand

        rng = _rand.Random(seed)
        candidates = rng.sample(candidates, max_candidates)

    # ── Step 2: parallel rubric generation ──────────────────────────────────
    def _build_one(
        item: tuple[dict, int, list[Turn], str],
    ) -> tuple[dict, int, Turn, str, str, str] | None:
        sess, idx, turns, reason = item
        sid = str(sess.get("session_id", ""))
        turn = turns[idx]
        prompt = build_prompt_with_history(turns, idx, max_chars=3000)
        reference = build_action_reference(turn)
        if not prompt.strip() or not reference.strip():
            _log("discard_empty_fields", f"session={sid} turn={idx}")
            return None
        rubric = generate_behavioral_rubric(
            prompt=prompt,
            reference=reference,
            weak_model_name=weak_model,
            judge_model=judge_model,
            api_key=api_key,
        )
        if not rubric:
            _log("discard_rubric_empty", f"session={sid} turn={idx}")
            return None
        return sess, idx, turn, prompt, reference, rubric

    with_rubrics: list[tuple[dict, int, Turn, str, str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_build_one, item) for item in candidates]
        for fut in as_completed(futures):
            try:
                r = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"[behavioral_builder] rubric_error: {exc}", file=sys.stderr)
                continue
            if r is not None:
                with_rubrics.append(r)

    # ── Step 3: calibration (subprocess-heavy; fewer workers) ───────────────
    def _calibrate_one(
        item: tuple[dict, int, Turn, str, str, str],
    ) -> dict | None:
        sess, idx, turn, prompt, reference, rubric = item
        sid = str(sess.get("session_id", ""))
        baseline_score = calibrate_eval(
            prompt=prompt,
            rubric=rubric,
            bundle_dir=bundle_dir,
            weak_model=weak_model,
            api_key=api_key,
            judge_model=judge_model,
            seed=seed + idx,
            n_rollouts=3,
        )
        # Behavioral keep-band: 0.0 < score < 0.9.
        if baseline_score <= 0.0:
            _log("discard_calibration_zero", f"session={sid} turn={idx}")
            return None
        if baseline_score >= 0.9:
            _log(
                "discard_calibration_solved",
                f"session={sid} turn={idx} score={baseline_score:.2f}",
            )
            return None
        return {
            "id": uuid.uuid4().hex[:8],
            "type": "behavioral_action",
            "prompt": prompt,
            "reference": reference,
            "rubric": rubric,
            "baseline_score": baseline_score,
            "baseline_completion_len_tokens": 0,
            "accepted": True,
            "source_session": sid,
            "source_skill": turn.skill_name,
        }

    cal_workers = max(1, max_workers // 2)
    survivors: list[dict] = []
    with ThreadPoolExecutor(max_workers=cal_workers) as pool:
        futures = [pool.submit(_calibrate_one, item) for item in with_rubrics]
        for fut in as_completed(futures):
            try:
                r = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"[behavioral_builder] calibrate_error: {exc}", file=sys.stderr)
                continue
            if r is not None:
                survivors.append(r)

    return survivors
