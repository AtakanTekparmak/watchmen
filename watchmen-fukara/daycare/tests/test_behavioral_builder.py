"""Unit tests for daycare.behavioral_builder (Group A)."""

from __future__ import annotations

from daycare.corpus import Turn
from daycare.behavioral_builder import (
    _BEHAVIORAL_RUBRIC_SYSTEM,
    build_action_reference,
    build_prompt_with_history,
    is_behavioral_decision_point,
)


def _mk_turn(
    user_text: str = "",
    assistant_text: str = "",
    tool_calls: list[dict] | None = None,
    skill_name: str | None = None,
) -> Turn:
    """Construct a minimal Turn for tests."""
    return Turn(
        user_text=user_text,
        assistant_text=assistant_text,
        tool_calls=tool_calls or [],
        skill_name=skill_name,
        timestamp="",
        accepted=None,
    )


# ─── is_behavioral_decision_point ──────────────────────────────────────────


def test_is_behavioral_decision_point_skill_invoke():
    """Turn with a Skill tool_use → (True, "skill_invoke")."""
    turn = _mk_turn(
        user_text="run the thing",
        assistant_text="Invoking the skill now.",
        tool_calls=[{"name": "Skill", "input": {"skill": "do-thing"}}],
    )
    keep, reason = is_behavioral_decision_point(turn, None)
    assert keep is True
    assert reason == "skill_invoke"


def test_is_behavioral_decision_point_multi_tool():
    """Turn with 3 tool_calls → (True, "multi_tool")."""
    turn = _mk_turn(
        user_text="do several things",
        assistant_text="Running multiple tools.",
        tool_calls=[
            {"name": "Bash", "input": {"command": "ls"}},
            {"name": "Edit", "input": {"file_path": "/x"}},
            {"name": "Write", "input": {"file_path": "/y"}},
        ],
    )
    keep, reason = is_behavioral_decision_point(turn, None)
    assert keep is True
    assert reason == "multi_tool"


def test_is_behavioral_decision_point_error_recovery():
    """User turn contains 'Traceback' → (True, "error_recovery")."""
    turn = _mk_turn(
        user_text="I hit a Traceback when running it",
        assistant_text="Let me diagnose this for you carefully.",
        tool_calls=[{"name": "Bash", "input": {"command": "python -c 'pass'"}}],
    )
    keep, reason = is_behavioral_decision_point(turn, None)
    assert keep is True
    assert reason == "error_recovery"


def test_rejects_trivial_read_only():
    """Single Read tool_call + short assistant_text → rejected."""
    turn = _mk_turn(
        user_text="show me the file",
        assistant_text="ok",
        tool_calls=[{"name": "Read", "input": {"file_path": "/tmp/x"}}],
    )
    keep, reason = is_behavioral_decision_point(turn, None)
    assert keep is False
    assert reason == "rejected_trivial_read"


def test_rejects_safety_refusal():
    """Safety refusal text → rejected."""
    turn = _mk_turn(
        user_text="do something bad",
        assistant_text="I cannot assist with that request, sorry.",
        tool_calls=[],
    )
    keep, reason = is_behavioral_decision_point(turn, None)
    assert keep is False
    assert reason == "rejected_safety"


# ─── build_prompt_with_history ─────────────────────────────────────────────


def test_build_prompt_with_history_truncates_from_front():
    """10 fake turns, max_chars=300 → result ends with most-recent user turn
    and total length ≤ 300."""
    turns = []
    for i in range(10):
        turns.append(
            _mk_turn(
                user_text=f"u{i}: " + ("x" * 80),
                assistant_text=f"a{i}: " + ("y" * 80),
            )
        )
    # idx=9 — the last user turn is the held-back one.
    result = build_prompt_with_history(turns, 9, max_chars=300)
    assert len(result) <= 300
    # Most recent user turn should be at/near the end.
    assert "u9:" in result


# ─── build_action_reference ────────────────────────────────────────────────


def test_build_action_reference_tool_call():
    """Turn with one Bash tool_call → reference starts with the canonical prefix."""
    turn = _mk_turn(
        user_text="run ls",
        assistant_text="Listing files.",
        tool_calls=[{"name": "Bash", "input": {"command": "ls -la"}}],
    )
    ref = build_action_reference(turn)
    assert ref.startswith("ACTION: invoke Bash\nINPUT: ")


def test_build_action_reference_text_only():
    """Turn with empty tool_calls → reference starts with text_response prefix."""
    turn = _mk_turn(
        user_text="explain X",
        assistant_text="Here is a careful explanation of X with detail.",
        tool_calls=[],
    )
    ref = build_action_reference(turn)
    assert ref.startswith("ACTION: text_response\nTEXT: ")


def test_is_behavioral_decision_point_action_tool():
    """Single Bash tool_call (CC corpus format) → (True, 'action_tool').

    This is the primary positive signal for Claude Code corpus where turns
    carry exactly one tool call and assistant_text is empty for tool-using turns.
    """
    turn = _mk_turn(
        user_text="run the tests",
        assistant_text="",
        tool_calls=[{"name": "Bash", "input": {"command": "pytest tests/ -q"}}],
    )
    keep, reason = is_behavioral_decision_point(turn, None)
    assert keep is True
    assert reason == "action_tool"


def test_action_tool_covers_all_action_names():
    """Every name in _ACTION_TOOLS produces a keep on a single-tool turn."""
    from daycare.behavioral_builder import _ACTION_TOOLS

    for name in _ACTION_TOOLS:
        turn = _mk_turn(
            user_text="do something",
            assistant_text="",
            tool_calls=[{"name": name, "input": {}}],
        )
        keep, reason = is_behavioral_decision_point(turn, None)
        assert keep is True, f"Expected keep for tool {name!r}, got rejected ({reason})"
        assert reason in ("skill_invoke", "action_tool"), f"Unexpected reason {reason!r} for {name!r}"


# ─── rubric template ──────────────────────────────────────────────────────


def test_behavioral_rubric_template_contains_action_clause():
    """The system prompt must call out 'behavioral action' explicitly."""
    assert "behavioral action" in _BEHAVIORAL_RUBRIC_SYSTEM
