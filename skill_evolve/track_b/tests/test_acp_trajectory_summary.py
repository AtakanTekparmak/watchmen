"""v9d patch: tests for the ACP-trajectory summarizer in bench_cli.py.

The summarizer is the load-bearing piece of the v9d patch — it pulls
the agent's last few execute commands and final message out of
``trajectory/acp_trajectory.jsonl`` so they can flow into
``failures_render`` and reach the mutator.

We hand-build a minimal JSONL fixture that mirrors the real format
observed on the pod (Gemini CLI ACP, benchflow 0.3.x).
"""

from __future__ import annotations

import json
from pathlib import Path

from skill_evolve.agents.bench_cli import (
    _build_failure_last_msg,
    _summarize_acp_trajectory,
)


def _write_fixture(trial_dir: Path, events: list[dict]) -> None:
    out = trial_dir / "trajectory" / "acp_trajectory.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def test_summary_extracts_final_message_and_commands(tmp_path: Path) -> None:
    events = [
        {"type": "tool_call", "kind": "execute", "title": "ls /root", "status": "completed"},
        {"type": "agent_thought", "text": "Long internal CoT — this should NOT appear in the excerpt."},
        {"type": "tool_call", "kind": "think", "title": "Plan: scan files"},
        {"type": "tool_call", "kind": "execute", "title": "head -n 5 /root/data.csv"},
        {"type": "tool_call", "kind": "edit", "title": "/root/answer.json"},
        {"type": "tool_call", "kind": "execute", "title": "python3 my_skill/scripts/extract_records.py /root/data.csv"},
        {"type": "agent_message", "text": "Final answer: 42 rows extracted."},
    ]
    _write_fixture(tmp_path, events)
    s = _summarize_acp_trajectory(tmp_path)

    assert "Final answer: 42 rows extracted." in s["final_message"]
    assert "ls /root" in s["excerpt"]
    assert "head -n 5" in s["excerpt"]
    assert "extract_records.py" in s["excerpt"]
    # CoT must NOT leak.
    assert "internal CoT" not in s["excerpt"]
    assert "internal CoT" not in s["final_message"]
    # Edit titles present in excerpt's edits line.
    assert "answer.json" in s["excerpt"]
    # Skill discovery from script invocation.
    assert "my_skill" in s["skills_invoked"]


def test_summary_handles_missing_file(tmp_path: Path) -> None:
    s = _summarize_acp_trajectory(tmp_path)
    assert s == {"excerpt": "", "final_message": "", "skills_invoked": []}


def test_summary_handles_malformed_lines(tmp_path: Path) -> None:
    out = tmp_path / "trajectory" / "acp_trajectory.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(
            [
                "not json at all",
                json.dumps({"type": "tool_call", "kind": "execute", "title": "echo ok"}),
                "{trailing comma,}",
                json.dumps({"type": "agent_message", "text": "done"}),
            ]
        )
    )
    s = _summarize_acp_trajectory(tmp_path)
    assert "echo ok" in s["excerpt"]
    assert s["final_message"] == "done"


def test_summary_truncates_to_caps(tmp_path: Path) -> None:
    long_msg = "X" * 10_000
    events = [
        {"type": "tool_call", "kind": "execute", "title": "Y" * 800},
        {"type": "agent_message", "text": long_msg},
    ]
    _write_fixture(tmp_path, events)
    s = _summarize_acp_trajectory(
        tmp_path,
        max_excerpt_chars=200,
        max_final_chars=500,
        last_n_commands=6,
    )
    assert len(s["final_message"]) <= 500
    assert len(s["excerpt"]) <= 220  # 200 + truncation marker overhead


def test_summary_keeps_only_last_n_commands(tmp_path: Path) -> None:
    events = [
        {"type": "tool_call", "kind": "execute", "title": f"cmd_{i}"} for i in range(15)
    ]
    _write_fixture(tmp_path, events)
    s = _summarize_acp_trajectory(tmp_path, last_n_commands=3)
    # Only last 3 commands present; earlier ones dropped.
    assert "cmd_14" in s["excerpt"]
    assert "cmd_13" in s["excerpt"]
    assert "cmd_12" in s["excerpt"]
    assert "cmd_5" not in s["excerpt"]
    assert "cmd_0" not in s["excerpt"]


def test_summary_falls_back_to_agent_path(tmp_path: Path) -> None:
    """Older bench-cli versions emit agent/acp_trajectory.jsonl."""
    out = tmp_path / "agent" / "acp_trajectory.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"type": "agent_message", "text": "via agent path"}) + "\n")
    s = _summarize_acp_trajectory(tmp_path)
    assert s["final_message"] == "via agent path"


def test_build_failure_last_msg_layout() -> None:
    summary = {
        "excerpt": "  1. ls\n  2. cat foo",
        "final_message": "I think I solved it.",
        "skills_invoked": ["my_skill"],
    }
    blob = _build_failure_last_msg(
        verifier_status="failed",
        error=None,
        verifier_error="reward 0.0 < 1.0",
        summary=summary,
    )
    # Sections appear in priority order.
    assert blob.index("[verifier]") < blob.index("[final agent message]")
    assert blob.index("[final agent message]") < blob.index("[last 6 commands]")
    assert "failed" in blob
    assert "reward 0.0" in blob
    assert "I think I solved it." in blob
    assert "ls" in blob


def test_build_failure_last_msg_skips_empty_sections() -> None:
    blob = _build_failure_last_msg(
        verifier_status="agent_error",
        error="boom",
        verifier_error=None,
        summary={"excerpt": "", "final_message": "", "skills_invoked": []},
    )
    assert "[verifier]" in blob
    assert "boom" in blob
    assert "[final agent message]" not in blob
    assert "[last 6 commands]" not in blob


def test_build_failure_last_msg_caps_total() -> None:
    summary = {
        "excerpt": "X" * 5000,
        "final_message": "Y" * 5000,
        "skills_invoked": [],
    }
    blob = _build_failure_last_msg(
        verifier_status="failed",
        error=None,
        verifier_error="z",
        summary=summary,
        max_total=1000,
    )
    assert len(blob) <= 1000 + len("\n[hard-truncated]")
    assert "[hard-truncated]" in blob
