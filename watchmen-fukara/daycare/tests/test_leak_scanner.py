"""Unit tests for daycare.leak_scanner (Stream 6.3)."""

from __future__ import annotations

import json
from pathlib import Path

from daycare.leak_scanner import (
    build_fingerprint_set,
    enforce_policy,
    scan,
)


def _write_eval_set(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_build_fingerprint_set_catches_recurring_ngram(tmp_path):
    """Plant a 15-char identifier in two different prompts → it must appear
    in the fingerprint set via the n-gram cross-eval-recurrence pass."""
    eval_set = tmp_path / "eval_set.jsonl"
    projects_json = tmp_path / "projects.json"
    projects_json.write_text("{}", encoding="utf-8")

    # 15 chars long, identifier-shaped: 'EXFIL-DATA-7K9Q3'
    secret = "EXFIL-DATA-7K9Q"  # 15 chars
    assert len(secret) == 15

    rows = [
        {"prompt": f"Use {secret} for the analysis", "source_session": "s1"},
        {"prompt": f"Now apply {secret} again here", "source_session": "s2"},
    ]
    _write_eval_set(eval_set, rows)

    fps = build_fingerprint_set(eval_set, projects_json)
    # Either the full 15-char string or a substring of length ≥ 12 must
    # be in the set — the spec requires "any ≥12-char n-gram appearing in
    # ≥2 different prompts". Verify the exact secret appears.
    assert secret in fps


def test_build_fingerprint_set_includes_source_session(tmp_path):
    eval_set = tmp_path / "eval_set.jsonl"
    projects_json = tmp_path / "projects.json"
    projects_json.write_text("{}", encoding="utf-8")
    rows = [
        {"prompt": "alpha", "source_session": "session-abcdef-1234"},
        {"prompt": "beta", "source_session": "session-zzzzzz-9999"},
    ]
    _write_eval_set(eval_set, rows)

    fps = build_fingerprint_set(eval_set, projects_json)
    assert "session-abcdef-1234" in fps
    assert "session-zzzzzz-9999" in fps


def test_scan_finds_planted_leak(tmp_path):
    """Plant a 15-char identifier in two prompts, then plant it inside a
    candidate bundle file — scan() must flag it."""
    eval_set = tmp_path / "eval_set.jsonl"
    projects_json = tmp_path / "projects.json"
    projects_json.write_text("{}", encoding="utf-8")

    secret = "LEAK-MARK-XYZ99"  # 15 chars
    rows = [
        {"prompt": f"do {secret} first", "source_session": "s1"},
        {"prompt": f"now {secret} again", "source_session": "s2"},
    ]
    _write_eval_set(eval_set, rows)

    fps = build_fingerprint_set(eval_set, projects_json)
    assert secret in fps

    bundle = tmp_path / "bundle"
    (bundle / "scripts").mkdir(parents=True)
    (bundle / "SKILL.md").write_text(
        f"This SKILL.md innocently mentions {secret} verbatim.\n",
        encoding="utf-8",
    )
    leaks = scan(bundle, fps)
    found = [lk for lk in leaks if lk.fingerprint == secret]
    assert found, f"expected to find {secret} in scan output"
    assert found[0].file == "SKILL.md"


def test_scan_clean_bundle_no_leaks(tmp_path):
    fps = {"VERY-RARE-MARKER-Q42"}
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "SKILL.md").write_text("nothing sensitive here\n", encoding="utf-8")
    leaks = scan(bundle, fps)
    assert leaks == []


def test_enforce_policy_zero_rejects_on_any_leak(tmp_path):
    """`zero` policy → True (reject) on ANY leak."""
    from daycare.leak_scanner import Leak

    leaks = [Leak(file="SKILL.md", line=1, fingerprint="abc", context="abc")]
    log_path = tmp_path / "leak_log.md"
    assert enforce_policy(leaks, "zero", leak_log_path=log_path) is True
    # And the log file is written.
    assert log_path.exists()


def test_enforce_policy_zero_no_leaks_accepts():
    """zero policy + zero leaks → False (no rejection)."""
    assert enforce_policy([], "zero", leak_log_path=None) is False


def test_enforce_policy_warn_proceeds_but_logs(tmp_path):
    """warn → False (proceeds) but writes the log."""
    from daycare.leak_scanner import Leak

    leaks = [Leak(file="SKILL.md", line=1, fingerprint="abc", context="abc")]
    log_path = tmp_path / "leak_log.md"
    result = enforce_policy(leaks, "warn", leak_log_path=log_path)
    assert result is False
    assert log_path.exists()
    text = log_path.read_text(encoding="utf-8")
    assert "abc" in text
