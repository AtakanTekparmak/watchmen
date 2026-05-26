"""Tests for the behavioral eval-extraction integration (Group D).

Covers the wiring between ``corpus.query_sessions`` →
``behavioral_builder.extract_behavioral_evals`` → ``eval_builder.semantic_dedup``
→ ``verifier.score_bundle``. The judge / weak-model / embedding subprocesses
are mocked so these tests run offline.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from daycare import behavioral_builder, eval_builder, verifier
from daycare.runner import RunResult


FIXTURE = Path(__file__).parent / "fixtures" / "transcript_behavioral.jsonl"


# ─── helpers ──────────────────────────────────────────────────────────────


def _fake_session(transcript_path: Path) -> dict:
    """Mimic the dict shape produced by corpus.query_sessions for one row."""
    return {
        "session_id": "sess-behavioral-0",
        "project_dir": "/tmp/proj",
        "started_at": "2026-02-01T10:00:00Z",
        "ended_at": "2026-02-01T11:00:00Z",
        "transcript_path": str(transcript_path),
        "cost_usd": 0.5,
        "is_subagent": 0,
        "model": "claude",
        "total_turns": 6,
    }


# ─── 1. behavioral extraction keeps 4 of 6 ────────────────────────────────


def test_behavioral_extraction_keeps_4_of_6(tmp_path, monkeypatch):
    """End-to-end: fixture has 6 turns, 4 are behavioral decision points."""

    # Stub query_sessions so corpus.db is bypassed entirely. We use
    # monkeypatch on the symbol imported INTO behavioral_builder.
    monkeypatch.setattr(
        behavioral_builder,
        "query_sessions",
        lambda db_path, source_repo, days: [_fake_session(FIXTURE)],
    )

    # Canned rubric — generate_behavioral_rubric is the judge call we
    # want to skip.
    monkeypatch.setattr(
        behavioral_builder,
        "generate_behavioral_rubric",
        lambda **kwargs: "Score 0.0-1.0. Award 1.0 if action matches.",
    )

    # Calibrate returns a fixed mid-band score so every candidate survives
    # the keep-band gate (0.0 < score < 0.9).
    monkeypatch.setattr(
        behavioral_builder,
        "calibrate_eval",
        lambda **kwargs: 0.3,
    )

    survivors = behavioral_builder.extract_behavioral_evals(
        db_path=tmp_path / "corpus.db",
        source_repo="/tmp/proj",
        bundle_dir=tmp_path / "bundle",
        weak_model="weak",
        judge_model="judge",
        api_key="dummy",
        seed=0,
        days=60,
        run_dir=tmp_path / "run",
        max_candidates=None,
        max_workers=1,
    )

    assert len(survivors) == 4, f"expected 4 survivors, got {len(survivors)}"
    assert all(e["type"] == "behavioral_action" for e in survivors)


# ─── 2. each eval has the 10 required fields ──────────────────────────────


REQUIRED_KEYS = {
    "id",
    "type",
    "prompt",
    "reference",
    "rubric",
    "baseline_score",
    "baseline_completion_len_tokens",
    "accepted",
    "source_session",
    "source_skill",
}


def test_behavioral_eval_has_required_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(
        behavioral_builder,
        "query_sessions",
        lambda db_path, source_repo, days: [_fake_session(FIXTURE)],
    )
    monkeypatch.setattr(
        behavioral_builder,
        "generate_behavioral_rubric",
        lambda **kwargs: "Score 0.0-1.0. Award 1.0 if action matches.",
    )
    monkeypatch.setattr(
        behavioral_builder,
        "calibrate_eval",
        lambda **kwargs: 0.3,
    )

    survivors = behavioral_builder.extract_behavioral_evals(
        db_path=tmp_path / "corpus.db",
        source_repo="/tmp/proj",
        bundle_dir=tmp_path / "bundle",
        weak_model="weak",
        judge_model="judge",
        api_key="dummy",
        seed=0,
        days=60,
        run_dir=tmp_path / "run",
        max_candidates=None,
        max_workers=1,
    )

    assert survivors, "expected at least one behavioral eval"
    for ev in survivors:
        missing = REQUIRED_KEYS - set(ev.keys())
        assert not missing, f"eval missing required keys: {missing}"


# ─── 3. dedup with loosened min_clusters ──────────────────────────────────


def _fake_embed_diverse(texts: list[str]) -> list[list[float]]:
    """Return one-hot vectors so every prompt forms its own cluster.

    Avoids the fastembed dependency in tests. With cosine >=0.92 threshold
    and orthogonal one-hot vectors, every eval ends up in a distinct
    cluster (cosine == 0 between any two non-identical vectors).
    """
    n = len(texts)
    out: list[list[float]] = []
    for i in range(n):
        vec = [0.0] * max(n, 1)
        vec[i % max(n, 1)] = 1.0
        out.append(vec)
    return out


def _fake_evals(n: int, prompt_prefix: str = "behavioral prompt") -> list[dict]:
    return [
        {
            "id": f"e{i}",
            "type": "behavioral_action",
            "prompt": f"{prompt_prefix} {i} — distinct content {i * 7}",
            "anonymized_prompt": f"{prompt_prefix} {i} — distinct content {i * 7}",
            "reference": f"ACTION: invoke Bash\nINPUT: cmd-{i}",
            "rubric": "Score 0-1",
            "baseline_score": 0.1 + (i % 10) * 0.05,
            "baseline_completion_len_tokens": 50 + i,
            "accepted": True,
            "source_session": "s",
            "source_skill": None,
        }
        for i in range(n)
    ]


def test_behavioral_dedup_min_clusters_loosened(monkeypatch):
    """22 distinct prompts + min_clusters=20 should NOT raise."""
    monkeypatch.setattr(eval_builder, "_embed_texts", _fake_embed_diverse)

    evals = _fake_evals(22)
    # Must not raise.
    out = eval_builder.semantic_dedup(evals, min_clusters=20)
    # Each prompt is distinct → 22 clusters → all survive.
    assert len(out) == 22


# ─── 4. default min_clusters=30 still enforced ────────────────────────────


def test_semantic_dedup_default_still_requires_30_clusters(monkeypatch):
    """Regression guard: default min_clusters=30 stays strict."""
    monkeypatch.setattr(eval_builder, "_embed_texts", _fake_embed_diverse)

    # 25 distinct prompts → 25 clusters < 30 → must raise.
    evals = _fake_evals(25)
    with pytest.raises(ValueError, match="insufficient_distillable_surface"):
        eval_builder.semantic_dedup(evals)


# ─── 5. behavioral=False routes through pull_and_classify ─────────────────


def test_run_eval_build_legacy_path_unchanged(tmp_path, monkeypatch):
    """run_eval_build(behavioral=False) must hit pull_and_classify and NOT
    extract_behavioral_evals."""
    calls = {"pull_and_classify": 0, "extract_behavioral_evals": 0}

    def _fake_pull_and_classify(**kwargs):
        calls["pull_and_classify"] += 1
        return []

    def _fake_extract_behavioral(**kwargs):
        calls["extract_behavioral_evals"] += 1
        return []

    monkeypatch.setattr(eval_builder, "pull_and_classify", _fake_pull_and_classify)
    monkeypatch.setattr(behavioral_builder, "extract_behavioral_evals", _fake_extract_behavioral)

    # build_context reads projects.json + bundle_dir — give it minimal disk.
    projects_json = tmp_path / "projects.json"
    projects_json.write_text('{"proj": {"source_repo": "/tmp/proj"}}', encoding="utf-8")
    bundle_dir = tmp_path / "bundles" / "proj" / "skills" / "myskill"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # With pull_and_classify returning [], build_eval_set will raise on
    # semantic_dedup (empty corpus). We catch any downstream error after
    # asserting pull_and_classify ran.
    try:
        eval_builder.run_eval_build(
            db_path=tmp_path / "corpus.db",
            source_repo="/tmp/proj",
            projects_json=projects_json,
            bundle_dir=bundle_dir,
            weak_model="weak",
            judge_model="judge",
            api_key="dummy",
            seed=0,
            days=60,
            run_dir=tmp_path / "run",
            max_candidates=None,
            max_workers=1,
            behavioral=False,
        )
    except ValueError:
        # build_eval_set raises on empty input — that's downstream of the
        # routing decision we care about.
        pass

    assert calls["pull_and_classify"] == 1
    assert calls["extract_behavioral_evals"] == 0


# ─── 6. verifier.score_bundle accepts behavioral_action ───────────────────


def test_score_bundle_accepts_behavioral_action_type(tmp_path):
    """A behavioral_action row should flow through score_bundle without
    a type-specific code-path raising."""
    row = {
        "id": "test1",
        "split": "holdout",
        "type": "behavioral_action",
        "prompt": "Refactor this module",
        "anonymized_prompt": "Refactor this module",
        "reference": "ACTION: invoke Bash\nINPUT: {}",
        "rubric": "Score 1.0 if the candidate invokes Bash with equivalent args.",
        "accepted": True,
        "baseline_score": 0.3,
        "baseline_completion_len_tokens": 10,
    }

    # Stub the inner score_one_eval so the network is not touched. Return
    # a constant mid-band score so the FM#3 90% gate passes (1/1 == 100%).
    def fake_score_one(row_in, bundle_dir, skill_prompt, model, judge_model, api_key, rollouts, seed):
        result = RunResult(score=0.7, completion="ok", completion_len_tokens=1, error=None)
        return row_in.get("id", ""), 0.7, [result]

    with patch("daycare.verifier._score_one_eval", side_effect=fake_score_one):
        summary = verifier.score_bundle(
            eval_rows=[row],
            split="holdout",
            bundle_dir=tmp_path,
            model="weak",
            judge_model="judge",
            api_key="dummy",
            rollouts=1,
            max_workers=1,
        )

    assert isinstance(summary.holdout_score, float)
    assert 0.0 <= summary.holdout_score <= 1.0
    assert summary.holdout_score == pytest.approx(0.7)
    # Confirm the behavioral_action type made it into the by_type aggregate.
    assert "behavioral_action" in summary.by_type
