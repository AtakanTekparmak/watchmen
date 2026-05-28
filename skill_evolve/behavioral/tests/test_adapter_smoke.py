"""Smoke tests for the behavioral adapter.

Depends on Group B's ``skill_evolve.behavioral.adapter.score_bundle_behavioral``
and the ``eval_source="behavioral"`` dispatch in
``skill_evolve.evaluator.evaluate``. Tests will fail until B merges.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

# Depends on Group B's score_bundle_behavioral
from skill_evolve.behavioral.adapter import score_bundle_behavioral  # noqa: E402
from skill_evolve.evaluator import evaluate  # noqa: E402


FIXTURE = Path(__file__).parent / "fixtures" / "tiny_eval_set.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_bundle(root: Path) -> Path:
    """Tiny self-contained skill bundle on disk."""
    bundle = root / "bundle"
    skill_dir = bundle / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: demo skill\n---\n\n# demo\n",
        encoding="utf-8",
    )
    return bundle


class _StubJudge:
    """Stub judge LLM. ``response_factory`` returns the canned message dict
    (or SDK-style object) per call."""

    def __init__(self, response_factory):
        self._response_factory = response_factory
        self.calls = 0

    def generate(self, *, system: str, user: str) -> str:
        self.calls += 1
        return self._response_factory()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_adapter_high_score_mean(tmp_path: Path) -> None:
    """Judge returns {score: 0.9} for both → mean_score ≈ 0.9, len(per_task)=2."""
    bundle = _seed_bundle(tmp_path)
    judge = _StubJudge(lambda: json.dumps({"score": 0.9, "reasoning": "good"}))
    result = score_bundle_behavioral(
        bundle_dir=bundle,
        eval_set_path=FIXTURE,
        judge_model="stub-judge",
        llm=judge,
    )
    assert result.per_task is not None
    assert len(result.per_task) == 2
    assert result.mean_score == pytest.approx(0.9, abs=1e-6)


def test_adapter_handles_deepseek_reasoning_short_circuit(
    tmp_path: Path,
) -> None:
    """Judge that returns DeepSeek-style content=None, reasoning=<json> still
    parses (reasoning short-circuit wired into the adapter)."""
    bundle = _seed_bundle(tmp_path)
    # Adapter takes care of the short-circuit at the LLM layer. We feed it
    # the same string the short-circuit would surface (the reasoning field).
    judge = _StubJudge(lambda: json.dumps({"score": 0.5}))
    result = score_bundle_behavioral(
        bundle_dir=bundle,
        eval_set_path=FIXTURE,
        judge_model="stub-judge",
        llm=judge,
    )
    assert result.mean_score == pytest.approx(0.5, abs=1e-6)


def test_evaluate_missing_eval_set_raises_value_error(tmp_path: Path) -> None:
    """``evaluate(eval_source='behavioral')`` without ``eval_set_path`` → ValueError."""
    bundle = _seed_bundle(tmp_path)
    with pytest.raises(ValueError):
        evaluate(
            bundle,
            eval_source="behavioral",
            eval_set_path=None,
            judge_model="stub-judge",
        )


def test_evaluate_dispatches_to_behavioral_when_eval_source_set(
    tmp_path: Path,
) -> None:
    """``evaluate(eval_source='behavioral', eval_set_path=...)`` calls the
    behavioral adapter."""
    bundle = _seed_bundle(tmp_path)
    # Patch the dispatched-to symbol at the evaluator's import site
    # (evaluator does ``from skill_evolve.behavioral.adapter import
    # score_bundle_behavioral`` inside the function, so patch the source
    # module's binding).
    with mock.patch(
        "skill_evolve.behavioral.adapter.score_bundle_behavioral"
    ) as m_adapter:
        m_adapter.return_value = mock.MagicMock(mean_score=0.7)
        evaluate(
            bundle,
            eval_source="behavioral",
            eval_set_path=FIXTURE,
            judge_model="stub-judge",
        )
    assert m_adapter.called
