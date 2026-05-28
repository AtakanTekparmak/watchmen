"""Validation-holdout test for track_b.

A synthetic LLM produces a clean winner. With ``--validation-task-list``
set, the run records a ``validation_score`` on the winning artifact that
is distinct from ``train_score``.

Depends on Group B's ``--validation-task-list`` CLI flag plumbed through
``track_b/run.py``, ``controller.py``, ``iteration.py``, and the
artifact-level ``train_score`` / ``validation_score`` fields. Tests will
fail until B merges.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Depends on Group B's --validation-task-list plumbing.
# TODO: once B's controller.py exposes RunConfig.validation_task_list /
# Program.validation_score, replace the mocked controller with the
# real one.
from skill_evolve.track_b.openevolve_skills.controller import (  # noqa: E402
    RunConfig,
    run_evolution,
)
from skill_evolve.track_b.openevolve_skills.evaluator import (
    EvaluationResult,
    SkillFolderEvaluator,
)
from skill_evolve.track_b.openevolve_skills.llm_client import SyntheticLLM
from skill_evolve.track_b.openevolve_skills.prompt_sampler import PromptSampler


REPO_ROOT = Path(__file__).resolve().parents[3]
SEED = REPO_ROOT / "seed_skills"


def _write_two_task_fixture(tmp_path: Path) -> Path:
    """Write a tiny validation task list with two task IDs."""
    val_path = tmp_path / "val_2.json"
    val_path.write_text(json.dumps(["mock_task_a", "mock_task_b"]))
    return val_path


def test_validation_score_recorded_distinct_from_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run produces a winner; validation_score recorded on artifact and is
    not auto-equal to train_score (uses a fresh eval pass over val list)."""
    pytest.importorskip(
        "skill_evolve.track_b.openevolve_skills.controller",
        reason="Group B's --validation-task-list plumbing must be present",
    )
    # Depends on Group B's RunConfig.validation_task_list field.
    rc_fields = getattr(RunConfig, "__dataclass_fields__", {})
    if "validation_task_list" not in rc_fields:
        pytest.skip("RunConfig.validation_task_list not yet present (Group B)")

    if not SEED.is_dir():
        pytest.skip(f"seed_skills not found at {SEED}")

    val_path = _write_two_task_fixture(tmp_path)
    out = tmp_path / "run"

    evaluator = SkillFolderEvaluator(
        force_synthetic=True,
        verify=False,
        cascade=True,
        max_workers=1,
        validation_task_list=val_path,
    )

    # SyntheticLLM emits valid patches but the synthetic eval path returns
    # composite=0.0 for both parent and child, so score_delta is always 0.
    # The record-mode gate requires score_delta > 0 to fire the validation
    # eval; ditto strict mode requires score_delta >= 0 plus val > best.
    # Monkeypatch evaluate_artifact to produce a strictly increasing
    # composite so the gate triggers and validation_score gets recorded.
    counter = {"n": 0}
    _orig_eval = evaluator.evaluate_artifact

    def _increasing_eval(artifact, *, program_id: str = ""):
        counter["n"] += 1
        n = counter["n"]
        return EvaluationResult(
            metrics={
                "composite": 0.1 * n,
                "success_rate": 0.0,
                "tool_calls_per_success": 0.0,
                "n_tasks": 1.0,
                "verified_count": 0.0,
                "unverified_count": 1.0,
            },
            artifacts={},
        )

    monkeypatch.setattr(evaluator, "evaluate_artifact", _increasing_eval)

    # Also stub validation eval so it returns a deterministic value
    # without spinning up the real skillsbench evaluation path.
    def _val_eval(artifact, *, program_id: str = ""):
        return {
            "validation_composite": 0.42,
            "validation_mean_score": 0.42,
            "validation_success_rate": 1.0,
            "validation_n": 2,
        }

    monkeypatch.setattr(evaluator, "evaluate_validation", _val_eval)

    llm = SyntheticLLM(seed=0)
    # Test predates Group E strict gate; explicitly request record mode to
    # preserve original assertion (eval val whenever train improved).
    config = RunConfig(
        num_generations=2,
        num_islands=2,
        migration_interval=5,
        rng_seed=0,
        validation_task_list=val_path,
        validation_gate="record",
    )
    result = run_evolution(
        seed_path=SEED,
        out_dir=out,
        config=config,
        evaluator=evaluator,
        llm=llm,
        prompt_sampler=PromptSampler(),
    )

    best = result.best
    assert best is not None
    # validation_score must be recorded distinct from train metrics.
    val_score = best.metrics.get("validation_score")
    train_score = best.metrics.get("composite") or best.metrics.get("train_score")
    assert val_score is not None
    # They may coincidentally be the same value under synthetic eval, but
    # the recording channel must be distinct.
    assert "validation_score" in best.metrics
    assert train_score is not None
