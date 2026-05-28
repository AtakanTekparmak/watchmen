"""Strict / record / relaxed validation-gate behavior for run_iteration.

Mirrors :mod:`track_b.tests.test_iteration_smoke_guard` — mock the
evaluator, drive a single iteration with a synthetic LLM that emits a
deterministic clean patch, and assert the gate's accept / reject
decision + the resulting :class:`RejectedEdit` push.

Plan section E.8: covers all three modes, the tie-rejection contract
in strict, and the per-rejection buffer push.
"""

from __future__ import annotations

from unittest import mock

import pytest

from skill_evolve.shared.rejected_buffer import RejectedBuffer
from skill_evolve.track_b.openevolve_skills.database import (
    Program,
    ProgramDatabase,
    new_program_id,
)
from skill_evolve.track_b.openevolve_skills.folder_artifact import (
    FolderArtifact,
)
from skill_evolve.track_b.openevolve_skills.iteration import (
    GateState,
    run_iteration,
)
from skill_evolve.track_b.openevolve_skills.prompt_sampler import PromptSampler


class _ReplaceLLM:
    """Synthetic LLM that emits a single EDIT_FILE replacing SKILL.md.

    The patch is syntactically clean (no scripts) so the smoke gate is a
    no-op; the validation eval is mocked so its outcome drives the gate
    decision in the test, not the actual evaluator path.
    """

    def __init__(self) -> None:
        self._parent: FolderArtifact | None = None

    def set_parent(self, artifact: FolderArtifact) -> None:
        self._parent = artifact

    def generate(self, *, system: str, user: str) -> str:
        return (
            "<<<EDIT_FILE demo/SKILL.md>>>\n"
            "---\nname: demo\ndescription: a demo (mutated)\n---\n\n# demo\n"
            "<<<END_FILE>>>\n"
        )


def _seed_database(parent_composite: float = 0.40) -> ProgramDatabase:
    db = ProgramDatabase(num_islands=1, migration_interval=5, rng_seed=0)
    seed_artifact = FolderArtifact(
        files={
            "demo/SKILL.md": "---\nname: demo\ndescription: a demo\n---\n\n# demo\n",
        }
    )
    seed = Program(
        id=new_program_id(),
        artifact=seed_artifact,
        parent_id=None,
        generation=0,
        metrics={"composite": parent_composite},
    )
    db.add(seed, island=0)
    return db


def _build_mock_evaluator(
    *,
    child_composite: float,
    validation_composite: float | None,
):
    """Returns a mocked SkillFolderEvaluator-like object."""
    evaluator = mock.MagicMock()
    evaluator.anonymize_tasks = False
    evaluator.validation_task_list = "/tmp/fake_val.json"

    eval_res = mock.MagicMock()
    eval_res.metrics = {"composite": child_composite}
    eval_res.artifacts = {}
    evaluator.evaluate_artifact = mock.MagicMock(return_value=eval_res)
    if validation_composite is not None:
        evaluator.evaluate_validation = mock.MagicMock(
            return_value={
                "validation_composite": validation_composite,
                "validation_mean_score": validation_composite,
                "validation_success_rate": 1.0,
                "validation_n": 2,
            }
        )
    else:
        evaluator.evaluate_validation = mock.MagicMock(return_value={})
    return evaluator


@pytest.fixture
def patch_synthetic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the iteration loop call ``set_parent`` on our LLM."""
    import skill_evolve.track_b.openevolve_skills.iteration as itmod

    monkeypatch.setattr(itmod, "SyntheticLLM", _ReplaceLLM)


def _run(
    *,
    evaluator,
    gate_state: GateState,
    buffer: RejectedBuffer,
    mode: str,
):
    db = _seed_database(parent_composite=0.40)
    return run_iteration(
        generation=1,
        database=db,
        evaluator=evaluator,
        llm=_ReplaceLLM(),
        prompt_sampler=PromptSampler(),
        island=0,
        validation_gate=mode,
        rejected_buffer=buffer,
        gate_state=gate_state,
    )


def test_strict_accepts_train_up_val_up(patch_synthetic: None) -> None:
    ev = _build_mock_evaluator(child_composite=0.50, validation_composite=0.60)
    gs = GateState(best_val_score_seen_so_far=0.50)
    buf = RejectedBuffer(capacity=10)
    res = _run(evaluator=ev, gate_state=gs, buffer=buf, mode="strict")
    assert res.op_type == "patch"
    assert len(buf) == 0
    # Best-seen advanced.
    assert gs.best_val_score_seen_so_far == 0.60


def test_strict_rejects_train_up_val_tie(patch_synthetic: None) -> None:
    ev = _build_mock_evaluator(child_composite=0.50, validation_composite=0.50)
    gs = GateState(best_val_score_seen_so_far=0.50)
    buf = RejectedBuffer(capacity=10)
    res = _run(evaluator=ev, gate_state=gs, buffer=buf, mode="strict")
    assert res.op_type == "gate_rejected"
    assert len(buf) == 1
    assert buf.recent(1)[0].rejection_reason == "val_tie"


def test_strict_rejects_train_up_val_down(patch_synthetic: None) -> None:
    ev = _build_mock_evaluator(child_composite=0.50, validation_composite=0.40)
    gs = GateState(best_val_score_seen_so_far=0.50)
    buf = RejectedBuffer(capacity=10)
    res = _run(evaluator=ev, gate_state=gs, buffer=buf, mode="strict")
    assert res.op_type == "gate_rejected"
    assert len(buf) == 1
    assert buf.recent(1)[0].rejection_reason == "val_not_strict_gt"


def test_strict_rejects_train_down(patch_synthetic: None) -> None:
    # Child composite < parent (0.40) → train_no_improve.
    ev = _build_mock_evaluator(child_composite=0.10, validation_composite=0.99)
    gs = GateState(best_val_score_seen_so_far=0.50)
    buf = RejectedBuffer(capacity=10)
    res = _run(evaluator=ev, gate_state=gs, buffer=buf, mode="strict")
    assert res.op_type == "gate_rejected"
    assert len(buf) == 1
    assert buf.recent(1)[0].rejection_reason == "train_no_improve"
    # Val eval should NOT have run on train regression.
    ev.evaluate_validation.assert_not_called()


def test_record_mode_train_up_val_tie_accepts(patch_synthetic: None) -> None:
    """Plan_0 Group-B behavior preserved under ``record`` mode."""
    ev = _build_mock_evaluator(child_composite=0.50, validation_composite=0.50)
    gs = GateState(best_val_score_seen_so_far=0.50)
    buf = RejectedBuffer(capacity=10)
    res = _run(evaluator=ev, gate_state=gs, buffer=buf, mode="record")
    assert res.op_type == "patch"
    assert len(buf) == 0


def test_relaxed_train_up_val_tie_accepts(patch_synthetic: None) -> None:
    ev = _build_mock_evaluator(child_composite=0.50, validation_composite=0.50)
    gs = GateState(best_val_score_seen_so_far=0.50)
    buf = RejectedBuffer(capacity=10)
    res = _run(evaluator=ev, gate_state=gs, buffer=buf, mode="relaxed")
    assert res.op_type == "patch"
    assert len(buf) == 0


def test_relaxed_train_up_val_down_rejects(patch_synthetic: None) -> None:
    ev = _build_mock_evaluator(child_composite=0.50, validation_composite=0.40)
    gs = GateState(best_val_score_seen_so_far=0.50)
    buf = RejectedBuffer(capacity=10)
    res = _run(evaluator=ev, gate_state=gs, buffer=buf, mode="relaxed")
    assert res.op_type == "gate_rejected"
    assert len(buf) == 1
    assert buf.recent(1)[0].rejection_reason == "val_not_strict_gt"


def test_one_push_per_rejection(patch_synthetic: None) -> None:
    """Each gate rejection pushes exactly ONE entry."""
    ev = _build_mock_evaluator(child_composite=0.50, validation_composite=0.50)
    gs = GateState(best_val_score_seen_so_far=0.50)
    buf = RejectedBuffer(capacity=10)
    _run(evaluator=ev, gate_state=gs, buffer=buf, mode="strict")
    _run(evaluator=ev, gate_state=gs, buffer=buf, mode="strict")
    assert len(buf) == 2


def test_invalid_gate_raises() -> None:
    db = _seed_database()
    ev = _build_mock_evaluator(child_composite=0.5, validation_composite=0.5)
    with pytest.raises(ValueError):
        run_iteration(
            generation=1,
            database=db,
            evaluator=ev,
            llm=_ReplaceLLM(),
            prompt_sampler=PromptSampler(),
            island=0,
            validation_gate="garbage",  # type: ignore[arg-type]
        )
