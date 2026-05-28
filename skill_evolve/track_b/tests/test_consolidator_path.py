"""Group G — consolidator path fires every K iters (plan §7l).

Coverage:
* Every K iters the consolidator fires (K=2: iters 2, 4 fire; 1, 3 don't).
* meta_skill.md is appended after a successful consolidator iter.
* Consolidator output is fence-only mutation (out-of-fence bytes preserved).
* Failed consolidator output (e.g. parse error) doesn't crash the run AND
  the meta-skill is not appended.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from skill_evolve.shared.meta_skill import MetaSkill
from skill_evolve.shared.slow_update import (
    extract_slow_update_field,
    has_slow_update_field,
    inject_slow_update_field,
)
from skill_evolve.track_b.openevolve_skills.database import (
    Program,
    ProgramDatabase,
    new_program_id,
)
from skill_evolve.track_b.openevolve_skills.folder_artifact import (
    FolderArtifact,
)
from skill_evolve.shared.rejected_buffer import RejectedBuffer
from skill_evolve.track_b.openevolve_skills.iteration import (
    GateState,
    _run_consolidator_iteration,
    run_iteration,
)
from skill_evolve.track_b.openevolve_skills.prompt_sampler import PromptSampler


# ─── Synthetic LLMs ──────────────────────────────────────────────────────


class _ConsolidatorLLM:
    """Emits a single EDIT_FILE SKILL.md with new in-fence content."""

    def __init__(self, in_fence_payload: str = "NEW LESSONS") -> None:
        self._parent = None
        self._in_fence = in_fence_payload
        self.generate_calls: list[tuple[str, str]] = []

    def set_parent(self, artifact: FolderArtifact) -> None:
        self._parent = artifact

    def generate(self, *, system: str, user: str) -> str:
        self.generate_calls.append((system, user))
        # Build a SKILL.md that contains the fence with new content.
        fenced = inject_slow_update_field(
            "# Best Skill\n\nProse.\n", content=self._in_fence
        )
        return f"<<<EDIT_FILE SKILL.md>>>\n{fenced}<<<END_FILE>>>\n"


class _BrokenConsolidatorLLM:
    """Emits unparseable garbage."""

    def __init__(self) -> None:
        self._parent = None

    def set_parent(self, artifact: FolderArtifact) -> None:
        self._parent = artifact

    def generate(self, *, system: str, user: str) -> str:
        # Two EDIT_FILE blocks — violates the "exactly one" contract.
        return (
            "<<<EDIT_FILE SKILL.md>>>\n# A\n<<<END_FILE>>>\n"
            "<<<EDIT_FILE SKILL.md>>>\n# B\n<<<END_FILE>>>\n"
        )


def _seed_database_with_fence() -> ProgramDatabase:
    """1-island db whose seed SKILL.md already has a fence."""
    db = ProgramDatabase(num_islands=1, migration_interval=5, rng_seed=0)
    fenced = inject_slow_update_field("# Best Skill\n\nProse.\n", content="ORIGINAL")
    seed_artifact = FolderArtifact(files={"SKILL.md": fenced})
    seed = Program(
        id=new_program_id(),
        artifact=seed_artifact,
        parent_id=None,
        generation=0,
        metrics={"composite": 0.1},
    )
    db.add(seed, island=0)
    return db


# ─── unit-level: helper directly ─────────────────────────────────────────


def test_consolidator_helper_accepts_well_formed_output(tmp_path: Path) -> None:
    db = _seed_database_with_fence()
    llm = _ConsolidatorLLM(in_fence_payload="NEW LESSONS")
    meta_skill = MetaSkill(max_entries=20)
    meta_path = tmp_path / "meta_skill.md"

    meta = _run_consolidator_iteration(
        generation=4,
        database=db,
        llm=llm,
        rejected_buffer=None,
        meta_skill=meta_skill,
        meta_skill_path=meta_path,
        persistent_failure_streaks={},
        persistent_failure_window=3,
        consolidator_model=None,
    )
    assert meta["fired"] is True
    assert meta["accepted"] is True
    assert meta["reason"] == "accepted"
    assert meta["meta_skill_appended"] is True
    assert len(llm.generate_calls) == 1

    # SKILL.md fence-content updated; out-of-fence preserved.
    skill_md = db.best().artifact.files["SKILL.md"]
    assert has_slow_update_field(skill_md)
    assert extract_slow_update_field(skill_md) == "NEW LESSONS"
    assert skill_md.startswith("# Best Skill")

    # Meta-skill persisted to disk.
    assert meta_path.exists()
    assert "Iteration 4" in meta_path.read_text(encoding="utf-8")
    assert len(meta_skill) == 1


def test_consolidator_helper_rejects_invalid_op_shape(tmp_path: Path) -> None:
    db = _seed_database_with_fence()
    llm = _BrokenConsolidatorLLM()
    meta_skill = MetaSkill(max_entries=20)
    meta_path = tmp_path / "meta_skill.md"
    skill_md_before = db.best().artifact.files["SKILL.md"]

    # Two EDIT_FILEs on SKILL.md — second triggers parse_error
    # (sentinel parser has structural defenses) OR
    # consolidator_invalid_op_shape (too many EDIT_FILE ops).
    meta = _run_consolidator_iteration(
        generation=4,
        database=db,
        llm=llm,
        rejected_buffer=None,
        meta_skill=meta_skill,
        meta_skill_path=meta_path,
        persistent_failure_streaks={},
        persistent_failure_window=3,
        consolidator_model=None,
    )
    assert meta["fired"] is True
    assert meta["accepted"] is False
    assert meta["meta_skill_appended"] is False
    assert len(meta_skill) == 0
    # SKILL.md unchanged.
    assert db.best().artifact.files["SKILL.md"] == skill_md_before


# ─── run_iteration: K-gated firing cadence ─────────────────────────────


@pytest.mark.parametrize(
    ("k", "firing_iters", "non_firing_iters"),
    [(2, (2, 4), (1, 3)), (3, (3, 6), (1, 2, 4, 5))],
)
def test_consolidator_fires_every_k_iters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    k: int,
    firing_iters: tuple[int, ...],
    non_firing_iters: tuple[int, ...],
) -> None:
    """Verify the K-cadence: consolidator only runs on iters where
    iter_n > 0 and iter_n % K == 0.
    """
    import skill_evolve.track_b.openevolve_skills.iteration as itmod

    # Patch _run_consolidator_iteration to record fire calls without
    # actually exercising the LLM-call path (that's covered by the
    # unit-level test above).
    fire_calls: list[int] = []

    def _fake_consolidator(**kwargs):
        fire_calls.append(kwargs["generation"])
        return {
            "fired": True,
            "accepted": True,
            "reason": "accepted",
            "meta_skill_appended": True,
        }

    monkeypatch.setattr(itmod, "_run_consolidator_iteration", _fake_consolidator)

    # Use a synthetic LLM that emits a tiny, valid EDIT_FILE patch — the
    # fast-edit path will fail smoke (or whatever) but consolidator
    # firing is independent of fast-edit outcome since it runs at the
    # TOP of the loop body.
    db = _seed_database_with_fence()

    class _SimpleLLM:
        def __init__(self) -> None:
            self._parent = None

        def set_parent(self, artifact: FolderArtifact) -> None:
            self._parent = artifact

        def generate(self, *, system: str, user: str) -> str:
            # Emit a valid sentinel block that passes smoke (no scripts).
            return (
                "<<<EDIT_FILE SKILL.md>>>\n"
                "# Best Skill\n\nProse updated by fast proposer.\n"
                "<<<END_FILE>>>\n"
            )

    monkeypatch.setattr(itmod, "SyntheticLLM", _SimpleLLM)

    evaluator = mock.MagicMock()
    evaluator.anonymize_tasks = False
    evaluator.validation_task_list = None
    evaluator.task_id_map = mock.MagicMock(return_value={})
    fake_result = mock.MagicMock()
    fake_result.metrics = {"composite": 0.5}
    fake_result.artifacts = {}
    evaluator.evaluate_artifact = mock.MagicMock(return_value=fake_result)

    meta_skill = MetaSkill(max_entries=20)
    meta_path = tmp_path / "meta_skill.md"
    streaks: dict = {}

    for gen in range(1, max(firing_iters) + 1):
        run_iteration(
            generation=gen,
            database=db,
            evaluator=evaluator,
            llm=_SimpleLLM(),
            prompt_sampler=PromptSampler(),
            island=0,
            slow_update_every=k,
            meta_skill=meta_skill,
            meta_skill_path=meta_path,
            persistent_failure_streaks=streaks,
            persistent_failure_window=3,
        )

    for iter_n in firing_iters:
        assert iter_n in fire_calls, (
            f"K={k}: iter {iter_n} should have fired consolidator"
        )
    for iter_n in non_firing_iters:
        assert iter_n not in fire_calls, (
            f"K={k}: iter {iter_n} should NOT have fired consolidator"
        )


def test_consolidator_disabled_when_K_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """slow_update_every=0 disables consolidator firing entirely."""
    import skill_evolve.track_b.openevolve_skills.iteration as itmod

    fire_calls: list[int] = []

    def _fake_consolidator(**kwargs):
        fire_calls.append(kwargs["generation"])
        return {
            "fired": True,
            "accepted": True,
            "reason": "accepted",
            "meta_skill_appended": False,
        }

    monkeypatch.setattr(itmod, "_run_consolidator_iteration", _fake_consolidator)

    db = _seed_database_with_fence()

    class _SimpleLLM:
        def __init__(self) -> None:
            self._parent = None

        def set_parent(self, artifact: FolderArtifact) -> None:
            self._parent = artifact

        def generate(self, *, system: str, user: str) -> str:
            return "<<<EDIT_FILE SKILL.md>>>\n# Best Skill\n\nupdated\n<<<END_FILE>>>\n"

    monkeypatch.setattr(itmod, "SyntheticLLM", _SimpleLLM)

    evaluator = mock.MagicMock()
    evaluator.anonymize_tasks = False
    evaluator.validation_task_list = None
    evaluator.task_id_map = mock.MagicMock(return_value={})
    fake_result = mock.MagicMock()
    fake_result.metrics = {"composite": 0.5}
    fake_result.artifacts = {}
    evaluator.evaluate_artifact = mock.MagicMock(return_value=fake_result)

    meta_skill = MetaSkill(max_entries=20)

    for gen in range(1, 6):
        run_iteration(
            generation=gen,
            database=db,
            evaluator=evaluator,
            llm=_SimpleLLM(),
            prompt_sampler=PromptSampler(),
            island=0,
            slow_update_every=0,  # disabled
            meta_skill=meta_skill,
            meta_skill_path=tmp_path / "meta.md",
            persistent_failure_streaks={},
            persistent_failure_window=3,
        )

    assert fire_calls == []


# ─── Strict validation gate: consolidator rejection (fix-round, plan §G.6) ─


class _StubGateEvaluator:
    """Minimal evaluator stub for exercising the consolidator gate.

    ``train_composite`` controls what ``evaluate_artifact`` reports for
    the consolidator candidate; ``validation_task_list`` being non-None
    triggers the val-side branch (set to None to keep the test focused
    on train-side regression).
    """

    def __init__(
        self,
        *,
        train_composite: float,
        validation_task_list: object = None,
        val_composite: float | None = None,
    ) -> None:
        self.train_composite = train_composite
        self.validation_task_list = validation_task_list
        self.val_composite = val_composite
        self.evaluate_calls: list[FolderArtifact] = []
        self.validate_calls: list[FolderArtifact] = []

    def evaluate_artifact(self, artifact: FolderArtifact, *, program_id: str = ""):
        self.evaluate_calls.append(artifact)
        result = mock.MagicMock()
        result.metrics = {"composite": self.train_composite}
        result.artifacts = {}
        return result

    def evaluate_validation(
        self, artifact: FolderArtifact, *, program_id: str = ""
    ) -> dict:
        self.validate_calls.append(artifact)
        if self.val_composite is None:
            return {}
        return {
            "validation_composite": float(self.val_composite),
            "validation_n": 1,
        }


def test_consolidator_rejected_on_train_regression(tmp_path: Path) -> None:
    """When the consolidator's output regresses train composite under
    strict mode: the candidate is rejected, ``best.artifact.files`` is
    untouched, meta_skill is NOT appended, and the RejectedBuffer gains
    one entry whose reason starts with ``consolidator_``.
    """
    db = _seed_database_with_fence()
    # Parent's composite is 0.1 (set in _seed_database_with_fence). The
    # stub evaluator returns 0.05 < 0.1 for the consolidator candidate
    # so the strict gate triggers the train-regression branch.
    evaluator = _StubGateEvaluator(train_composite=0.05)
    llm = _ConsolidatorLLM(in_fence_payload="NEW LESSONS")
    meta_skill = MetaSkill(max_entries=20)
    meta_path = tmp_path / "meta_skill.md"
    rejected_buffer = RejectedBuffer(capacity=10)
    gate_state = GateState()

    skill_before = db.best().artifact.files["SKILL.md"]
    files_before = dict(db.best().artifact.files)

    meta = _run_consolidator_iteration(
        generation=4,
        database=db,
        llm=llm,
        rejected_buffer=rejected_buffer,
        meta_skill=meta_skill,
        meta_skill_path=meta_path,
        persistent_failure_streaks={},
        persistent_failure_window=3,
        consolidator_model=None,
        evaluator=evaluator,
        validation_gate_mode="strict",
        gate_state=gate_state,
        parent_train_score=0.1,
        rejected_buffer_path=None,
    )

    assert meta["fired"] is True
    assert meta["accepted"] is False
    assert meta["reason"] == "consolidator_train_regression"
    assert meta["meta_skill_appended"] is False

    # Evaluator was called (the gate ran an evaluation pass) but the
    # val side was NOT reached because train regressed first.
    assert len(evaluator.evaluate_calls) == 1
    assert len(evaluator.validate_calls) == 0

    # Bundle was NOT mutated.
    assert db.best().artifact.files == files_before
    assert db.best().artifact.files["SKILL.md"] == skill_before

    # MetaSkill log was NOT appended; file not written.
    assert len(meta_skill) == 0
    assert not meta_path.exists()

    # RejectedBuffer gained exactly one entry whose reason is
    # consolidator-prefixed.
    assert len(rejected_buffer) == 1
    items = rejected_buffer.recent(len(rejected_buffer))
    assert items[0].rejection_reason.startswith("consolidator_")
    assert items[0].rejection_reason == "consolidator_train_regression"
    assert items[0].iteration == 4


def test_consolidator_rejected_on_val_not_strict_gt(tmp_path: Path) -> None:
    """When train passes but val ties best-seen under strict mode: the
    consolidator candidate is rejected with ``consolidator_val_tie``
    (strict requires strict > on val).
    """
    db = _seed_database_with_fence()
    evaluator = _StubGateEvaluator(
        train_composite=0.15,  # > parent 0.1 — train passes
        validation_task_list=tmp_path / "fake_val.json",
        val_composite=0.5,
    )
    llm = _ConsolidatorLLM(in_fence_payload="NEW LESSONS")
    meta_skill = MetaSkill(max_entries=20)
    meta_path = tmp_path / "meta_skill.md"
    rejected_buffer = RejectedBuffer(capacity=10)
    # Seed best_val_score_seen_so_far at 0.5 so the consolidator's
    # val=0.5 triggers a tie → val_tie under strict.
    gate_state = GateState(best_val_score_seen_so_far=0.5)

    files_before = dict(db.best().artifact.files)

    meta = _run_consolidator_iteration(
        generation=4,
        database=db,
        llm=llm,
        rejected_buffer=rejected_buffer,
        meta_skill=meta_skill,
        meta_skill_path=meta_path,
        persistent_failure_streaks={},
        persistent_failure_window=3,
        consolidator_model=None,
        evaluator=evaluator,
        validation_gate_mode="strict",
        gate_state=gate_state,
        parent_train_score=0.1,
        rejected_buffer_path=None,
    )

    assert meta["accepted"] is False
    assert meta["reason"] == "consolidator_val_tie"
    assert meta["meta_skill_appended"] is False

    # Both train and val sides were evaluated.
    assert len(evaluator.evaluate_calls) == 1
    assert len(evaluator.validate_calls) == 1

    # Bundle NOT mutated.
    assert db.best().artifact.files == files_before
    assert len(meta_skill) == 0
    assert not meta_path.exists()

    # Rejection recorded with consolidator-prefixed reason.
    assert len(rejected_buffer) == 1
    assert rejected_buffer.recent(1)[0].rejection_reason == "consolidator_val_tie"


def test_consolidator_accepted_under_strict_gate(tmp_path: Path) -> None:
    """Strict gate passes when train >= parent AND val > best_val_seen.
    Confirms the gate path doesn't over-reject — accepted candidates
    still mutate the bundle and append meta_skill.
    """
    db = _seed_database_with_fence()
    evaluator = _StubGateEvaluator(
        train_composite=0.2,  # > parent 0.1 — train passes
        validation_task_list=tmp_path / "fake_val.json",
        val_composite=0.7,  # > best_val 0.5
    )
    llm = _ConsolidatorLLM(in_fence_payload="NEW LESSONS")
    meta_skill = MetaSkill(max_entries=20)
    meta_path = tmp_path / "meta_skill.md"
    rejected_buffer = RejectedBuffer(capacity=10)
    gate_state = GateState(best_val_score_seen_so_far=0.5)

    meta = _run_consolidator_iteration(
        generation=4,
        database=db,
        llm=llm,
        rejected_buffer=rejected_buffer,
        meta_skill=meta_skill,
        meta_skill_path=meta_path,
        persistent_failure_streaks={},
        persistent_failure_window=3,
        consolidator_model=None,
        evaluator=evaluator,
        validation_gate_mode="strict",
        gate_state=gate_state,
        parent_train_score=0.1,
        rejected_buffer_path=None,
    )

    assert meta["accepted"] is True
    assert meta["reason"] == "accepted"
    assert meta["meta_skill_appended"] is True

    # Bundle mutated; meta_skill written.
    assert (
        extract_slow_update_field(db.best().artifact.files["SKILL.md"]) == "NEW LESSONS"
    )
    assert len(meta_skill) == 1
    assert meta_path.exists()

    # Strict gate advanced best_val_score_seen_so_far.
    assert gate_state.best_val_score_seen_so_far == 0.7
    # No rejection pushed.
    assert len(rejected_buffer) == 0
