"""Single-mode reflection in track_b's iteration loop (Group H).

Back-compat guard: ``--reflection-mode single`` (or the function-default
when no flag is threaded) MUST still produce exactly ONE proposer call
and behave identically to the plan_0 Group D smoke-guard expectations.
"""

from __future__ import annotations

from unittest import mock

import pytest

from skill_evolve.track_b.openevolve_skills.database import (
    Program,
    ProgramDatabase,
    new_program_id,
)
from skill_evolve.track_b.openevolve_skills.folder_artifact import (
    FolderArtifact,
)
from skill_evolve.track_b.openevolve_skills.iteration import run_iteration
from skill_evolve.track_b.openevolve_skills.prompt_sampler import PromptSampler


class _CountingLLM:
    """Records every call. Returns a fixed valid sentinel patch."""

    def __init__(self) -> None:
        self._parent: FolderArtifact | None = None
        self.calls: list[dict] = []

    def set_parent(self, artifact: FolderArtifact) -> None:
        self._parent = artifact

    def generate(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        # Trivial single-edit patch — keeps the smoke gate happy.
        return (
            "<<<EDIT_FILE demo/SKILL.md>>>\n"
            "---\nname: demo\ndescription: edited\n---\n\n# demo (single)\n"
            "<<<END_FILE>>>\n"
        )


def _seed_database() -> ProgramDatabase:
    db = ProgramDatabase(num_islands=1, migration_interval=5, rng_seed=0)
    seed_artifact = FolderArtifact(
        files={
            "demo/SKILL.md": ("---\nname: demo\ndescription: a demo\n---\n\n# demo\n"),
        }
    )
    seed = Program(
        id=new_program_id(),
        artifact=seed_artifact,
        parent_id=None,
        generation=0,
        metrics={"composite": 0.0},
    )
    db.add(seed, island=0)
    return db


def test_single_mode_fires_exactly_one_proposer_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _seed_database()
    llm = _CountingLLM()

    import skill_evolve.track_b.openevolve_skills.iteration as itmod

    monkeypatch.setattr(itmod, "SyntheticLLM", _CountingLLM)
    monkeypatch.setattr(itmod, "_run_smoke_gate", lambda artifact: None)

    evaluator = mock.MagicMock()
    evaluator.anonymize_tasks = False
    evaluator.validation_task_list = None
    eval_res = mock.MagicMock()
    eval_res.metrics = {"composite": 0.6}
    eval_res.artifacts = {}
    evaluator.evaluate_artifact = mock.MagicMock(return_value=eval_res)

    result = run_iteration(
        generation=1,
        database=db,
        evaluator=evaluator,
        llm=llm,
        prompt_sampler=PromptSampler(),
        island=0,
        reflection_mode="single",
    )

    assert len(llm.calls) == 1
    # Single-mode does NOT use reflection prompts.
    system = llm.calls[0]["system"]
    assert "FAILURE TRAJECTORIES" not in system
    assert "SUCCESS TRAJECTORIES" not in system
    # No reflection metadata in single mode.
    assert result.reflection is None


def test_default_reflection_mode_is_single(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_iteration's function-default for reflection_mode is ``single``
    so existing callers (e.g. Group D's smoke-guard test) still see the
    one-proposer-call back-compat path.
    """
    db = _seed_database()
    llm = _CountingLLM()

    import skill_evolve.track_b.openevolve_skills.iteration as itmod

    monkeypatch.setattr(itmod, "SyntheticLLM", _CountingLLM)
    monkeypatch.setattr(itmod, "_run_smoke_gate", lambda artifact: None)

    evaluator = mock.MagicMock()
    evaluator.anonymize_tasks = False
    evaluator.validation_task_list = None
    eval_res = mock.MagicMock()
    eval_res.metrics = {"composite": 0.6}
    eval_res.artifacts = {}
    evaluator.evaluate_artifact = mock.MagicMock(return_value=eval_res)

    # No reflection_mode kwarg passed -> function-default applies.
    result = run_iteration(
        generation=1,
        database=db,
        evaluator=evaluator,
        llm=llm,
        prompt_sampler=PromptSampler(),
        island=0,
    )
    assert len(llm.calls) == 1
    assert result.reflection is None
