"""Group F integration test — iteration-level L_t clip behavior.

A synthetic LLM emits a 5-op sentinel patch; the iteration runs with
``--edit-budget constant:2``; the apply path must see exactly 2 ops,
the artifact must record ``edit_budget.parsed_op_count = 5`` and
``edit_budget.applied_op_count = 2`` (with ``clipped = True``), and
the evaluator MUST still be called once (clipping is NOT a smoke-gate
short-circuit — the surviving ops produce a valid candidate that should
be evaluated).

Mirrors the test-shape used by ``test_iteration_smoke_guard.py``.
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


_FIVE_OP_PATCH = (
    "<<<ADD_FILE demo/scripts/a.py>>>\n"
    "# op 0\n"
    "<<<END_FILE>>>\n"
    "<<<ADD_FILE demo/scripts/b.py>>>\n"
    "# op 1\n"
    "<<<END_FILE>>>\n"
    "<<<ADD_FILE demo/scripts/c.py>>>\n"
    "# op 2\n"
    "<<<END_FILE>>>\n"
    "<<<ADD_FILE demo/scripts/d.py>>>\n"
    "# op 3\n"
    "<<<END_FILE>>>\n"
    "<<<ADD_FILE demo/scripts/e.py>>>\n"
    "# op 4\n"
    "<<<END_FILE>>>\n"
)


class _FiveOpLLM:
    """Synthetic LLM that emits a fixed 5-op ADD_FILE patch every call."""

    def __init__(self) -> None:
        self._parent: FolderArtifact | None = None

    def set_parent(self, artifact: FolderArtifact) -> None:
        self._parent = artifact

    def generate(self, *, system: str, user: str) -> str:
        return _FIVE_OP_PATCH


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


def _make_evaluator() -> mock.MagicMock:
    evaluator = mock.MagicMock()
    evaluator.anonymize_tasks = False
    evaluator.validation_task_list = None
    # ``evaluate_artifact`` returns an object with ``.metrics`` and
    # ``.artifacts`` attrs; mimic with a SimpleNamespace.
    from types import SimpleNamespace

    evaluator.evaluate_artifact.return_value = SimpleNamespace(
        metrics={"composite": 0.5},
        artifacts={},
    )
    return evaluator


def test_edit_budget_clips_five_ops_to_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _seed_database()
    llm = _FiveOpLLM()

    import skill_evolve.track_b.openevolve_skills.iteration as itmod

    monkeypatch.setattr(itmod, "SyntheticLLM", _FiveOpLLM)
    # The smoke gate touches the filesystem; for this test we just stub it
    # so the focus stays on clipping behavior.
    monkeypatch.setattr(itmod, "_run_smoke_gate", lambda artifact: None)

    evaluator = _make_evaluator()

    result = run_iteration(
        generation=3,
        database=db,
        evaluator=evaluator,
        llm=llm,
        prompt_sampler=PromptSampler(),
        island=0,
        edit_budget="constant:2",
        max_iters=12,
    )

    # Clipping does NOT short-circuit the evaluator.
    assert evaluator.evaluate_artifact.call_count == 1

    # The artifact records the clip.
    assert result.edit_budget is not None
    assert result.edit_budget["parsed_op_count"] == 5
    assert result.edit_budget["applied_op_count"] == 2
    assert result.edit_budget["lt"] == 2
    assert result.edit_budget["clipped"] is True

    # And the actual applied artifact has exactly the first two new files
    # under demo/scripts (proving clip_ops took the proposer-emit order).
    child_id = result.child_id
    assert child_id is not None
    child_program = db.programs[child_id]
    new_files = set(child_program.artifact.files.keys()) - {
        "demo/SKILL.md",
    }
    assert new_files == {"demo/scripts/a.py", "demo/scripts/b.py"}


def test_edit_budget_disabled_when_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Back-compat: ``edit_budget=None`` keeps the legacy ``mutate`` path."""
    db = _seed_database()
    llm = _FiveOpLLM()

    import skill_evolve.track_b.openevolve_skills.iteration as itmod

    monkeypatch.setattr(itmod, "SyntheticLLM", _FiveOpLLM)
    monkeypatch.setattr(itmod, "_run_smoke_gate", lambda artifact: None)

    evaluator = _make_evaluator()

    result = run_iteration(
        generation=1,
        database=db,
        evaluator=evaluator,
        llm=llm,
        prompt_sampler=PromptSampler(),
        island=0,
        # edit_budget omitted (None) — should NOT populate the meta block
        # and should NOT clip.
    )

    assert result.edit_budget is None
    # All 5 ops should have been applied via the legacy ``mutate`` path.
    child_program = db.programs[result.child_id]
    new_files = set(child_program.artifact.files.keys()) - {
        "demo/SKILL.md",
    }
    assert len(new_files) == 5


def test_edit_budget_under_budget_records_not_clipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When parsed <= L_t, clipped=False and counts match."""
    db = _seed_database()
    llm = _FiveOpLLM()

    import skill_evolve.track_b.openevolve_skills.iteration as itmod

    monkeypatch.setattr(itmod, "SyntheticLLM", _FiveOpLLM)
    monkeypatch.setattr(itmod, "_run_smoke_gate", lambda artifact: None)

    evaluator = _make_evaluator()

    result = run_iteration(
        generation=1,
        database=db,
        evaluator=evaluator,
        llm=llm,
        prompt_sampler=PromptSampler(),
        island=0,
        edit_budget="constant:10",  # well above the 5-op patch
        max_iters=12,
    )

    assert result.edit_budget is not None
    assert result.edit_budget["parsed_op_count"] == 5
    assert result.edit_budget["applied_op_count"] == 5
    assert result.edit_budget["lt"] == 10
    assert result.edit_budget["clipped"] is False
