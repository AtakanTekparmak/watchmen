"""Smoke-gate guard for track_b's iteration loop.

A synthetic LLM emits a patch that adds a syntactically-broken Python
file. The iteration must:
  * Record a smoke-failure on the artifact (op_type=parse_error,
    notes prefixed with ``smoke_rejected:``).
  * NOT call ``evaluator.evaluate_artifact`` (mock and assert
    call_count == 0).

Depends on Group A's ``_run_smoke_gate`` insertion in
``track_b/openevolve_skills/iteration.py`` between the anonymize block
and the evaluator call. Group A has landed this — see plan section 7e.
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


class _BrokenScriptLLM:
    """Synthetic LLM that emits an ADD_FILE block containing a broken .py."""

    def __init__(self) -> None:
        self._parent: FolderArtifact | None = None

    def set_parent(self, artifact: FolderArtifact) -> None:
        self._parent = artifact

    def generate(self, *, system: str, user: str) -> str:
        # Intentionally broken Python syntax (``def foo(:``).
        return (
            "<<<ADD_FILE demo/scripts/broken.py>>>\n"
            "def foo(:\n    pass\n"
            "<<<END_FILE>>>\n"
        )


def _seed_database() -> ProgramDatabase:
    """Build a 1-island ProgramDatabase with a single seed Program."""
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
        metrics={"composite": 0.0},
    )
    db.add(seed, island=0)
    return db


# Make SyntheticLLM's isinstance() check happy by registering our class
# under the LLMClient Protocol. We don't need to inherit; the iteration
# loop only checks ``isinstance(llm, SyntheticLLM)`` to call set_parent.
# Provide a set_parent so the iteration calls it via the isinstance hook —
# we monkey-patch the SyntheticLLM symbol used in iteration.
def test_smoke_gate_skips_evaluator_on_broken_py(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _seed_database()
    llm = _BrokenScriptLLM()

    # Trick the iteration's ``isinstance(llm, SyntheticLLM)`` check so it
    # calls set_parent on our fake LLM.
    import skill_evolve.track_b.openevolve_skills.iteration as itmod

    monkeypatch.setattr(itmod, "SyntheticLLM", _BrokenScriptLLM)

    evaluator = mock.MagicMock()
    evaluator.anonymize_tasks = False
    evaluator.evaluate_artifact = mock.MagicMock()

    result = run_iteration(
        generation=1,
        database=db,
        evaluator=evaluator,
        llm=llm,
        prompt_sampler=PromptSampler(),
        island=0,
    )

    assert evaluator.evaluate_artifact.call_count == 0
    assert result.op_type == "parse_error"
    assert "smoke_rejected" in result.notes
