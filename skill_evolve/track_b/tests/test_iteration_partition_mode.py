"""Partition-mode reflection in track_b's iteration loop (Group H).

Drives ``run_iteration`` with ``reflection_mode="partition"`` and a
synthetic LLM that emits DIFFERENT sentinel patches depending on whether
the system prompt looks like a failure-reflection prompt or a
success-reflection prompt. Asserts:

  * TWO proposer calls fire (one failure-side, one success-side).
  * The merged op list contains ops from BOTH sides (one collision
    resolved failure-priority, one disjoint pair survives).
  * The iteration result's ``reflection`` field captures the per-side
    counts + collisions + merged_count.
  * Per-iteration reflection JSON files are persisted to the
    ``reflection_artifact_dir``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List
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


class _PartitionAwareLLM:
    """Records every call. Returns failure or success patches based on prompt."""

    def __init__(self) -> None:
        self._parent: FolderArtifact | None = None
        self.calls: List[dict] = []

    def set_parent(self, artifact: FolderArtifact) -> None:
        self._parent = artifact

    def generate(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        if "FAILURE TRAJECTORIES" in system or "failure-analysis" in system:
            # Failure side: edit demo SKILL.md + add a new failure-side
            # script. The EDIT_FILE will collide with the success side's
            # EDIT_FILE on the same path and win by failure-priority.
            return (
                "<<<EDIT_FILE demo/SKILL.md>>>\n"
                "---\nname: demo\ndescription: failure-side edit\n---\n\n"
                "# demo (FAILURE side)\n"
                "<<<END_FILE>>>\n"
                "<<<ADD_FILE demo/scripts/fail_helper.sh>>>\n"
                "#!/usr/bin/env bash\nset -euo pipefail\necho fail\n"
                "<<<END_FILE>>>\n"
            )
        # Success side: same EDIT_FILE path (collides) + a disjoint
        # ADD_FILE on a different path.
        return (
            "<<<EDIT_FILE demo/SKILL.md>>>\n"
            "---\nname: demo\ndescription: success-side edit\n---\n\n"
            "# demo (SUCCESS side)\n"
            "<<<END_FILE>>>\n"
            "<<<ADD_FILE demo/scripts/success_helper.sh>>>\n"
            "#!/usr/bin/env bash\nset -euo pipefail\necho success\n"
            "<<<END_FILE>>>\n"
        )


def _seed_database_with_per_task() -> ProgramDatabase:
    """1-island database whose parent carries mixed pass/fail per_task."""
    db = ProgramDatabase(num_islands=1, migration_interval=5, rng_seed=0)
    seed_artifact = FolderArtifact(
        files={
            "demo/SKILL.md": ("---\nname: demo\ndescription: a demo\n---\n\n# demo\n"),
        }
    )
    per_task = [
        {
            "task_id": "t_fail_1",
            "score": 0.0,
            "verifier_status": "fail",
            "last_msg": "boom",
        },
        {
            "task_id": "t_fail_2",
            "score": 0.1,
            "verifier_status": "fail",
            "last_msg": "boom",
        },
        {
            "task_id": "t_pass_1",
            "score": 0.8,
            "verifier_status": "pass",
            "last_msg": "ok",
        },
        {
            "task_id": "t_pass_2",
            "score": 1.0,
            "verifier_status": "pass",
            "last_msg": "ok",
        },
    ]
    seed = Program(
        id=new_program_id(),
        artifact=seed_artifact,
        parent_id=None,
        generation=0,
        metrics={"composite": 0.5},
        eval_artifacts={"per_task": json.dumps(per_task)},
    )
    db.add(seed, island=0)
    return db


def test_partition_mode_fires_two_proposer_calls_and_merges(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db = _seed_database_with_per_task()
    llm = _PartitionAwareLLM()

    import skill_evolve.track_b.openevolve_skills.iteration as itmod

    monkeypatch.setattr(itmod, "SyntheticLLM", _PartitionAwareLLM)
    # Skip the smoke gate so we can focus on the partition behavior; the
    # gate is exercised in test_iteration_smoke_guard.py.
    monkeypatch.setattr(itmod, "_run_smoke_gate", lambda artifact: None)

    evaluator = mock.MagicMock()
    evaluator.anonymize_tasks = False
    evaluator.validation_task_list = None
    eval_res = mock.MagicMock()
    eval_res.metrics = {"composite": 0.6}
    eval_res.artifacts = {}
    evaluator.evaluate_artifact = mock.MagicMock(return_value=eval_res)

    artifact_dir = tmp_path / "reflection"

    result = run_iteration(
        generation=1,
        database=db,
        evaluator=evaluator,
        llm=llm,
        prompt_sampler=PromptSampler(),
        island=0,
        reflection_mode="partition",
        reflection_success_threshold=0.5,
        reflection_artifact_dir=artifact_dir,
        edit_budget="constant:10",  # don't clip; we want the full merged list
        max_iters=12,
    )

    # Two proposer calls fired (one failure-side, one success-side).
    assert len(llm.calls) == 2
    systems = [c["system"] for c in llm.calls]
    # One contains the failure trajectories header, the other the success
    # trajectories header.
    assert any("FAILURE TRAJECTORIES" in s for s in systems)
    assert any("SUCCESS TRAJECTORIES" in s for s in systems)

    # Reflection metadata captured.
    assert result.reflection is not None
    assert result.reflection["mode"] == "partition"
    assert result.reflection["failure_count"] == 2  # 2 ops per side
    assert result.reflection["success_count"] == 2
    # One collision (same EDIT_FILE path on both sides) -> failure wins.
    assert len(result.reflection["collisions"]) == 1
    # Merged: failure[0..1] (edit + add) + success[1] (the non-colliding
    # add) = 3 ops total.
    assert result.reflection["merged_count"] == 3

    # Reflection JSON artifacts written to disk.
    assert (artifact_dir / "iter_0001_reflection_failure.json").exists()
    assert (artifact_dir / "iter_0001_reflection_success.json").exists()
    assert (artifact_dir / "iter_0001_reflection_merge.json").exists()
    # Merge file carries the collision list.
    merge_blob = json.loads(
        (artifact_dir / "iter_0001_reflection_merge.json").read_text()
    )
    assert len(merge_blob["collisions"]) == 1


def test_partition_mode_empty_per_task_falls_back_to_single(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When the parent has NO per_task data, partition mode falls through
    to a single proposer call per §7l empty-side handling.
    """
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
        metrics={"composite": 0.5},
        # No eval_artifacts -> no per_task data -> both sides empty.
    )
    db.add(seed, island=0)

    llm = _PartitionAwareLLM()
    import skill_evolve.track_b.openevolve_skills.iteration as itmod

    monkeypatch.setattr(itmod, "SyntheticLLM", _PartitionAwareLLM)
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
        reflection_mode="partition",
        reflection_artifact_dir=tmp_path / "reflection",
        edit_budget="constant:10",
        max_iters=12,
    )

    # Both sides empty -> single fallback -> exactly ONE proposer call
    # (the legacy SENTINEL_PROPOSER prompt, NOT a reflection prompt).
    assert len(llm.calls) == 1
    system = llm.calls[0]["system"]
    assert "FAILURE TRAJECTORIES" not in system
    assert "SUCCESS TRAJECTORIES" not in system
    # ``reflection`` field stays None on the result (we fell through).
    assert result.reflection is None
