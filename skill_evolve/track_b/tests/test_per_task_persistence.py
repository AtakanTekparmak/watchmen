"""Per-task detail persistence through the OpenEvolve translation layer.

After Fix 2, :meth:`SkillFolderEvaluator._translate` must emit a
``per_task`` entry in ``EvaluationResult.artifacts`` whose JSON payload
carries the fields needed to answer "which task did evolution crack?"
(``task_id``, ``verified``, ``success``, ``elapsed_s``, ``tool_calls``,
``skills_invoked``, ``verifier_status``). These tests lock in:

* ``_translate`` populates the per-task list under the ``per_task`` key.
* The list has one entry per :class:`TaskOutcome`, with the three-state
  ``verified`` preserved (``True`` / ``False`` / ``None``).
* The payload round-trips through ``json.dumps`` / ``json.loads``.
* Aggregate ``metrics`` / pre-existing ``artifacts`` keys are untouched.
"""

from __future__ import annotations

import json

from skill_evolve.evaluator import EvalResult, TaskOutcome
from skill_evolve.track_b.openevolve_skills.evaluator import (
    SkillFolderEvaluator,
)
from skill_evolve.track_b.openevolve_skills.folder_artifact import (
    FolderArtifact,
)


def _make_artifact() -> FolderArtifact:
    """Minimal two-skill artifact that passes ``validate()``."""
    return FolderArtifact(
        files={
            "alpha/SKILL.md": (
                "---\nname: alpha\ndescription: stub alpha skill\n---\n\nBody."
            ),
            "beta/SKILL.md": (
                "---\nname: beta\ndescription: stub beta skill\n---\n\nBody."
            ),
        }
    )


def _make_eval_result() -> EvalResult:
    """Three synthetic TaskOutcomes covering verified=True/False/None."""
    outcomes = [
        TaskOutcome(
            task_id="tblite/t1",
            success=True,
            tool_calls=4,
            elapsed_s=12.5,
            skills_invoked=["alpha"],
            last_msg="done",
            verified=True,
            verifier_status="verified_pass",
        ),
        TaskOutcome(
            task_id="tblite/t2",
            success=False,
            tool_calls=7,
            elapsed_s=31.0,
            skills_invoked=["alpha", "beta"],
            last_msg="failed sanity check",
            verified=False,
            verifier_status="verified_fail",
        ),
        TaskOutcome(
            task_id="swebench/t3",
            success=False,
            tool_calls=0,
            elapsed_s=0.4,
            skills_invoked=[],
            last_msg="docker missing",
            verified=None,
            verifier_status="docker_unavailable",
        ),
    ]
    return EvalResult(
        success_rate=1 / 3,
        tool_calls_per_success=4.0,
        composite=0.31,
        per_task=[o.to_dict() for o in outcomes],
        failures=[
            {"task_id": "tblite/t2", "last_msg": "failed sanity check"},
            {"task_id": "swebench/t3", "last_msg": "docker missing"},
        ],
        skills_folder="/tmp/irrelevant",
        n_tasks=3,
        verified_count=1,
        unverified_count=1,
    )


def test_translate_adds_per_task_json():
    ev = SkillFolderEvaluator(force_synthetic=True, verify=False)
    res = _make_eval_result()
    artifact = _make_artifact()

    out = ev._translate(res, artifact, program_id="test-pid")

    assert "per_task" in out.artifacts, (
        "expected per_task key in artifacts side-channel"
    )
    per_task = json.loads(out.artifacts["per_task"])
    assert isinstance(per_task, list)
    assert len(per_task) == 3

    # Row 0: verified pass.
    assert per_task[0]["task_id"] == "tblite/t1"
    assert per_task[0]["verified"] is True
    assert per_task[0]["success"] is True
    assert per_task[0]["tool_calls"] == 4
    assert per_task[0]["elapsed_s"] == 12.5
    assert per_task[0]["skills_invoked"] == ["alpha"]
    assert per_task[0]["verifier_status"] == "verified_pass"

    # Row 1: verified fail.
    assert per_task[1]["task_id"] == "tblite/t2"
    assert per_task[1]["verified"] is False
    assert per_task[1]["success"] is False
    assert per_task[1]["verifier_status"] == "verified_fail"
    assert per_task[1]["skills_invoked"] == ["alpha", "beta"]

    # Row 2: unverified (three-state None preserved through JSON).
    assert per_task[2]["task_id"] == "swebench/t3"
    assert per_task[2]["verified"] is None
    assert per_task[2]["verifier_status"] == "docker_unavailable"


def test_per_task_serializable():
    """The per_task list must round-trip through json.dumps / json.loads."""
    ev = SkillFolderEvaluator(force_synthetic=True, verify=False)
    res = _make_eval_result()
    artifact = _make_artifact()

    out = ev._translate(res, artifact, program_id="")
    blob = out.artifacts["per_task"]

    # Round-trip: the stored JSON parses, and re-serializing the parsed
    # value then re-parsing it gives the same structure.
    parsed = json.loads(blob)
    assert json.loads(json.dumps(parsed)) == parsed


def test_translate_preserves_existing_artifacts_and_metrics():
    """Appending per_task must not clobber sibling artifact keys or metrics."""
    ev = SkillFolderEvaluator(force_synthetic=True, verify=False)
    res = _make_eval_result()
    artifact = _make_artifact()

    out = ev._translate(res, artifact, program_id="")

    # Metrics unchanged shape.
    for key in (
        "composite",
        "success_rate",
        "tool_calls_per_success",
        "n_tasks",
        "verified_count",
        "unverified_count",
    ):
        assert key in out.metrics, f"metric {key} missing"

    # Sibling artifact keys still present.
    for key in (
        "failures",
        "invocation_counts",
        "unused_skills",
        "synthetic",
        "cascade_truncated",
    ):
        assert key in out.artifacts, f"artifact key {key} was clobbered"


def test_controller_decode_eval_artifacts_parses_per_task():
    """The controller helper must decode JSON-string blobs to native types
    so best_meta.json is a single well-typed document (not JSON-inside-JSON).
    """
    from skill_evolve.track_b.openevolve_skills.controller import (
        _decode_eval_artifacts,
    )

    raw = {
        "per_task": json.dumps(
            [
                {"task_id": "a", "verified": True, "success": True},
                {"task_id": "b", "verified": None, "success": False},
            ]
        ),
        "invocation_counts": json.dumps({"alpha": 2}),
        "synthetic": "0",
        "cascade_truncated": "1",
    }
    decoded = _decode_eval_artifacts(raw)

    assert isinstance(decoded["per_task"], list)
    assert decoded["per_task"][0]["verified"] is True
    assert decoded["per_task"][1]["verified"] is None
    assert decoded["invocation_counts"] == {"alpha": 2}
    # Scalars stay as strings (can't be mistaken for JSON objects).
    assert decoded["synthetic"] == "0"
    assert decoded["cascade_truncated"] == "1"
