"""Regression tests for ``SkillFolderEvaluator.evaluate_artifact`` plumbing.

Phase E bug: ``evaluate_artifact`` stored ``task_source`` / ``task_list`` /
``agent_backend`` on ``self`` but never passed them to the inner
``skill_evolve.evaluator.evaluate()`` call. Result: evolution silently
fell back to the default tblite manifest (10 tasks) instead of the
configured SkillsBench hot-12 subset.

These tests verify (without any real LLM / Docker / bench CLI calls):

  1. ``task_source="skillsbench"`` forwards ``sources=["skillsbench"]``
     to the inner ``evaluate()``.
  2. A populated ``task_list`` JSON is hydrated and forwarded as
     ``task_ids=[...]`` (matching the file contents).
  3. The default tblite path forwards ``sources=["tblite"]`` and
     leaves ``task_ids`` as ``None`` (no filter) — bit-identical to
     the pre-Phase-E behavior except for the explicit source pin.
  4. ``agent_backend`` is plumbed through so the inner evaluator can
     route per-task execution to the configured backend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pytest

from skill_evolve.track_b.openevolve_skills.evaluator import (
    SkillFolderEvaluator,
)
from skill_evolve.track_b.openevolve_skills.folder_artifact import (
    FolderArtifact,
)


def _make_artifact() -> FolderArtifact:
    """Minimal valid artifact with one well-formed SKILL.md."""
    return FolderArtifact(
        files={
            "placeholder/SKILL.md": (
                "---\nname: placeholder\ndescription: placeholder skill\n---\n\nbody\n"
            ),
        }
    )


def _fake_eval_result(skills_dir: Path) -> Any:
    """Return a minimal-fields stand-in for ``EvalResult``."""
    return mock.MagicMock(
        composite=0.0,
        success_rate=0.0,
        tool_calls_per_success=0.0,
        n_tasks=0,
        verified_count=0,
        unverified_count=0,
        mean_score=None,
        scored_task_count=0,
        per_task=[],
        failures=[],
        skills_folder=str(skills_dir),
        cascade_truncated=False,
        synthetic=False,
    )


def test_skillsbench_path_forwards_sources_and_task_ids(tmp_path: Path) -> None:
    """``task_source="skillsbench"`` + populated ``task_list`` JSON
    must reach the inner ``evaluate()`` as
    ``sources=["skillsbench"]`` and ``task_ids=[...]`` matching the
    JSON contents.
    """
    task_list_path = tmp_path / "hot_12.json"
    payload = [
        "skillsbench/forensics-disk-recovery",
        "skillsbench/azure-bgp-oscillation",
        "skillsbench/dialogue-parser",
    ]
    task_list_path.write_text(json.dumps(payload), encoding="utf-8")

    ev = SkillFolderEvaluator(
        force_synthetic=True,
        task_source="skillsbench",
        agent_backend="bench-cli",
        task_list=task_list_path,
    )

    captured: Dict[str, Any] = {}

    def _fake_evaluate(skills_dir: Path, **kwargs: Any) -> Any:
        captured["skills_dir"] = skills_dir
        captured["kwargs"] = kwargs
        return _fake_eval_result(skills_dir)

    with mock.patch(
        "skill_evolve.track_b.openevolve_skills.evaluator._skill_evaluator.evaluate",
        side_effect=_fake_evaluate,
    ):
        ev.evaluate_artifact(_make_artifact())

    kw = captured["kwargs"]
    assert kw["sources"] == ["skillsbench"]
    assert kw["task_ids"] == payload
    assert kw["agent_backend"] == "bench-cli"


def test_tblite_path_does_not_pass_skillsbench_filter() -> None:
    """The default ``task_source="tblite"`` path must NOT pass a
    skillsbench source filter, must NOT set ``task_ids``, and must
    forward the configured agent_backend (default ``"hermes"``).
    """
    ev = SkillFolderEvaluator(
        force_synthetic=True,
        # All Phase E fields default: task_source="tblite",
        # agent_backend="hermes", task_list=None.
    )

    captured: Dict[str, Any] = {}

    def _fake_evaluate(skills_dir: Path, **kwargs: Any) -> Any:
        captured["skills_dir"] = skills_dir
        captured["kwargs"] = kwargs
        return _fake_eval_result(skills_dir)

    with mock.patch(
        "skill_evolve.track_b.openevolve_skills.evaluator._skill_evaluator.evaluate",
        side_effect=_fake_evaluate,
    ):
        ev.evaluate_artifact(_make_artifact())

    kw = captured["kwargs"]
    # tblite path pins the source explicitly so the inner evaluator
    # can't accidentally pull non-tblite manifest entries; task_ids
    # stays None so the existing 10-task tblite subset is preserved.
    assert kw["sources"] == ["tblite"]
    assert kw["task_ids"] is None
    assert kw["agent_backend"] == "hermes"


def test_skillsbench_with_missing_task_list_falls_back_gracefully(
    tmp_path: Path,
) -> None:
    """A non-existent ``task_list`` path should NOT crash; it falls
    back to ``task_ids=None`` (no allowlist) while still pinning
    ``sources=["skillsbench"]``.
    """
    ev = SkillFolderEvaluator(
        force_synthetic=True,
        task_source="skillsbench",
        agent_backend="bench-cli",
        task_list=tmp_path / "does_not_exist.json",
    )

    captured: Dict[str, Any] = {}

    def _fake_evaluate(skills_dir: Path, **kwargs: Any) -> Any:
        captured["kwargs"] = kwargs
        return _fake_eval_result(skills_dir)

    with mock.patch(
        "skill_evolve.track_b.openevolve_skills.evaluator._skill_evaluator.evaluate",
        side_effect=_fake_evaluate,
    ):
        ev.evaluate_artifact(_make_artifact())

    kw = captured["kwargs"]
    assert kw["sources"] == ["skillsbench"]
    assert kw["task_ids"] is None


def test_load_subset_filters_by_task_ids() -> None:
    """``load_subset`` honors a ``task_ids=`` allowlist (post-filter)."""
    from skill_evolve.benchmark.load import load_subset

    # offline_only=True so no HF lookups; we only care about the IDs.
    full = load_subset(offline_only=True)
    assert len(full) > 0
    chosen = [full[0]["task_id"]]
    filtered = load_subset(offline_only=True, task_ids=chosen)
    assert {t["task_id"] for t in filtered} == set(chosen)


def test_load_subset_task_ids_accepts_bare_segment() -> None:
    """``task_ids`` matches both fully-qualified IDs and bare segments."""
    from skill_evolve.benchmark.load import load_subset

    full = load_subset(offline_only=True)
    if not full:
        pytest.skip("manifest is empty in this environment")
    full_id = full[0]["task_id"]
    bare = full_id.rsplit("/", 1)[-1] if "/" in full_id else full_id
    filtered = load_subset(offline_only=True, task_ids=[bare])
    assert any(t["task_id"] == full_id for t in filtered)


def test_evaluate_signature_accepts_new_kwargs() -> None:
    """The inner ``evaluate()`` signature must accept the new kwargs
    so we don't get a ``TypeError`` at runtime."""
    import inspect

    from skill_evolve.evaluator import evaluate

    sig = inspect.signature(evaluate)
    assert "sources" in sig.parameters
    assert "task_ids" in sig.parameters
    assert "agent_backend" in sig.parameters
