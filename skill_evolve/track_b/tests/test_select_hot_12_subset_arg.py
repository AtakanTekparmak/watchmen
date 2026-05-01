"""Verify the Phase E fix to ``scripts/select_hot_12.py``.

Two regressions are locked in:

  1. The script accepts ANY task-list JSON (not just ``subset_20.json``)
     — Phase D v2 ships ``subset_17.json`` and the runner must not
     hardcode the size-20 path.
  2. The score extractor handles the Phase D ``by_condition`` summary
     shape (``{"with-skills": {"score_mean": ...}}``) in addition to
     the flat ``with_skills_score`` shape used by the older fixture.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"


def _load_script_module(name: str) -> ModuleType:
    """Import a script-module by file path so it doesn't need to be a package."""
    path = _SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_scripts_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_by_condition_summary(
    subset: List[str],
    *,
    n_failing: int,
    n_partial: int,
    n_passing: int,
) -> Dict[str, Any]:
    """Mirror the Phase D summary.json shape (by_condition.with-skills.score_mean)."""
    assert n_failing + n_partial + n_passing == len(subset)
    per_task: List[Dict[str, Any]] = []
    cursor = 0
    for tid in subset[cursor : cursor + n_failing]:
        per_task.append(
            {
                "task_id": tid,
                "by_condition": {
                    "with-skills": {"n": 5, "pass_rate": 0.0, "score_mean": 0.0},
                    "no-skills": {"n": 5, "pass_rate": 0.0, "score_mean": 0.0},
                },
            }
        )
    cursor += n_failing
    for tid in subset[cursor : cursor + n_partial]:
        per_task.append(
            {
                "task_id": tid,
                "by_condition": {
                    "with-skills": {"n": 5, "pass_rate": 0.4, "score_mean": 0.5},
                    "no-skills": {"n": 5, "pass_rate": 0.0, "score_mean": 0.0},
                },
            }
        )
    cursor += n_partial
    for tid in subset[cursor : cursor + n_passing]:
        per_task.append(
            {
                "task_id": tid,
                "by_condition": {
                    "with-skills": {"n": 5, "pass_rate": 1.0, "score_mean": 1.0},
                    "no-skills": {"n": 5, "pass_rate": 0.2, "score_mean": 0.2},
                },
            }
        )
    return {"per_task": per_task}


def test_extract_handles_by_condition_shape() -> None:
    """Phase D summaries use ``by_condition.with-skills.score_mean``."""
    select_hot_12 = _load_script_module("select_hot_12")

    rec = {
        "task_id": "skillsbench/example",
        "by_condition": {
            "with-skills": {"n": 5, "pass_rate": 0.4, "score_mean": 0.55},
            "no-skills": {"n": 5, "pass_rate": 0.0, "score_mean": 0.0},
        },
    }
    score = select_hot_12._extract_with_skills_score(rec)
    assert score == 0.55


def test_extract_falls_back_to_flat_shape() -> None:
    """Old fixture shape still works."""
    select_hot_12 = _load_script_module("select_hot_12")
    rec = {"task_id": "skillsbench/example", "with_skills_score": 0.6}
    score = select_hot_12._extract_with_skills_score(rec)
    assert score == 0.6


def test_select_hot_12_against_subset_17(tmp_path: Path) -> None:
    """The script accepts a subset of ANY size; not just 20.

    Build a 17-task list (matching subset_17.json) and a Phase D-shaped
    summary; assert ``main`` succeeds and emits a 12-task hot subset.
    """
    select_hot_12 = _load_script_module("select_hot_12")

    subset_17 = [f"skillsbench/task-{i:02d}" for i in range(17)]
    summary = _build_by_condition_summary(
        subset_17,
        n_failing=8,
        n_partial=7,
        n_passing=2,
    )

    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    subset_path = tmp_path / "subset_17.json"
    subset_path.write_text(json.dumps(subset_17), encoding="utf-8")
    out_path = tmp_path / "hot_12.json"

    rc = select_hot_12.main(
        [
            str(summary_path),
            str(subset_path),
            "-o",
            str(out_path),
        ]
    )
    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert len(payload) == 12
    assert set(payload).issubset(set(subset_17))


def test_real_baseline_v2_summary_against_subset_17() -> None:
    """End-to-end: the real Phase D v2 baseline summary + subset_17 must
    yield 12 tasks. Locks in the bug fix.

    Skipped if the real summary is not committed (some test environments
    won't have ``runs/`` populated).
    """
    select_hot_12 = _load_script_module("select_hot_12")

    summary_p = _REPO_ROOT / "runs" / "skillsbench_baseline_v2" / "summary.json"
    subset_p = _REPO_ROOT / "skill_evolve" / "skillsbench" / "subset_17.json"
    if not (summary_p.exists() and subset_p.exists()):
        import pytest

        pytest.skip("baseline_v2 summary or subset_17 missing")

    summary = json.loads(summary_p.read_text(encoding="utf-8"))
    subset = json.loads(subset_p.read_text(encoding="utf-8"))
    hot = select_hot_12.select_hot_12(summary, subset, target=12)
    assert len(hot) == 12
    assert set(hot).issubset(set(subset))


def test_error_message_does_not_say_subset_20() -> None:
    """When there's no overlap, the error message should be generic
    (the old hardcoded ``subset_20.json`` was misleading)."""
    select_hot_12 = _load_script_module("select_hot_12")
    summary = {"per_task": [{"task_id": "totally-different", "with_skills_score": 0.5}]}
    subset = ["skillsbench/missing-task"]
    import pytest

    with pytest.raises(RuntimeError) as exc:
        select_hot_12.select_hot_12(summary, subset, target=12)
    assert "subset_20" not in str(exc.value)
