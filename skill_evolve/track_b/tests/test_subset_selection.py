"""Tests for ``scripts/select_subset_20.py`` and ``scripts/select_hot_12.py``.

We import the script modules directly (rather than shelling out) so the
tests stay hermetic and fast. The scripts are pure-Python with a
``main(argv)`` entrypoint and helper functions (``select_subset``,
``select_hot_12``) suitable for direct invocation.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import pytest


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


# ---------------------------------------------------------------------------
# Vendor fixture
# ---------------------------------------------------------------------------


_TASK_TOML_TPL = """\
version = "1.0"

[metadata]
author_name = "Test"
difficulty = "medium"
category = "{domain}"
tags = ["{domain}"]

[verifier]
timeout_sec = 600.0

[agent]
timeout_sec = 600.0
"""


def _build_fake_vendor(
    tmp_path: Path,
    total: int = 30,
    n_domains: int = 5,
) -> Path:
    """Lay out ``total`` fake tasks under ``<vendor>/tasks/`` across N domains."""
    vendor = tmp_path / "vendor"
    tasks_root = vendor / "tasks"
    tasks_root.mkdir(parents=True)
    domain_names = [f"domain{ord('a') + i:c}" for i in range(n_domains)]
    for i in range(total):
        domain = domain_names[i % n_domains]
        # Deterministic, alpha-sortable name.
        task_dir = tasks_root / f"task-{i:02d}-{domain}"
        task_dir.mkdir()
        (task_dir / "task.toml").write_text(
            _TASK_TOML_TPL.format(domain=domain),
            encoding="utf-8",
        )
        (task_dir / "instruction.md").write_text("test", encoding="utf-8")
        (task_dir / "environment").mkdir()
        (task_dir / "tests").mkdir()
    return vendor


# ---------------------------------------------------------------------------
# select_subset_20
# ---------------------------------------------------------------------------


def test_select_subset_20_count_and_diversity(tmp_path: Path) -> None:
    vendor = _build_fake_vendor(tmp_path, total=30, n_domains=5)
    select_subset_20 = _load_script_module("select_subset_20")

    subset = select_subset_20.select_subset(vendor, target_count=20)
    assert len(subset) == 20
    # All entries are fully-qualified.
    assert all(t.startswith("skillsbench/") for t in subset)
    # Diversity: every domain contributes at least one task in the first
    # pass (≤ 5 tasks).
    first_five_domains = {tid.rsplit("-", 1)[-1] for tid in subset[:5]}
    assert len(first_five_domains) == 5


def test_select_subset_20_is_deterministic(tmp_path: Path) -> None:
    vendor = _build_fake_vendor(tmp_path, total=30, n_domains=5)
    select_subset_20 = _load_script_module("select_subset_20")

    s1 = select_subset_20.select_subset(vendor, target_count=20)
    s2 = select_subset_20.select_subset(vendor, target_count=20)
    assert s1 == s2


def test_select_subset_20_main_writes_json(tmp_path: Path) -> None:
    vendor = _build_fake_vendor(tmp_path, total=30, n_domains=5)
    out = tmp_path / "out" / "subset_20.json"
    select_subset_20 = _load_script_module("select_subset_20")

    rc = select_subset_20.main(
        [
            "--vendor-dir",
            str(vendor),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert len(payload) == 20

    # Round-trip determinism via main entrypoint.
    out2 = tmp_path / "out" / "subset_20_b.json"
    select_subset_20.main(
        [
            "--vendor-dir",
            str(vendor),
            "--out",
            str(out2),
        ]
    )
    assert json.loads(out.read_text()) == json.loads(out2.read_text())


def test_select_subset_20_fails_if_too_few(tmp_path: Path) -> None:
    vendor = _build_fake_vendor(tmp_path, total=10, n_domains=2)
    select_subset_20 = _load_script_module("select_subset_20")
    with pytest.raises(RuntimeError):
        select_subset_20.select_subset(vendor, target_count=20)


# ---------------------------------------------------------------------------
# select_hot_12
# ---------------------------------------------------------------------------


def _build_synthetic_summary(
    subset_20: List[str],
    *,
    n_failing: int = 8,
    n_partial: int = 8,
    n_passing: int = 4,
) -> Dict[str, Any]:
    assert n_failing + n_partial + n_passing == len(subset_20)
    per_task = []
    cursor = 0
    for tid in subset_20[cursor : cursor + n_failing]:
        per_task.append({"task_id": tid, "with_skills_score": 0.0})
    cursor += n_failing
    for tid in subset_20[cursor : cursor + n_partial]:
        per_task.append({"task_id": tid, "with_skills_score": 0.5})
    cursor += n_partial
    for tid in subset_20[cursor : cursor + n_passing]:
        per_task.append({"task_id": tid, "with_skills_score": 1.0})
    return {"per_task": per_task}


def test_select_hot_12_picks_4_per_bin() -> None:
    select_hot_12 = _load_script_module("select_hot_12")
    subset_20 = [f"skillsbench/task-{i:02d}" for i in range(20)]
    summary = _build_synthetic_summary(
        subset_20,
        n_failing=8,
        n_partial=8,
        n_passing=4,
    )

    hot = select_hot_12.select_hot_12(summary, subset_20)
    assert len(hot) == 12
    # Every selected task is in subset_20.
    assert set(hot).issubset(set(subset_20))

    # Look up scores.
    score_map = {r["task_id"]: r["with_skills_score"] for r in summary["per_task"]}
    bins = {"failing": 0, "partial": 0, "passing": 0}
    for tid in hot:
        s = score_map[tid]
        if s < 0.2:
            bins["failing"] += 1
        elif s < 0.8:
            bins["partial"] += 1
        else:
            bins["passing"] += 1
    assert bins["failing"] == 4
    assert bins["partial"] == 4
    assert bins["passing"] == 4


def test_select_hot_12_overflow_when_bin_short() -> None:
    """If passing has 2, partial overflow fills the remaining 2 slots."""
    select_hot_12 = _load_script_module("select_hot_12")
    subset_20 = [f"skillsbench/task-{i:02d}" for i in range(20)]
    summary = _build_synthetic_summary(
        subset_20,
        n_failing=8,
        n_partial=10,
        n_passing=2,
    )

    hot = select_hot_12.select_hot_12(summary, subset_20)
    assert len(hot) == 12
    score_map = {r["task_id"]: r["with_skills_score"] for r in summary["per_task"]}
    bins = {"failing": 0, "partial": 0, "passing": 0}
    for tid in hot:
        s = score_map[tid]
        if s < 0.2:
            bins["failing"] += 1
        elif s < 0.8:
            bins["partial"] += 1
        else:
            bins["passing"] += 1
    # Passing only had 2; partial overflow should have filled the remainder.
    assert bins["passing"] == 2
    assert bins["partial"] == 6
    assert bins["failing"] == 4


def test_select_hot_12_main_writes_json(tmp_path: Path) -> None:
    select_hot_12 = _load_script_module("select_hot_12")
    subset_20 = [f"skillsbench/task-{i:02d}" for i in range(20)]
    summary = _build_synthetic_summary(subset_20)

    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    subset_path = tmp_path / "subset_20.json"
    subset_path.write_text(json.dumps(subset_20), encoding="utf-8")
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
    assert set(payload).issubset(set(subset_20))
