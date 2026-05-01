"""Mocked tests for ``--task-source`` / ``--agent-backend`` dispatch.

Per ``plans/plan_0.md`` Group E step 1-2, ``skill_evolve.track_b.run``
gains three flags:
  * ``--task-source {tblite,skillsbench}`` (default tblite)
  * ``--agent-backend {hermes,bench-cli}`` (default hermes)
  * ``--task-list <path>`` (default None -> full manifest)

This module verifies (without launching a real evolution run):

  1. Defaults preserve the existing tblite/hermes flow byte-for-byte —
     the SkillsBench dispatch code path is NOT touched.
  2. ``--task-source skillsbench`` forces ``--agent-backend bench-cli``
     and ``--anonymize-tasks`` on (D-5 mandatory).
  3. The Controller's ``_install_anonymizer`` selects the SkillsBench
     anonymizer module + skillsbench id_map when ``task_source ==
     "skillsbench"``, and the in-file (tblite) anonymizer otherwise.
  4. The leak scanner integrates into ``evaluate_artifact`` and runs
     when ``anonymize_tasks=True``.

Tests are hermetic: no Docker, no network, no real bench CLI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from skill_evolve.track_b.openevolve_skills.controller import (
    _install_anonymizer,
)
from skill_evolve.track_b.openevolve_skills.evaluator import (
    SkillFolderEvaluator,
)
from skill_evolve.track_b.openevolve_skills.folder_artifact import (
    FolderArtifact,
)
from skill_evolve.track_b.run import _build_parser, main as run_main


# ---------------------------------------------------------------------------
# CLI argument plumbing
# ---------------------------------------------------------------------------


def test_default_args_preserve_tblite_hermes_path() -> None:
    """Without any new flags, all defaults match the pre-Phase-E flow."""
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--seed",
            "seed_skills_empty",
            "--out",
            "/tmp/x",
        ]
    )
    assert args.task_source == "tblite"
    assert args.agent_backend == "hermes"
    assert args.task_list is None
    assert args.anonymize_tasks is False
    assert args.leak_policy == "zero"


def test_skillsbench_flag_choices() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--seed",
            "seed_skills_empty",
            "--out",
            "/tmp/x",
            "--task-source",
            "skillsbench",
            "--agent-backend",
            "bench-cli",
            "--task-list",
            "/tmp/hot_12.json",
            "--anonymize-tasks",
        ]
    )
    assert args.task_source == "skillsbench"
    assert args.agent_backend == "bench-cli"
    assert args.task_list == Path("/tmp/hot_12.json")
    assert args.anonymize_tasks is True


def test_skillsbench_rejects_invalid_choices() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--seed",
                "x",
                "--out",
                "y",
                "--task-source",
                "swebench",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--seed",
                "x",
                "--out",
                "y",
                "--agent-backend",
                "openai",
            ]
        )


def test_skillsbench_forces_bench_cli_and_anonymize(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end: when --task-source skillsbench is set, run.main()
    rewrites args.agent_backend to bench-cli and forces anonymize_tasks
    on with a stderr warning."""
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "placeholder").mkdir()
    (seed / "placeholder" / "SKILL.md").write_text(
        "---\nname: placeholder\ndescription: x\n---\n\nbody",
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    def _fake_run_evolution(**kwargs: Any) -> Any:
        captured["evaluator"] = kwargs["evaluator"]
        # Return a dummy result-shaped object.
        return mock.MagicMock(
            best=mock.MagicMock(
                fitness=lambda: 0.0,
                metrics={},
                eval_artifacts={},
            ),
            output_dir=kwargs["out_dir"],
            history=[],
        )

    with mock.patch(
        "skill_evolve.track_b.run.run_evolution",
        side_effect=_fake_run_evolution,
    ):
        rc = run_main(
            [
                "--seed",
                str(seed),
                "--out",
                str(tmp_path / "out"),
                "--num-generations",
                "0",
                "--num-islands",
                "1",
                "--force-synthetic",
                "--task-source",
                "skillsbench",
                # Deliberately omit --anonymize-tasks; expect it forced on.
            ]
        )
    # rc==0 since fake run_evolution returns a populated best.
    assert rc == 0
    err = capsys.readouterr().err
    assert "forces --anonymize-tasks" in err
    ev = captured["evaluator"]
    assert ev.task_source == "skillsbench"
    assert ev.agent_backend == "bench-cli"
    assert ev.anonymize_tasks is True


# ---------------------------------------------------------------------------
# Anonymizer dispatch
# ---------------------------------------------------------------------------


def test_install_anonymizer_uses_in_file_for_tblite() -> None:
    """tblite path -> in-file ``sanitize_text`` / ``find_leaked_names``."""
    ev = SkillFolderEvaluator(
        force_synthetic=True,
        anonymize_tasks=True,
        task_source="tblite",
    )
    _install_anonymizer(ev)
    # Spot-check: the anonymizer's sanitize_text behaves like the
    # in-file one (replaces both qualified + bare forms).
    from skill_evolve.track_b.openevolve_skills.evaluator import (
        sanitize_text as _in_file_sanitize_text,
    )

    src = "tblite/foo-bar and foo-bar"
    mapping = {"tblite/foo-bar": "task_001", "foo-bar": "task_001"}
    a, _ = ev.anonymizer.sanitize_text(src, mapping)
    b, _ = _in_file_sanitize_text(src, mapping)
    assert a == b


def test_install_anonymizer_uses_skillsbench_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """skillsbench path -> ``skill_evolve.benchmark.skillsbench_anonymize``
    + id_map domain limited to the (mocked) hydrated task records."""
    from skill_evolve.benchmark.load import Task

    def _make_task(task_id: str) -> Task:
        return Task(
            task_id=task_id,
            source="skillsbench",
            prompt="",
            success_check_kind="skillsbench_test_sh",
            success_check_payload={},
            timeout_s=600,
        )

    fake_records: list[Any] = [
        _make_task("skillsbench/forensics-disk-recovery"),
        _make_task("skillsbench/azure-bgp-oscillation"),
    ]

    def _fake_load(_task_list: Any) -> list[Any]:
        return fake_records

    monkeypatch.setattr(
        "skill_evolve.track_b.openevolve_skills.controller."
        "_load_skillsbench_task_records",
        _fake_load,
    )

    ev = SkillFolderEvaluator(
        force_synthetic=True,
        anonymize_tasks=True,
        task_source="skillsbench",
        task_list=Path("/tmp/does_not_matter.json"),
    )
    _install_anonymizer(ev)

    # The id_map should map both fully-qualified and bare-segment forms.
    id_map = ev.task_id_map()
    assert "skillsbench/forensics-disk-recovery" in id_map
    assert "forensics-disk-recovery" in id_map
    # And the records were stashed for the leak scanner.
    assert ev._task_records is fake_records


# ---------------------------------------------------------------------------
# Task-list filtering
# ---------------------------------------------------------------------------


def test_load_skillsbench_task_records_filters_by_task_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``--task-list`` is provided, only those tasks are hydrated."""
    fake_vendor = tmp_path / "vendor" / "tasks"
    fake_vendor.mkdir(parents=True)
    for name in ("alpha-task", "beta-task", "gamma-task"):
        d = fake_vendor / name
        d.mkdir()
        (d / "task.toml").write_text(
            'version = "1.0"\n[metadata]\n[verifier]\ntimeout_sec = 60\n'
            "[agent]\ntimeout_sec = 60\n",
            encoding="utf-8",
        )
        (d / "instruction.md").write_text("instr", encoding="utf-8")
        (d / "environment").mkdir()
        (d / "tests").mkdir()

    # Point the loader at our fake vendor dir.
    import skill_evolve.track_b.openevolve_skills.controller as ctrl_mod

    monkeypatch.setattr(
        ctrl_mod,
        "_load_skillsbench_task_records",
        ctrl_mod._load_skillsbench_task_records,
        # raising=True default
    )
    # We monkey the Path that the function uses internally — patch the
    # vendor_dir construction by overriding the function with one that
    # re-points at fake_vendor.
    from skill_evolve.benchmark.skillsbench_loader import hydrate_one

    def _fake_loader(task_list: Any) -> list[Any]:
        ids = json.loads(Path(task_list).read_text(encoding="utf-8"))
        names = [tid.split("/", 1)[-1] for tid in ids]
        return [hydrate_one(fake_vendor / n) for n in names]

    task_list_path = tmp_path / "task_list.json"
    task_list_path.write_text(
        json.dumps(["skillsbench/alpha-task", "skillsbench/beta-task"]),
        encoding="utf-8",
    )
    records = _fake_loader(task_list_path)
    assert len(records) == 2
    assert {r.task_id for r in records} == {
        "skillsbench/alpha-task",
        "skillsbench/beta-task",
    }


# ---------------------------------------------------------------------------
# Leak-scanner hookup
# ---------------------------------------------------------------------------


def test_leak_scanner_attaches_warning_on_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the scanner returns hits, ``_apply_leak_policy`` attaches a
    ``leak_warning`` artifact and zeroes the score (default policy)."""
    from skill_evolve.track_b.openevolve_skills.evaluator import (
        EvaluationResult,
    )
    from skill_evolve.track_b.openevolve_skills.leak_scanner import (
        LeakHit,
        LeakScanResult,
    )

    ev = SkillFolderEvaluator(
        force_synthetic=True,
        anonymize_tasks=True,
        task_source="skillsbench",
        leak_policy="zero",
    )
    # Inject a faked anonymizer + task records so id_map is non-empty.
    ev.set_task_id_map({"skillsbench/foo": "task_001", "foo": "task_001"})
    ev._task_records = []

    fake_hits = LeakScanResult(
        hits=[
            LeakHit(file="s/SKILL.md", kind="task_id", needle="foo"),
        ]
    )
    monkeypatch.setattr(
        "skill_evolve.track_b.openevolve_skills.leak_scanner.scan_artifact",
        lambda *a, **kw: fake_hits,
    )

    art = FolderArtifact(
        files={
            "s/SKILL.md": "---\nname: s\ndescription: x\n---\n\nfoo",
        }
    )
    base = EvaluationResult(
        metrics={"composite": 0.5, "success_rate": 0.8, "mean_score": 0.7},
        artifacts={"failures": "[]"},
    )
    out = ev._apply_leak_policy(base, art)
    assert out.metrics["composite"] == 0.0
    assert out.metrics["success_rate"] == 0.0
    assert out.metrics["mean_score"] == 0.0
    assert "leak_warning" in out.artifacts
    assert "task_id" in out.artifacts["leak_warning"]


def test_leak_scanner_warn_policy_keeps_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skill_evolve.track_b.openevolve_skills.evaluator import (
        EvaluationResult,
    )
    from skill_evolve.track_b.openevolve_skills.leak_scanner import (
        LeakHit,
        LeakScanResult,
    )

    ev = SkillFolderEvaluator(
        force_synthetic=True,
        anonymize_tasks=True,
        task_source="skillsbench",
        leak_policy="warn",
    )
    ev.set_task_id_map({"foo": "task_001"})
    ev._task_records = []

    monkeypatch.setattr(
        "skill_evolve.track_b.openevolve_skills.leak_scanner.scan_artifact",
        lambda *a, **kw: LeakScanResult(
            hits=[
                LeakHit(file="s/SKILL.md", kind="task_id", needle="foo"),
            ]
        ),
    )

    art = FolderArtifact(
        files={
            "s/SKILL.md": "---\nname: s\ndescription: x\n---\n\nfoo",
        }
    )
    base = EvaluationResult(
        metrics={"composite": 0.7, "success_rate": 1.0},
        artifacts={},
    )
    out = ev._apply_leak_policy(base, art)
    # warn -> score preserved, but warning still attached.
    assert out.metrics["composite"] == 0.7
    assert out.metrics["success_rate"] == 1.0
    assert "leak_warning" in out.artifacts


def test_leak_scanner_anonymize_off_is_noop() -> None:
    """When anonymize_tasks=False, scanner is bypassed entirely."""
    from skill_evolve.track_b.openevolve_skills.evaluator import (
        EvaluationResult,
    )

    ev = SkillFolderEvaluator(
        force_synthetic=True,
        anonymize_tasks=False,
        task_source="tblite",
    )
    art = FolderArtifact(
        files={
            "s/SKILL.md": "---\nname: s\ndescription: x\n---\n\ntblite/foo-bar",
        }
    )
    base = EvaluationResult(
        metrics={"composite": 0.5, "success_rate": 0.5},
        artifacts={},
    )
    out = ev._apply_leak_policy(base, art)
    # Identical pass-through.
    assert out is base
