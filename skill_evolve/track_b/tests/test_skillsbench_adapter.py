"""Unit tests for the SkillsBench task source adapter (Group C).

Covers:
  * ``hydrate_one`` returns a Task with the expected fields and payload.
  * ``build_skillsbench_id_map`` produces stable ``task_NNN`` aliases
    for both fully-qualified and bare-segment forms.
  * ``sanitize_text_skillsbench`` round-trips via a reverse map.
  * ``find_leaked_skillsbench_names`` only flags ``*.md`` files.
  * The verifier shim returns ``passed=True`` when ``docker build`` and
    ``docker run`` both rc=0 (subprocess.run mocked). benchflow 0.3.2
    has no ``bench eval verify`` subcommand; verification is dockerized
    directly against the task's bundled ``environment/Dockerfile``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
from unittest import mock

import pytest

from skill_evolve.benchmark.load import Task
from skill_evolve.benchmark.skillsbench_anonymize import (
    build_skillsbench_id_map,
    find_leaked_skillsbench_names,
    sanitize_artifact_skillsbench,
    sanitize_text_skillsbench,
)
from skill_evolve.benchmark.skillsbench_loader import hydrate_one
from skill_evolve.track_b.openevolve_skills.folder_artifact import FolderArtifact


_TASK_TOML_BODY = """\
version = "1.0"

[metadata]
author_name = "Test"
difficulty = "medium"
category = "engineering"
tags = ["python", "test"]

[verifier]
timeout_sec = 600.0

[agent]
timeout_sec = 600.0
"""


def _build_fixture_task(tmp_path: Path, name: str = "fixture-task") -> Path:
    task_dir = tmp_path / "tasks" / name
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "solution").mkdir()
    (task_dir / "task.toml").write_text(_TASK_TOML_BODY, encoding="utf-8")
    (task_dir / "instruction.md").write_text("Solve foo.", encoding="utf-8")
    (task_dir / "solution" / "solve.sh").write_text("exit 0\n", encoding="utf-8")
    return task_dir


# ---------------------------------------------------------------------------
# hydrate_one
# ---------------------------------------------------------------------------


def test_hydrate_one_returns_task(tmp_path: Path) -> None:
    task_dir = _build_fixture_task(tmp_path)
    task = hydrate_one(task_dir)

    assert isinstance(task, Task)
    assert task.task_id == "skillsbench/fixture-task"
    assert task.source == "skillsbench"
    assert task.success_check_kind == "skillsbench_test_sh"
    assert task.prompt.strip() == "Solve foo."
    # Payload shape.
    payload = task.success_check_payload
    assert payload["task_dir"] == str(task_dir)
    assert payload["environment_dir"] == str(task_dir / "environment")
    assert payload["tests_dir"] == str(task_dir / "tests")
    # No skills/ subdir in the fixture.
    assert payload["skills_dir"] is None
    # Domain falls through tags[0] (no top-level domain set).
    assert payload["domain"] in {"python", "engineering"}
    assert payload["timeout_sec"] == 600
    # Misc fields.
    assert task.timeout_s == 600
    assert task.extra["dataset_task_name"] == "fixture-task"


def test_hydrate_one_picks_up_skills_dir(tmp_path: Path) -> None:
    task_dir = _build_fixture_task(tmp_path, name="fixture-with-skills")
    (task_dir / "environment" / "skills").mkdir()
    task = hydrate_one(task_dir)
    assert task.success_check_payload["skills_dir"] == str(
        task_dir / "environment" / "skills"
    )


# ---------------------------------------------------------------------------
# Anonymizer
# ---------------------------------------------------------------------------


def _fake_tasks(*ids: str) -> list:
    return [
        Task(
            task_id=tid,
            source="skillsbench",
            prompt="",
            success_check_kind="skillsbench_test_sh",
            success_check_payload={},
            timeout_s=600,
        )
        for tid in ids
    ]


def test_build_skillsbench_id_map_alpha_sort_and_aliases() -> None:
    tasks = _fake_tasks(
        "skillsbench/zebra-task",
        "skillsbench/alpha-task",
        "skillsbench/middle-task",
    )
    id_map = build_skillsbench_id_map(tasks)

    # Alpha-sorted: alpha < middle < zebra → 001, 002, 003.
    assert id_map["skillsbench/alpha-task"] == "task_001"
    assert id_map["skillsbench/middle-task"] == "task_002"
    assert id_map["skillsbench/zebra-task"] == "task_003"
    # Bare segments map to the same alias.
    assert id_map["alpha-task"] == "task_001"
    assert id_map["middle-task"] == "task_002"
    assert id_map["zebra-task"] == "task_003"


def test_sanitize_text_round_trip_via_reverse_map() -> None:
    tasks = _fake_tasks(
        "skillsbench/forensics-disk-recovery",
        "skillsbench/parser-task",
        "skillsbench/network-task",
    )
    id_map = build_skillsbench_id_map(tasks)

    text = (
        "We routed forensics-disk-recovery and skillsbench/parser-task "
        "in the prompt; network-task is unrelated."
    )
    sanitized, hits = sanitize_text_skillsbench(text, id_map)

    # All real names redacted.
    assert "forensics-disk-recovery" not in sanitized
    assert "parser-task" not in sanitized
    assert "network-task" not in sanitized
    # Hits captured.
    assert "forensics-disk-recovery" in hits
    assert "skillsbench/parser-task" in hits
    assert "network-task" in hits

    # Build a reverse map (alias -> first canonical name); substituting
    # back yields the original modulo prefix conventions.
    reverse: Dict[str, str] = {}
    for k, v in id_map.items():
        # Prefer fully-qualified form when present.
        if "/" in k:
            reverse[v] = k
        else:
            reverse.setdefault(v, k)

    desanitized = sanitized
    for alias, original in reverse.items():
        desanitized = desanitized.replace(alias, original)

    # We can recover the fully-qualified prefix for the qualified original;
    # the bare originals come back as fully-qualified (since reverse map
    # prefers qualified). That's fine — same task IDs round-trip semantically.
    assert "forensics-disk-recovery" in desanitized
    assert "parser-task" in desanitized
    assert "network-task" in desanitized


def test_find_leaked_skillsbench_names_only_md(tmp_path: Path) -> None:
    tasks = _fake_tasks("skillsbench/forensics-disk-recovery")
    id_map = build_skillsbench_id_map(tasks)

    artifact = FolderArtifact(
        files={
            "skill/SKILL.md": (
                "---\nname: skill\ndescription: x\n---\n\n"
                "Handles forensics-disk-recovery.\n"
            ),
            "skill/scripts/run.sh": (
                "#!/usr/bin/env bash\n# forensics-disk-recovery in script - allowed\n"
            ),
        }
    )
    leaks = find_leaked_skillsbench_names(artifact, id_map)
    assert any(p.endswith("SKILL.md") for p, _ in leaks)
    assert not any("scripts" in p for p, _ in leaks)
    assert any(name == "forensics-disk-recovery" for _, name in leaks)


def test_sanitize_artifact_only_mutates_md() -> None:
    tasks = _fake_tasks("skillsbench/foo-task")
    id_map = build_skillsbench_id_map(tasks)
    art = FolderArtifact(
        files={
            "s/SKILL.md": "Use foo-task here.",
            "s/scripts/run.sh": "echo foo-task",
        }
    )
    out, detail = sanitize_artifact_skillsbench(art, id_map)
    # md was mutated, script left alone.
    assert "foo-task" not in out.files["s/SKILL.md"]
    assert "task_001" in out.files["s/SKILL.md"]
    assert out.files["s/scripts/run.sh"] == "echo foo-task"
    # Detail records the md hit.
    assert any(p.endswith("SKILL.md") for p, _ in detail)


# ---------------------------------------------------------------------------
# Verifier shim — mocked subprocess
# ---------------------------------------------------------------------------


def _mk_skillsbench_task(tmp_path: Path) -> Task:
    """Tiny in-memory Task for verifier tests (no real env needed)."""
    return Task(
        task_id="skillsbench/dummy",
        source="skillsbench",
        prompt="",
        success_check_kind="skillsbench_test_sh",
        success_check_payload={
            "task_dir": str(tmp_path / "task"),
            "environment_dir": str(tmp_path / "task" / "environment"),
            "tests_dir": str(tmp_path / "task" / "tests"),
            "skills_dir": None,
            "domain": "test",
            "timeout_sec": 60,
        },
        timeout_s=60,
        extra={"dataset_task_name": "dummy"},
    )


def test_verify_passes_via_dockerfile(tmp_path: Path) -> None:
    """docker build + docker run both rc=0 → passed=True, score=1.0.

    benchflow 0.3.2 has no ``bench eval verify`` subcommand, so the
    verifier shim goes straight to the Dockerfile-build path.
    """
    from skill_evolve.benchmark import skillsbench_verifier as sv

    task = _mk_skillsbench_task(tmp_path)
    (tmp_path / "task" / "environment").mkdir(parents=True)
    (tmp_path / "task" / "tests").mkdir(parents=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    docker_build_proc = mock.Mock(returncode=0, stdout="", stderr="")
    docker_run_proc = mock.Mock(
        returncode=0,
        stdout="=== 5 passed in 0.4s ===\n",
        stderr="",
    )

    call_log = []

    def _fake_run(cmd, *args, **kwargs):
        call_log.append(cmd)
        if cmd[0] == "docker" and cmd[1] == "build":
            return docker_build_proc
        if cmd[0] == "docker" and cmd[1] == "run":
            return docker_run_proc
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    with mock.patch.object(sv.subprocess, "run", side_effect=_fake_run):
        result = sv.verify(task, run_dir)
    assert result.passed is True
    assert result.status == "passed"
    # Continuous score parsed from pytest summary (5 passed / 5).
    assert result.score == 1.0
    # Verifier never shells out to ``bench`` — only docker.
    assert all(c[0] == "docker" for c in call_log)


def test_verify_failed_via_dockerfile(tmp_path: Path) -> None:
    """docker run rc=1 with partial pytest fails → passed=False, 0 < score < 1."""
    from skill_evolve.benchmark import skillsbench_verifier as sv

    task = _mk_skillsbench_task(tmp_path)
    (tmp_path / "task" / "environment").mkdir(parents=True)
    (tmp_path / "task" / "tests").mkdir(parents=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    docker_build_proc = mock.Mock(returncode=0, stdout="", stderr="")
    docker_run_proc = mock.Mock(
        returncode=1,
        stdout="=== 3 passed, 2 failed in 0.5s ===\n",
        stderr="",
    )

    def _fake_run(cmd, *args, **kwargs):
        if cmd[0] == "docker" and cmd[1] == "build":
            return docker_build_proc
        if cmd[0] == "docker" and cmd[1] == "run":
            return docker_run_proc
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    with mock.patch.object(sv.subprocess, "run", side_effect=_fake_run):
        result = sv.verify(task, run_dir)
    assert result.passed is False
    assert result.status == "failed"
    assert result.score is not None
    assert 0.0 < result.score < 1.0


def test_verify_unavailable_when_docker_missing(tmp_path: Path) -> None:
    """No docker on PATH → status='verifier_unavailable', passed=None."""
    from skill_evolve.benchmark import skillsbench_verifier as sv

    task = _mk_skillsbench_task(tmp_path)
    (tmp_path / "task" / "environment").mkdir(parents=True)
    (tmp_path / "task" / "tests").mkdir(parents=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with mock.patch.object(
        sv.subprocess, "run", side_effect=FileNotFoundError("docker")
    ):
        result = sv.verify(task, run_dir)
    assert result.passed is None
    assert result.status == "verifier_unavailable"


@pytest.mark.skip(reason="requires Docker daemon; integration test, not unit")
def test_verify_real_docker_solve_sh(tmp_path: Path) -> None:  # pragma: no cover
    raise NotImplementedError
