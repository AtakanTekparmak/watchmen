"""Mocked tests for :class:`skill_evolve.agents.bench_cli.BenchCliBackend`.

All tests run without any real ``bench`` CLI installed and without
network calls. The single seam under test is the ``subprocess.run``
call inside ``bench_cli.py``; we patch it so we control argv shape and
also have it write a synthetic ``result.json`` (matching the on-disk
schema bench emits) into the workdir's ``jobs/<job_name>/<trial>/``
subtree. The backend then reads that file and projects it onto a
``TrajectoryResult`` — same code path that runs against the real CLI.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from skill_evolve.agents.bench_cli import (
    BenchCliBackend,
    BenchRateLimitedError,
    _classify,
    _is_rate_limit_error,
)
from skill_evolve.agents.base import TrajectoryResult


# Realistic 429 error string sampled from a Phase D
# ``runs/skillsbench_baseline_v0/.../result.json``. Both the bare
# ``429`` and the ``rate_limit_error`` substring appear; either is
# enough to trigger detection.
_REAL_429_ERROR = (
    "ACP error -32603: Internal error: API Error: 429 "
    '{"type":"error","error":{"type":"rate_limit_error",'
    '"message":"This request would exceed your organization\'s rate limit '
    "of 20,000,000 prompt bytes per hour (org: 0acf070f-..., model: "
    "claude-haiku-4-5-20251001). For details, refer to: "
    'https://docs.claude.com/en/api/rate-limits."},'
    '"request_id":"req_011CaYRKcgnV5PiDzJqVUXXP"}'
)


def _make_task(
    task_id: str = "skillsbench/forensics-disk-recovery",
    task_dir: str = "/tmp/vendor/skillsbench/tasks/forensics-disk-recovery",
) -> Dict[str, Any]:
    """Minimal hydrated task dict matching the loader's shape."""
    return {
        "task_id": task_id,
        "source": "skillsbench",
        "prompt": "do the thing",
        "success_check_kind": "skillsbench_test_sh",
        "success_check_payload": {"task_dir": task_dir},
        "timeout_s": 600,
        "stage": 1,
        "skill_relevance": "",
        "extra": {},
    }


def _completed_proc(stdout: str = "", stderr: str = "", rc: int = 0) -> MagicMock:
    cp = MagicMock(spec=subprocess.CompletedProcess)
    cp.stdout = stdout
    cp.stderr = stderr
    cp.returncode = rc
    return cp


def _make_result_json(
    *,
    task_name: str = "forensics-disk-recovery",
    reward: float = 1.0,
    n_tool_calls: int = 7,
    n_prompts: int = 1,
    error: Optional[str] = None,
    verifier_error: Optional[str] = None,
    elapsed_s: float = 12.3,
    partial_trajectory: bool = False,
    trajectory_source: str = "acp",
) -> Dict[str, Any]:
    """Build a synthetic ``result.json`` matching bench's on-disk schema."""
    return {
        "task_name": task_name,
        "trial_name": f"{task_name}__deadbeef",
        "rewards": {"reward": reward},
        # SG-1: bench writes the AgentConfig.name we register, which is now
        # "claude-code" (was "claude-agent-acp" under benchflow's built-in
        # registry). agent_name stays at the underlying ACP shim package
        # since we still launch through @zed-industries/claude-agent-acp.
        "agent": "claude-code",
        "agent_name": "@zed-industries/claude-agent-acp",
        "model": "claude-haiku-4-5",
        "n_tool_calls": n_tool_calls,
        "n_prompts": n_prompts,
        "error": error,
        "verifier_error": verifier_error,
        "partial_trajectory": partial_trajectory,
        "trajectory_source": trajectory_source,
        "started_at": "2026-04-29 14:40:54.714600",
        "finished_at": "2026-04-29 14:48:44.366280",
        "timing": {
            "environment_setup": 3.5,
            "agent_setup": 0.9,
            "agent_execution": elapsed_s - 4.4 - 3.3,
            "verifier": 3.3,
            "total": elapsed_s,
        },
    }


def _make_writer(
    workdir: Path,
    *,
    result_json: Dict[str, Any],
    pytest_tail: str = "",
    stdout: str = "Score: 1/1 (100.0%), errors=0",
    stderr: str = "",
    rc: int = 0,
) -> Any:
    """Build a ``subprocess.run`` side-effect that writes ``result.json``.

    Mimics what real bench does: drops a per-trial directory under
    ``<jobs_dir>/<job_name>/<trial_name>/`` containing ``result.json``
    (and optionally ``verifier/pytest_output.txt``).
    """
    jobs_dir = workdir / "jobs"

    def _side_effect(*args: Any, **kwargs: Any) -> MagicMock:
        # bench writes its own job_name. Pick one that *differs* from
        # whatever the backend pre-computed, to exercise the fallback
        # "newest subdir under jobs_dir" branch.
        job_dir = jobs_dir / "2099-99-99__99-99-99"
        trial_dir = job_dir / result_json["trial_name"]
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "result.json").write_text(
            json.dumps(result_json), encoding="utf-8"
        )
        if pytest_tail:
            verifier_dir = trial_dir / "verifier"
            verifier_dir.mkdir(parents=True, exist_ok=True)
            (verifier_dir / "pytest_output.txt").write_text(
                pytest_tail, encoding="utf-8"
            )
        return _completed_proc(stdout=stdout, stderr=stderr, rc=rc)

    return _side_effect


@pytest.fixture(autouse=True)
def _no_orphan_sweep():
    """No-op the best-effort docker orphan-sweep for deterministic counts.

    ``_run_task_once`` calls ``_sweep_orphaned_compose_projects`` pre- and
    post-dispatch (each issues ``docker ps``/``rm`` via ``subprocess.run``).
    These tests mock ``subprocess.run`` to drive only the bench-eval call, so
    the sweep would otherwise inflate ``run_mock.call_count`` (the pre-existing
    ``3 != 1`` failures). Patching it out isolates the eval calls.
    """
    with patch(
        "skill_evolve.agents.bench_cli._sweep_orphaned_compose_projects",
        return_value=0,
    ):
        yield


def test_argv_shape(tmp_path: Path) -> None:
    """``subprocess.run`` receives the exact argv the plan specifies."""
    task = _make_task()
    rj = _make_result_json(task_name="forensics-disk-recovery", reward=1.0)

    with patch("skill_evolve.agents.bench_cli.subprocess.run") as run_mock:
        run_mock.side_effect = _make_writer(tmp_path, result_json=rj)
        BenchCliBackend().run_task(
            task,
            skills_dir=tmp_path / "skills",
            model="claude-haiku-4-5",
            timeout_s=600,
            budget_usd=1.0,
            anonymize_map=None,
            workdir=tmp_path,
        )

    assert run_mock.call_count == 1
    argv: List[str] = run_mock.call_args.args[0]
    # SG-2 fix: bench is invoked through ``python -c <shim>`` so the
    # deploy_skills monkey-patch is installed in the subprocess
    # interpreter before benchflow.cli.main resolves deploy_skills.
    # Argv shape:
    #   [<python>, "-c", <shim>, "eval", "create",
    #    --config, --tasks-dir, --agent, --model, ...]
    assert argv[0].endswith("python") or argv[0].endswith("python3")
    assert argv[1] == "-c"
    # SG-1 shim: subprocess pre-imports register_claude_code so
    # ``--agent claude-code`` resolves against benchflow's runtime registry.
    assert "register_claude_code" in argv[2]
    # SG-2 shim: subprocess pre-imports the deploy_skills monkey-patch.
    assert "_benchflow_patches" in argv[2]
    assert argv[3:5] == ["eval", "create"]
    # benchflow 0.3.4 accepts only the long flags; the short -f/-t/-a/-m forms
    # were dropped and silently zeroed every candidate (2026-05-28 silent-fail).
    assert "--config" in argv
    assert "--tasks-dir" in argv
    assert "--agent" in argv
    assert "--model" in argv
    f_idx = argv.index("--config")
    t_idx = argv.index("--tasks-dir")
    a_idx = argv.index("--agent")
    m_idx = argv.index("--model")
    # --config points at a YAML the backend materialized under the workdir.
    assert argv[f_idx + 1].endswith(".yaml")
    assert str(tmp_path) in argv[f_idx + 1]
    # --tasks-dir carries the real task_dir (not anonymized).
    assert argv[t_idx + 1] == task["success_check_payload"]["task_dir"]
    # --agent is fixed to the SG-1-registered claude-code agent; --model is the model.
    assert argv[a_idx + 1] == "claude-code"
    assert argv[m_idx + 1] == "claude-haiku-4-5"


def test_yaml_jobs_dir_overridden_to_workdir(tmp_path: Path) -> None:
    """Materialized YAML's ``jobs_dir`` must point under the workdir."""
    task = _make_task()
    rj = _make_result_json(reward=1.0)

    with patch("skill_evolve.agents.bench_cli.subprocess.run") as run_mock:
        run_mock.side_effect = _make_writer(tmp_path, result_json=rj)
        BenchCliBackend().run_task(
            task,
            skills_dir=tmp_path / "skills",
            model="claude-haiku-4-5",
            timeout_s=600,
            budget_usd=0.0,
            anonymize_map=None,
            workdir=tmp_path,
        )

    argv = run_mock.call_args.args[0]
    yaml_path = Path(argv[argv.index("--config") + 1])
    yaml_text = yaml_path.read_text(encoding="utf-8")
    expected_jobs_dir = str(tmp_path / "jobs")
    # Forced override: jobs_dir line points under the workdir,
    # regardless of what the template hardcoded.
    assert f"jobs_dir: {expected_jobs_dir}" in yaml_text
    # And only one jobs_dir line — we replace, not append.
    jobs_lines = [
        line for line in yaml_text.splitlines() if line.lstrip().startswith("jobs_dir:")
    ]
    assert len(jobs_lines) == 1


def test_trajectory_result_fields_from_disk(tmp_path: Path) -> None:
    """Reading ``result.json`` off disk maps cleanly onto ``TrajectoryResult``."""
    task = _make_task(
        task_id="skillsbench/dialogue-parser",
        task_dir=str(tmp_path / "task_dialogue"),
    )
    rj = _make_result_json(
        task_name="dialogue-parser",
        reward=1.0,
        n_tool_calls=48,
        n_prompts=1,
        elapsed_s=469.7,
    )

    with patch("skill_evolve.agents.bench_cli.subprocess.run") as run_mock:
        run_mock.side_effect = _make_writer(tmp_path, result_json=rj)
        result = BenchCliBackend().run_task(
            task,
            skills_dir=None,
            model="claude-haiku-4-5",
            timeout_s=600,
            budget_usd=0.0,
            anonymize_map=None,
            workdir=tmp_path,
        )

    assert isinstance(result, TrajectoryResult)
    # Bug 14: ``task_id`` MUST be the manifest-canonical id (slash form),
    # NOT bench's ``result.json["task_name"]`` (the underscore-flattened
    # symlink basename). The outer aggregator buckets by this.
    assert result.task_id == "skillsbench/dialogue-parser"
    assert result.success is True
    assert result.score == pytest.approx(1.0)
    assert result.tool_calls == 48
    assert result.elapsed_s == pytest.approx(469.7)
    assert result.verified is True
    assert result.verifier_status == "passed"
    # bench does not emit cost; backend leaves it None.
    assert result.cost_usd is None
    # raw_completed should mirror "not partial_trajectory".
    assert result.raw_completed is True


def test_failed_run_clean_no_errors(tmp_path: Path) -> None:
    """reward<1.0 with no error → verified=False, verifier_status='failed'."""
    task = _make_task(task_id="x", task_dir=str(tmp_path / "task_x"))
    rj = _make_result_json(task_name="x", reward=0.5, error=None, verifier_error=None)

    with patch("skill_evolve.agents.bench_cli.subprocess.run") as run_mock:
        run_mock.side_effect = _make_writer(
            tmp_path, result_json=rj, pytest_tail="3 failed, 30 passed in 4.2s"
        )
        result = BenchCliBackend().run_task(
            task,
            skills_dir=None,
            model="claude-haiku-4-5",
            timeout_s=600,
            budget_usd=0.0,
            anonymize_map=None,
            workdir=tmp_path,
        )

    assert result.success is False
    assert result.verified is False
    assert result.verifier_status == "failed"
    assert result.score == pytest.approx(0.5)
    # pytest tail leaks into verifier_detail.
    assert "3 failed, 30 passed" in result.verifier_detail


def test_agent_error_classification(tmp_path: Path) -> None:
    """error!=None → verified=None, verifier_status='agent_error'."""
    task = _make_task(task_id="x", task_dir=str(tmp_path / "task_x"))
    rj = _make_result_json(
        task_name="x",
        reward=0.0,
        error="agent crashed: out of context",
        verifier_error=None,
    )

    with patch("skill_evolve.agents.bench_cli.subprocess.run") as run_mock:
        run_mock.side_effect = _make_writer(tmp_path, result_json=rj)
        result = BenchCliBackend().run_task(
            task,
            skills_dir=None,
            model="claude-haiku-4-5",
            timeout_s=600,
            budget_usd=0.0,
            anonymize_map=None,
            workdir=tmp_path,
        )

    assert result.success is False
    assert result.verified is None
    assert result.verifier_status == "agent_error"
    assert "agent crashed" in result.last_msg


def test_verifier_error_classification(tmp_path: Path) -> None:
    """verifier_error!=None → verified=None, status='verifier_unavailable'."""
    task = _make_task(task_id="x", task_dir=str(tmp_path / "task_x"))
    rj = _make_result_json(
        task_name="x",
        reward=0.0,
        error=None,
        verifier_error="docker exec failed: container died",
    )

    with patch("skill_evolve.agents.bench_cli.subprocess.run") as run_mock:
        run_mock.side_effect = _make_writer(tmp_path, result_json=rj)
        result = BenchCliBackend().run_task(
            task,
            skills_dir=None,
            model="claude-haiku-4-5",
            timeout_s=600,
            budget_usd=0.0,
            anonymize_map=None,
            workdir=tmp_path,
        )

    assert result.success is False
    assert result.verified is None
    assert result.verifier_status == "verifier_unavailable"
    assert "docker exec failed" in result.verifier_detail


def test_notes_capture_trajectory_source_and_n_prompts(tmp_path: Path) -> None:
    """notes carry trajectory_source plus n_prompts when n_prompts>1."""
    task = _make_task(task_id="x", task_dir=str(tmp_path / "task_x"))
    rj = _make_result_json(
        task_name="x", reward=1.0, n_prompts=3, trajectory_source="acp"
    )

    with patch("skill_evolve.agents.bench_cli.subprocess.run") as run_mock:
        run_mock.side_effect = _make_writer(tmp_path, result_json=rj)
        result = BenchCliBackend().run_task(
            task,
            skills_dir=None,
            model="claude-haiku-4-5",
            timeout_s=600,
            budget_usd=0.0,
            anonymize_map=None,
            workdir=tmp_path,
        )

    assert "trajectory_source=acp" in result.notes
    assert "n_prompts=3" in result.notes


def test_budget_usd_emits_warning_but_does_not_enforce(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """budget_usd>0 logs a warning; cost stays None (bench doesn't emit it)."""
    task = _make_task(task_id="x", task_dir=str(tmp_path / "task_x"))
    rj = _make_result_json(task_name="x", reward=1.0)

    with caplog.at_level("WARNING", logger="skill_evolve.agents.bench_cli"):
        with patch("skill_evolve.agents.bench_cli.subprocess.run") as run_mock:
            run_mock.side_effect = _make_writer(tmp_path, result_json=rj)
            result = BenchCliBackend().run_task(
                task,
                skills_dir=None,
                model="claude-haiku-4-5",
                timeout_s=600,
                budget_usd=0.50,
                anonymize_map=None,
                workdir=tmp_path,
            )

    assert any(
        "budget cap is enforced at runner level" in rec.message
        for rec in caplog.records
    )
    assert result.cost_usd is None
    # And the backend annotates notes so callers can grep for it.
    assert "budget_unenforced_at_backend" in result.notes


def test_timeout_handling(tmp_path: Path) -> None:
    """``subprocess.TimeoutExpired`` returns a clean failure outcome."""
    task = _make_task()

    with patch("skill_evolve.agents.bench_cli.subprocess.run") as run_mock:
        run_mock.side_effect = subprocess.TimeoutExpired(cmd="bench", timeout=5)
        result = BenchCliBackend().run_task(
            task,
            skills_dir=None,
            model="claude-haiku-4-5",
            timeout_s=5,
            budget_usd=1.0,
            anonymize_map=None,
            workdir=tmp_path,
        )

    assert result.success is False
    assert result.verifier_status == "timeout"
    assert result.verified is None
    assert "timeout" in (result.notes or "").lower()


def test_calledprocesserror_handling(tmp_path: Path) -> None:
    """Non-zero rc maps to ``verifier_status='bench_cli_error'``."""
    task = _make_task()

    with patch("skill_evolve.agents.bench_cli.subprocess.run") as run_mock:
        run_mock.return_value = _completed_proc(
            stdout="",
            stderr="bench: command not found",
            rc=2,
        )
        result = BenchCliBackend().run_task(
            task,
            skills_dir=None,
            model="claude-haiku-4-5",
            timeout_s=60,
            budget_usd=1.0,
            anonymize_map=None,
            workdir=tmp_path,
        )

    assert result.success is False
    assert result.verifier_status == "bench_cli_error"
    assert "bench: command not found" in (result.last_msg or "")


def test_result_json_missing_falls_back_to_stdout_score(tmp_path: Path) -> None:
    """If bench writes no result.json, fall back to parsing stdout score."""
    task = _make_task(task_id="x", task_dir=str(tmp_path / "task_x"))

    with patch("skill_evolve.agents.bench_cli.subprocess.run") as run_mock:
        # rc=0 but no result.json written anywhere; stdout has the score line.
        run_mock.return_value = _completed_proc(
            stdout="Score: 1/1 (100.0%), errors=0",
            rc=0,
        )
        result = BenchCliBackend().run_task(
            task,
            skills_dir=None,
            model="claude-haiku-4-5",
            timeout_s=60,
            budget_usd=0.0,
            anonymize_map=None,
            workdir=tmp_path,
        )

    assert result.success is True
    assert result.verified is True
    assert result.verifier_status == "passed"
    assert result.score == pytest.approx(1.0)
    assert "result_json_missing" in result.notes
    assert "stdout_parsed" in result.notes


def test_result_json_missing_and_unparseable_stdout_is_bench_cli_error(
    tmp_path: Path,
) -> None:
    """Missing result.json + unparseable stdout → bench_cli_error."""
    task = _make_task(task_id="x", task_dir=str(tmp_path / "task_x"))

    with patch("skill_evolve.agents.bench_cli.subprocess.run") as run_mock:
        run_mock.return_value = _completed_proc(
            stdout="garbage no score line here",
            rc=0,
        )
        result = BenchCliBackend().run_task(
            task,
            skills_dir=None,
            model="claude-haiku-4-5",
            timeout_s=60,
            budget_usd=0.0,
            anonymize_map=None,
            workdir=tmp_path,
        )

    assert result.success is False
    assert result.verifier_status == "bench_cli_error"
    assert "result_json_missing" in result.notes


def test_path_resolution_finds_result_under_unexpected_job_name(
    tmp_path: Path,
) -> None:
    """Backend finds result.json even when bench uses its own job_name.

    Regression guard for the fallback "newest subdir under jobs_dir"
    path: the writer in ``_make_writer`` always uses an unexpected
    job_name (``2099-99-99__99-99-99``), and the backend must still
    locate the result.
    """
    task = _make_task(task_id="x", task_dir=str(tmp_path / "task_x"))
    rj = _make_result_json(task_name="x", reward=0.833)

    with patch("skill_evolve.agents.bench_cli.subprocess.run") as run_mock:
        run_mock.side_effect = _make_writer(tmp_path, result_json=rj)
        result = BenchCliBackend().run_task(
            task,
            skills_dir=None,
            model="claude-haiku-4-5",
            timeout_s=600,
            budget_usd=0.0,
            anonymize_map=None,
            workdir=tmp_path,
        )

    assert result.task_id == "x"
    assert result.score == pytest.approx(0.833)
    # Score below 1.0 with no errors → verifier_status=failed.
    assert result.verifier_status == "failed"


def test_anonymize_map_applied_lazily(tmp_path: Path) -> None:
    """``anonymize_map`` must be applied to ``last_msg`` when present.

    If the SkillsBench anonymizer module hasn't been created yet
    (Group C of plan_0), the import is silently caught and
    ``last_msg`` is returned unmodified. If the module is present,
    real-id occurrences in ``last_msg`` are rewritten to the alias.
    """
    task = _make_task()
    rj = _make_result_json(
        task_name="forensics-disk-recovery",
        reward=0.0,
        error="could not solve forensics-disk-recovery",
    )
    anonymize_map = {"forensics-disk-recovery": "task_001"}

    # Try with a fake anonymizer module installed; if real one is
    # already present we use it instead.
    try:
        import skill_evolve.benchmark.skillsbench_anonymize as _sba  # noqa: F401

        anonymizer_present = True
    except ImportError:
        anonymizer_present = False

    with patch("skill_evolve.agents.bench_cli.subprocess.run") as run_mock:
        run_mock.side_effect = _make_writer(tmp_path, result_json=rj)
        result = BenchCliBackend().run_task(
            task,
            skills_dir=None,
            model="claude-haiku-4-5",
            timeout_s=60,
            budget_usd=1.0,
            anonymize_map=anonymize_map,
            workdir=tmp_path,
        )

    if anonymizer_present:
        # Real anonymizer should rewrite the leak.
        assert "forensics-disk-recovery" not in (result.last_msg or "")
        assert "task_001" in (result.last_msg or "")
    else:
        # Lazy-import path: anonymizer absent → unmodified last_msg.
        assert result.last_msg == "could not solve forensics-disk-recovery"


# ---------------------------------------------------------------------------
# 429 rate-limit handling (SG-3)
# ---------------------------------------------------------------------------


def test_is_rate_limit_error_predicate() -> None:
    """The 429 detector matches both substrings, ignoring case."""
    assert _is_rate_limit_error(_REAL_429_ERROR) is True
    assert _is_rate_limit_error("429 Too Many Requests") is True
    assert _is_rate_limit_error("rate_limit_error") is True
    assert _is_rate_limit_error("RATE_LIMIT_ERROR") is True
    # Should NOT match other agent errors.
    assert _is_rate_limit_error("agent crashed: out of context") is False
    assert _is_rate_limit_error("verifier docker exec failed") is False
    assert _is_rate_limit_error("HTTP 4290 something") is False  # \b boundary
    assert _is_rate_limit_error("") is False
    assert _is_rate_limit_error(None) is False


def test_classify_raises_on_429() -> None:
    """``_classify`` raises ``BenchRateLimitedError`` for 429-shaped errors."""
    with pytest.raises(BenchRateLimitedError) as ei:
        _classify(0.0, _REAL_429_ERROR, None)
    # The original error string is preserved on the exception so the
    # caller can stash it after exhausting retries.
    assert "429" in ei.value.original_error
    assert "rate_limit_error" in ei.value.original_error


def test_classify_passes_through_non_429_errors() -> None:
    """Non-429 errors keep returning ``agent_error`` (no behavior change)."""
    success, verified, status = _classify(0.0, "agent crashed", None)
    assert success is False
    assert verified is None
    assert status == "agent_error"


def test_429_triggers_retry_then_returns_error_after_max(tmp_path: Path) -> None:
    """5 consecutive 429s → 4 sleeps with exponential backoff, final agent_error."""
    task = _make_task(task_id="x", task_dir=str(tmp_path / "task_x"))

    # The writer is stateful across attempts: each attempt writes a
    # fresh ``result.json`` under a unique trial subdir, so the
    # backend's path resolver finds *the latest one* (it picks the
    # newest subdir under jobs_dir). We simply rewrite a single
    # 429-flavoured payload into a unique trial name on every call.
    call_counter = {"n": 0}

    def _writer(*args: Any, **kwargs: Any) -> MagicMock:
        call_counter["n"] += 1
        n = call_counter["n"]
        jobs_dir = tmp_path / "jobs"
        job_dir = jobs_dir / f"2099-99-99__99-99-{n:02d}"
        trial_dir = job_dir / f"x__attempt{n}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        rj = _make_result_json(task_name="x", reward=0.0, error=_REAL_429_ERROR)
        (trial_dir / "result.json").write_text(json.dumps(rj), encoding="utf-8")
        return _completed_proc(stdout="", stderr="", rc=0)

    with patch("skill_evolve.agents.bench_cli.subprocess.run") as run_mock:
        run_mock.side_effect = _writer
        with patch("skill_evolve.agents.bench_cli.time.sleep") as sleep_mock:
            result = BenchCliBackend().run_task(
                task,
                skills_dir=None,
                model="claude-haiku-4-5",
                timeout_s=600,
                budget_usd=0.0,
                anonymize_map=None,
                workdir=tmp_path,
            )

    # Five subprocess invocations: one initial + four retries.
    assert run_mock.call_count == 5
    # Four sleeps between them, with exponential backoff.
    assert sleep_mock.call_count == 4
    sleep_seconds = [call.args[0] for call in sleep_mock.call_args_list]
    assert sleep_seconds == [60, 120, 240, 480]

    # After max retries, the trial isn't lost: it's reported as a
    # rate-limited agent_error so the runner's summary still reflects
    # the 429 pressure.
    assert result.task_id == "x"
    assert result.success is False
    assert result.verified is None
    assert result.verifier_status == "agent_error"
    assert "rate_limited_max_retries" in (result.notes or "")
    # The original 429 message survives in last_msg (truncated to 500 chars).
    assert "429" in (result.last_msg or "")


def test_429_succeeds_on_2nd_attempt(tmp_path: Path) -> None:
    """One 429 then a clean run → final TrajectoryResult is the clean one."""
    task = _make_task(task_id="x", task_dir=str(tmp_path / "task_x"))

    call_counter = {"n": 0}

    def _writer(*args: Any, **kwargs: Any) -> MagicMock:
        call_counter["n"] += 1
        n = call_counter["n"]
        jobs_dir = tmp_path / "jobs"
        job_dir = jobs_dir / f"2099-99-99__99-99-{n:02d}"
        trial_dir = job_dir / f"x__attempt{n}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        if n == 1:
            rj = _make_result_json(task_name="x", reward=0.0, error=_REAL_429_ERROR)
        else:
            rj = _make_result_json(task_name="x", reward=1.0)
        (trial_dir / "result.json").write_text(json.dumps(rj), encoding="utf-8")
        return _completed_proc(stdout="", stderr="", rc=0)

    with patch("skill_evolve.agents.bench_cli.subprocess.run") as run_mock:
        run_mock.side_effect = _writer
        with patch("skill_evolve.agents.bench_cli.time.sleep") as sleep_mock:
            result = BenchCliBackend().run_task(
                task,
                skills_dir=None,
                model="claude-haiku-4-5",
                timeout_s=600,
                budget_usd=0.0,
                anonymize_map=None,
                workdir=tmp_path,
            )

    # Two subprocess calls (one 429, one clean), one 60s sleep between.
    assert run_mock.call_count == 2
    assert sleep_mock.call_count == 1
    assert sleep_mock.call_args_list[0].args[0] == 60

    # Result reflects the clean 2nd attempt — success, score=1.0.
    assert result.success is True
    assert result.verified is True
    assert result.verifier_status == "passed"
    assert result.score == pytest.approx(1.0)
    # ``notes`` records the recovery so callers can grep for retried trials.
    assert "recovered_after_429_retries=1" in (result.notes or "")


def test_429_no_retry_when_first_attempt_clean(tmp_path: Path) -> None:
    """A clean first attempt should NOT trigger any retries or sleeps."""
    task = _make_task(task_id="x", task_dir=str(tmp_path / "task_x"))
    rj = _make_result_json(task_name="x", reward=1.0)

    with patch("skill_evolve.agents.bench_cli.subprocess.run") as run_mock:
        run_mock.side_effect = _make_writer(tmp_path, result_json=rj)
        with patch("skill_evolve.agents.bench_cli.time.sleep") as sleep_mock:
            result = BenchCliBackend().run_task(
                task,
                skills_dir=None,
                model="claude-haiku-4-5",
                timeout_s=600,
                budget_usd=0.0,
                anonymize_map=None,
                workdir=tmp_path,
            )

    assert run_mock.call_count == 1
    assert sleep_mock.call_count == 0
    assert result.success is True
    # No retry annotation in notes when no retries happened.
    assert "recovered_after_429_retries" not in (result.notes or "")
