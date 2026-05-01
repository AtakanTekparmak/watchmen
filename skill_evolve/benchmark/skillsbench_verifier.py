"""Verifier shim for SkillsBench tasks.

Strategy
--------

benchflow 0.3.2 (the version pinned in this repo's lockfile) does NOT
expose a ``bench eval verify`` subcommand. The eval Typer subapp at
``benchflow/cli/eval.py`` only registers ``create``, ``list``, and
``retrieve``; the same is true of the legacy entrypoint
``benchflow/cli/main.py``. Verification of an already-edited workdir
must therefore go through the task's bundled Docker environment
directly.

Implementation
--------------

For each task we ``docker build`` the task's
``environment/Dockerfile`` into a local image
(``skillsbench-<dataset_task_name>:local``), then ``docker run`` that
image with the agent's edited workdir bind-mounted at ``/workdir`` and
execute the task's ``tests/test.sh``. The script writes a CTRF-format
JSON summary which we parse via the standard ``=== N passed ===``
pytest-summary regex (the upstream test.sh always tees pytest output
even when the JSON path fails).

If ``docker`` itself is missing on PATH we surface
``status='verifier_unavailable'`` so the dispatcher can route around it
(same shape ``verify_tblite`` / ``verify_swebench`` use).
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from skill_evolve.benchmark.load import Task
from skill_evolve.verifiers import VerifyResult

logger = logging.getLogger(__name__)


_PYTEST_SUMMARY_RE = re.compile(
    r"=+\s*"
    r"(?:(?P<passed>\d+)\s+passed)?"
    r"(?:[^=]*?(?P<failed>\d+)\s+failed)?"
    r"[^=]*?in\s+[\d.]+\s*s\s*=+",
)


def _parse_pytest_summary(output: str) -> Tuple[Optional[int], Optional[int]]:
    """Best-effort parse of ``=== N passed, M failed in X.Xs ===``.

    Returns ``(passed, failed)`` ints or (None, None) when no summary
    line could be located. Both fields may be partially present
    (e.g. ``=== 5 passed in 0.4s ===`` returns ``(5, 0)``).
    """
    if not output:
        return None, None
    matches = list(_PYTEST_SUMMARY_RE.finditer(output))
    if not matches:
        return None, None
    last = matches[-1]
    p = last.group("passed")
    f = last.group("failed")
    passed = int(p) if p is not None else 0
    failed = int(f) if f is not None else 0
    if passed == 0 and failed == 0:
        return None, None
    return passed, failed


def _verify_via_dockerfile(task: Task, run_dir: Path) -> VerifyResult:
    """``docker build`` the task's environment + run ``tests/test.sh``."""
    payload = task.success_check_payload
    image_tag = f"skillsbench-{task.extra.get('dataset_task_name', task.task_id.replace('/', '-'))}:local"
    environment_dir = payload["environment_dir"]
    tests_dir = payload["tests_dir"]

    try:
        build = subprocess.run(
            ["docker", "build", "-t", image_tag, str(environment_dir)],
            capture_output=True,
            text=True,
            timeout=task.timeout_s,
        )
    except FileNotFoundError:
        return VerifyResult(
            passed=None,
            status="verifier_unavailable",
            detail="docker not available on PATH",
        )
    except subprocess.TimeoutExpired:
        return VerifyResult(
            passed=False,
            status="docker_build_timeout",
            detail=f"docker build exceeded {task.timeout_s}s",
        )

    if build.returncode != 0:
        return VerifyResult(
            passed=False,
            status="docker_build_failed",
            detail=(build.stderr or "")[-500:],
        )

    test_sh = str(Path(tests_dir) / "test.sh")
    try:
        run = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{run_dir}:/workdir",
                "-w",
                "/workdir",
                image_tag,
                "bash",
                test_sh,
            ],
            capture_output=True,
            text=True,
            timeout=task.timeout_s,
            cwd="/workdir" if Path("/workdir").exists() else None,
        )
    except subprocess.TimeoutExpired:
        return VerifyResult(
            passed=False,
            status="test_sh_timeout",
            detail=f"docker run/test.sh exceeded {task.timeout_s}s",
        )

    out = (run.stdout or "") + (run.stderr or "")
    passed_n, failed_n = _parse_pytest_summary(out)
    score: Optional[float] = None
    score_detail = None
    if passed_n is not None and failed_n is not None:
        total = passed_n + failed_n
        if total > 0:
            score = passed_n / total
            score_detail = (passed_n, total)

    if run.returncode == 0:
        return VerifyResult(
            passed=True,
            status="passed",
            detail=out[-500:],
            score=1.0 if score is None else score,
            score_detail=score_detail,
        )
    return VerifyResult(
        passed=False,
        status="failed",
        detail=out[-500:],
        score=0.0 if score is None else score,
        score_detail=score_detail,
    )


def verify(task: Task, run_dir: Path) -> VerifyResult:
    """Run the SkillsBench verifier for ``task`` against ``run_dir``.

    Builds the task's ``environment/Dockerfile`` and runs
    ``tests/test.sh`` against the agent's edited workdir. See module
    docstring for why no ``bench eval verify`` primary path exists.
    """
    return _verify_via_dockerfile(task, run_dir)
