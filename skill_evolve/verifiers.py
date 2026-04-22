"""Real pass/fail verifiers for benchmark tasks.

This module replaces the old "agent finished without crashing" placeholder
with actual benchmark verification:

  * ``tblite_test_sh``       — run the task's bundled ``test.sh`` inside the
                                task's docker image (with the agent's
                                workspace bind-mounted as ``/app``) and
                                report ``exit_code == 0`` (or
                                ``/logs/verifier/reward.txt == "1"``).
  * ``swebench_patch_tests`` — extract a ``git diff`` from the agent's
                                workspace (which was pre-staged with the
                                instance's repo at ``base_commit``) and
                                hand it to the official ``swebench.harness``
                                runner. Pass = all FAIL_TO_PASS tests now
                                pass AND all PASS_TO_PASS tests still pass.

Hard requirement: a working Docker daemon. If Docker is unavailable the
dispatcher returns ``None`` for ``passed`` (third state — distinct from
"verified fail") so the evaluator can downgrade the score without lying
about the outcome. See :func:`verify_task` and :class:`VerifyResult`.

The agent is run on the host in a per-task workspace; for TBLite tasks the
workspace is seeded by extracting ``/app`` out of the task container before
the agent starts, then re-mounted into a fresh container at verification
time so the agent's edits are what test.sh sees. For SWE-bench tasks the
workspace is a ``git clone`` of the instance repo at the base commit.

Both staging routines live in :func:`stage_workspace`. The evaluator calls
that BEFORE ``run_agent.py`` so the agent operates on a real workspace
matching what the verifier will inspect.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------

@dataclass
class VerifyResult:
    """Outcome of a single verifier invocation.

    Attributes:
        passed: True (verified pass), False (verified fail), or None
            (verifier could not run — e.g. Docker daemon down). Three-state
            on purpose so callers can distinguish "the agent failed" from
            "we couldn't tell".
        status: short string for logging/JSON ("ok", "verifier_unavailable",
            "harness_error", "patch_extract_error", ...).
        detail: human-readable message (last 500 chars of stderr / report
            summary). Safe to dump in the per-task report.
        artifacts: paths to files the verifier wrote that may aid debugging
            (test logs, harness reports, extracted patch). Optional.
        score: continuous pass-rate in [0,1] when the verifier exposes a
            structured breakdown (pytest CTRF JSON or parsed "X passed,
            Y failed" summary). ``None`` when no breakdown is available —
            callers should treat None as "fall back to the binary
            ``passed`` signal". Adding this lifts evaluation noise from
            ±0.125 (binary pass/fail per task over 8 tasks) to roughly
            ±0.025 because a task with 10 assertions can now score 0.3
            instead of collapsing to 0.0.
        score_detail: optional (passed, total) tuple when the structured
            breakdown is available. Preserved for debugging; the composite
            only uses ``score``.
    """

    passed: Optional[bool]
    status: str = "ok"
    detail: str = ""
    artifacts: Dict[str, str] = None  # type: ignore[assignment]
    score: Optional[float] = None
    score_detail: Optional[Tuple[int, int]] = None

    def __post_init__(self) -> None:
        if self.artifacts is None:
            self.artifacts = {}


# ---------------------------------------------------------------------------
# Docker availability
# ---------------------------------------------------------------------------

_DOCKER_AVAILABLE_CACHE: Optional[bool] = None


def docker_available() -> bool:
    """Cheap probe — runs ``docker info`` once and caches the result.

    A return of False means we can't run TBLite or SWE-bench verifiers
    against real containers. Callers should fall back to
    ``passed=None, status="verifier_unavailable"``.
    """
    global _DOCKER_AVAILABLE_CACHE
    if _DOCKER_AVAILABLE_CACHE is not None:
        return _DOCKER_AVAILABLE_CACHE
    docker = shutil.which("docker")
    if docker is None:
        _DOCKER_AVAILABLE_CACHE = False
        return False
    try:
        proc = subprocess.run(
            [docker, "info"], capture_output=True, text=True, timeout=15,
        )
        _DOCKER_AVAILABLE_CACHE = (proc.returncode == 0)
    except Exception as exc:
        logger.warning("docker info probe failed: %s", exc)
        _DOCKER_AVAILABLE_CACHE = False
    return _DOCKER_AVAILABLE_CACHE


def _reset_docker_cache() -> None:
    """Test hook — re-probe Docker on next call."""
    global _DOCKER_AVAILABLE_CACHE
    _DOCKER_AVAILABLE_CACHE = None


# ---------------------------------------------------------------------------
# Workspace staging (called BEFORE the agent runs)
# ---------------------------------------------------------------------------

def stage_workspace(task: Dict[str, Any], workspace: Path) -> Dict[str, Any]:
    """Prepare ``workspace`` for the agent and return staging metadata.

    For TBLite: pull the docker image (if not present) and copy ``/app/``
    out of a throw-away container into ``workspace``. Returns a dict with
    ``container_app_dir`` so the verifier knows what to bind-mount back
    in.

    For SWE-bench: run ``git clone`` of the instance repo into
    ``workspace/repo`` and ``git checkout base_commit``. Returns a dict
    with ``repo_dir`` so the verifier knows where to compute ``git diff``.

    For unknown kinds (or if Docker is unavailable) returns an empty
    metadata dict and a warning is logged — the agent can still run
    against an empty workspace; the verifier will report
    ``verifier_unavailable``.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    kind = task.get("success_check_kind")
    payload = task.get("success_check_payload") or {}

    if kind == "tblite_test_sh":
        return _stage_tblite_workspace(payload, workspace)
    if kind == "swebench_patch_tests":
        return _stage_swebench_workspace(payload, workspace)
    logger.info("no staging for kind=%s; agent will run against empty cwd", kind)
    return {}


def _stage_tblite_workspace(
    payload: Dict[str, Any], workspace: Path,
) -> Dict[str, Any]:
    """Copy /app/ out of the task container into ``workspace``.

    The agent will then operate on these files; at verify time we mount
    ``workspace`` back to /app in a fresh container and run test.sh.
    """
    image = payload.get("docker_image")
    if not image:
        logger.warning("tblite task has no docker_image; cannot stage workspace")
        return {}
    if not docker_available():
        logger.warning("docker unavailable; skipping tblite workspace staging")
        return {"docker_image": image, "container_app_dir": "/app",
                "staging_skipped": True}

    cname = f"skillevolve-stage-{uuid.uuid4().hex[:10]}"
    try:
        # `docker create` doesn't start the container — just makes one we
        # can `cp` files out of. Faster + safer than `docker run sleep`.
        proc = subprocess.run(
            ["docker", "create", "--name", cname, image, "/bin/true"],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            logger.warning("docker create failed for %s: %s",
                           image, proc.stderr[-300:])
            return {"docker_image": image, "container_app_dir": "/app",
                    "staging_skipped": True,
                    "staging_error": proc.stderr[-300:]}

        # Copy /app/. (the dot copies *contents* not the parent dir).
        cp = subprocess.run(
            ["docker", "cp", f"{cname}:/app/.", str(workspace)],
            capture_output=True, text=True, timeout=600,
        )
        if cp.returncode != 0:
            logger.warning("docker cp /app failed for %s: %s",
                           image, cp.stderr[-300:])
            return {"docker_image": image, "container_app_dir": "/app",
                    "staging_skipped": True,
                    "staging_error": cp.stderr[-300:]}
    finally:
        subprocess.run(
            ["docker", "rm", "-f", cname],
            capture_output=True, text=True, timeout=60,
        )
    return {"docker_image": image, "container_app_dir": "/app"}


def _stage_swebench_workspace(
    payload: Dict[str, Any], workspace: Path,
) -> Dict[str, Any]:
    """Clone the instance repo at ``base_commit`` into ``workspace/repo``."""
    repo = payload.get("repo")
    base = payload.get("base_commit")
    if not (repo and base):
        logger.warning("swebench task missing repo/base_commit; skipping stage")
        return {}
    repo_dir = workspace / "repo"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    url = f"https://github.com/{repo}.git"
    proc = subprocess.run(
        ["git", "clone", "--quiet", url, str(repo_dir)],
        capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0:
        logger.warning("git clone %s failed: %s", url, proc.stderr[-300:])
        return {"repo_dir": str(repo_dir), "staging_skipped": True,
                "staging_error": proc.stderr[-300:]}
    co = subprocess.run(
        ["git", "-C", str(repo_dir), "checkout", "--quiet", base],
        capture_output=True, text=True, timeout=120,
    )
    if co.returncode != 0:
        logger.warning("git checkout %s failed: %s", base, co.stderr[-300:])
        return {"repo_dir": str(repo_dir), "staging_skipped": True,
                "staging_error": co.stderr[-300:]}
    return {"repo_dir": str(repo_dir), "base_commit": base, "repo": repo}


# ---------------------------------------------------------------------------
# TBLite verifier
# ---------------------------------------------------------------------------

_PYTEST_SUMMARY_RE = re.compile(
    r"=+\s*([^=]+?)\s+in\s+[\d.]+\s*s\s*=+\s*$", re.MULTILINE
)


# Stderr fingerprints that indicate the container lost DNS/network during
# apt-get. These come from colima networking hiccups, NOT from the agent's
# output being wrong, so we retry the test.sh invocation a few times
# before scoring the task as a real fail.
_NETWORK_ERROR_MARKERS = (
    "Could not resolve",
    "Temporary failure resolving",
    "Temporary failure in name resolution",
    "Network is unreachable",
    "Connection timed out",
    "No address associated with hostname",
)


def _looks_like_network_failure(stdout: str, stderr: str) -> bool:
    """True iff the test.sh stdout+stderr contains a DNS/network marker.

    We OR stdout and stderr because apt-get writes some errors to stdout
    with ``2>&1`` redirection in test scripts.
    """
    blob = (stdout or "") + "\n" + (stderr or "")
    return any(marker in blob for marker in _NETWORK_ERROR_MARKERS)


# Default retry count for test.sh invocations that fail on network. Each
# retry costs one `docker exec` + the full test.sh timeout so we keep it
# small. Overridden by tests.
_TEST_SH_NETWORK_RETRIES = 2


def _parse_pytest_stdout(stdout: str) -> Optional[Tuple[int, int]]:
    """Extract (passed, total_meaningful) from a pytest ``-rA`` tail.

    Matches lines like ``===== 3 passed, 1 failed in 0.5s =====`` and
    returns ``(passed, passed+failed+error)``. Skipped and xfailed tests
    are ignored (not counted as failures or successes) so scoring tracks
    the problem-under-test, not optional / expected-fail suites.

    Returns ``None`` if no summary line is present (collection error,
    container crash, or the test.sh never reached pytest). Callers should
    fall back to the binary signal in that case.
    """
    if not stdout:
        return None
    last_match = None
    for m in _PYTEST_SUMMARY_RE.finditer(stdout):
        last_match = m
    if last_match is None:
        return None
    counts: Dict[str, int] = {}
    for chunk in last_match.group(1).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        # "3 passed", "1 warning" — first token is the number, last is the label.
        parts = chunk.split()
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        counts[parts[-1]] = int(parts[0])
    passed = counts.get("passed", 0)
    failed = counts.get("failed", 0)
    errors = counts.get("error", 0) + counts.get("errors", 0)
    total = passed + failed + errors
    if total == 0:
        return None
    return passed, total


def _parse_ctrf(ctrf_text: str) -> Optional[Tuple[int, int]]:
    """Extract (passed, total) from a pytest-json-ctrf report.

    The CTRF format documents ``results.summary.{tests,passed,failed,...}``.
    We prefer ``passed + failed`` as the denominator (ignoring ``skipped``
    and ``pending``) to match the pytest-stdout parser. Returns ``None``
    if the JSON is malformed or counts are missing / zero.
    """
    try:
        data = json.loads(ctrf_text)
    except Exception:
        return None
    summary = ((data.get("results") or {}).get("summary")) or {}
    try:
        passed = int(summary.get("passed", 0))
        failed = int(summary.get("failed", 0))
        other = int(summary.get("other", 0))
    except (TypeError, ValueError):
        return None
    total = passed + failed + other
    if total == 0:
        return None
    return passed, total


def _extract_base64_tar(b64_data: str, target_dir: Path) -> None:
    """Inverse of upstream ``_extract_base64_tar`` — decode + extract.

    Keeps the same on-disk semantics as
    ``hermes-agent/.../terminalbench_2/terminalbench2_env.py`` so test
    payloads land in the directory the test scripts expect (``/tests``).
    """
    if not b64_data:
        return
    raw = base64.b64decode(b64_data)
    buf = io.BytesIO(raw)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        tar.extractall(target_dir)


def verify_tblite(
    task: Dict[str, Any],
    run_dir: Path,
    *,
    workspace_subdir: str = "workspace",
) -> VerifyResult:
    """Run the task's ``test.sh`` against the agent's edited workspace.

    Strategy:
      1. ``docker run --rm -v {workspace}:/app {image}`` so the agent's
         edits replace the container's pre-baked ``/app``.
      2. Inject ``test_sh`` + extracted ``tests_tar`` into ``/tests/`` via
         a tar piped on stdin to ``docker cp -``.
      3. Execute ``bash /tests/test.sh`` inside the container.
      4. Read ``/logs/verifier/reward.txt`` if present, else fall back to
         the pytest exit code.
    """
    if not docker_available():
        return VerifyResult(
            passed=None, status="verifier_unavailable",
            detail="docker daemon not reachable; install/start Docker",
        )

    payload = task.get("success_check_payload") or {}
    image = payload.get("docker_image")
    test_sh = payload.get("test_sh") or ""
    tests_tar = payload.get("tests_tar") or ""
    timeout = float(payload.get("test_timeout_sec") or 600)

    if not image:
        return VerifyResult(passed=False, status="missing_docker_image",
                            detail="task payload has no docker_image")
    if not test_sh:
        return VerifyResult(passed=False, status="missing_test_sh",
                            detail="task payload has no test_sh")

    workspace = run_dir / workspace_subdir
    workspace.mkdir(parents=True, exist_ok=True)

    # Stage tests on host so we can `docker cp` them into a long-running
    # container (mounting tests into /tests would shadow the entrypoint).
    tests_host = run_dir / "tests"
    if tests_host.exists():
        shutil.rmtree(tests_host)
    tests_host.mkdir(parents=True)
    if tests_tar:
        try:
            _extract_base64_tar(tests_tar, tests_host)
        except Exception as exc:
            return VerifyResult(passed=False, status="tests_tar_extract_error",
                                detail=f"failed to extract tests_tar: {exc}")
    (tests_host / "test.sh").write_text(test_sh)
    os.chmod(tests_host / "test.sh", 0o755)

    cname = f"skillevolve-verify-{uuid.uuid4().hex[:10]}"
    try:
        # Start container with workspace mounted at /app, sleeping so we
        # can `docker cp` tests in then exec test.sh.
        run_proc = subprocess.run(
            ["docker", "run", "-d", "--rm",
             "--name", cname,
             "-v", f"{workspace.resolve()}:/app",
             image, "sleep", "3600"],
            capture_output=True, text=True, timeout=120,
        )
        if run_proc.returncode != 0:
            return VerifyResult(
                passed=False, status="container_start_error",
                detail=run_proc.stderr[-500:],
            )

        # Inject /tests + /logs/verifier
        subprocess.run(["docker", "exec", cname, "mkdir", "-p",
                        "/tests", "/logs/verifier"],
                       capture_output=True, text=True, timeout=30)
        cp = subprocess.run(
            ["docker", "cp", f"{tests_host}/.", f"{cname}:/tests/"],
            capture_output=True, text=True, timeout=120,
        )
        if cp.returncode != 0:
            return VerifyResult(
                passed=False, status="tests_copy_error",
                detail=cp.stderr[-500:],
            )

        # Execute test.sh. Retry up to ``_TEST_SH_NETWORK_RETRIES`` times
        # if the failure stderr fingerprints as a DNS/network error — those
        # come from colima networking hiccups during apt-get, not from the
        # agent's output being wrong, so we do NOT want to score them as
        # real fails. Any non-network failure returns on the first attempt.
        network_retry_count = 0
        network_retry_details: List[str] = []
        tproc: subprocess.CompletedProcess[str]
        try:
            for attempt in range(_TEST_SH_NETWORK_RETRIES + 1):
                tproc = subprocess.run(
                    ["docker", "exec", cname, "bash", "/tests/test.sh"],
                    capture_output=True, text=True,
                    timeout=timeout,
                )
                if tproc.returncode == 0:
                    break
                if not _looks_like_network_failure(tproc.stdout, tproc.stderr):
                    break
                network_retry_count += 1
                network_retry_details.append(
                    f"attempt {attempt + 1}: rc={tproc.returncode} "
                    f"(network fingerprint matched)"
                )
                logger.warning(
                    "verify_tblite(%s): test.sh attempt %d hit network "
                    "error; retrying (%d/%d).",
                    task.get("task_id"), attempt + 1,
                    network_retry_count, _TEST_SH_NETWORK_RETRIES,
                )
        except subprocess.TimeoutExpired:
            return VerifyResult(
                passed=False, status="test_sh_timeout",
                detail=f"test.sh exceeded {timeout}s",
            )
        exit_code = tproc.returncode
        out_tail = (tproc.stdout or "")[-400:] + (tproc.stderr or "")[-400:]

        # If all retries exhausted AND the final attempt still fingerprints
        # as network-layer failure, report it under a distinct status so
        # callers can filter infrastructure noise from real task fails.
        if (
            exit_code != 0
            and network_retry_count >= _TEST_SH_NETWORK_RETRIES
            and _looks_like_network_failure(tproc.stdout, tproc.stderr)
        ):
            detail = (
                f"network failure persisted across "
                f"{network_retry_count + 1} attempts; " +
                "; ".join(network_retry_details) +
                f"\nlast stderr tail: {(tproc.stderr or '')[-300:]}"
            )
            return VerifyResult(
                passed=False,
                status="network_fail",
                detail=detail,
                score=None,
            )

        # Continuous score: prefer pytest-json-ctrf (harder tasks opt into
        # this at --ctrf /logs/verifier/ctrf.json), else parse the pytest
        # stdout summary. Collapses to None when neither is available so
        # the caller falls back to the binary signal.
        score: Optional[float] = None
        score_detail: Optional[Tuple[int, int]] = None
        ctrf_proc = subprocess.run(
            ["docker", "exec", cname, "cat", "/logs/verifier/ctrf.json"],
            capture_output=True, text=True, timeout=30,
        )
        if ctrf_proc.returncode == 0 and ctrf_proc.stdout.strip():
            parsed = _parse_ctrf(ctrf_proc.stdout)
            if parsed is not None:
                passed_n, total_n = parsed
                score = passed_n / total_n
                score_detail = parsed
        if score is None:
            parsed = _parse_pytest_stdout(tproc.stdout or "")
            if parsed is not None:
                passed_n, total_n = parsed
                score = passed_n / total_n
                score_detail = parsed

        # Prefer reward.txt over exit code (matches upstream TBLite logic).
        reward_proc = subprocess.run(
            ["docker", "exec", cname, "cat", "/logs/verifier/reward.txt"],
            capture_output=True, text=True, timeout=30,
        )
        reward_txt = reward_proc.stdout.strip() if reward_proc.returncode == 0 else ""

        def _score_detail_suffix() -> str:
            if score_detail is None:
                return ""
            return f" score={score_detail[0]}/{score_detail[1]}"

        if reward_txt in {"1", "1.0"}:
            return VerifyResult(
                passed=True, status="ok",
                detail=f"reward.txt=1 (exit={exit_code}){_score_detail_suffix()}",
                score=score if score is not None else 1.0,
                score_detail=score_detail,
            )
        if reward_txt in {"0", "0.0"}:
            return VerifyResult(
                passed=False, status="test_failed",
                detail=(
                    f"reward.txt=0 (exit={exit_code}){_score_detail_suffix()}\n"
                    f"{out_tail}"
                ),
                score=score if score is not None else 0.0,
                score_detail=score_detail,
            )
        try:
            rv = float(reward_txt)
            return VerifyResult(
                passed=rv >= 0.5, status="ok",
                detail=f"reward.txt={reward_txt} (exit={exit_code}){_score_detail_suffix()}",
                score=score if score is not None else rv,
                score_detail=score_detail,
            )
        except (ValueError, TypeError):
            passed = (exit_code == 0)
            return VerifyResult(
                passed=passed, status="ok" if passed else "test_failed",
                detail=f"no reward.txt; exit={exit_code}{_score_detail_suffix()}\n{out_tail}",
                score=score if score is not None else (1.0 if passed else 0.0),
                score_detail=score_detail,
            )
    finally:
        subprocess.run(["docker", "rm", "-f", cname],
                       capture_output=True, text=True, timeout=60)


# ---------------------------------------------------------------------------
# SWE-bench verifier
# ---------------------------------------------------------------------------

def extract_git_diff(repo_dir: Path) -> str:
    """Return ``git diff`` (unstaged + staged) for ``repo_dir``.

    We add untracked files first so ``git diff HEAD`` includes them. The
    output is the same shape as a SWE-bench gold ``patch`` field, suitable
    for piping into ``git apply`` or the swebench harness.
    """
    if not (repo_dir / ".git").exists():
        return ""
    # Stage untracked + tracked changes so a single `diff --cached` covers
    # both. We avoid `git add -A` here because we don't want to disturb
    # the working tree state for callers; use a temporary index instead.
    #
    # Simpler approach: `git add -N` (intent-to-add) makes untracked files
    # appear in `git diff` without staging their contents.
    subprocess.run(
        ["git", "-C", str(repo_dir), "add", "-N", "."],
        capture_output=True, text=True, timeout=60,
    )
    proc = subprocess.run(
        ["git", "-C", str(repo_dir), "diff", "--no-color", "HEAD"],
        capture_output=True, text=True, timeout=120,
    )
    return proc.stdout


def verify_swebench(
    task: Dict[str, Any],
    run_dir: Path,
    *,
    workspace_subdir: str = "workspace",
) -> VerifyResult:
    """Apply the agent's diff and run the official SWE-bench harness.

    Requires ``pip install swebench`` and a working Docker daemon. The
    harness pulls (or builds) per-instance images on first run and caches
    them — first invocation per instance can take 10+ minutes.
    """
    if not docker_available():
        return VerifyResult(
            passed=None, status="verifier_unavailable",
            detail="docker daemon not reachable; install/start Docker",
        )
    try:
        from swebench.harness.run_evaluation import main as swe_main  # noqa: F401
    except ImportError as exc:
        return VerifyResult(
            passed=None, status="verifier_unavailable",
            detail=f"swebench not installed: {exc}; pip install swebench",
        )

    payload = task.get("success_check_payload") or {}
    instance_id = task.get("extra", {}).get("instance_id") or \
        task["task_id"].split("/", 1)[-1]

    repo_dir = run_dir / workspace_subdir / "repo"
    if not repo_dir.exists():
        return VerifyResult(
            passed=False, status="repo_missing",
            detail=f"expected staged repo at {repo_dir}; "
                   "did stage_workspace run?",
        )
    patch = extract_git_diff(repo_dir)
    if not patch.strip():
        return VerifyResult(
            passed=False, status="empty_patch",
            detail="agent produced no diff against base_commit",
        )

    # Persist the patch for debugging / replay.
    patch_path = run_dir / "agent.patch"
    patch_path.write_text(patch)

    # SWE-bench harness needs a predictions JSONL with one record.
    preds = [{
        "instance_id": instance_id,
        "model_patch": patch,
        "model_name_or_path": "skill_evolve",
    }]
    preds_path = run_dir / "predictions.jsonl"
    with preds_path.open("w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")

    report_dir = run_dir / "swebench_report"
    report_dir.mkdir(exist_ok=True)
    run_id = f"skillevolve-{uuid.uuid4().hex[:10]}"

    # We run the harness via its CLI so import-time side effects don't
    # leak into the parent (it monkey-patches docker, sets ulimits, etc.).
    cmd = [
        "python", "-m", "swebench.harness.run_evaluation",
        "--dataset_name", "princeton-nlp/SWE-bench_Verified",
        "--predictions_path", str(preds_path),
        "--max_workers", "1",
        "--instance_ids", instance_id,
        "--run_id", run_id,
        "--report_dir", str(report_dir),
        "--cache_level", "env",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=int(task.get("timeout_s", 1800)),
        )
    except subprocess.TimeoutExpired:
        return VerifyResult(
            passed=False, status="harness_timeout",
            detail="swebench harness exceeded task timeout",
            artifacts={"patch": str(patch_path)},
        )

    # Parse the harness report. The harness writes
    # {report_dir}/{model}.{run_id}.json with resolved instance lists.
    report_glob = list(report_dir.glob("*.json"))
    if not report_glob:
        return VerifyResult(
            passed=False, status="harness_no_report",
            detail=f"no report json under {report_dir}; "
                   f"stderr_tail={proc.stderr[-500:]}",
            artifacts={"patch": str(patch_path)},
        )
    report = json.loads(report_glob[0].read_text())
    resolved = report.get("resolved_ids") or []
    passed = instance_id in resolved
    return VerifyResult(
        passed=passed,
        status="ok" if passed else "test_failed",
        detail=(f"resolved={resolved} report={report_glob[0].name} "
                f"rc={proc.returncode}"),
        artifacts={"patch": str(patch_path),
                   "report": str(report_glob[0])},
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_VERIFIERS = {
    "tblite_test_sh": verify_tblite,
    "swebench_patch_tests": verify_swebench,
}


def verify_task(task: Dict[str, Any], run_dir: Path) -> VerifyResult:
    """Dispatch to the right verifier based on ``success_check_kind``.

    Unknown kinds return ``passed=None`` so the cascade can downgrade
    cleanly without crashing the whole eval. The Docker-unavailable
    fallback lives in each verifier so it can craft a kind-specific
    detail message.
    """
    kind = task.get("success_check_kind")
    fn = _VERIFIERS.get(kind)
    if fn is None:
        return VerifyResult(
            passed=None, status="unknown_kind",
            detail=f"no verifier registered for kind={kind!r}",
        )
    try:
        return fn(task, run_dir)
    except Exception as exc:  # pragma: no cover — verifier crashes
        logger.exception("verifier %s crashed", kind)
        return VerifyResult(
            passed=None, status="verifier_crashed",
            detail=f"{type(exc).__name__}: {exc}",
        )
