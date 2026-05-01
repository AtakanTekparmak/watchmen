"""Bench CLI agent backend — subprocess-wraps ``bench eval create``.

This is the only path used by the SkillsBench harness (per
``plans/plan_0.md`` D-1). The harness invokes the ``claude-code`` agent
(Anthropic's first-party CLI, pinned to the paper's version 2.1.19) —
registered at runtime by ``skill_evolve.agents.register_claude_code``,
which is imported by the subprocess shim before ``benchflow.cli.main``
resolves ``-a`` against its registry. The agent runs with skills mounted
via the materialized scene YAML, and writes per-task artifacts to disk
under ``<jobs_dir>/<job_name>/<trial_name>/`` (notably ``result.json``).

Argv shape:

    bench eval create -f <yaml> -t <task_dir> -a claude-code -m <model>

The ``-f`` YAML is materialized per-task from one of the templates in
``skill_evolve/skillsbench/scenes/`` (with-skills vs no-skills). The
``-t`` task_dir flows from the hydrated task's success-check payload.

bench's stdout is just a one-line ``Score: N/M (X%), errors=K`` summary
— structured per-task data lives on disk:

    <jobs_dir>/<job_name>/<task_name>__<hash>/result.json
    <jobs_dir>/summary.json

We control ``jobs_dir`` and ``job_name`` ourselves (via the rendered
YAML and a pre-computed timestamped subdir respectively) so we always
know where to look. A robust fallback walks ``<jobs_dir>`` for the
newest job + sole result.json if bench overrides our path.

Cost note: bench does not currently emit per-call USD cost. The
``budget_usd`` kwarg is preserved for forward-compat but is **not
enforced** by this backend — Phase D's runner enforces budget caps at
the runner level via Anthropic billing / token-count estimation.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from skill_evolve.agents.base import AgentBackend, TrajectoryResult


# Shim invoked by ``python -c`` so SG-1 + SG-2 are installed in the bench
# subprocess *before* ``benchflow.cli.main`` imports ``deploy_skills`` or
# resolves ``-a`` against its agent registry:
#
#   * ``register_claude_code`` (SG-1) registers the ``claude-code`` agent —
#     not in benchflow's built-in registry — so ``-a claude-code`` resolves.
#   * ``_benchflow_patches`` (SG-2) wraps ``deploy_skills`` to chown the
#     sandbox user's home dir after the root-side ``cp -r`` of skills.
#
# Without these pre-imports the parent process's runtime registrations and
# monkey-patches don't reach the bench subprocess (which loads its own
# benchflow module copy).
_BENCH_SHIM_CODE = (
    "import sys; "
    "import skill_evolve.agents.register_claude_code as _cc; "
    "from benchflow.agents.registry import AGENTS as _A; "
    "assert 'claude-code' in _A, 'SG-1 claude-code registration failed'; "
    "import skill_evolve.agents._benchflow_patches as _p; "
    "assert _p.is_applied(), 'SG-2 deploy_skills patch failed to apply'; "
    "from benchflow.cli.main import app; "
    "sys.exit(app())"
)


def _bench_cli_argv(
    yaml_path: "Path | str",
    task_dir: str,
    model: str,
) -> List[str]:
    """Build the argv that runs bench through the SG-1/SG-2 shim.

    Uses ``sys.executable -c`` instead of the bare ``bench`` console
    script so:

      * ``claude-code`` is registered in the subprocess's
        ``benchflow.agents.registry.AGENTS`` (SG-1) before the CLI
        resolves ``-a``.
      * Our monkey-patch on ``benchflow._agent_setup.deploy_skills``
        (SG-2) is installed before the bench CLI pulls in
        ``benchflow.trial``/``benchflow.sdk`` (which import
        ``deploy_skills`` by name).
    """
    return [
        sys.executable,
        "-c",
        _BENCH_SHIM_CODE,
        "eval",
        "create",
        "-f",
        str(yaml_path),
        "-t",
        str(task_dir),
        "-a",
        "claude-code",
        "-m",
        model,
    ]


if TYPE_CHECKING:  # pragma: no cover
    pass


_log = logging.getLogger(__name__)


# 429 rate-limit detection. The error string we see on disk looks like:
#   "ACP error -32603: Internal error: API Error: 429 {\"type\":\"error\",
#    \"error\":{\"type\":\"rate_limit_error\",\"message\":\"...20,000,000
#    prompt bytes per hour...\"}}"
# Either substring is a strong signal; we match on the first hit.
_RATE_LIMIT_RE = re.compile(r"\b429\b|rate_limit_error", re.IGNORECASE)


# Exponential backoff schedule (in seconds) for 429 retries inside
# ``BenchCliBackend.run_task``. Four retries total: 60s → 120s → 240s →
# 480s. Tests patch ``time.sleep`` to keep the suite fast.
_RATE_LIMIT_BACKOFF_S: Tuple[int, ...] = (60, 120, 240, 480)


class BenchRateLimitedError(Exception):
    """Raised when bench's ``result.json`` reports an Anthropic 429.

    The classifier (``_classify``) raises this in lieu of returning an
    ``agent_error`` ``TrajectoryResult`` so the caller can decide
    whether to retry-with-backoff or surface the failure. After the
    backoff schedule is exhausted, the wrapper falls back to the
    original ``agent_error`` shape with ``notes="rate_limited_max_retries"``.
    """

    def __init__(self, message: str, *, original_error: str) -> None:
        super().__init__(message)
        self.original_error = original_error


def _is_rate_limit_error(error: Optional[str]) -> bool:
    """True if ``error`` looks like an Anthropic 429 rate-limit message."""
    if not error:
        return False
    return bool(_RATE_LIMIT_RE.search(str(error)))


# Resolve the scenes dir relative to this file. The plan places
# templates at ``skill_evolve/skillsbench/scenes/`` (sibling package
# under ``skill_evolve``), so we walk up to the package root.
_PKG_ROOT = Path(__file__).resolve().parent.parent
_SCENES_DIR = _PKG_ROOT / "skillsbench" / "scenes"


# bench stdout fast-sanity-check pattern, e.g.:
#   "Score: 0/1 (0.0%), errors=0"
#   "Score: 1/1 (100.0%), errors=0"
_STDOUT_SCORE_RE = re.compile(
    r"Score:\s*(?P<passed>\d+)\s*/\s*(?P<total>\d+)\s*"
    r"\(\s*(?P<pct>[\d.]+)%\s*\)\s*,\s*errors\s*=\s*(?P<errors>\d+)"
)


def _load_scene_template(with_skills: bool) -> str:
    """Read the appropriate scene YAML template."""
    name = (
        "baseline_with_skills.yaml.tmpl"
        if with_skills
        else "baseline_no_skills.yaml.tmpl"
    )
    path = _SCENES_DIR / name
    if not path.exists():
        # Fall back to a minimal inline template so the backend is
        # self-contained when the scenes/ folder hasn't been laid down
        # yet (Group A of plan_0 ships those templates separately).
        if with_skills:
            return (
                "tasks_dir: <task_dir>\n"
                "jobs_dir: <jobs_dir>\n"
                "agent: claude-code\n"
                "model: <model>\n"
                "environment: docker\n"
                "concurrency: 1\n"
                "max_retries: 0\n"
                "sandbox_user: agent\n"
                "skills_dir: <skills_dir>\n"
            )
        return (
            "tasks_dir: <task_dir>\n"
            "jobs_dir: <jobs_dir>\n"
            "agent: claude-code\n"
            "model: <model>\n"
            "environment: docker\n"
            "concurrency: 1\n"
            "max_retries: 0\n"
            "sandbox_user: agent\n"
        )
    return path.read_text(encoding="utf-8")


def _force_jobs_dir(yaml_text: str, jobs_dir: str) -> str:
    """Forcibly override the rendered YAML's ``jobs_dir`` to a known path.

    The repo's templates hardcode ``jobs_dir: runs/baseline/jobs`` (and
    similar) — that's fine for the bash smoke driver but defeats this
    backend's path-resolution because bench then writes results
    relative to the caller's CWD. We rewrite the line to a workdir-
    relative path we own, so ``result.json`` is at a deterministic
    location.
    """
    lines = yaml_text.splitlines()
    out: List[str] = []
    seen = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("jobs_dir:"):
            out.append(f"jobs_dir: {jobs_dir}")
            seen = True
        else:
            out.append(line)
    if not seen:
        out.append(f"jobs_dir: {jobs_dir}")
    return "\n".join(out) + ("\n" if yaml_text.endswith("\n") else "")


def _render_scene(
    template: str,
    *,
    task_dir: str,
    model: str,
    skills_dir: Optional[str],
    jobs_dir: str,
) -> str:
    """Substitute placeholders and force ``jobs_dir`` to a known path."""
    out = template.replace("<task_dir>", task_dir).replace("<model>", model)
    if skills_dir is not None:
        out = out.replace("<skills_dir>", skills_dir)
    out = out.replace("<jobs_dir>", jobs_dir)
    out = _force_jobs_dir(out, jobs_dir)
    return out


def _stage_single_task_wrapper(workdir: Path, task_id: str, task_dir: str) -> Path:
    """Create a wrapper dir containing a single symlink to the real task.

    benchflow's ``Job._get_task_dirs`` (job.py:355) expects ``tasks_dir`` to
    be a parent directory whose children each have ``task.toml``. To run
    one task at a time via ``-f <yaml>`` we stage
    ``<workdir>/tasks/<safe_id>/`` as a symlink to the real task dir.
    """
    safe_id = task_id.replace("/", "_")
    wrapper_root = workdir / "tasks" / safe_id
    wrapper_root.mkdir(parents=True, exist_ok=True)
    link = wrapper_root / safe_id
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path(task_dir).resolve(), target_is_directory=True)
    return wrapper_root


def _materialize_scene_yaml(
    workdir: Path,
    *,
    task_id: str,
    task_dir: str,
    model: str,
    skills_dir: Optional[str],
    jobs_dir: Path,
) -> Path:
    """Write the per-task scene YAML to ``<workdir>/scenes/<task_id>.yaml``.

    The YAML's ``tasks_dir`` is set to a single-task wrapper dir (parent
    containing one symlinked task), since benchflow's ``Job.from_yaml`` /
    ``Job._get_task_dirs`` expect a directory of tasks. ``jobs_dir`` is
    forced to a workdir-relative path so we always know where bench
    writes ``result.json``.
    """
    scenes = workdir / "scenes"
    scenes.mkdir(parents=True, exist_ok=True)
    template = _load_scene_template(with_skills=skills_dir is not None)
    wrapper_dir = _stage_single_task_wrapper(workdir, task_id, task_dir)
    rendered = _render_scene(
        template,
        task_dir=str(wrapper_dir),
        model=model,
        skills_dir=skills_dir,
        jobs_dir=str(jobs_dir),
    )
    # ``task_id`` may contain slashes (e.g. ``skillsbench/<id>``);
    # normalize to a flat filename.
    safe_id = task_id.replace("/", "_")
    yaml_path = scenes / f"{safe_id}.yaml"
    yaml_path.write_text(rendered, encoding="utf-8")
    return yaml_path


def _apply_anonymize(
    text: str,
    anonymize_map: Optional[Dict[str, str]],
) -> str:
    """Sanitize ``text`` via the SkillsBench anonymizer, if available.

    Imports the anonymizer module lazily so the backend stays usable
    even when Group C of plan_0 (which ships
    ``skill_evolve.benchmark.skillsbench_anonymize``) hasn't landed yet.
    """
    if not anonymize_map or not text:
        return text
    try:
        from skill_evolve.benchmark.skillsbench_anonymize import (
            sanitize_text_skillsbench,
        )
    except ImportError:
        return text
    try:
        # ``sanitize_text_skillsbench`` returns a ``(sanitized, hits)``
        # tuple; we only need the sanitized string here.
        result = sanitize_text_skillsbench(text, anonymize_map)
    except Exception:
        # If the anonymizer raises for any reason, prefer leaving the
        # text un-redacted over crashing the backend mid-eval.
        return text
    if isinstance(result, tuple):
        return result[0]
    return result  # forward-compat if the helper ever returns a bare str


def _task_dir_from_task(task: Any) -> str:
    """Pull the task_dir out of the hydrated task's payload."""
    payload = getattr(task, "success_check_payload", None)
    if payload is None and isinstance(task, dict):
        payload = task.get("success_check_payload") or {}
    payload = payload or {}
    td = payload.get("task_dir")
    if td:
        return str(td)
    # Fall back to ``environment_dir`` / ``tests_dir`` parents if the
    # SkillsBench loader populated those instead.
    for k in ("environment_dir", "tests_dir"):
        v = payload.get(k)
        if v:
            return str(Path(v).parent)
    raise KeyError(
        "BenchCliBackend.run_task: task payload missing 'task_dir' "
        "(set success_check_payload['task_dir'] when hydrating)."
    )


def _locate_result_json(
    jobs_dir: Path,
    expected_job_name: str,
) -> Optional[Path]:
    """Find the per-task ``result.json`` under ``jobs_dir``.

    Resolution order:

      1. ``jobs_dir / expected_job_name / */result.json`` — the path we
         pre-computed and passed to bench. This is the happy path.
      2. Newest subdir under ``jobs_dir`` (by mtime) — covers the case
         where bench rewrote our ``job_name`` (it shouldn't, but the
         YAML schema currently doesn't surface a ``job_name`` field).
      3. Recursive ``jobs_dir.rglob("result.json")`` — last-ditch.

    Returns ``None`` if no ``result.json`` could be located.
    """
    if not jobs_dir.exists():
        return None

    # 1. expected job_name path
    expected = jobs_dir / expected_job_name
    if expected.is_dir():
        for child in sorted(expected.iterdir()):
            if child.is_dir():
                rj = child / "result.json"
                if rj.is_file():
                    return rj

    # 2. newest subdir under jobs_dir
    subdirs = [d for d in jobs_dir.iterdir() if d.is_dir()]
    if subdirs:
        subdirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        for job_dir in subdirs:
            for child in sorted(job_dir.iterdir()):
                if child.is_dir():
                    rj = child / "result.json"
                    if rj.is_file():
                        return rj

    # 3. recursive rglob fallback
    for rj in jobs_dir.rglob("result.json"):
        if rj.is_file():
            return rj

    return None


def _parse_stdout_score(stdout: str) -> Optional[Dict[str, Any]]:
    """Parse bench's ``Score: N/M (X%), errors=K`` line as a fallback."""
    if not stdout:
        return None
    m = _STDOUT_SCORE_RE.search(stdout)
    if not m:
        return None
    return {
        "passed": int(m.group("passed")),
        "total": int(m.group("total")),
        "errors": int(m.group("errors")),
    }


def _read_pytest_tail(trial_dir: Path, max_chars: int = 500) -> str:
    """Return the tail of ``verifier/pytest_output.txt`` for verifier_detail."""
    pytest_path = trial_dir / "verifier" / "pytest_output.txt"
    if not pytest_path.is_file():
        return ""
    try:
        text = pytest_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def _classify(
    reward: float,
    error: Optional[str],
    verifier_error: Optional[str],
) -> Tuple[bool, Optional[bool], str]:
    """Map ``reward``/``error``/``verifier_error`` → (success, verified, status).

    Per the plan's mapping table:
      success  = error is None and verifier_error is None and reward >= 1.0
      verified = True  if error is None and verifier_error is None
                 None  if error is not None       (errored)
                 False if test failed (reward<1.0, no errors)
      verifier_status = "passed" if reward>=1.0 (clean run)
                        "failed" if reward<1.0 (clean run)
                        "verifier_unavailable" if verifier_error
                        "agent_error" if error

    Special-case: when ``error`` looks like an Anthropic 429 rate-limit
    response, raise :class:`BenchRateLimitedError` so the calling
    backend can decide whether to retry-with-backoff. Other agent
    errors fall through to the existing ``agent_error`` status.
    """
    if error is not None:
        if _is_rate_limit_error(error):
            raise BenchRateLimitedError(
                "bench result.json reports a 429 rate_limit_error",
                original_error=str(error),
            )
        return (False, None, "agent_error")
    if verifier_error is not None:
        return (False, None, "verifier_unavailable")
    if reward >= 1.0:
        return (True, True, "passed")
    return (False, False, "failed")


def _result_to_trajectory(
    *,
    task_id: str,
    result_json: Dict[str, Any],
    trial_dir: Path,
    anonymize_map: Optional[Dict[str, str]],
    budget_usd: Optional[float],
) -> TrajectoryResult:
    """Project bench's per-task ``result.json`` onto ``TrajectoryResult``."""
    rewards = result_json.get("rewards") or {}
    try:
        reward = float(rewards.get("reward", 0.0) or 0.0)
    except (TypeError, ValueError):
        reward = 0.0

    error = result_json.get("error")
    verifier_error = result_json.get("verifier_error")
    success, verified, verifier_status = _classify(reward, error, verifier_error)

    timing = result_json.get("timing") or {}
    try:
        elapsed_s = float(timing.get("total", 0.0) or 0.0)
    except (TypeError, ValueError):
        elapsed_s = 0.0

    try:
        tool_calls = int(result_json.get("n_tool_calls", 0) or 0)
    except (TypeError, ValueError):
        tool_calls = 0

    # last_msg: prefer agent error, then verifier error, then ""
    last_msg_src = ""
    if error:
        last_msg_src = str(error)
    elif verifier_error:
        last_msg_src = str(verifier_error)
    last_msg = _apply_anonymize(last_msg_src[:500], anonymize_map)

    # verifier_detail: stringified verifier_error or pytest_output.txt tail
    if verifier_error:
        verifier_detail = str(verifier_error)
    else:
        verifier_detail = _read_pytest_tail(trial_dir)

    # Notes: trajectory_source, n_prompts (if >1), and budget warning.
    notes_parts: List[str] = []
    trajectory_source = result_json.get("trajectory_source")
    if trajectory_source:
        notes_parts.append(f"trajectory_source={trajectory_source}")
    try:
        n_prompts = int(result_json.get("n_prompts", 1) or 1)
    except (TypeError, ValueError):
        n_prompts = 1
    if n_prompts > 1:
        notes_parts.append(f"n_prompts={n_prompts}")
    # bench does not emit cost; budget_usd is unenforced by this
    # backend. The runner is responsible for cumulative budget caps.
    if budget_usd is not None and budget_usd > 0:
        notes_parts.append("budget_unenforced_at_backend")
    notes = "; ".join(notes_parts)

    # Bug 14: bench's ``result.json["task_name"]`` is the underscore-flattened
    # symlink name (e.g. ``skillsbench_citation-check``), but the manifest
    # canonical id is slash-form (e.g. ``skillsbench/citation-check``). The
    # outer aggregator (skill_evolve/evaluator.py:_run_batch) buckets outcomes
    # by the result's ``task_id`` and looks up by manifest form — using the
    # bench-flattened name silently misses every bucket and zeroes the
    # composite. Always preserve the manifest-canonical id we were called with.
    return TrajectoryResult(
        task_id=task_id,
        success=success,
        tool_calls=tool_calls,
        elapsed_s=elapsed_s,
        skills_invoked=[],  # bench doesn't surface this directly
        last_msg=last_msg,
        raw_completed=not bool(result_json.get("partial_trajectory", False)),
        notes=notes,
        verified=verified,
        verifier_status=verifier_status,
        verifier_detail=verifier_detail[:500],
        score=reward,
        repeats_detail=[],
        cost_usd=None,  # bench does not emit per-call cost (see module docstring)
    )


class BenchCliBackend(AgentBackend):
    """Run a single task via the official ``bench eval create`` CLI."""

    def run_task(
        self,
        task: Any,
        skills_dir: Optional[Path],
        *,
        model: str,
        timeout_s: int,
        budget_usd: float,
        anonymize_map: Optional[Dict[str, str]],
        workdir: Optional[Path] = None,
    ) -> TrajectoryResult:
        """Dispatch a task with 429-aware retry-with-backoff.

        Anthropic's org-wide ``20,000,000 prompt bytes/hour`` cap can
        knock out individual trials with a 429. ``_run_task_once``
        signals these by raising :class:`BenchRateLimitedError`; this
        wrapper sleeps with the schedule in :data:`_RATE_LIMIT_BACKOFF_S`
        (60s → 120s → 240s → 480s) and retries up to four times. After
        max retries, the trial is reported as ``agent_error`` with
        ``notes`` containing ``rate_limited_max_retries`` so the runner's
        summary can still surface the rate-limit pressure.
        """
        task_id: str = getattr(task, "task_id", None) or (
            task.get("task_id") if isinstance(task, dict) else "unknown"
        )

        last_error: Optional[str] = None
        for attempt in range(len(_RATE_LIMIT_BACKOFF_S) + 1):
            try:
                tr = self._run_task_once(
                    task,
                    skills_dir,
                    model=model,
                    timeout_s=timeout_s,
                    budget_usd=budget_usd,
                    anonymize_map=anonymize_map,
                    workdir=workdir,
                )
            except BenchRateLimitedError as exc:
                last_error = exc.original_error
                if attempt >= len(_RATE_LIMIT_BACKOFF_S):
                    _log.error(
                        "rate_limited: %s exhausted %d retries; giving up",
                        task_id,
                        len(_RATE_LIMIT_BACKOFF_S),
                    )
                    break
                sleep_s = _RATE_LIMIT_BACKOFF_S[attempt]
                _log.warning(
                    "rate_limited: %s 429 on attempt %d/%d; sleeping %ds",
                    task_id,
                    attempt + 1,
                    len(_RATE_LIMIT_BACKOFF_S) + 1,
                    sleep_s,
                )
                time.sleep(sleep_s)
                continue
            else:
                if attempt > 0:
                    # Annotate notes so the caller sees we recovered.
                    suffix = f"recovered_after_429_retries={attempt}"
                    tr.notes = f"{tr.notes}; {suffix}" if tr.notes else suffix
                return tr

        # Max retries exhausted. Surface the original 429 as an
        # ``agent_error`` so the trial isn't lost from the summary.
        truncated_err = (last_error or "rate_limit_error")[:500]
        return TrajectoryResult(
            task_id=task_id,
            success=False,
            tool_calls=0,
            elapsed_s=0.0,
            last_msg=truncated_err,
            notes="rate_limited_max_retries",
            verified=None,
            verifier_status="agent_error",
            cost_usd=None,
        )

    def _run_task_once(
        self,
        task: Any,
        skills_dir: Optional[Path],
        *,
        model: str,
        timeout_s: int,
        budget_usd: float,
        anonymize_map: Optional[Dict[str, str]],
        workdir: Optional[Path] = None,
    ) -> TrajectoryResult:
        """Single attempt at running a task through the bench CLI.

        Raises :class:`BenchRateLimitedError` if the resulting
        ``result.json`` reports an Anthropic 429.
        """
        # ``Task`` may arrive as the dataclass or its dict form; normalize.
        task_id: str = getattr(task, "task_id", None) or (
            task.get("task_id") if isinstance(task, dict) else "unknown"
        )
        task_dir = _task_dir_from_task(task)

        wd = (
            Path(workdir)
            if workdir is not None
            else Path(tempfile.mkdtemp(prefix="bench_cli_"))
        )
        wd.mkdir(parents=True, exist_ok=True)

        # Pre-pick the jobs_dir under our workdir so we know exactly
        # where bench will write result.json. Pre-compute the
        # job_name (a timestamp matching bench's default format) so the
        # primary-path lookup hits without scanning.
        jobs_dir = wd / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        expected_job_name = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")

        yaml_path = _materialize_scene_yaml(
            wd,
            task_id=task_id,
            task_dir=task_dir,
            model=model,
            skills_dir=str(skills_dir) if skills_dir is not None else None,
            jobs_dir=jobs_dir,
        )

        if budget_usd is not None and budget_usd > 0:
            # bench does not emit per-call cost; the budget_usd kwarg
            # is preserved for forward-compat but is not enforced
            # here. Phase D's runner enforces cumulative caps via
            # Anthropic billing / token-count estimation.
            _log.warning(
                "BenchCliBackend: budget_usd=%.4f provided but bench CLI does "
                "not surface per-call cost; budget cap is enforced at runner "
                "level only.",
                float(budget_usd),
            )

        argv = _bench_cli_argv(yaml_path, task_dir, model)

        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return TrajectoryResult(
                task_id=task_id,
                success=False,
                tool_calls=0,
                elapsed_s=float(timeout_s),
                last_msg="bench cli timeout",
                notes="bench cli timeout",
                verified=None,
                verifier_status="timeout",
                cost_usd=None,
            )

        if proc.returncode != 0:
            stderr = proc.stderr or ""
            return TrajectoryResult(
                task_id=task_id,
                success=False,
                tool_calls=0,
                elapsed_s=0.0,
                last_msg=stderr[:500],
                notes=f"bench cli rc={proc.returncode}",
                verified=None,
                verifier_status="bench_cli_error",
                cost_usd=None,
            )

        # Locate the per-task result.json on disk. bench's stdout is
        # just a one-line summary, not JSON.
        result_path = _locate_result_json(jobs_dir, expected_job_name)
        if result_path is None:
            # Fall back to parsing the stdout score line so the eval
            # loop sees a reasonable status rather than crashing.
            stdout_summary = _parse_stdout_score(proc.stdout or "")
            if stdout_summary is not None:
                # Single-task runs only — passed = result for this task.
                # Mark verifier_status with a clear "fallback" tag in
                # notes so downstream sees this wasn't a clean parse.
                passed = stdout_summary["passed"] >= stdout_summary["total"]
                errors = stdout_summary["errors"]
                if errors > 0:
                    return TrajectoryResult(
                        task_id=task_id,
                        success=False,
                        tool_calls=0,
                        elapsed_s=0.0,
                        last_msg=(proc.stderr or "")[:500],
                        notes="result_json_missing; stdout_parsed; agent_error",
                        verified=None,
                        verifier_status="agent_error",
                        cost_usd=None,
                    )
                return TrajectoryResult(
                    task_id=task_id,
                    success=passed,
                    tool_calls=0,
                    elapsed_s=0.0,
                    last_msg="",
                    notes="result_json_missing; stdout_parsed",
                    verified=passed,
                    verifier_status="passed" if passed else "failed",
                    score=1.0 if passed else 0.0,
                    cost_usd=None,
                )
            return TrajectoryResult(
                task_id=task_id,
                success=False,
                tool_calls=0,
                elapsed_s=0.0,
                last_msg=(proc.stdout or "")[:500],
                notes="result_json_missing; stdout_unparseable",
                verified=None,
                verifier_status="bench_cli_error",
                cost_usd=None,
            )

        try:
            result_json: Dict[str, Any] = json.loads(
                result_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            return TrajectoryResult(
                task_id=task_id,
                success=False,
                tool_calls=0,
                elapsed_s=0.0,
                last_msg=f"result.json read/parse failed: {exc}",
                notes="result_json_unreadable",
                verified=None,
                verifier_status="bench_cli_error",
                cost_usd=None,
            )

        return _result_to_trajectory(
            task_id=task_id,
            result_json=result_json,
            trial_dir=result_path.parent,
            anonymize_map=anonymize_map,
            budget_usd=budget_usd,
        )
