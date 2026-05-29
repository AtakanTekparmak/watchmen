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

    bench eval create --config <yaml> --tasks-dir <task_dir> \
        --agent claude-code --model <model>

The ``--config`` YAML is materialized per-task from one of the templates
in ``skill_evolve/skillsbench/scenes/`` (with-skills vs no-skills). The
``--tasks-dir`` task_dir flows from the hydrated task's success-check
payload.

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
import os
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
    "assert 'claude-code' in _A or 'gemini' in _A, 'SG-1 agent registration failed'; "
    "import skill_evolve.agents._benchflow_patches as _p; "
    "assert _p.is_applied(), 'SG-2 deploy_skills patch failed to apply'; "
    "from benchflow.cli.main import app; "
    "sys.exit(app())"
)


def _agent_env_args(model: str) -> List[str]:
    """Thread the inner agent's required provider credentials via ``--agent-env``.

    benchflow does NOT inherit the parent process's ANTHROPIC_API_KEY into
    the agent sandbox — it fails pre-dispatch with ``agent_error |
    ANTHROPIC_API_KEY required for model '<m>' but not set`` (surfaced by
    the 2026-05-28 smoke once the flag fix let the call reach the agent).
    The credentials must be passed explicitly as ``--agent-env KEY=VALUE``.

    Routing:
      * Anthropic-native slugs (``claude*`` / ``anthropic*``) → forward
        ``ANTHROPIC_API_KEY`` (and ``ANTHROPIC_BASE_URL`` if set).
      * Anything else → forward ``BENCHFLOW_PROVIDER_{BASE_URL,API_KEY,MODEL}``
        when present (the provider-shim path, e.g. qwen via OpenRouter).

    Values are read from ``os.environ`` at argv-build time; on the
    ephemeral single-tenant eval VM the resulting ``ps`` exposure is
    acceptable. Missing vars are simply omitted — the run's pre-flight
    (skillsbench/evolve.py) is responsible for failing loud on a missing
    inner key before the loop starts.
    """
    args: List[str] = []
    m = model.lower()
    if m.startswith("claude") or m.startswith("anthropic"):
        forward = ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL")
    else:
        forward = (
            "BENCHFLOW_PROVIDER_BASE_URL",
            "BENCHFLOW_PROVIDER_API_KEY",
            "BENCHFLOW_PROVIDER_MODEL",
        )
    for var in forward:
        val = os.environ.get(var)
        if val:
            args += ["--agent-env", f"{var}={val}"]
    return args


def _bench_cli_argv(
    yaml_path: "Path | str",
    task_dir: str,
    model: str,
    agent: str = "claude-code",
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
    # Long flags only: installed benchflow 0.3.4 ``bench eval create``
    # rejects the short -f/-t/-a/-m forms at arg-parse, which silently
    # zeroed every candidate in the 2026-05-28 smoke. Mapping verified by
    # a live single-task repro: -f→--config, -t→--tasks-dir, -a→--agent,
    # -m→--model.
    return [
        sys.executable,
        "-c",
        _BENCH_SHIM_CODE,
        "eval",
        "create",
        "--config",
        str(yaml_path),
        "--tasks-dir",
        str(task_dir),
        "--agent",
        agent,
        "--model",
        model,
        *_agent_env_args(model),
    ]


if TYPE_CHECKING:  # pragma: no cover
    pass


_log = logging.getLogger(__name__)
logger = _log


def _sweep_orphaned_compose_projects(
    timeout_s: float = 30, min_age_s: int = 900
) -> int:
    """Force-remove orphaned bench-cli eval containers older than ``min_age_s``.

    Filters by name prefix ``skillsbench_`` (matches benchflow's docker-compose
    project naming for SkillsBench tasks: ``skillsbench_<task>__<hash>-main-N``).
    Only kills containers older than ``min_age_s`` to avoid nuking active
    workers. Returns count removed. Best-effort; logs but never raises.
    """
    import subprocess as _sp

    try:
        # Use {{.RunningFor}} only for human-readable; rely on CreatedAt for parse.
        proc = _sp.run(
            [
                "docker",
                "ps",
                "--filter",
                "name=skillsbench_",
                "--format",
                "{{.ID}} {{.CreatedAt}}",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        now = time.time()
        stale: List[str] = []
        for line in proc.stdout.strip().splitlines():
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            cid, created_at = parts[0], parts[1]
            # docker CreatedAt format: "2026-05-04 19:29:25 +0000 UTC"
            try:
                # Strip timezone parts; parse first 19 chars as UTC.
                ts = datetime.strptime(created_at[:19], "%Y-%m-%d %H:%M:%S")
                age = now - ts.timestamp()
            except Exception:
                continue
            if age >= min_age_s:
                stale.append(cid)
        if not stale:
            return 0
        _sp.run(["docker", "rm", "-f", *stale], capture_output=True, timeout=timeout_s)
        logger.warning(
            "orphan sweep: removed %d stale containers (older than %ds)",
            len(stale),
            min_age_s,
        )
        return len(stale)
    except Exception as exc:
        logger.warning("orphan sweep failed: %s", exc)
        return 0


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
                "agent: <agent>\n"
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
            "agent: <agent>\n"
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
    agent: str = "claude-code",
) -> str:
    """Substitute placeholders and force ``jobs_dir`` to a known path."""
    out = template.replace("<task_dir>", task_dir).replace("<model>", model)
    if skills_dir is not None:
        out = out.replace("<skills_dir>", skills_dir)
    out = out.replace("<jobs_dir>", jobs_dir)
    out = out.replace("<agent>", agent)
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
    agent: str = "claude-code",
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
        agent=agent,
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


# v9d patch (2026-05-07): mutator-prompt enrichment.
#
# E1 + E2 meta-probes confirmed that the bottleneck of v5–v8 evolution was
# ``failures_render`` containing only ``success_votes=0/3`` strings — no task
# content, no trace excerpt, no agent final message. Given the actual trace,
# DeepSeek-v4-pro authors task-specific patches in 4/5 hand-built E2 calls. We
# pull a small excerpt out of ``acp_trajectory.jsonl`` so it can flow through
# ``last_msg`` → ``failures_blob`` → mutator prompt.
#
# Also fixes the long-standing instrumentation gap where ``skills_invoked`` was
# hardcoded to ``[]`` in this backend, masking actual invocation rates from
# every prior evolution run's prompts.

_SCRIPT_INVOKE_RE = re.compile(
    r"\bpython3?\b\s+(?:[^\s'\"]*?/)?([\w-]+)/scripts/([\w.-]+\.py)\b"
)


def _summarize_acp_trajectory(
    trial_dir: Path,
    *,
    max_excerpt_chars: int = 1500,
    max_final_chars: int = 1200,
    last_n_commands: int = 6,
) -> Dict[str, Any]:
    """Parse ``trajectory/acp_trajectory.jsonl`` and return a small summary.

    Schema (Gemini CLI ACP, observed in benchflow 0.3.x):
      - ``type=tool_call`` ``kind=execute`` → ``title`` is the bash command.
      - ``type=tool_call`` ``kind=think`` → planning block (skipped — verbose).
      - ``type=tool_call`` ``kind=edit`` → ``title`` is the file path.
      - ``type=agent_message`` → ``text`` is the final user-facing answer.
      - ``type=agent_thought`` → chain-of-thought (skipped — usually >5KB).

    Returns ``{excerpt, final_message, skills_invoked}``. All values are
    safe defaults (empty string / list) when the file is missing or
    malformed; never raises. ``skills_invoked`` is the list of skill
    folder names whose ``scripts/<x>.py`` appears in any execute title.
    """
    paths = (
        trial_dir / "trajectory" / "acp_trajectory.jsonl",
        trial_dir / "agent" / "acp_trajectory.jsonl",
    )
    jsonl: Optional[Path] = next((p for p in paths if p.is_file()), None)
    if jsonl is None:
        return {"excerpt": "", "final_message": "", "skills_invoked": []}

    execute_titles: List[str] = []
    edit_titles: List[str] = []
    final_message: str = ""
    skills: List[str] = []

    try:
        text = jsonl.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"excerpt": "", "final_message": "", "skills_invoked": []}

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        ev_type = ev.get("type")
        if ev_type == "tool_call":
            kind = ev.get("kind")
            title = (ev.get("title") or "").strip()
            if not title:
                continue
            if kind == "execute":
                execute_titles.append(title)
                for m in _SCRIPT_INVOKE_RE.finditer(title):
                    skills.append(m.group(1))
            elif kind == "edit":
                edit_titles.append(title)
        elif ev_type == "agent_message":
            txt = ev.get("text") or ""
            if isinstance(txt, str) and txt.strip():
                final_message = txt[:max_final_chars]

    # Compose excerpt: last N execute titles, then any edit titles.
    tail = execute_titles[-last_n_commands:]
    parts: List[str] = []
    for i, t in enumerate(tail, 1):
        # Hard-cap each title so a single multi-line heredoc can't blow the budget.
        parts.append(f"  {i}. {t[:300]}")
    if edit_titles:
        parts.append("")
        parts.append(f"  edits: {', '.join(edit_titles[-4:])[:240]}")
    excerpt = "\n".join(parts)
    if len(excerpt) > max_excerpt_chars:
        excerpt = excerpt[:max_excerpt_chars] + "\n  [hard-truncated]"

    # Dedupe skills, preserve order.
    seen: set = set()
    skills_invoked: List[str] = []
    for s in skills:
        if s not in seen:
            seen.add(s)
            skills_invoked.append(s)

    return {
        "excerpt": excerpt,
        "final_message": final_message,
        "skills_invoked": skills_invoked,
    }


def _build_failure_last_msg(
    *,
    verifier_status: str,
    error: Optional[str],
    verifier_error: Optional[str],
    summary: Dict[str, Any],
    max_total: int = 3000,
) -> str:
    """Assemble the enriched ``last_msg`` for a failed bench-cli trial.

    Layout (sections only emitted when non-empty):

      [verifier] <status> | <short err>
      [final agent message]
      <up to max_final_chars>
      [last 6 commands]
      <execute titles>

    Capped at ``max_total`` chars. Ordering reflects what's most useful
    to the mutator: status first, then the agent's own self-report,
    then the bash trace.
    """
    short_err = ""
    if error:
        short_err = str(error)[:240]
    elif verifier_error:
        short_err = str(verifier_error)[:240]
    sections: List[str] = []
    if short_err:
        sections.append(f"[verifier] {verifier_status} | {short_err}")
    else:
        sections.append(f"[verifier] {verifier_status}")
    if summary.get("final_message"):
        sections.append("[final agent message]")
        sections.append(summary["final_message"])
    if summary.get("excerpt"):
        sections.append("[last 6 commands]")
        sections.append(summary["excerpt"])
    blob = "\n".join(sections)
    if len(blob) > max_total:
        blob = blob[:max_total] + "\n[hard-truncated]"
    return blob


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

    # v9d patch (2026-05-07): for failed trials, enrich last_msg with the
    # ACP trajectory excerpt + agent final message. For passed trials, keep
    # the historical short shape (errors only) — the mutator only needs
    # rich context for failures it should fix.
    summary = _summarize_acp_trajectory(trial_dir)
    if not success:
        last_msg_raw = _build_failure_last_msg(
            verifier_status=verifier_status,
            error=error,
            verifier_error=verifier_error,
            summary=summary,
        )
    else:
        last_msg_raw = ""
        if error:
            last_msg_raw = str(error)
        elif verifier_error:
            last_msg_raw = str(verifier_error)
        last_msg_raw = last_msg_raw[:500]
    last_msg = _apply_anonymize(last_msg_raw, anonymize_map)

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
        # v9d patch (2026-05-07): parsed from acp_trajectory.jsonl by
        # _summarize_acp_trajectory. Was hardcoded ``[]`` since Phase E v5;
        # that gap masked invocation rates from every prior mutator prompt
        # via ``unused_skills`` and ``invocation_counts``.
        skills_invoked=list(summary.get("skills_invoked") or []),
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
        agent: str = "claude-code",
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
                    agent=agent,
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
        agent: str = "claude-code",
    ) -> TrajectoryResult:
        """Single attempt at running a task through the bench CLI.

        Raises :class:`BenchRateLimitedError` if the resulting
        ``result.json`` reports an Anthropic 429.
        """
        # Pre-dispatch defensive sweep: if a previous attempt orphaned a
        # docker-compose project (e.g. due to TimeoutExpired or other
        # subprocess failure), clean it up before launching a new eval.
        pre_swept = _sweep_orphaned_compose_projects()
        if pre_swept > 0:
            _log.warning(
                "orphan sweep (pre-dispatch): removed %d container(s)", pre_swept
            )

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
            agent=agent,
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

        argv = _bench_cli_argv(yaml_path, task_dir, model, agent=agent)

        try:
            try:
                proc = subprocess.run(
                    argv,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )
            except subprocess.TimeoutExpired:
                # Fail-loud: a broken/hung harness must be VISIBLE in the
                # run log, not silently scored as a bad-skill failure
                # (2026-05-28 silent-fail incident).
                logger.warning(
                    "bench-cli task %s failed: status=%s detail=%s",
                    task_id,
                    "timeout",
                    f"timed out after {timeout_s}s",
                )
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
        finally:
            # Container-leak guard: every subprocess.run path (success,
            # nonzero rc, TimeoutExpired, or any unexpected exception)
            # must sweep orphaned docker-compose projects so retries
            # don't accumulate zombie eval containers.
            swept = _sweep_orphaned_compose_projects()
            if swept > 0:
                _log.warning(
                    "orphan sweep (post-dispatch): removed %d container(s)", swept
                )

        if proc.returncode != 0:
            stderr = proc.stderr or ""
            # Fail-loud: a nonzero bench-cli rc (e.g. arg-parse death) is a
            # broken harness, not a bad skill — log it so an all-errored
            # eval is visible in the run log (2026-05-28 silent-fail).
            logger.warning(
                "bench-cli task %s failed: status=%s detail=%s",
                task_id,
                "bench_cli_error",
                f"rc={proc.returncode} stderr={stderr[:500]!r}",
            )
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
                    # Fail-loud: bench reported agent-side errors with no
                    # result.json — surface it rather than scoring a silent
                    # zero (2026-05-28 silent-fail incident).
                    logger.warning(
                        "bench-cli task %s failed: status=%s detail=%s",
                        task_id,
                        "agent_error",
                        f"result_json_missing; stdout_errors={errors} "
                        f"stderr={(proc.stderr or '')[:500]!r}",
                    )
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
            # Fail-loud: no result.json and stdout score line unparseable —
            # the harness produced nothing usable, which is a broken-harness
            # signal, not a bad skill (2026-05-28 silent-fail incident).
            logger.warning(
                "bench-cli task %s failed: status=%s detail=%s",
                task_id,
                "bench_cli_error",
                f"result_json_missing; stdout_unparseable "
                f"stdout={(proc.stdout or '')[:500]!r}",
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
            # Fail-loud: result.json exists but is unreadable/corrupt — a
            # broken harness, not a bad skill (2026-05-28 silent-fail).
            logger.warning(
                "bench-cli task %s failed: status=%s detail=%s",
                task_id,
                "bench_cli_error",
                f"result_json_unreadable path={result_path} err={exc}",
            )
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
