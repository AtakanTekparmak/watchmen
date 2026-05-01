"""Evaluator harness — score a candidate skills folder.

Design choices (so downstream tracks can change them with eyes open):

1. **Subprocess vs in-process.** We invoke ``hermes-agent/run_agent.py`` as a
   subprocess. Reasons:
     * AIAgent registers tools at import time and mutates global state
       (registry, env-driven model selection, signal handlers). Importing it
       repeatedly inside one parent process makes parallel evaluation flaky.
     * The subprocess boundary doubles as our HERMES_HOME enforcement —
       the child literally cannot see another candidate's skills.
   Cost: ~300ms cold start per task. Acceptable vs minutes of LLM latency.

2. **Composite score formula.**
       composite = success_rate - 0.05 * normalized_tool_call_overhead
   where normalized_tool_call_overhead = clamp(avg_tool_calls/10, 0, 1).
   Rationale: weight success heavily, mildly penalize tool-call bloat to
   discourage skills that succeed via brute-force exploration. Override
   by setting ``EvalResult.composite`` after construction or by replacing
   :func:`compute_composite`.

3. **Cascade.** Stage-1 (the two cheapest tasks) runs first; if 0/2 succeed,
   we short-circuit and return. Saves ~80% on candidates that simply break
   the agent (bad SKILL.md, infinite loops, etc.).

4. **No live LLM keys -> synthetic placeholder.** When no provider key is
   set in the environment we return a ``synthetic`` EvalResult so the rest
   of the pipeline (sandbox, manifest, scoring, plumbing) can be exercised
   for free. The placeholder marks success_rate=0.0 and tool_calls=0 for
   every task, plus ``synthetic=True`` on EvalResult.

5. **Real verifiers (see :mod:`skill_evolve.verifiers`).** By default each
   task's ``success`` is now the real ``test.sh`` / SWE-bench harness
   result, not "trajectory exists". The verifier dispatcher returns a
   three-state outcome: ``True`` (verified pass), ``False`` (verified
   fail), or ``None`` (could not verify — e.g. Docker daemon down).
   ``None`` is treated as a fail for scoring (0 toward ``success_rate``)
   but flagged separately on each ``TaskOutcome`` (``verified=None``,
   ``verifier_status`` spelling out why) and counted on ``EvalResult``
   as ``unverified_count`` so callers can distinguish "3 failed" from
   "2 failed + 1 unverifiable". Pass ``verify=False`` (or ``--no-verify``)
   to fall back to the legacy "trajectory exists" signal for fast
   plumbing-only tests.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .benchmark import load_subset
from .sandbox import sandbox
from .verifiers import VerifyResult, stage_workspace, verify_task

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
HERMES_DIR = REPO_ROOT / "hermes-agent"
RUN_AGENT_PY = HERMES_DIR / "run_agent.py"

# Provider keys we look at to decide if we can do live LLM calls.
_LIVE_KEY_ENV_VARS = (
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "NOUS_API_KEY",
    "NOUS_PORTAL_API_KEY",
)

# Default toolset for evals — matches what TBLite uses upstream.
DEFAULT_ENABLED_TOOLSETS = "terminal,file,skills"

# Cap turns per task so a runaway agent can't burn the whole budget.
DEFAULT_MAX_TURNS = 20


def _hermes_default_model() -> Optional[str]:
    """Read ~/.hermes/config.yaml's model.default, if present.

    Provides a defensive default so the evaluator picks up the user's
    chosen model instead of silently overriding it with sonnet 4.6.
    Returns None on any read/parse failure.
    """
    cfg_path = Path.home() / ".hermes" / "config.yaml"
    if not cfg_path.exists():
        return None
    try:
        import yaml  # local import to keep top-level imports lean

        loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    if not isinstance(loaded, dict):
        return None
    model_block = loaded.get("model") or {}
    if not isinstance(model_block, dict):
        return None
    val = model_block.get("default")
    return val if isinstance(val, str) and val else None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TaskOutcome:
    task_id: str
    success: bool
    tool_calls: int
    elapsed_s: float
    skills_invoked: List[str] = field(default_factory=list)
    last_msg: str = ""
    raw_completed: bool = False
    notes: str = ""
    # New (back-compatible — defaults preserve old behaviour for callers
    # that constructed TaskOutcome positionally before these existed):
    verified: Optional[bool] = None
    """Three-state verification result.

    ``True``  → the verifier asserted pass (test.sh or harness).
    ``False`` → the verifier asserted fail.
    ``None``  → verifier could not run (Docker missing, harness install
                missing, --no-verify, or synthetic placeholder). Treated
                as ``False`` for scoring but flagged in ``verifier_status``
                so callers can tell the two apart.
    """
    verifier_status: str = "not_run"
    verifier_detail: str = ""
    # Continuous score in [0,1] when the verifier exposes a structured
    # assertion breakdown (pytest CTRF JSON or "X passed, Y failed"
    # summary). ``None`` when no breakdown is available — ``success``
    # stays authoritative in that case. For ``repeats > 1`` this is the
    # mean of per-repeat scores.
    score: Optional[float] = None
    # When ``repeats > 1`` in :func:`evaluate`, each per-repeat raw outcome
    # is captured here (``task_id``, ``success``, ``tool_calls``,
    # ``elapsed_s``, ``skills_invoked``, ``verified``, ``verifier_status``)
    # so callers can inspect within-candidate variance post-hoc. Empty for
    # the default ``repeats=1`` path.
    repeats_detail: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class EvalResult:
    success_rate: float
    tool_calls_per_success: float
    composite: float
    per_task: List[Dict[str, Any]]
    failures: List[Dict[str, str]]
    skills_folder: str
    n_tasks: int
    cascade_truncated: bool = False
    synthetic: bool = False
    notes: str = ""
    # New, back-compatible fields. ``verified_count``/``unverified_count``
    # exist so downstream tracks can tell "3/5 passed, 1 couldn't be
    # verified because Docker was down" apart from "4/5 passed".
    verified_count: int = 0
    unverified_count: int = 0
    verifier_disabled: bool = False
    # Continuous-score mean across tasks (``None`` → none of the tasks
    # exposed a structured breakdown, so fall back to ``success_rate``).
    # This is what the composite prefers when available; it drops
    # composite noise from ±0.125 (binary/8 tasks) to roughly ±0.025.
    mean_score: Optional[float] = None
    scored_task_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


def compute_composite(
    success_rate: float,
    avg_tool_calls: float,
    *,
    mean_score: Optional[float] = None,
) -> float:
    """``<score> - 0.05 * normalized_tool_call_overhead``.

    ``<score>`` is ``mean_score`` when available (continuous per-assertion
    pass rate) and falls back to ``success_rate`` (binary per-task) when
    it isn't — preserving historical composite values on folders that pre-
    date continuous scoring. The tool-call overhead term is unchanged.
    Documented in module docstring. Downstream tracks may swap this out.
    """
    base = mean_score if mean_score is not None else success_rate
    overhead = max(0.0, min(1.0, avg_tool_calls / 10.0))
    return base - 0.05 * overhead


# ---------------------------------------------------------------------------
# Live-key detection
# ---------------------------------------------------------------------------


def have_live_keys(env: Optional[Dict[str, str]] = None) -> bool:
    e = env if env is not None else os.environ
    return any(e.get(k) for k in _LIVE_KEY_ENV_VARS)


# ---------------------------------------------------------------------------
# Trajectory parsing
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.debug("skipping malformed jsonl line in %s", path)


def _parse_tool_calls_from_value(value: str) -> List[Dict[str, Any]]:
    """Pull tool_call JSON blobs out of a ShareGPT 'gpt' turn value."""
    out: List[Dict[str, Any]] = []
    for m in _TOOL_CALL_RE.finditer(value or ""):
        try:
            out.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            continue
    return out


def parse_trajectory_files(run_dir: Path) -> Dict[str, Any]:
    """Aggregate trajectory_samples.jsonl + failed_trajectories.jsonl.

    Returns the *most recently appended* trajectory of either kind plus
    an aggregate count of tool_calls and a list of skill IDs invoked
    via the ``skill_view`` tool.
    """
    success_path = run_dir / "trajectory_samples.jsonl"
    fail_path = run_dir / "failed_trajectories.jsonl"

    last_entry: Optional[Dict[str, Any]] = None
    success = False
    for entry in _iter_jsonl(success_path):
        last_entry = entry
        success = True
    for entry in _iter_jsonl(fail_path):
        # If both files have entries we keep success=True (more recent
        # successful trajectory wins) but still surface the latest entry
        # for last_msg context if no success file existed.
        if last_entry is None:
            last_entry = entry

    tool_calls = 0
    skills_invoked: List[str] = []
    last_msg = ""

    if last_entry:
        for turn in last_entry.get("conversations", []):
            if turn.get("from") != "gpt":
                continue
            value = turn.get("value", "") or ""
            calls = _parse_tool_calls_from_value(value)
            tool_calls += len(calls)
            for c in calls:
                if c.get("name") == "skill_view":
                    args = c.get("arguments") or {}
                    name = args.get("name")
                    if isinstance(name, str) and name:
                        skills_invoked.append(name)
        # The "last message" is whichever non-tool turn ended the conversation.
        for turn in reversed(last_entry.get("conversations", [])):
            if turn.get("from") in ("gpt", "tool"):
                last_msg = (turn.get("value") or "")[:400]
                break

    return {
        "success": success,
        "tool_calls": tool_calls,
        "skills_invoked": skills_invoked,
        "last_msg": last_msg,
        "raw_completed": bool(last_entry and last_entry.get("completed")),
    }


# ---------------------------------------------------------------------------
# Per-task runner
# ---------------------------------------------------------------------------


def _run_one_task(
    task: Dict[str, Any],
    skills_folder: Path,
    *,
    model: str,
    max_turns: int,
    enabled_toolsets: str,
    extra_run_agent_args: Sequence[str] = (),
    verify: bool = True,
    keep_sandbox: bool = False,
) -> TaskOutcome:
    """Run a single benchmark task in its own sandbox + subprocess.

    Flow:
      1. Stage the workspace (clone repo for SWE-bench; extract /app from
         the docker image for TBLite). Agent works inside that workspace.
      2. Invoke ``run_agent.py`` with cwd = workspace.
      3. Parse trajectory files for tool-call / skill-usage stats.
      4. If ``verify=True``, dispatch to the real verifier and fold its
         pass/fail into ``TaskOutcome.success``.
      5. If ``verify=False`` or the verifier can't run, ``success`` falls
         back to "agent finished a saved trajectory" and ``verified``
         is left as ``None``.
    """
    rid = f"{task['task_id'].replace('/', '_')}-{uuid.uuid4().hex[:6]}"
    started = time.monotonic()

    with sandbox(skills_folder, run_id=rid, keep_on_exit=keep_sandbox) as h:
        # Stage the workspace *before* the agent starts. For TBLite this
        # pulls /app out of the task container; for SWE-bench it clones the
        # repo at base_commit. The agent runs with cwd = workspace so its
        # edits land where the verifier will look for them.
        workspace = h.run_dir / "workspace"
        staging_meta: Dict[str, Any] = {}
        if verify:
            try:
                staging_meta = stage_workspace(task, workspace)
            except Exception as exc:  # pragma: no cover — staging crashes
                logger.warning(
                    "stage_workspace failed for %s: %s", task["task_id"], exc
                )
                staging_meta = {"staging_error": str(exc)}
        else:
            workspace.mkdir(parents=True, exist_ok=True)

        agent_cwd = workspace if verify else h.run_dir

        # When running real verifiers, wire the Hermes agent through the
        # per-task Docker container so agent commands execute inside the same
        # filesystem test.sh will later verify. Without this, the agent runs
        # on the macOS host, its /app references fail, and it can't dogfood
        # its own edits.
        docker_backend_enabled = False
        docker_setup_note = ""
        if verify:
            from . import hermes_docker

            if hermes_docker.docker_backend_available():
                try:
                    image = task["success_check_payload"]["docker_image"]
                    hermes_docker.ensure_image_pulled(image)
                    env_patch = hermes_docker.build_env_patch(
                        task,
                        Path(workspace),
                    )
                    h.env.update(env_patch)
                    docker_backend_enabled = True
                except Exception as exc:
                    # Fall back gracefully: local backend is still functional,
                    # we just won't get real pass/fail fidelity on this task.
                    logger.warning(
                        "hermes docker backend setup failed for %s: %s",
                        task.get("task_id"),
                        exc,
                    )
                    docker_setup_note = f"docker_setup_failed: {exc!r}"

        # We always run via run_agent.py for one-shot per-task evals — it
        # owns the trajectory file writes (trajectory_samples.jsonl and
        # failed_trajectories.jsonl) which we parse afterwards.
        # python-fire (run_agent.py uses fire.Fire) calls ast.literal_eval on
        # values after '='. Bare strings with commas/punctuation get parsed as
        # tuples or fail. Wrap in literal double-quotes so fire parses them as
        # str. The model name has a slash (e.g. minimax/minimax-m2.7) which
        # already forces fire to treat as string, but quote it for consistency.
        def _fq(v: str) -> str:
            # Escape any embedded backslashes/quotes per Python literal rules.
            return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'

        cmd = [
            sys.executable,
            str(RUN_AGENT_PY),
            f"--query={_fq(task['prompt'])}",
            f"--max_turns={max_turns}",
            f"--enabled_toolsets={_fq(enabled_toolsets)}",
            f"--model={_fq(model)}",
            "--save_trajectories=True",
        ]
        cmd.extend(extra_run_agent_args)

        notes = ""
        try:
            proc = subprocess.run(
                cmd,
                env=h.env,
                cwd=agent_cwd,
                capture_output=True,
                text=True,
                timeout=task.get("timeout_s", 900),
            )
            stderr_tail = proc.stderr[-500:] if proc.stderr else ""
            notes = (
                ""
                if proc.returncode == 0
                else (f"run_agent rc={proc.returncode}; stderr_tail={stderr_tail!r}")
            )
            if docker_backend_enabled:
                # Give Hermes' background container cleanup a moment before we tear
                # down HERMES_HOME (it fires docker rm -f in a detached Popen).
                time.sleep(1)
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - started
            return TaskOutcome(
                task_id=task["task_id"],
                success=False,
                tool_calls=0,
                elapsed_s=elapsed,
                last_msg="timeout",
                notes=f"hard timeout after {task.get('timeout_s')}s",
                verified=None,
                verifier_status="agent_timeout",
            )

        # Trajectory files are written in the agent's cwd.
        parsed = parse_trajectory_files(agent_cwd)

        # Verification stage. We keep the "trajectory exists" signal as a
        # fallback for back-compat (when --no-verify is set or a verifier
        # couldn't run), but the primary success signal is the verifier.
        verified: Optional[bool] = None
        verifier_status = "not_run"
        verifier_detail = ""
        score: Optional[float] = None
        if verify:
            vres: VerifyResult = verify_task(task, h.run_dir)
            verified = vres.passed
            verifier_status = vres.status
            verifier_detail = vres.detail[:400]
            score = vres.score

        elapsed = time.monotonic() - started

        if verify and verified is not None:
            success = bool(verified)
        else:
            # Back-compat: "did the agent finish a saved trajectory?"
            # Used when --no-verify is set OR the verifier couldn't run.
            success = parsed["success"]

        staging_note = ""
        if staging_meta.get("staging_error"):
            staging_note = f" staging_error={staging_meta['staging_error'][:160]}"

        docker_note = f" {docker_setup_note}" if docker_setup_note else ""

        return TaskOutcome(
            task_id=task["task_id"],
            success=success,
            tool_calls=parsed["tool_calls"],
            elapsed_s=elapsed,
            skills_invoked=sorted(set(parsed["skills_invoked"])),
            last_msg=parsed["last_msg"],
            raw_completed=parsed["raw_completed"],
            notes=(notes + staging_note + docker_note).strip(),
            verified=verified,
            verifier_status=verifier_status,
            verifier_detail=verifier_detail,
            score=score,
        )


# ---------------------------------------------------------------------------
# Repeat aggregation
# ---------------------------------------------------------------------------


def _median(xs: Sequence[float]) -> float:
    """Plain median (no numpy)."""
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2)


def _aggregate_repeats(outs: List[TaskOutcome]) -> TaskOutcome:
    """Collapse k repeated TaskOutcomes into one via the rules in the brief.

    * ``success`` — majority vote; conservative tie-break toward ``False``
      when ``k`` is even and the vote ties.
    * ``tool_calls`` / ``elapsed_s`` — median.
    * ``skills_invoked`` — union (sorted).
    * ``verified`` / ``verifier_status`` — taken from one of the runs whose
      ``success`` matches the majority outcome so "verified_count" reflects
      the majority verifier state.
    * ``repeats_detail`` — per-repeat slim record list.
    """
    if not outs:
        raise ValueError("cannot aggregate empty repeat list")
    if len(outs) == 1:
        return outs[0]

    k = len(outs)
    succ = sum(1 for o in outs if o.success)
    # Majority vote: conservative tie-break toward False.
    majority_success = succ > (k / 2)

    # Pick a representative outcome matching the majority for verifier state
    # + last_msg / notes. Prefer one whose `verified` isn't None if any.
    matching = [o for o in outs if o.success == majority_success]
    if not matching:
        matching = outs  # shouldn't happen, but be safe
    # Prefer a verified-not-None rep inside the majority bucket.
    rep = next((o for o in matching if o.verified is not None), matching[0])

    skills_union: List[str] = sorted({s for o in outs for s in o.skills_invoked})
    tool_calls_med = int(round(_median([float(o.tool_calls) for o in outs])))
    elapsed_med = float(_median([o.elapsed_s for o in outs]))

    detail = [
        {
            "task_id": o.task_id,
            "success": bool(o.success),
            "tool_calls": int(o.tool_calls),
            "elapsed_s": float(o.elapsed_s),
            "skills_invoked": list(o.skills_invoked),
            "verified": o.verified,
            "verifier_status": o.verifier_status,
            "score": o.score,
            "last_msg": (o.last_msg or "")[:200],
        }
        for o in outs
    ]

    notes = rep.notes
    if notes:
        notes = f"{notes} | repeats={k} success_votes={succ}/{k}"
    else:
        notes = f"repeats={k} success_votes={succ}/{k}"

    # Mean of per-repeat scores, ignoring repeats where the verifier
    # couldn't produce one (score=None). If none produced a score, the
    # aggregated task also has no continuous signal -> falls back to
    # the binary ``success`` vote downstream.
    scored = [o.score for o in outs if o.score is not None]
    score_mean = float(sum(scored) / len(scored)) if scored else None

    return TaskOutcome(
        task_id=outs[0].task_id,
        success=bool(majority_success),
        tool_calls=tool_calls_med,
        elapsed_s=elapsed_med,
        skills_invoked=skills_union,
        last_msg=rep.last_msg,
        raw_completed=bool(rep.raw_completed),
        notes=notes,
        verified=rep.verified,
        verifier_status=rep.verifier_status,
        verifier_detail=rep.verifier_detail,
        score=score_mean,
        repeats_detail=detail,
    )


# ---------------------------------------------------------------------------
# Synthetic placeholder (no live keys / dry-run)
# ---------------------------------------------------------------------------


def _synthetic_result(
    skills_folder: Path,
    tasks: List[Dict[str, Any]],
    note: str,
) -> EvalResult:
    per_task = [
        TaskOutcome(
            task_id=t["task_id"],
            success=False,
            tool_calls=0,
            elapsed_s=0.0,
            skills_invoked=[],
            last_msg="synthetic placeholder — no live LLM keys",
            notes="synthetic",
            verified=None,
            verifier_status="synthetic",
        ).to_dict()
        for t in tasks
    ]
    return EvalResult(
        success_rate=0.0,
        tool_calls_per_success=0.0,
        composite=0.0,
        per_task=per_task,
        failures=[
            {"task_id": p["task_id"], "last_msg": p["last_msg"]} for p in per_task
        ],
        skills_folder=str(skills_folder),
        n_tasks=len(tasks),
        cascade_truncated=False,
        synthetic=True,
        notes=note,
        verified_count=0,
        unverified_count=len(per_task),
        verifier_disabled=True,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def evaluate(
    skills_folder_path: Path,
    *,
    cascade: bool = True,
    max_workers: int = 1,
    model: Optional[str] = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    enabled_toolsets: str = DEFAULT_ENABLED_TOOLSETS,
    sources: Optional[List[str]] = None,
    task_ids: Optional[List[str]] = None,
    agent_backend: Optional[str] = None,
    extra_run_agent_args: Sequence[str] = (),
    force_synthetic: bool = False,
    verify: bool = True,
    repeats: int = 3,
    keep_sandbox: bool = False,
) -> EvalResult:
    """Score a candidate skills folder.

    Args:
        skills_folder_path: directory of SKILL.md skills.
        cascade: enable stage-1 short-circuit (default on).
        max_workers: parallel task workers (each gets its own sandbox).
        model: model string passed to run_agent (defaults to its own default).
        max_turns: per-task tool-iteration cap.
        enabled_toolsets: comma-separated toolset list.
        sources: filter benchmark sources (e.g. ``["tblite"]``).
        task_ids: optional allowlist of task IDs (fully qualified or
            bare segment). Forwarded to :func:`load_subset`. Used by
            Phase E SkillsBench dispatch so evolution operates on the
            hot-12 subset rather than the default tblite section of
            the manifest.
        agent_backend: name of the agent backend to dispatch each
            per-task call to. ``"hermes"`` (default when ``None``)
            preserves the legacy ``_run_one_task`` flow unchanged
            (tblite/swebench paths). ``"bench-cli"`` shells out to
            :class:`skill_evolve.agents.bench_cli.BenchCliBackend` for
            SkillsBench tasks (whose ``success_check_payload`` doesn't
            carry the ``docker_image`` Hermes expects). Resolved via
            :func:`skill_evolve.agents.get_backend`. Unknown values
            raise.
        extra_run_agent_args: extra CLI args appended to run_agent.py.
        force_synthetic: emit a synthetic result even if keys are present.
        verify: run the real pass/fail verifiers (test.sh / swebench
            harness) for each task. When False, fall back to the legacy
            "agent finished without crashing" signal and skip workspace
            staging (faster; useful for plumbing-only runs).
        repeats: number of independent repetitions per task; when >1 each
            task is rolled-out ``repeats`` times and the per-task outcome
            is aggregated by majority-vote success + median tool-calls /
            elapsed + union skills. Raw per-repeat records are kept on
            ``TaskOutcome.repeats_detail``. Default 1 is fully
            back-compat with single-shot evaluation.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    skills_folder_path = Path(skills_folder_path).expanduser().resolve()
    if not skills_folder_path.is_dir():
        raise FileNotFoundError(skills_folder_path)

    if agent_backend is not None and agent_backend not in ("hermes", "bench-cli"):
        raise ValueError(
            f"unknown agent_backend: {agent_backend!r} (known: 'hermes', 'bench-cli')"
        )

    # Hydrate offline first (cheap) so we have something to summarize even
    # if HF lookups fail, then upgrade to full hydration when keys exist.
    online = have_live_keys() and not force_synthetic
    tasks = load_subset(offline_only=not online, sources=sources, task_ids=task_ids)

    if not online:
        return _synthetic_result(
            skills_folder_path,
            tasks,
            note="no live LLM keys (set OPENROUTER_API_KEY etc.) — "
            "returning synthetic placeholder",
        )

    if not RUN_AGENT_PY.exists():
        raise FileNotFoundError(
            f"hermes-agent run_agent.py missing at {RUN_AGENT_PY}; "
            "did you clone the fork?"
        )

    chosen_model = (
        model
        or os.environ.get("SKILL_EVOLVE_MODEL")
        or _hermes_default_model()
        or "anthropic/claude-sonnet-4.6"
    )

    # Resolve the agent backend once; every per-task call dispatches
    # through this object. ``None`` defaults to the historical Hermes
    # path so the tblite manifest stays bit-identical. Bench-cli is
    # selected by Phase E for SkillsBench tasks (whose payloads lack
    # the ``docker_image`` field Hermes expects).
    from skill_evolve.agents import get_backend

    backend_name = agent_backend or "hermes"
    backend = get_backend(backend_name)

    def _dispatch(task: Dict[str, Any]) -> TaskOutcome:
        """Run a single task through the configured backend.

        For ``hermes`` we forward the evaluator's full kwarg set
        (max_turns / enabled_toolsets / verify / keep_sandbox /
        extra_run_agent_args) so the tblite path stays bit-identical
        to the pre-dispatch direct ``_run_one_task`` call. For
        ``bench-cli`` those kwargs are ignored (the bench CLI owns
        its own runtime knobs via the rendered scene YAML).
        """
        logger.info(
            "evaluate: dispatching to %s for task %s",
            type(backend).__name__,
            task.get("task_id", "<unknown>"),
        )
        if backend_name == "hermes":
            traj = backend.run_task(
                task,
                skills_dir=skills_folder_path,
                model=chosen_model,
                timeout_s=int(task.get("timeout_s", 0) or 0),
                budget_usd=0.0,
                anonymize_map=None,
                max_turns=max_turns,
                enabled_toolsets=enabled_toolsets,
                extra_run_agent_args=extra_run_agent_args,
                verify=verify,
                keep_sandbox=keep_sandbox,
            )
        else:
            traj = backend.run_task(
                task,
                skills_dir=skills_folder_path,
                model=chosen_model,
                timeout_s=int(task.get("timeout_s", 600) or 600),
                budget_usd=0.0,
                anonymize_map=None,
            )
        return traj.to_task_outcome()

    # Cascade: stage-1 first, short-circuit if all-zero.
    stage1 = [t for t in tasks if t.get("stage", 1) == 1]
    rest = [t for t in tasks if t.get("stage", 1) != 1]

    outcomes: List[TaskOutcome] = []

    def _run_batch(batch: List[Dict[str, Any]]) -> List[TaskOutcome]:
        if not batch:
            return []
        # Expand tasks by ``repeats`` so the ThreadPoolExecutor can saturate
        # with ``max_workers`` regardless of how big the manifest is.
        expanded: List[Dict[str, Any]] = []
        for t in batch:
            for _ in range(repeats):
                expanded.append(t)

        raw: List[TaskOutcome]
        if max_workers <= 1:
            raw = [_dispatch(t) for t in expanded]
        else:
            raw = []
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                # Use an explicit submission list (dict-keyed-by-task would
                # collapse duplicates when repeats > 1). Preserve the
                # manifest-order → repeat-order correspondence so we can
                # still sort by task_id below.
                futures = [pool.submit(_dispatch, t) for t in expanded]
                for fut in as_completed(futures):
                    raw.append(fut.result())

        # Bucket by task_id (raw may be out of order due to as_completed).
        by_id: Dict[str, List[TaskOutcome]] = {}
        for o in raw:
            by_id.setdefault(o.task_id, []).append(o)

        # Aggregate + preserve manifest order.
        order = {t["task_id"]: i for i, t in enumerate(batch)}
        aggregated: List[TaskOutcome] = []
        for t in batch:
            bucket = by_id.get(t["task_id"], [])
            if not bucket:
                continue
            aggregated.append(_aggregate_repeats(bucket))
        aggregated.sort(key=lambda o: order.get(o.task_id, 1_000_000))
        return aggregated

    cascade_truncated = False
    s1_outcomes = _run_batch(stage1)
    outcomes.extend(s1_outcomes)

    # Cascade short-circuit. Uses continuous ``score`` when available
    # (threshold 0.3 on per-task mean), binary ``success`` as fallback.
    # Previous behavior (binary-only) was wrong under continuous scoring:
    # a folder whose stage-1 tasks all hit 0.92 mean_score but fail the
    # binary gate would still short-circuit, truncating the composite to
    # a 2-task mean and producing inflated-looking scores.
    def _effective_score(o: TaskOutcome) -> float:
        if o.score is not None:
            return float(o.score)
        return 1.0 if o.success else 0.0

    if cascade and stage1 and all(_effective_score(o) < 0.3 for o in s1_outcomes):
        cascade_truncated = True
    else:
        outcomes.extend(_run_batch(rest))

    # --- aggregate ---
    n = len(outcomes)
    n_succ = sum(1 for o in outcomes if o.success)
    success_rate = (n_succ / n) if n else 0.0
    success_calls = [o.tool_calls for o in outcomes if o.success]
    tool_calls_per_success = (
        sum(success_calls) / len(success_calls) if success_calls else 0.0
    )
    avg_tool_calls = sum(o.tool_calls for o in outcomes) / n if n else 0.0

    # Continuous score: per-task score for tasks that have one (CTRF or
    # parsed pytest summary), binary fallback (1.0 if success else 0.0)
    # for tasks that don't. This keeps the mean interpretable as
    # "assertion-level pass rate averaged across the benchmark" without
    # biasing toward whichever tasks happened to expose a breakdown.
    #
    # When cascade truncates, ``outcomes`` contains only stage-1 tasks,
    # so a mean over it is not comparable to a full-benchmark mean. We
    # return ``mean_score=None`` in that case so naive cross-run
    # comparisons can't silently mix 2-task and 10-task means. Callers
    # that want the truncated mean can compute it from ``per_task``.
    per_task_scores: List[float] = []
    for o in outcomes:
        if o.score is not None:
            per_task_scores.append(float(o.score))
        else:
            per_task_scores.append(1.0 if o.success else 0.0)
    scored_task_count = sum(1 for o in outcomes if o.score is not None)
    if cascade_truncated:
        mean_score = None
    else:
        mean_score = sum(per_task_scores) / n if (n and scored_task_count) else None

    composite = compute_composite(success_rate, avg_tool_calls, mean_score=mean_score)

    failures = [
        {"task_id": o.task_id, "last_msg": o.last_msg or o.notes}
        for o in outcomes
        if not o.success
    ]

    verified_count = sum(1 for o in outcomes if o.verified is True)
    unverified_count = sum(1 for o in outcomes if o.verified is None)

    return EvalResult(
        success_rate=success_rate,
        tool_calls_per_success=tool_calls_per_success,
        composite=composite,
        per_task=[o.to_dict() for o in outcomes],
        failures=failures,
        skills_folder=str(skills_folder_path),
        n_tasks=n,
        cascade_truncated=cascade_truncated,
        verified_count=verified_count,
        unverified_count=unverified_count,
        verifier_disabled=(not verify),
        mean_score=mean_score,
        scored_task_count=scored_task_count,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> int:
    ap = argparse.ArgumentParser(
        prog="python -m skill_evolve.evaluator",
        description="Evaluate a skills folder against the curated benchmark.",
    )
    ap.add_argument(
        "--skills",
        required=True,
        type=Path,
        help="path to skills folder containing SKILL.md dirs",
    )
    ap.add_argument(
        "--no-cascade", action="store_true", help="disable stage-1 short-circuit"
    )
    ap.add_argument("--max-workers", type=int, default=1)
    ap.add_argument(
        "--model",
        default=None,
        help="override model (default: env SKILL_EVOLVE_MODEL or "
        "anthropic/claude-sonnet-4.6)",
    )
    ap.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    ap.add_argument("--toolsets", default=DEFAULT_ENABLED_TOOLSETS)
    ap.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="restrict to specific sources, e.g. tblite swebench",
    )
    ap.add_argument(
        "--force-synthetic",
        action="store_true",
        help="emit synthetic placeholder result (no LLM calls)",
    )
    ap.add_argument(
        "--no-verify",
        action="store_true",
        help="skip real verification (test.sh / swebench harness); "
        "fall back to legacy 'agent finished without crashing' "
        "signal. Useful for fast plumbing tests.",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="k repeat roll-outs per task for majority-vote aggregation (default: 1).",
    )
    ap.add_argument(
        "--json", action="store_true", help="dump full EvalResult as JSON to stdout"
    )
    ap.add_argument(
        "--keep-sandbox",
        action="store_true",
        help="preserve per-task HERMES_HOME dirs under ~/.cache/skill_evolve "
        "after the run so trajectories + skill-dir state can be inspected. "
        "Default cleans them up on sandbox exit.",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    res = evaluate(
        args.skills,
        cascade=not args.no_cascade,
        max_workers=args.max_workers,
        model=args.model,
        max_turns=args.max_turns,
        enabled_toolsets=args.toolsets,
        sources=args.sources,
        force_synthetic=args.force_synthetic,
        verify=not args.no_verify,
        repeats=args.repeats,
        keep_sandbox=args.keep_sandbox,
    )

    if args.json:
        print(json.dumps(res.to_dict(), indent=2, default=str))
    else:
        print("─" * 70)
        print(f"skills_folder       : {res.skills_folder}")
        print(f"n_tasks             : {res.n_tasks}")
        print(f"success_rate        : {res.success_rate:.3f}")
        if res.mean_score is not None:
            print(
                f"mean_score          : {res.mean_score:.4f} "
                f"(continuous; {res.scored_task_count}/{res.n_tasks} tasks scored)"
            )
        print(f"tool_calls/success  : {res.tool_calls_per_success:.2f}")
        print(f"composite           : {res.composite:.4f}")
        if res.cascade_truncated:
            print("cascade_truncated   : YES (stage-1 0/N — skipped rest)")
        if res.synthetic:
            print(f"synthetic           : YES — {res.notes}")
        if res.verifier_disabled:
            print("verifier            : DISABLED (fallback to trajectory-exists)")
        else:
            print(
                f"verified            : {res.verified_count}/{res.n_tasks} "
                f"(unverified={res.unverified_count})"
            )
        print("─ per task ─")
        for p in res.per_task:
            mark = "✓" if p["success"] else "✗"
            sk = ",".join(p.get("skills_invoked") or []) or "—"
            v = p.get("verified")
            vstr = (
                "verified"
                if v is True
                else ("VERIFIED_FAIL" if v is False else p.get("verifier_status", "?"))
            )
            score = p.get("score")
            sstr = f"score={score:.3f} " if isinstance(score, (int, float)) else ""
            print(
                f"  {mark} {p['task_id']:42s} "
                f"calls={p['tool_calls']:>3d} "
                f"elapsed={p['elapsed_s']:6.1f}s "
                f"{sstr}"
                f"skills={sk} "
                f"[{vstr}]"
            )
        if res.failures:
            print("─ failures ─")
            for f in res.failures:
                msg = (f["last_msg"] or "").replace("\n", " ")[:120]
                print(f"  {f['task_id']:42s}  {msg}")

    return 0


if __name__ == "__main__":
    sys.exit(_cli())
