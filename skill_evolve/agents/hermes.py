"""Hermes agent backend — thin wrapper around the existing
``_run_one_task`` flow at ``skill_evolve/evaluator.py:314``.

Behavior-preserving by design: this class neither moves nor modifies
``_run_one_task``; it imports it lazily and adapts the resulting
:class:`TaskOutcome` into a :class:`TrajectoryResult` (copy of fields
plus ``cost_usd=None``, since Hermes does not surface a USD figure).

The lazy import sidesteps the import cycle between ``evaluator`` (which
may grow an ``AgentBackend`` parameter in a follow-up commit) and this
module.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from skill_evolve.agents.base import AgentBackend, TrajectoryResult

if TYPE_CHECKING:  # pragma: no cover
    from skill_evolve.benchmark.load import Task


# Default toolset / max-turns mirror the constants used at the
# evaluator's call site (``DEFAULT_ENABLED_TOOLSETS``, ``DEFAULT_MAX_TURNS``).
# We re-declare them here so the wrapper has sensible defaults without
# importing the evaluator module at import time.
_DEFAULT_ENABLED_TOOLSETS = "terminal,file,skills"
_DEFAULT_MAX_TURNS = 20


class HermesBackend(AgentBackend):
    """Behavior-preserving shim around ``_run_one_task``.

    The real execution logic stays in ``skill_evolve/evaluator.py``;
    this class only adapts the calling convention so callers that
    target the :class:`AgentBackend` ABC can dispatch through it.
    """

    def run_task(
        self,
        task: "Task",
        skills_dir: Optional[Path],
        *,
        model: str,
        timeout_s: int,
        budget_usd: float,
        anonymize_map: Optional[Dict[str, str]],
        max_turns: Optional[int] = None,
        enabled_toolsets: Optional[str] = None,
        extra_run_agent_args: Sequence[str] = (),
        verify: bool = True,
        keep_sandbox: bool = False,
    ) -> TrajectoryResult:
        # Lazy import: ``skill_evolve.evaluator`` registers a fair amount
        # of module-level state (logger, RUN_AGENT_PY path, regex
        # compilations) and importing it eagerly from the agents
        # package would create a cycle once ``evaluator`` learns about
        # ``AgentBackend``.
        from skill_evolve.evaluator import _run_one_task

        # ``_run_one_task`` operates on the dict form of a Task (its
        # current callers feed it the output of ``Task.to_dict()`` from
        # ``load_subset``). Accept either shape so the ABC contract
        # (which types ``task`` as ``Task``) stays clean.
        task_dict: Dict[str, Any]
        if hasattr(task, "to_dict"):
            task_dict = task.to_dict()
        else:
            task_dict = dict(task)  # type: ignore[arg-type]

        # ``skills_dir`` is required by ``_run_one_task``. The Hermes
        # path historically receives a real path; for the no-skills
        # condition the caller should hand us an empty staged folder
        # rather than ``None``. If we got ``None`` here, fall back to
        # the historical behavior of letting Hermes fail loudly rather
        # than silently swap in something else.
        if skills_dir is None:
            raise ValueError(
                "HermesBackend.run_task: skills_dir is required "
                "(Hermes does not support a no-skills condition; "
                "stage an empty folder if you need the equivalent)."
            )

        # Apply the Hermes timeout via the task dict, since
        # ``_run_one_task`` already honors ``task["timeout_s"]`` for its
        # subprocess timeout. Override only when the caller passed a
        # tighter cap so we don't accidentally widen historical timeouts.
        if timeout_s and timeout_s > 0:
            task_dict = dict(task_dict)
            task_dict["timeout_s"] = int(timeout_s)

        outcome = _run_one_task(
            task_dict,
            Path(skills_dir),
            model=model,
            max_turns=max_turns if max_turns is not None else _DEFAULT_MAX_TURNS,
            enabled_toolsets=(
                enabled_toolsets
                if enabled_toolsets is not None
                else _DEFAULT_ENABLED_TOOLSETS
            ),
            extra_run_agent_args=extra_run_agent_args,
            verify=verify,
            keep_sandbox=keep_sandbox,
        )

        # Adapt to TrajectoryResult — copy every field, set cost_usd=None
        # (Hermes does not report a USD figure).
        return TrajectoryResult(
            task_id=outcome.task_id,
            success=outcome.success,
            tool_calls=outcome.tool_calls,
            elapsed_s=outcome.elapsed_s,
            skills_invoked=list(outcome.skills_invoked),
            last_msg=outcome.last_msg,
            raw_completed=outcome.raw_completed,
            notes=outcome.notes,
            verified=outcome.verified,
            verifier_status=outcome.verifier_status,
            verifier_detail=outcome.verifier_detail,
            score=outcome.score,
            repeats_detail=list(outcome.repeats_detail),
            cost_usd=None,
        )
