"""Agent backend ABC and trajectory result dataclass.

Defines the abstract interface every agent backend (Hermes, bench-cli)
must implement so the evaluator harness can dispatch tasks against
multiple agent runtimes through a single seam.

The ``TrajectoryResult`` dataclass mirrors the shape of
:class:`skill_evolve.evaluator.TaskOutcome` (line 120 of evaluator.py)
field-for-field, plus a single new ``cost_usd`` field that the bench
CLI's ``total_cost_usd`` reporting flows into. A
``to_task_outcome()`` adapter drops the extra and returns the existing
``TaskOutcome`` so legacy code paths (everything that already consumes
``TaskOutcome`` today) keep working untouched.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover — import-time-only refs
    from skill_evolve.benchmark.load import Task
    from skill_evolve.evaluator import TaskOutcome


@dataclass
class TrajectoryResult:
    """Result of running a single task through an agent backend.

    Field shape mirrors :class:`skill_evolve.evaluator.TaskOutcome`
    (defined at ``skill_evolve/evaluator.py:120``) exactly, so
    ``to_task_outcome()`` is a pure projection. The single addition is
    ``cost_usd`` for backends (notably bench-cli) that report a
    per-task USD spend in their JSON output.
    """

    task_id: str
    success: bool
    tool_calls: int
    elapsed_s: float
    skills_invoked: List[str] = field(default_factory=list)
    last_msg: str = ""
    raw_completed: bool = False
    notes: str = ""
    verified: Optional[bool] = None
    verifier_status: str = "not_run"
    verifier_detail: str = ""
    score: Optional[float] = None
    repeats_detail: List[Dict[str, Any]] = field(default_factory=list)
    # New: per-task USD spend for backends that report one (bench CLI).
    # ``None`` when the backend doesn't expose cost (e.g. Hermes).
    cost_usd: Optional[float] = None

    def to_task_outcome(self) -> "TaskOutcome":
        """Project to the legacy ``TaskOutcome`` (drops ``cost_usd``).

        Existing call sites consume ``TaskOutcome``; this adapter lets
        backends return the richer ``TrajectoryResult`` while keeping
        the legacy seam working without modification.
        """
        # Lazy import to avoid a hard cycle: evaluator imports
        # benchmark which may eventually import this module.
        from skill_evolve.evaluator import TaskOutcome

        return TaskOutcome(
            task_id=self.task_id,
            success=self.success,
            tool_calls=self.tool_calls,
            elapsed_s=self.elapsed_s,
            skills_invoked=list(self.skills_invoked),
            last_msg=self.last_msg,
            raw_completed=self.raw_completed,
            notes=self.notes,
            verified=self.verified,
            verifier_status=self.verifier_status,
            verifier_detail=self.verifier_detail,
            score=self.score,
            repeats_detail=list(self.repeats_detail),
        )


class AgentBackend(ABC):
    """Abstract base class every agent backend must implement.

    Implementations:

      * :class:`skill_evolve.agents.hermes.HermesBackend` — wraps the
        existing ``_run_one_task`` Hermes/Docker subprocess flow at
        ``skill_evolve/evaluator.py:314``. Behavior-preserving.
      * :class:`skill_evolve.agents.bench_cli.BenchCliBackend` — shells
        out to the official ``bench eval create -f <yaml> -t <task_dir>
        -a claude-code -m <model>`` CLI used by SkillsBench.

    There is no ``ClaudeCodeBackend`` — see plan_0.md D-1.
    """

    @abstractmethod
    def run_task(
        self,
        task: "Task",
        skills_dir: Optional[Path],
        *,
        model: str,
        timeout_s: int,
        budget_usd: float,
        anonymize_map: Optional[Dict[str, str]],
    ) -> TrajectoryResult:
        """Run one task and return its trajectory result.

        Args:
            task: hydrated task from ``skill_evolve.benchmark.load.Task``.
            skills_dir: candidate skill bundle to mount, or ``None`` for
                the no-skills condition.
            model: model identifier passed to the underlying agent.
            timeout_s: hard wall-clock cap for the task.
            budget_usd: per-task USD ceiling. Backends that expose
                ``total_cost_usd`` (bench-cli) should annotate
                ``notes="budget_exceeded"`` if the cap is breached.
            anonymize_map: optional ``real_id → alias`` map. Backends
                must apply it to any outer-LLM-visible fields (notably
                ``last_msg``) before returning.
        """
        raise NotImplementedError
