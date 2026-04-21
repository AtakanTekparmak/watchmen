"""Wrapper that adapts ``skill_evolve.evaluator.evaluate`` to the
openevolve :class:`EvaluationResult` shape.

Responsibilities:

1. Take a :class:`Program`, write its ``FolderArtifact`` to a scratch
   directory, invoke :func:`skill_evolve.evaluator.evaluate`, collect
   the :class:`EvalResult`.
2. Surface the metrics the evolution loop needs to sort/bin programs
   (``composite``, ``success_rate``, ``tool_calls_per_success``).
3. Put failure messages + per-task ``skills_invoked`` into the
   ``artifacts`` channel so the prompt sampler can show them as
   feedback in the next mutation.

Design note: we keep this layer *thin*. All the real work (sandbox,
verifier, Docker, etc.) is inside :mod:`skill_evolve.evaluator`. Our job
here is purely translation.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from skill_evolve import evaluator as _skill_evaluator

from .database import Program
from .folder_artifact import FolderArtifact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local EvaluationResult mirror of openevolve's (metrics + artifacts channel).
# We redefine instead of importing to keep this sub-package standalone.
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    metrics: Dict[str, float]
    artifacts: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class SkillFolderEvaluator:
    """Evaluate a FolderArtifact by delegating to ``skill_evolve.evaluator``."""

    def __init__(
        self,
        *,
        force_synthetic: bool = False,
        verify: bool = True,
        cascade: bool = True,
        max_workers: int = 1,
        model: Optional[str] = None,
        repeats: int = 1,
    ) -> None:
        self.force_synthetic = force_synthetic
        self.verify = verify
        self.cascade = cascade
        self.max_workers = max_workers
        self.model = model
        self.repeats = repeats

    def evaluate_program(self, program: Program) -> EvaluationResult:
        return self.evaluate_artifact(program.artifact, program_id=program.id)

    def evaluate_artifact(
        self, artifact: FolderArtifact, *, program_id: str = ""
    ) -> EvaluationResult:
        """Materialize, evaluate, translate."""
        artifact.validate()
        with tempfile.TemporaryDirectory(prefix="track_b_eval_") as tmp:
            skills_dir = Path(tmp) / "skills"
            artifact.write_to(skills_dir)
            res = _skill_evaluator.evaluate(
                skills_dir,
                cascade=self.cascade,
                max_workers=self.max_workers,
                model=self.model,
                force_synthetic=self.force_synthetic,
                verify=self.verify,
                repeats=self.repeats,
            )
        return self._translate(res, artifact, program_id=program_id)

    # -------------------- translation --------------------

    def _translate(
        self,
        res: "_skill_evaluator.EvalResult",
        artifact: FolderArtifact,
        *,
        program_id: str,
    ) -> EvaluationResult:
        metrics: Dict[str, float] = {
            # Primary fitness.
            "composite": float(res.composite),
            # Secondary signal (useful for logging + diagnostics).
            "success_rate": float(res.success_rate),
            "tool_calls_per_success": float(res.tool_calls_per_success),
            # Coverage signals.
            "n_tasks": float(res.n_tasks),
            "verified_count": float(res.verified_count),
            "unverified_count": float(res.unverified_count),
        }
        # Continuous-scoring signal (added 2026-04-20). ``mean_score`` is
        # ``None`` when no task exposed a structured breakdown — omit from
        # metrics in that case so downstream dashboards don't pick up a
        # spurious zero.
        if res.mean_score is not None:
            metrics["mean_score"] = float(res.mean_score)
            metrics["scored_task_count"] = float(res.scored_task_count)

        # Feedback to fold into the next prompt.
        failures_blob = json.dumps(res.failures, indent=2)
        # Per-task detail. This is the structured record that downstream
        # consumers (Track B/C best_meta.json, Track C tournament rebuild)
        # read to answer "which task did evolution actually crack?". Keep
        # the schema stable — the three-state ``verified`` (True/False/None)
        # plus ``verifier_status`` is what distinguishes "we proved a pass"
        # from "we couldn't verify". ``last_msg`` is retained (trimmed) for
        # the Track A critic feedback path.
        per_task_list = [
            {
                "task_id": t.get("task_id"),
                "verified": t.get("verified"),
                "success": t.get("success"),
                "elapsed_s": t.get("elapsed_s"),
                "tool_calls": t.get("tool_calls"),
                "skills_invoked": list(t.get("skills_invoked") or []),
                "verifier_status": t.get("verifier_status"),
                "last_msg": (t.get("last_msg") or "")[:240],
            }
            for t in res.per_task
        ]

        # Aggregate invocation counts (what skills actually fired?).
        invocation_counts: Dict[str, int] = {}
        for t in per_task_list:
            for name in t["skills_invoked"]:
                invocation_counts[name] = invocation_counts.get(name, 0) + 1
        # Skills present but never invoked → dead weight signal.
        present = set(artifact.skill_names())
        invoked = set(invocation_counts.keys())
        unused = sorted(present - invoked)

        artifacts = {
            "failures": failures_blob,
            "per_task": json.dumps(per_task_list),
            "invocation_counts": json.dumps(invocation_counts, indent=2),
            "unused_skills": json.dumps(unused),
            "synthetic": "1" if res.synthetic else "0",
            "cascade_truncated": "1" if res.cascade_truncated else "0",
        }

        logger.info(
            "eval %s: composite=%.4f success=%.2f (%d tasks, %d verified)%s",
            (program_id or "")[:8],
            metrics["composite"],
            metrics["success_rate"],
            int(metrics["n_tasks"]),
            int(metrics["verified_count"]),
            " [synthetic]" if res.synthetic else "",
        )
        return EvaluationResult(metrics=metrics, artifacts=artifacts)
