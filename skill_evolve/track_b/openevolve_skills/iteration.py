"""Single-iteration loop — fork of openevolve/iteration.py.

A Track-B iteration:

1. Pick an island (round-robin — simple, stable, good enough for 3
   islands × 30 generations).
2. Sample a parent uniformly from that island's MAP-Elites archive.
3. Sample a second "inspiration" program (same island, different from
   parent if possible).
4. Build a prompt (:mod:`.prompt_sampler`).
5. Ask the LLM for a patch; feed it through :mod:`.patch_parser.mutate`.
6. Evaluate the child; place it in the island archive.
7. Periodically migrate (every ``migration_interval`` generations).

Everything is synchronous — we intentionally don't replicate
openevolve's ProcessPoolExecutor worker layer, because the evaluator
itself subprocesses out to ``run_agent.py`` per task. Adding another
process layer here would be fork-overhead without benefit.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from .database import Program, ProgramDatabase, new_program_id
from .evaluator import SkillFolderEvaluator
from .folder_artifact import FolderArtifactError
from .llm_client import LLMClient, SyntheticLLM
from .patch_parser import PatchParseError, mutate
from .prompt_sampler import PromptSampler

logger = logging.getLogger(__name__)


@dataclass
class IterationResult:
    generation: int
    island: int
    parent_id: str
    child_id: Optional[str]
    op_type: (
        str  # "patch" | "rewrite" | "parse_error" | "validate_error" | "eval_error"
    )
    score_delta: float
    cell: Optional[tuple] = None
    notes: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)


def run_iteration(
    generation: int,
    database: ProgramDatabase,
    evaluator: SkillFolderEvaluator,
    llm: LLMClient,
    prompt_sampler: PromptSampler,
    *,
    island: int,
) -> IterationResult:
    """Run one evolution iteration on ``island``."""
    t0 = time.monotonic()

    # 1. Sample parent + inspiration from the island's archive.
    parent = database.sample_parent(island)
    inspiration = database.sample_inspiration(island, exclude=parent.id)

    # 2. Build prompt. If our LLM is synthetic, feed it the parent directly
    #    so it can emit a deterministic valid patch.
    if isinstance(llm, SyntheticLLM):
        llm.set_parent(parent.artifact)
    prompt = prompt_sampler.build(parent, inspiration=inspiration)

    # 3. Ask the LLM.
    try:
        response = llm.generate(system=prompt.system, user=prompt.user)
    except Exception as exc:  # pragma: no cover — network-level failures
        logger.warning("LLM generate() failed: %s", exc)
        return IterationResult(
            generation=generation,
            island=island,
            parent_id=parent.id,
            child_id=None,
            op_type="eval_error",
            score_delta=0.0,
            notes=f"llm_error: {exc}",
        )

    # 4. Parse + apply.
    try:
        child_artifact = mutate(parent.artifact, response)
    except (PatchParseError, FolderArtifactError) as exc:
        logger.warning("patch rejected: %s", exc)
        return IterationResult(
            generation=generation,
            island=island,
            parent_id=parent.id,
            child_id=None,
            op_type="parse_error",
            score_delta=0.0,
            notes=str(exc),
        )

    # kai-skills patch (2026-04-27): validate-on-write guard against
    # task-name leakage. When the evaluator is in anonymize mode, the
    # outer LLM should never see real task IDs — but a misbehaving model
    # could still emit them by guessing. Reject any patch that
    # reintroduces a redacted name into the *.md prose layer (the
    # router-readable surface). Scripts/refs are not scanned because they
    # legitimately reference task fixtures by name.
    if getattr(evaluator, "anonymize_tasks", False):
        # kai-skills patch (Phase E, 2026-04-29): dispatch through the
        # anonymizer module the Controller installed (in-file for
        # tblite, skillsbench_anonymize for skillsbench). Falls back to
        # the in-file functions if Controller wiring was bypassed
        # (synthetic LLM tests etc.).
        if getattr(evaluator, "anonymizer", None) is not None:
            _find_leaked_names = evaluator.anonymizer.find_leaked_names
        else:
            from .evaluator import find_leaked_names as _find_leaked_names
        leaks = _find_leaked_names(child_artifact, evaluator.task_id_map())
        if leaks:
            file_, name = leaks[0]
            msg = (
                f"patch reintroduces redacted task name `{name}` in "
                f"`{file_}`; rejected ({len(leaks)} total leak(s))"
            )
            logger.warning(msg)
            return IterationResult(
                generation=generation,
                island=island,
                parent_id=parent.id,
                child_id=None,
                op_type="parse_error",
                score_delta=0.0,
                notes=msg,
            )
    # kai-skills patch end

    # 5. Evaluate child.
    try:
        eval_res = evaluator.evaluate_artifact(child_artifact, program_id="")
    except Exception as exc:  # pragma: no cover
        logger.exception("evaluator crashed: %s", exc)
        return IterationResult(
            generation=generation,
            island=island,
            parent_id=parent.id,
            child_id=None,
            op_type="eval_error",
            score_delta=0.0,
            notes=f"eval_error: {exc}",
        )

    # 6. Wrap in Program, place in archive.
    child = Program(
        id=new_program_id(),
        artifact=child_artifact,
        parent_id=parent.id,
        generation=parent.generation + 1,
        iteration_found=generation,
        metrics=eval_res.metrics,
        eval_artifacts=eval_res.artifacts,
        metadata={"iteration_time_s": time.monotonic() - t0},
    )
    database.add(child, island=island)
    database.tick_generation(island)

    score_delta = child.fitness() - parent.fitness()
    from .database import cell_key as _cell_key

    cell = _cell_key(child.artifact)

    logger.info(
        "iter %d island %d: %.4f -> %.4f (Δ=%+.4f) cell=%s",
        generation,
        island,
        parent.fitness(),
        child.fitness(),
        score_delta,
        cell,
    )

    return IterationResult(
        generation=generation,
        island=island,
        parent_id=parent.id,
        child_id=child.id,
        op_type="patch",
        score_delta=score_delta,
        cell=cell,
        metrics=child.metrics,
    )
