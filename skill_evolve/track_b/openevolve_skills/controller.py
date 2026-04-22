"""Top-level evolution controller — fork of openevolve/controller.py.

Glues Database + Evaluator + LLM + PromptSampler + Iteration together
into a single synchronous run. Persists history + archive on exit.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .database import (
    Program,
    ProgramDatabase,
    cell_key,
    new_program_id,
)
from .evaluator import SkillFolderEvaluator
from .folder_artifact import FolderArtifact
from .islands import seed_variants
from .iteration import IterationResult, run_iteration
from .llm_client import LLMClient
from .prompt_sampler import PromptSampler

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    num_generations: int = 30
    num_islands: int = 3
    migration_interval: int = 5
    rng_seed: Optional[int] = 0


@dataclass
class RunResult:
    best: Optional[Program]
    history: List[Dict[str, Any]]
    output_dir: Path


def run_evolution(
    *,
    seed_path: Path,
    out_dir: Path,
    config: RunConfig,
    evaluator: SkillFolderEvaluator,
    llm: LLMClient,
    prompt_sampler: Optional[PromptSampler] = None,
) -> RunResult:
    """End-to-end: seed → evaluate-seeds → iterate → dump archive.

    Persists:
      * ``out_dir/history.jsonl``  — one line per attempted iteration.
      * ``out_dir/archive/``       — MAP-Elites archive, one cell per dir.
      * ``out_dir/best/``          — single best program as a skill folder.
      * ``out_dir/run_meta.json``  — summary metadata.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_sampler = prompt_sampler or PromptSampler()

    history_path = out_dir / "history.jsonl"
    # Truncate if rerunning.
    history_path.write_text("", encoding="utf-8")

    # --- seed each island --------------------------------------------------
    # Include executable files alongside SKILL.md so code-bearing seeds
    # (hermes's built-in subdirs: scripts/, references/, templates/,
    # assets/) survive the load. Prior include_exts={".md"} silently
    # dropped every .sh / .py / .jq in the seed — see LOGIC_GAPS.md.
    base_artifact = FolderArtifact.from_path(
        seed_path,
        include_exts={".md", ".sh", ".py", ".jq", ".json", ".yaml", ".yml", ".txt"},
    )
    base_artifact.validate()
    variants = seed_variants(base_artifact, num_islands=config.num_islands)

    db = ProgramDatabase(
        num_islands=config.num_islands,
        migration_interval=config.migration_interval,
        rng_seed=config.rng_seed,
    )

    logger.info("seeding %d islands from %s", config.num_islands, seed_path)
    for i, variant in enumerate(variants):
        logger.info(
            "  island %d seed: %d skills, %d files, %d bytes",
            i,
            variant.num_skills(),
            len(variant),
            variant.total_bytes(),
        )
        eval_res = evaluator.evaluate_artifact(variant, program_id="")
        seed_prog = Program(
            id=new_program_id(),
            artifact=variant,
            parent_id=None,
            generation=0,
            iteration_found=0,
            metrics=eval_res.metrics,
            eval_artifacts=eval_res.artifacts,
            metadata={"role": "seed", "island": i},
        )
        db.add(seed_prog, island=i)
        _append_history(
            history_path,
            IterationResult(
                generation=0,
                island=i,
                parent_id="",
                child_id=seed_prog.id,
                op_type="seed",
                score_delta=0.0,
                cell=cell_key(variant),
                metrics=seed_prog.metrics,
            ),
        )

    # --- iterate -----------------------------------------------------------
    t_start = time.monotonic()
    for gen in range(1, config.num_generations + 1):
        island = (gen - 1) % config.num_islands
        try:
            res = run_iteration(
                gen,
                db,
                evaluator,
                llm,
                prompt_sampler,
                island=island,
            )
        except LookupError as exc:
            logger.warning("iter %d: island %d empty (%s); skipping", gen, island, exc)
            continue
        _append_history(history_path, res)

        # Periodic migration.
        if db.should_migrate():
            moves = db.migrate()
            for src, dst, pid in moves:
                _append_history(
                    history_path,
                    IterationResult(
                        generation=gen,
                        island=dst,
                        parent_id=pid,
                        child_id=pid,
                        op_type="migrate",
                        score_delta=0.0,
                        cell=cell_key(db.programs[pid].artifact),
                        notes=f"from_island={src}",
                        metrics=db.programs[pid].metrics,
                    ),
                )

    elapsed = time.monotonic() - t_start

    # --- persist archive + best --------------------------------------------
    archive_dir = out_dir / "archive"
    db.dump_archive(archive_dir)

    best = db.best()
    if best is not None:
        best_dir = out_dir / "best"
        if best_dir.exists():
            import shutil

            shutil.rmtree(best_dir)
        best.artifact.write_to(best_dir)
        # Surface the evaluator's side-channel artifacts (per-task detail,
        # invocation counts, unused-skills list, etc.) into best_meta so
        # post-run inspection can answer "which task did evolution crack?"
        # without having to re-evaluate. ``per_task`` / ``invocation_counts``
        # / ``unused_skills`` are JSON-encoded inside eval_artifacts; decode
        # so the best_meta.json is a single well-typed document.
        eval_artifacts_payload = _decode_eval_artifacts(best.eval_artifacts)
        (out_dir / "best_meta.json").write_text(
            json.dumps(
                {
                    "id": best.id,
                    "generation": best.generation,
                    "metrics": best.metrics,
                    "fitness": best.fitness(),
                    "cell": list(cell_key(best.artifact)),
                    "eval_artifacts": eval_artifacts_payload,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # --- summary -----------------------------------------------------------
    num_cells = sum(len(c) for c in db.islands)
    run_meta = {
        "seed_path": str(seed_path),
        "num_generations": config.num_generations,
        "num_islands": config.num_islands,
        "migration_interval": config.migration_interval,
        "elapsed_s": elapsed,
        "total_programs": len(db.programs),
        "archive_cells": num_cells,
        "best_fitness": best.fitness() if best else None,
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2), encoding="utf-8"
    )

    history = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return RunResult(best=best, history=history, output_dir=out_dir)


def _decode_eval_artifacts(raw: Dict[str, str]) -> Dict[str, Any]:
    """Decode the evaluator's ``artifacts`` side-channel for best_meta.json.

    ``EvaluationResult.artifacts`` is a flat ``Dict[str, str]`` because the
    openevolve artifact channel is defined that way, but several of those
    strings are themselves JSON blobs (``per_task``, ``invocation_counts``,
    ``unused_skills``, ``failures``). Decode them to native types so the
    final best_meta.json is a single well-typed document instead of a
    document-with-embedded-strings-of-JSON.
    """
    decoded: Dict[str, Any] = {}
    for k, v in (raw or {}).items():
        if not isinstance(v, str):
            decoded[k] = v
            continue
        stripped = v.lstrip()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                decoded[k] = json.loads(v)
                continue
            except json.JSONDecodeError:
                pass
        decoded[k] = v
    return decoded


def _append_history(path: Path, res: IterationResult) -> None:
    payload: Dict[str, Any] = {
        "generation": res.generation,
        "island": res.island,
        "parent_id": res.parent_id,
        "child_id": res.child_id,
        "op_type": res.op_type,
        "score_delta": res.score_delta,
        "cell": list(res.cell) if res.cell is not None else None,
        "notes": res.notes,
        "metrics": res.metrics,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
