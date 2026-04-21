"""Track C top-level controller.

Fork of :mod:`skill_evolve.track_b.openevolve_skills.controller` with:

* The patch-mutation iteration swapped for the A/B/AB tournament
  iteration in :mod:`.iteration`.
* Per-generation ``tournament_winner`` / ``tournament_role`` logged to
  ``history.jsonl``.
* A tournament-stats block (B-wins / AB-wins / A-wins per island) added
  to ``run_meta.json`` / final report.

The MAP-Elites archive, island seeding, and migration logic are imported
verbatim from Track B — we do not re-implement any of them here.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from skill_evolve.track_a.llm import LLMClient as TrackALLMClient

from skill_evolve.track_b.openevolve_skills.database import (
    Program,
    ProgramDatabase,
    cell_key,
    new_program_id,
)
from skill_evolve.track_b.openevolve_skills.evaluator import SkillFolderEvaluator
from skill_evolve.track_b.openevolve_skills.folder_artifact import FolderArtifact
from skill_evolve.track_b.openevolve_skills.islands import seed_variants

from .iteration import IterationResult, run_iteration

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
    tournament_stats: Dict[str, Any] = field(default_factory=dict)


def run_evolution(
    *,
    seed_path: Path,
    out_dir: Path,
    config: RunConfig,
    evaluator: SkillFolderEvaluator,
    llm: TrackALLMClient,
) -> RunResult:
    """End-to-end Track C run — seed → iterate (tournament) → persist.

    Outputs under ``out_dir``:
      * ``history.jsonl``      — one line per attempted iteration / migrate / seed.
      * ``archive/``           — MAP-Elites archive dump (per-cell dirs).
      * ``best/``              — the single best folder found.
      * ``run_meta.json``      — run summary including tournament stats.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history_path = out_dir / "history.jsonl"
    history_path.write_text("", encoding="utf-8")

    rng = random.Random(config.rng_seed)

    # --- seed each island --------------------------------------------------
    base_artifact = FolderArtifact.from_path(seed_path, include_exts={".md"})
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
        ev = evaluator.evaluate_artifact(variant)
        seed_prog = Program(
            id=new_program_id(),
            artifact=variant,
            parent_id=None,
            generation=0,
            iteration_found=0,
            metrics=ev.metrics,
            eval_artifacts=ev.artifacts,
            metadata={
                "role": "seed",
                "island": i,
                "a_wins_streak": 0,
                "saturated": False,
            },
        )
        db.add(seed_prog, island=i)
        _append_seed_line(history_path, i, seed_prog, cell_key(variant))

    # --- per-island tournament counters ------------------------------------
    win_counts: List[Dict[str, int]] = [
        {"A": 0, "B": 0, "AB": 0} for _ in range(config.num_islands)
    ]
    op_counts: Dict[str, int] = {}

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
                island=island,
                rng=rng,
            )
        except LookupError as exc:
            logger.warning("iter %d: island %d empty (%s); skipping", gen, island, exc)
            continue

        win_counts[island][res.tournament_winner] = (
            win_counts[island].get(res.tournament_winner, 0) + 1
        )
        if res.op:
            op_name = (
                res.op.get("op", "unknown") if isinstance(res.op, dict) else str(res.op)
            )
            op_counts[op_name] = op_counts.get(op_name, 0) + 1

        _append_iteration_line(history_path, res)

        # Periodic migration (Track B's implementation).
        if db.should_migrate():
            moves = db.migrate()
            for src, dst, pid in moves:
                _append_migrate_line(history_path, gen, dst, src, pid, db)

    elapsed = time.monotonic() - t_start

    # --- persist archive + best --------------------------------------------
    archive_dir = out_dir / "archive"
    db.dump_archive(archive_dir)

    best = db.best()
    if best is not None:
        best_dir = out_dir / "best"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        best.artifact.write_to(best_dir)
        # See Track B controller: decode eval_artifacts into native types
        # so best_meta.json is one well-typed document rather than having
        # JSON-strings-inside-JSON. Track C Programs come from
        # ``tournament_mutate`` which already routes ``ev.artifacts`` into
        # ``Program.eval_artifacts``; no additional plumbing needed here.
        eval_artifacts_payload = _decode_eval_artifacts(best.eval_artifacts)
        (out_dir / "best_meta.json").write_text(
            json.dumps(
                {
                    "id": best.id,
                    "generation": best.generation,
                    "metrics": best.metrics,
                    "fitness": best.fitness(),
                    "cell": list(cell_key(best.artifact)),
                    "tournament_role": best.metadata.get("tournament_role"),
                    "tournament_winner": best.metadata.get("tournament_winner"),
                    "eval_artifacts": eval_artifacts_payload,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # --- tournament-stats summary ------------------------------------------
    total = {"A": 0, "B": 0, "AB": 0}
    for counts in win_counts:
        for k, v in counts.items():
            total[k] = total.get(k, 0) + v
    denom = max(1, sum(total.values()))
    win_rate_by_role = {k: (v / denom) for k, v in total.items()}

    tournament_stats = {
        "per_island": win_counts,
        "totals": total,
        "win_rate_by_role": win_rate_by_role,
        "op_counts": op_counts,
    }

    num_cells = sum(len(c) for c in db.islands)
    run_meta = {
        "track": "C",
        "seed_path": str(seed_path),
        "num_generations": config.num_generations,
        "num_islands": config.num_islands,
        "migration_interval": config.migration_interval,
        "elapsed_s": elapsed,
        "total_programs": len(db.programs),
        "archive_cells": num_cells,
        "best_fitness": best.fitness() if best else None,
        "tournament_stats": tournament_stats,
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2), encoding="utf-8"
    )

    history = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return RunResult(
        best=best,
        history=history,
        output_dir=out_dir,
        tournament_stats=tournament_stats,
    )


# ---------------------------------------------------------------------------
# history.jsonl writers — one line per event, JSON object per line.
# ---------------------------------------------------------------------------


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


def _append_seed_line(path: Path, island: int, seed: Program, cell: tuple) -> None:
    payload = {
        "event": "seed",
        "generation": 0,
        "island": island,
        "parent_id": "",
        "child_id": seed.id,
        "op_type": "seed",
        "cell": list(cell),
        "metrics": seed.metrics,
        "tournament_winner": None,
        "tournament_role": "seed",
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _append_iteration_line(path: Path, res: IterationResult) -> None:
    payload: Dict[str, Any] = {
        "event": "iteration",
        "generation": res.generation,
        "island": res.island,
        "parent_id": res.parent_id,
        "children_ids": res.children_ids,
        "op_type": res.op_type,
        "tournament_winner": res.tournament_winner,
        "tournament_role_per_child": res.tournament_role_per_child,
        "cell_per_child": {cid: list(c) for cid, c in res.cell_per_child.items()},
        "score_A": res.score_A,
        "score_B": res.score_B,
        "score_AB": res.score_AB,
        "score_delta": res.score_delta,
        "op": res.op,
        "notes": res.notes,
        "metrics": res.metrics,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _append_migrate_line(
    path: Path,
    gen: int,
    dst_island: int,
    src_island: int,
    pid: str,
    db: ProgramDatabase,
) -> None:
    prog = db.programs[pid]
    payload = {
        "event": "migrate",
        "generation": gen,
        "island": dst_island,
        "src_island": src_island,
        "program_id": pid,
        "op_type": "migrate",
        "cell": list(cell_key(prog.artifact)),
        "metrics": prog.metrics,
        "tournament_winner": None,
        "tournament_role": prog.metadata.get("tournament_role"),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
