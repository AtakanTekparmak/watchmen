"""Track C single-iteration loop.

Fork of :mod:`skill_evolve.track_b.openevolve_skills.iteration` with one
surgical change: the LLM-patch-based mutation step is replaced by the
A/B/AB tournament from :mod:`.tournament`. The rest of the Track B
machinery (parent sampling, archive placement, migration, generation
counter) is reused unchanged — we import Track B's classes directly.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from skill_evolve.track_a.llm import LLMClient as TrackALLMClient

from skill_evolve.track_b.openevolve_skills.database import (
    Program,
    ProgramDatabase,
    cell_key,
)
from skill_evolve.track_b.openevolve_skills.evaluator import SkillFolderEvaluator

from .tournament import TournamentResult, tournament_mutate, SATURATION_STREAK

logger = logging.getLogger(__name__)


# Max re-draws when a parent-sampling attempt returns a "saturated" program.
# We cap this so we never infinite-loop when every archive cell is saturated.
MAX_SATURATED_RESAMPLES = 4


@dataclass
class IterationResult:
    generation: int
    island: int
    parent_id: str
    children_ids: List[str]
    tournament_winner: str            # "A" | "B" | "AB"
    tournament_role_per_child: Dict[str, str]  # child_id -> "B" | "AB"
    op_type: str                       # "tournament" | "tournament_error" | "seed" | "migrate"
    score_delta: float                 # winner_score - score_A (0 if A won)
    score_A: float
    score_B: Optional[float]
    score_AB: Optional[float]
    cell_per_child: Dict[str, tuple] = field(default_factory=dict)
    notes: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)  # winner metrics
    op: Optional[Dict[str, Any]] = None


def _sample_non_saturated_parent(
    db: ProgramDatabase,
    island: int,
    *,
    max_tries: int = MAX_SATURATED_RESAMPLES,
) -> Program:
    """Sample a parent from ``island``, re-drawing if the draw is saturated.

    Falls back to the saturated draw if we exceed ``max_tries`` — i.e.
    every cell in the island has been marked saturated. That's fine:
    at that point every parent deserves another shot anyway (archive
    residents can accumulate streak over time).
    """
    parent = db.sample_parent(island)
    for _ in range(max_tries):
        if not parent.metadata.get("saturated"):
            return parent
        parent = db.sample_parent(island)
    return parent


def run_iteration(
    generation: int,
    database: ProgramDatabase,
    evaluator: SkillFolderEvaluator,
    llm: TrackALLMClient,
    *,
    island: int,
    rng: Optional[random.Random] = None,
) -> IterationResult:
    """Run one Track-C evolution iteration on ``island``.

    Flow:

    1. Sample a parent (preferring non-saturated ones).
    2. Run the A/B/AB tournament — produces 0, 1, or 2 new Program
       objects (B and/or AB, subject to validation + evaluation).
    3. Place each new Program in the island's archive.
    4. Bump the island's generation counter.
    """
    t0 = time.monotonic()

    try:
        parent = _sample_non_saturated_parent(database, island)
    except LookupError:
        raise

    try:
        result: TournamentResult = tournament_mutate(
            parent, evaluator, llm, generation=generation, rng=rng,
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.exception("tournament_mutate crashed: %s", exc)
        return IterationResult(
            generation=generation, island=island,
            parent_id=parent.id, children_ids=[],
            tournament_winner="A",
            tournament_role_per_child={},
            op_type="tournament_error",
            score_delta=0.0,
            score_A=float(parent.fitness()),
            score_B=None, score_AB=None,
            notes=f"tournament_error: {exc}",
        )

    # Place every new Program in the archive (insert-all-valid).
    children_ids: List[str] = []
    role_per_child: Dict[str, str] = {}
    cell_per_child: Dict[str, tuple] = {}
    for prog in result.new_programs:
        database.add(prog, island=island)
        children_ids.append(prog.id)
        role_per_child[prog.id] = str(prog.metadata.get("tournament_role", ""))
        cell_per_child[prog.id] = cell_key(prog.artifact)

    database.tick_generation(island)

    # Metrics of the tournament's winner (A's known metrics if A wins).
    if result.winner_label == "A":
        winner_metrics = dict(parent.metrics)
    else:
        # Find the Program that corresponds to the winning label.
        winner_metrics = {}
        for prog in result.new_programs:
            if prog.metadata.get("tournament_role") == result.winner_label:
                winner_metrics = dict(prog.metrics)
                break

    elapsed = time.monotonic() - t0
    winner_metrics.setdefault("iteration_time_s", elapsed)

    logger.info(
        "iter %d island %d: winner=%s  A=%.4f B=%s AB=%s  children=%d  "
        "a_streak=%d saturated=%s",
        generation, island, result.winner_label,
        result.score_A,
        ("%.4f" % result.score_B) if result.score_B != float("-inf") else "—",
        ("%.4f" % result.score_AB) if result.score_AB != float("-inf") else "—",
        len(children_ids),
        int(parent.metadata.get("a_wins_streak", 0)),
        bool(parent.metadata.get("saturated", False)),
    )

    return IterationResult(
        generation=generation, island=island,
        parent_id=parent.id, children_ids=children_ids,
        tournament_winner=result.winner_label,
        tournament_role_per_child=role_per_child,
        op_type="tournament",
        score_delta=(result.winner_score - result.score_A),
        score_A=result.score_A,
        score_B=(None if result.score_B == float("-inf") else result.score_B),
        score_AB=(None if result.score_AB == float("-inf") else result.score_AB),
        cell_per_child=cell_per_child,
        notes=result.notes,
        metrics=winner_metrics,
        op=(result.op.to_dict() if result.op else None),
    )
