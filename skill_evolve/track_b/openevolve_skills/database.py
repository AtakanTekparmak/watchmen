"""Program + ProgramDatabase — forked from openevolve/database.py.

Key deltas vs upstream:

* ``Program.code: str`` → ``Program.artifact: FolderArtifact``. Every
  place that hashed / persisted code now routes through the artifact's
  canonical blob.
* Feature dimensions are hard-coded to the three Track-B features
  (``num_skills``, ``total_tokens``, ``avg_specificity``), with
  per-dimension bin edges defined in this module. Upstream uses
  min-max scaling over an unbounded feature space — overkill for us.
* Novelty judge + embeddings stripped. The MAP-Elites cell already
  enforces enough diversity for a 60-cell archive; adding an embedding
  pass at this scale would cost more than it buys.
* Added :meth:`migrate` (5-generation cadence default, configurable).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .folder_artifact import FolderArtifact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature binning
# ---------------------------------------------------------------------------

# Right-open bin edges (value < edge[i] → bin i).
FEATURE_BIN_EDGES: Dict[str, List[float]] = {
    # 4 bins: 1-3, 4-7, 8-12, 13-20.
    "num_skills": [4, 8, 13, float("inf")],
    # 5 bins: <1k, 1-3k, 3-6k, 6-10k, >10k tokens (rough tokens = chars/4).
    "total_tokens": [1_000, 3_000, 6_000, 10_000, float("inf")],
    # 3 bins: low/mid/high specificity. Score = avg description length
    # (chars) across SKILL.md frontmatter — shorter desc ≈ more generic.
    "avg_specificity": [120, 240, float("inf")],
}

FEATURE_DIMENSIONS: List[str] = list(FEATURE_BIN_EDGES.keys())

# Number of cells = 4 * 5 * 3 = 60 (as spec'd).


def assign_bin(value: float, edges: Sequence[float]) -> int:
    """First bin index ``i`` such that ``value < edges[i]`` (clamped)."""
    for i, edge in enumerate(edges):
        if value < edge:
            return i
    return len(edges) - 1


def compute_features(artifact: FolderArtifact) -> Dict[str, float]:
    """Raw (non-binned) feature vector for ``artifact``."""
    # Rough token count: chars/4. Close enough for binning.
    total_chars = artifact.total_bytes()
    total_tokens = total_chars / 4.0

    # Specificity heuristic: mean length of the `description:` frontmatter
    # field across all SKILL.md files. Proxy: short desc → generic skill.
    descs: List[int] = []
    for path, content in artifact.items():
        if not path.endswith("SKILL.md"):
            continue
        desc = _extract_frontmatter_field(content, "description")
        if desc is not None:
            descs.append(len(desc))
    avg_spec = (sum(descs) / len(descs)) if descs else 0.0

    return {
        "num_skills": float(artifact.num_skills()),
        "total_tokens": total_tokens,
        "avg_specificity": avg_spec,
    }


def _extract_frontmatter_field(content: str, key: str) -> Optional[str]:
    """Minimal YAML-ish frontmatter parser — pulls ``key: value``.

    We intentionally avoid a PyYAML dep; SKILL.md frontmatter is simple
    enough that a line-oriented scan covers every case in seed_skills/.
    Returns the *first* match after ``---`` and before the second ``---``.
    """
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for i, line in enumerate(lines[1:], start=1):
        stripped = line.strip()
        if stripped == "---":
            return None
        if stripped.startswith(f"{key}:"):
            return stripped.split(":", 1)[1].strip().strip('"').strip("'")
    return None


def cell_key(artifact: FolderArtifact) -> Tuple[int, ...]:
    """MAP-Elites cell key (tuple of bin indices)."""
    feats = compute_features(artifact)
    return tuple(
        assign_bin(feats[dim], FEATURE_BIN_EDGES[dim])
        for dim in FEATURE_DIMENSIONS
    )


# ---------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------

@dataclass
class Program:
    """Fork of openevolve.database.Program — ``artifact`` replaces ``code``."""

    id: str
    artifact: FolderArtifact
    parent_id: Optional[str] = None
    generation: int = 0
    timestamp: float = field(default_factory=time.time)
    iteration_found: int = 0

    # Fitness metrics (mapped from EvalResult).
    metrics: Dict[str, float] = field(default_factory=dict)

    # Derived scalar features (cached for sampling-time sort).
    complexity: float = 0.0
    diversity: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)

    # Artifacts side-channel from the evaluator (failures, skills_invoked).
    eval_artifacts: Dict[str, str] = field(default_factory=dict)

    def fitness(self) -> float:
        """Primary scalar used for 'best in cell' and 'best overall'.

        We use ``composite`` (the skill_evolve evaluator's final score)
        as the single fitness signal — per spec, no LLM-as-judge.
        """
        return float(self.metrics.get("composite", 0.0))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # FolderArtifact becomes a dict of files; restorable via from_dict.
        d["artifact"] = {"files": dict(self.artifact.files)}
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Program":
        art = data["artifact"]
        if isinstance(art, dict) and "files" in art:
            art = FolderArtifact(files=art["files"])
        elif isinstance(art, FolderArtifact):
            pass
        else:
            raise ValueError(f"unrecognized artifact payload: {type(art)}")
        kwargs = dict(data)
        kwargs["artifact"] = art
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

class ProgramDatabase:
    """Island-partitioned MAP-Elites archive of :class:`Program` objects."""

    def __init__(
        self,
        *,
        num_islands: int = 3,
        migration_interval: int = 5,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self._rng = _SeededRandom(rng_seed)

        self.programs: Dict[str, Program] = {}
        # Per-island MAP-Elites archive: cell_key → program_id.
        self.islands: List[Dict[Tuple[int, ...], str]] = [
            {} for _ in range(num_islands)
        ]
        self.island_generations: List[int] = [0] * num_islands
        self.last_migration_generation: int = 0

        self.best_program_id: Optional[str] = None

    # -------------------- add / place --------------------

    def add(self, program: Program, *, island: int) -> bool:
        """Place ``program`` in ``island``'s archive cell. Returns True if it
        *won* its cell (new or strictly better), False otherwise.
        """
        if not 0 <= island < self.num_islands:
            raise ValueError(f"island {island} out of range")
        self.programs[program.id] = program
        program.metadata["island"] = island

        key = cell_key(program.artifact)
        cell = self.islands[island]
        incumbent_id = cell.get(key)

        won = False
        if incumbent_id is None:
            cell[key] = program.id
            won = True
            logger.info(
                "island %d: new cell %s occupied by %s (fit=%.4f)",
                island, key, program.id[:8], program.fitness(),
            )
        else:
            incumbent = self.programs.get(incumbent_id)
            if incumbent is None or program.fitness() > incumbent.fitness():
                cell[key] = program.id
                won = True
                logger.info(
                    "island %d: cell %s improved %.4f -> %.4f (%s)",
                    island, key,
                    incumbent.fitness() if incumbent else float("-inf"),
                    program.fitness(), program.id[:8],
                )

        # Global best tracking.
        if (self.best_program_id is None
                or program.fitness() > self.programs[self.best_program_id].fitness()):
            self.best_program_id = program.id

        return won

    # -------------------- sampling --------------------

    def sample_parent(self, island: int) -> Program:
        """Uniform random sample from ``island``'s archive."""
        cell = self.islands[island]
        if not cell:
            raise LookupError(f"island {island} is empty — seed it first")
        pid = self._rng.choice(list(cell.values()))
        return self.programs[pid]

    def sample_inspiration(self, island: int,
                           exclude: Optional[str] = None) -> Optional[Program]:
        """Second sample for LLM-prompt 'inspiration' slot (may differ
        from the parent). Returns None if the island has only one cell.
        """
        cell = self.islands[island]
        pool = [pid for pid in cell.values() if pid != exclude]
        if not pool:
            return None
        return self.programs[self._rng.choice(pool)]

    def top_in_island(self, island: int, k: int = 3) -> List[Program]:
        cell = self.islands[island]
        progs = [self.programs[pid] for pid in cell.values()]
        progs.sort(key=lambda p: p.fitness(), reverse=True)
        return progs[:k]

    def best(self) -> Optional[Program]:
        if self.best_program_id is None:
            return None
        return self.programs.get(self.best_program_id)

    # -------------------- migration --------------------

    def tick_generation(self, island: int) -> None:
        self.island_generations[island] += 1

    def should_migrate(self) -> bool:
        cur = min(self.island_generations)
        return cur - self.last_migration_generation >= self.migration_interval

    def migrate(self) -> List[Tuple[int, int, str]]:
        """Top 1 program from each island migrates to (island+1) % N.

        Returns ``[(src_island, dst_island, program_id), ...]`` for logging.
        """
        moves: List[Tuple[int, int, str]] = []
        for src in range(self.num_islands):
            top = self.top_in_island(src, k=1)
            if not top:
                continue
            dst = (src + 1) % self.num_islands
            migrant = top[0]
            # We don't clone — we *re-place* the same Program in the
            # destination island (openevolve does a metadata-tag migration
            # too). The cell it occupies in dst is keyed by its artifact.
            key = cell_key(migrant.artifact)
            dst_cell = self.islands[dst]
            incumbent_id = dst_cell.get(key)
            if (incumbent_id is None
                    or migrant.fitness() > self.programs[incumbent_id].fitness()):
                dst_cell[key] = migrant.id
                moves.append((src, dst, migrant.id))
                logger.info(
                    "migration: %s  island %d -> %d  (cell=%s fit=%.4f)",
                    migrant.id[:8], src, dst, key, migrant.fitness(),
                )
        self.last_migration_generation = min(self.island_generations)
        return moves

    # -------------------- archive export --------------------

    def iter_archive_cells(self) -> List[Tuple[int, Tuple[int, ...], Program]]:
        out: List[Tuple[int, Tuple[int, ...], Program]] = []
        for i, cell in enumerate(self.islands):
            for key, pid in cell.items():
                prog = self.programs.get(pid)
                if prog is not None:
                    out.append((i, key, prog))
        return out

    def dump_archive(self, root: Path) -> None:
        """Write every archive cell to ``root/island_<i>/cell_<bins>/``."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        for island_idx, key, prog in self.iter_archive_cells():
            cell_dir = root / f"island_{island_idx}" / f"cell_{'_'.join(str(b) for b in key)}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            prog.artifact.write_to(cell_dir / "skills")
            (cell_dir / "meta.json").write_text(
                json.dumps({
                    "id": prog.id,
                    "parent_id": prog.parent_id,
                    "generation": prog.generation,
                    "metrics": prog.metrics,
                    "fitness": prog.fitness(),
                    "cell": list(key),
                    "island": island_idx,
                }, indent=2),
                encoding="utf-8",
            )


# ---------------------------------------------------------------------------
# Small seeded-RNG wrapper (avoid leaking global random state)
# ---------------------------------------------------------------------------

class _SeededRandom:
    def __init__(self, seed: Optional[int]) -> None:
        import random as _r
        self._r = _r.Random(seed)

    def choice(self, xs):
        return self._r.choice(xs)

    def random(self) -> float:
        return self._r.random()

    def shuffle(self, xs) -> None:
        self._r.shuffle(xs)


def new_program_id() -> str:
    return uuid.uuid4().hex
