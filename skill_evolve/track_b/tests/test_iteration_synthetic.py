"""Smoke test: 3 generations end-to-end under ``--force-synthetic``.

Asserts:
  * archive is populated (≥1 cell);
  * history.jsonl has the expected number of lines;
  * best.composite ≥ seed.composite (under synthetic eval they are
    both 0.0; we assert non-regression).
"""

from __future__ import annotations

import json
from pathlib import Path

from skill_evolve.track_b.openevolve_skills.controller import (
    RunConfig,
    run_evolution,
)
from skill_evolve.track_b.openevolve_skills.evaluator import (
    SkillFolderEvaluator,
)
from skill_evolve.track_b.openevolve_skills.llm_client import SyntheticLLM
from skill_evolve.track_b.openevolve_skills.prompt_sampler import PromptSampler


REPO_ROOT = Path(__file__).resolve().parents[3]
SEED = REPO_ROOT / "seed_skills"


def test_three_generation_synthetic_run(tmp_path: Path):
    assert SEED.is_dir(), f"seed_skills not found at {SEED}"
    out = tmp_path / "run"

    evaluator = SkillFolderEvaluator(
        force_synthetic=True, verify=False, cascade=True, max_workers=1,
    )
    llm = SyntheticLLM(seed=0)
    config = RunConfig(
        num_generations=3,
        num_islands=3,
        migration_interval=5,
        rng_seed=0,
    )
    result = run_evolution(
        seed_path=SEED,
        out_dir=out,
        config=config,
        evaluator=evaluator,
        llm=llm,
        prompt_sampler=PromptSampler(),
    )

    # Archive populated.
    assert result.best is not None
    assert (out / "archive").is_dir()
    cells = list((out / "archive").rglob("meta.json"))
    assert len(cells) >= 1

    # History: 3 seed lines + 3 iteration lines (migration only at 5+).
    history_path = out / "history.jsonl"
    assert history_path.exists()
    lines = [json.loads(l) for l in history_path.read_text().splitlines() if l.strip()]
    seed_lines = [l for l in lines if l["op_type"] == "seed"]
    iter_lines = [l for l in lines if l["op_type"] in ("patch", "rewrite")]
    assert len(seed_lines) == 3
    assert len(iter_lines) == 3

    # Non-regression: best fitness ≥ any seed fitness.
    seed_fitness = [l["metrics"].get("composite", 0.0) for l in seed_lines]
    assert result.best.fitness() >= max(seed_fitness)

    # Best folder persisted.
    best_dir = out / "best"
    assert best_dir.is_dir()
    assert list(best_dir.rglob("SKILL.md"))


def test_run_meta_written(tmp_path: Path):
    evaluator = SkillFolderEvaluator(force_synthetic=True, verify=False)
    llm = SyntheticLLM(seed=1)
    run_evolution(
        seed_path=SEED,
        out_dir=tmp_path / "meta",
        config=RunConfig(num_generations=2, num_islands=3, migration_interval=5, rng_seed=1),
        evaluator=evaluator,
        llm=llm,
    )
    meta = json.loads((tmp_path / "meta" / "run_meta.json").read_text())
    assert meta["num_generations"] == 2
    assert meta["num_islands"] == 3
    assert meta["archive_cells"] >= 1
