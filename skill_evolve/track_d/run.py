"""Track D — sequential Track B (explore) -> Track A (refine).

Hypothesis: Track B's MAP-Elites + islands explores skill-space wide and
produces a diverse archive. Track A's A/B/AB tournament then refines
Track B's best folder via greedy AB synthesis — exploiting patterns that
Track A's greedy search alone couldn't reach from a cold seed.

Artifacts:

  runs/<id>/phase_b/                — raw Track B output (archive, best, history)
  runs/<id>/phase_a/                — raw Track A output (history, pass_*, final)
  runs/<id>/final/                  — byte-copy of phase_a/final (the winner)
  runs/<id>/run_meta.json           — summary linking both phases

CLI:

  python -m skill_evolve.track_d.run \\
      --seed seed_skills_empty/ \\
      --out runs/track_d_<ts>/ \\
      --b-generations 10 \\
      --a-max-passes 8 \\
      [--outer-model anthropic/claude-sonnet-4.6] \\
      [--inner-model minimax/minimax-m2.7] \\
      [--max-workers 7] [--repeats 1] [--sources tblite]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

from skill_evolve.evaluator import have_live_keys
from skill_evolve.track_a.folder import SkillFolder
from skill_evolve.track_a.runner import run_loop as run_track_a
from skill_evolve.track_a.validate import validate as track_a_validate
from skill_evolve.track_b.openevolve_skills.controller import (
    RunConfig,
    run_evolution,
)
from skill_evolve.track_b.openevolve_skills.evaluator import SkillFolderEvaluator
from skill_evolve.track_b.openevolve_skills.llm_client import build_default_client
from skill_evolve.track_b.openevolve_skills.prompt_sampler import PromptSampler

logger = logging.getLogger(__name__)

DEFAULT_OUTER_MODEL = "anthropic/claude-sonnet-4.6"
DEFAULT_INNER_MODEL = "minimax/minimax-m2.7"
DEFAULT_B_GENERATIONS = 10
DEFAULT_A_MAX_PASSES = 8


def run_sequential(
    *,
    seed: Path,
    out: Path,
    b_generations: int,
    a_max_passes: int,
    outer_model: str,
    inner_model: str,
    max_workers: int,
    repeats: int,
    sources: Optional[List[str]],
    force_synthetic: bool,
    verify: bool,
    num_islands: int,
    migration_interval: int,
    rng_seed: Optional[int],
) -> dict:
    """Run Track B then Track A. Return a summary dict written to run_meta.json.

    Phase B feeds its ``best/`` folder as the seed for Phase A. If Phase B
    produced no best program (e.g. all iterations failed validation), we
    fall back to the original seed for Phase A so the run still yields a
    Track-A-only result rather than aborting.
    """
    out = Path(out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # --- Phase B: MAP-Elites explore ---
    phase_b = out / "phase_b"
    logger.info(
        "Track D phase B: %d generations, %d islands -> %s",
        b_generations, num_islands, phase_b,
    )
    t0 = time.monotonic()

    # SkillFolderEvaluator doesn't plumb ``sources`` (Track B assumes the
    # manifest is already curated). ``sources`` is still honored for
    # Phase A below — if we need source-filtering in Phase B we'd extend
    # SkillFolderEvaluator first.
    evaluator = SkillFolderEvaluator(
        force_synthetic=force_synthetic,
        verify=verify,
        cascade=True,
        max_workers=max_workers,
        model=inner_model if not force_synthetic else None,
        repeats=repeats,
    )
    llm = build_default_client(
        force_synthetic=force_synthetic,
        model=outer_model,
        seed=rng_seed,
    )

    b_config = RunConfig(
        num_generations=b_generations,
        num_islands=num_islands,
        migration_interval=migration_interval,
        rng_seed=rng_seed if rng_seed is not None else 0,
    )
    b_result = run_evolution(
        seed_path=seed,
        out_dir=phase_b,
        config=b_config,
        evaluator=evaluator,
        llm=llm,
    )
    t_b = time.monotonic() - t0

    # --- pick the seed for Phase A ---
    # openevolve patch mutations can produce SKILL.md with YAML that parses
    # under its own artifact.validate() but fails Track A's stricter
    # validate() (e.g. unquoted descriptions containing ":", frontmatter
    # name mismatching folder name). If that happens, fall back to the
    # original seed rather than crashing the whole Track D run.
    phase_b_best = phase_b / "best"
    phase_a_seed: Path
    phase_a_seed_note: str
    if phase_b_best.is_dir() and any(phase_b_best.iterdir()):
        try:
            folder = SkillFolder.load(phase_b_best)
            ok, errs = track_a_validate(folder)
        except Exception as exc:
            ok, errs = False, [f"SkillFolder.load crashed: {exc!r}"]
        if ok:
            phase_a_seed = phase_b_best
            phase_a_seed_note = "phase_b_best"
        else:
            logger.warning(
                "Track D: phase_b best/ does not pass Track A validation "
                "(%s); falling back to original seed for phase_a",
                errs,
            )
            phase_a_seed = Path(seed).expanduser().resolve()
            phase_a_seed_note = f"fallback_original_seed (phase_b_invalid: {errs})"
    else:
        logger.warning(
            "Track D: phase_b produced no /best/ folder; "
            "falling back to original seed for phase_a"
        )
        phase_a_seed = Path(seed).expanduser().resolve()
        phase_a_seed_note = "fallback_original_seed"

    # --- Phase A: autoreason refine ---
    phase_a = out / "phase_a"
    logger.info(
        "Track D phase A: max_passes=%d, seed=%s -> %s",
        a_max_passes, phase_a_seed, phase_a,
    )
    t1 = time.monotonic()
    a_history = run_track_a(
        seed=phase_a_seed,
        out=phase_a,
        max_passes=a_max_passes,
        force_synthetic=force_synthetic,
        verify=verify,
        outer_model=outer_model,
        inner_model=inner_model,
        max_workers=max_workers,
        repeats=repeats,
        sources=sources,
        rng_seed=rng_seed,
    )
    t_a = time.monotonic() - t1

    # --- pin final ---
    phase_a_final = phase_a / "final"
    final = out / "final"
    if final.exists():
        shutil.rmtree(final)
    if phase_a_final.is_dir():
        shutil.copytree(phase_a_final, final)

    # --- summary ---
    b_best_metrics = {}
    if b_result.best is not None:
        b_best_metrics = dict(b_result.best.metrics)

    a_passes = a_history.get("passes") or []
    a_final_pass = a_passes[-1] if a_passes else {}

    summary = {
        "seed": str(Path(seed).resolve()),
        "out": str(out),
        "phase_b": {
            "generations": b_generations,
            "num_islands": num_islands,
            "elapsed_s": t_b,
            "best_metrics": b_best_metrics,
            "archive_size": len(list((phase_b / "archive").glob("*")))
            if (phase_b / "archive").is_dir() else 0,
        },
        "phase_a_seed_note": phase_a_seed_note,
        "phase_a": {
            "max_passes": a_max_passes,
            "elapsed_s": t_a,
            "passes_recorded": len(a_passes),
            "final_score": a_final_pass.get("score_A"),
            "final_winner": a_final_pass.get("winner"),
        },
        "total_elapsed_s": t_b + t_a,
        "outer_model": outer_model,
        "inner_model": inner_model,
        "max_workers": max_workers,
        "repeats": repeats,
        "force_synthetic": force_synthetic,
        "verify": verify,
    }
    (out / "run_meta.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info(
        "Track D done in %.1fs (B: %.1fs, A: %.1fs). "
        "Final score=%s, summary -> %s",
        summary["total_elapsed_s"], t_b, t_a,
        summary["phase_a"]["final_score"], out / "run_meta.json",
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m skill_evolve.track_d.run",
        description="Track D — sequential Track B (explore) -> Track A (refine).",
    )
    p.add_argument("--seed", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--b-generations", type=int, default=DEFAULT_B_GENERATIONS)
    p.add_argument("--a-max-passes", type=int, default=DEFAULT_A_MAX_PASSES)
    p.add_argument("--num-islands", type=int, default=3)
    p.add_argument("--migration-interval", type=int, default=5)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--max-workers", type=int, default=1)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--sources", nargs="*", default=None)
    p.add_argument("--outer-model", default=DEFAULT_OUTER_MODEL)
    p.add_argument("--inner-model", default=DEFAULT_INNER_MODEL)
    p.add_argument("--force-synthetic", action="store_true")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not args.seed.is_dir():
        print(f"seed folder not found: {args.seed}", file=sys.stderr)
        return 2

    if not args.force_synthetic and not have_live_keys():
        print(
            "WARNING: no live LLM keys set. Pass --force-synthetic for a "
            "dry run, or export OPENROUTER_API_KEY.",
            file=sys.stderr,
        )

    run_sequential(
        seed=args.seed,
        out=args.out,
        b_generations=args.b_generations,
        a_max_passes=args.a_max_passes,
        outer_model=args.outer_model,
        inner_model=args.inner_model,
        max_workers=args.max_workers,
        repeats=args.repeats,
        sources=args.sources,
        force_synthetic=args.force_synthetic,
        verify=not args.no_verify,
        num_islands=args.num_islands,
        migration_interval=args.migration_interval,
        rng_seed=args.rng_seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
