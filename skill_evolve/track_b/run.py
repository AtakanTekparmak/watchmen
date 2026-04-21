"""CLI entry point: ``python -m skill_evolve.track_b.run``.

Examples:

    # Dev smoke (deterministic, no API keys needed):
    python -m skill_evolve.track_b.run \\
        --seed seed_skills/ \\
        --out runs/smoke/ \\
        --num-generations 3 \\
        --force-synthetic

    # Live run (costs money — prints estimate first):
    python -m skill_evolve.track_b.run \\
        --seed seed_skills/ \\
        --out runs/live/ \\
        --num-generations 30 \\
        --num-islands 3 \\
        --outer-model anthropic/claude-sonnet-4.6 \\
        --inner-model minimax/minimax-m2.7
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .openevolve_skills.controller import RunConfig, run_evolution
from .openevolve_skills.evaluator import SkillFolderEvaluator
from .openevolve_skills.llm_client import build_default_client
from .openevolve_skills.prompt_sampler import PromptSampler

# Asymmetric model defaults: Sonnet 4.6 for the patch-generating LLM
# (outer) and M2.7 for the inner Hermes agent rollouts.
DEFAULT_OUTER_MODEL = "anthropic/claude-sonnet-4.6"
DEFAULT_INNER_MODEL = "minimax/minimax-m2.7"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m skill_evolve.track_b.run",
        description="Evolve a Hermes-agent skills folder "
        "(folder-artifact fork of OpenEvolve).",
    )
    p.add_argument(
        "--seed",
        type=Path,
        required=True,
        help="path to seed skills folder (e.g. seed_skills/)",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output directory for history/archive/best",
    )
    p.add_argument("--num-generations", type=int, default=30)
    p.add_argument("--num-islands", type=int, default=3)
    p.add_argument("--migration-interval", type=int, default=5)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="parallel task workers per evaluation (>1 spawns "
        "concurrent Hermes subprocesses)",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="k repeat roll-outs per task in each evaluation "
        "for majority-vote aggregation (default: 1).",
    )

    # Cost / mode controls.
    p.add_argument(
        "--force-synthetic",
        action="store_true",
        help="always use the synthetic LLM + synthetic evaluator "
        "(no API calls, no Docker). Default for dev.",
    )
    p.add_argument(
        "--no-verify",
        action="store_true",
        help="skip real verifiers inside skill_evolve.evaluator",
    )
    p.add_argument(
        "--outer-model",
        default=DEFAULT_OUTER_MODEL,
        help="model slug for the patch-generating LLM (outer). "
        f"Default: {DEFAULT_OUTER_MODEL}",
    )
    p.add_argument(
        "--inner-model",
        default=DEFAULT_INNER_MODEL,
        help="model slug for the inner Hermes agent rollouts "
        "inside SkillFolderEvaluator. "
        f"Default: {DEFAULT_INNER_MODEL}",
    )
    p.add_argument(
        "--model",
        default=None,
        help="DEPRECATED alias — sets BOTH --outer-model and "
        "--inner-model. Prefer the split flags.",
    )

    p.add_argument("--verbose", "-v", action="store_true")
    return p


def _cost_warning(args: argparse.Namespace) -> None:
    """Very rough cost estimate — prints to stderr when running live."""
    if args.force_synthetic:
        return
    # ~Rough: per generation we do one LLM call (prompt ~4k tokens in,
    # ~2k tokens out) and one evaluator run. Evaluator cost varies wildly
    # with task count; we only cost-warn the LLM side here.
    n = args.num_generations
    print(
        f"[COST ESTIMATE] {n} generations × ~6k LLM tokens "
        f"(outer={args.outer_model}) + 1 evaluator run/gen "
        f"(inner={args.inner_model}).\n"
        f"                Evaluator cost depends on benchmark + model; live runs\n"
        f"                typically dominate the total. Re-run with\n"
        f"                --force-synthetic if you just want to smoke-test.",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Handle deprecated --model alias: when set, it overrides BOTH split
    # flags. Emit a warning so pipeline owners transition to the new flags.
    if args.model is not None:
        print(
            "WARNING: --model is deprecated, use --outer-model and --inner-model",
            file=sys.stderr,
        )
        args.outer_model = args.model
        args.inner_model = args.model

    if not args.seed.is_dir():
        print(f"seed folder not found: {args.seed}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    _cost_warning(args)

    evaluator = SkillFolderEvaluator(
        force_synthetic=args.force_synthetic,
        verify=not args.no_verify,
        cascade=True,
        max_workers=args.max_workers,
        model=args.inner_model if not args.force_synthetic else None,
        repeats=args.repeats,
    )
    llm = build_default_client(
        force_synthetic=args.force_synthetic,
        model=args.outer_model,
        seed=args.rng_seed,
    )
    prompt_sampler = PromptSampler()
    config = RunConfig(
        num_generations=args.num_generations,
        num_islands=args.num_islands,
        migration_interval=args.migration_interval,
        rng_seed=args.rng_seed,
    )

    result = run_evolution(
        seed_path=args.seed,
        out_dir=args.out,
        config=config,
        evaluator=evaluator,
        llm=llm,
        prompt_sampler=prompt_sampler,
    )

    best = result.best
    if best is not None:
        print("\n─── run summary ───")
        print(f"out_dir       : {result.output_dir}")
        print(f"best fitness  : {best.fitness():.4f}")
        print(f"best metrics  : {best.metrics}")
        print(f"history lines : {len(result.history)}")
    else:
        print("no best program produced — check logs", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
