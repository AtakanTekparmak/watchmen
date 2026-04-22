"""CLI entry: ``python -m skill_evolve.track_c.run``.

Examples
--------

Synthetic smoke run (no API keys, no Docker):

    python -m skill_evolve.track_c.run \\
        --seed seed_skills/ \\
        --out runs/smoke/ \\
        --num-generations 3 \\
        --force-synthetic

Live run (will print a cost estimate first):

    python -m skill_evolve.track_c.run \\
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

from skill_evolve.evaluator import have_live_keys
from skill_evolve.track_a.llm import LLMClient as TrackALLMClient
from skill_evolve.track_b.openevolve_skills.evaluator import SkillFolderEvaluator

from .controller import RunConfig, run_evolution

# Asymmetric model defaults: Sonnet 4.6 for the outer/meta LLM
# (tournament — critic / op-planner / body-writer / synthesizer) and
# M2.7 for the inner Hermes agent rollouts inside SkillFolderEvaluator.
DEFAULT_OUTER_MODEL = "moonshotai/kimi-k2.6"
DEFAULT_INNER_MODEL = "minimax/minimax-m2.7"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m skill_evolve.track_c.run",
        description="Track C — autoreason A/B/AB tournament inside a "
        "MAP-Elites + islands shell.",
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
        help="output directory (history.jsonl, archive/, best/, ...)",
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
        default=3,
        help="k repeat roll-outs per task in each evaluation "
        "for majority-vote aggregation (default: 1).",
    )

    p.add_argument(
        "--force-synthetic",
        action="store_true",
        help="canned LLM responses + synthetic evaluator (no spend). "
        "Default for dev / tests.",
    )
    p.add_argument(
        "--no-verify",
        action="store_true",
        help="skip real verifiers inside skill_evolve.evaluator",
    )
    p.add_argument(
        "--outer-model",
        default=DEFAULT_OUTER_MODEL,
        help="model slug for the outer/meta LLM (tournament). "
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
    """Rough, stderr-only cost hint for live runs.

    Track C burns ~3× the LLM budget per generation compared with Track B
    (critic + op-planner + author-B + synthesizer) and evaluates up to 2
    extra candidates (B, AB) on top of the parent. We just make sure the
    operator notices before the run starts.
    """
    if args.force_synthetic:
        return
    n = args.num_generations
    # ~1-2 critic/planner/author/synth calls + up to 2 extra evaluator
    # invocations per generation.
    per_gen_lo, per_gen_hi = 1.00, 8.00
    lo, hi = n * per_gen_lo, n * per_gen_hi
    print("─" * 70, file=sys.stderr)
    print("Track C — LIVE run  (autoreason tournament × MAP-Elites)", file=sys.stderr)
    print(f"  generations       : {n}", file=sys.stderr)
    print(f"  islands           : {args.num_islands}", file=sys.stderr)
    print(f"  verify (docker)   : {not args.no_verify}", file=sys.stderr)
    print(
        f"  est. LLM cost     : ~${lo:.2f}–${hi:.2f} USD "
        f"(~4 LLM calls + up to 2 evals per generation)",
        file=sys.stderr,
    )
    print("  Re-run with --force-synthetic for a free dry run.", file=sys.stderr)
    print("─" * 70, file=sys.stderr)


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

    if not args.force_synthetic:
        if have_live_keys():
            _cost_warning(args)
        else:
            print(
                "WARNING: no live LLM keys set. Pass --force-synthetic "
                "for a full dry run (canned LLM + synthetic evaluator).",
                file=sys.stderr,
            )

    evaluator = SkillFolderEvaluator(
        force_synthetic=args.force_synthetic,
        verify=not args.no_verify,
        cascade=True,
        max_workers=args.max_workers,
        model=args.inner_model if not args.force_synthetic else None,
        repeats=args.repeats,
    )
    # Track C uses Track A's LLMClient — the tournament is purely a
    # Track-A-style operation, and Track A's ``synthetic`` mode already
    # ships the canned responses we need for each tag (critic, op_planner,
    # rewrite_body, split, merge, new_body, synth).
    llm = TrackALLMClient(
        model=args.outer_model,
        synthetic=args.force_synthetic,
    )

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
    )

    print("\n─── Track C run summary ───")
    print(f"out_dir           : {result.output_dir}")
    if result.best is not None:
        print(f"best fitness      : {result.best.fitness():.4f}")
        print(f"best metrics      : {result.best.metrics}")
    else:
        print("no best program produced — check logs", file=sys.stderr)
    stats = result.tournament_stats
    totals = stats.get("totals", {})
    wr = stats.get("win_rate_by_role", {})
    print(
        f"tournament totals : A={totals.get('A', 0)}  "
        f"B={totals.get('B', 0)}  AB={totals.get('AB', 0)}"
    )
    print("tournament win-rate-by-role:")
    for role in ("A", "B", "AB"):
        print(f"  {role:>2s} : {wr.get(role, 0.0):.3f}")
    print(f"history lines     : {len(result.history)}")
    return 0 if result.best is not None else 1


if __name__ == "__main__":
    sys.exit(main())
