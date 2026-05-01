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
        --outer-model moonshotai/kimi-k2.6 \\
        --inner-model claude-haiku-4-5
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

# Asymmetric model defaults:
#   * outer (patch generator): Kimi K2.6 via OpenRouter.
#   * inner (Hermes/bench-cli agent rollouts): claude-haiku-4-5 via the
#     Anthropic-direct API. Bug 12: previously defaulted to
#     ``minimax/minimax-m2.7`` which silently turned every Phase-E run
#     into an M2.7 run when ``--inner-model`` was not passed. The slug
#     ``claude-haiku-4-5`` matches the SkillsBench leaderboard "Haiku 4.5"
#     entry — no ``@DATE`` suffix, no ``vertex_ai/`` prefix.
DEFAULT_OUTER_MODEL = "moonshotai/kimi-k2.6"
DEFAULT_INNER_MODEL = "claude-haiku-4-5"


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
        default=3,
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
    # Bug 12 belt-and-suspenders: track whether the user explicitly passed
    # ``--inner-model``. When ``--task-source=skillsbench`` is in play we
    # insist on an explicit slug so a future default-bump can't silently
    # change which inner model evolves on the leaderboard task corpus.
    p.add_argument(
        "--inner-model",
        default=None,
        help="model slug for the inner Hermes agent rollouts "
        "inside SkillFolderEvaluator. "
        f"Default: {DEFAULT_INNER_MODEL} (REQUIRED for "
        "--task-source=skillsbench).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="DEPRECATED alias — sets BOTH --outer-model and "
        "--inner-model. Prefer the split flags.",
    )

    # kai-skills patch (2026-04-27): task-name leakage prevention. When
    # set, every benchmark task ID is replaced with a stable
    # ``task_NNN`` alias in everything fed to the outer LLM (per_task,
    # failures, the seed's own SKILL.md prose, and child patches are
    # rejected if they reintroduce a redacted name). Default OFF for
    # v6 reproducibility.
    p.add_argument(
        "--anonymize-tasks",
        action="store_true",
        help="redact verbatim benchmark task names from the outer LLM's "
        "view (seed *.md, evaluator artifacts) and reject child "
        "patches that reintroduce them. Recommended for fresh runs.",
    )
    # kai-skills patch end

    # kai-skills patch (Phase E, 2026-04-29): SkillsBench task source
    # dispatch. When --task-source skillsbench is selected, the
    # evaluator runs against the vendored SkillsBench task corpus via
    # the bench-cli backend instead of the existing TBLite/Hermes path.
    # Defaults preserve the existing tblite/hermes flow byte-identically.
    p.add_argument(
        "--task-source",
        choices=("tblite", "skillsbench"),
        default="tblite",
        help="Task source manifest. tblite = existing manifest path "
        "(unchanged). skillsbench = vendored SkillsBench corpus. "
        "Default: %(default)s.",
    )
    p.add_argument(
        "--agent-backend",
        choices=("hermes", "bench-cli"),
        default="hermes",
        help="Agent backend that runs each inner trial. hermes = "
        "existing run_agent.py subprocess (TBLite). bench-cli = "
        "BenchCliBackend (SkillsBench). When --task-source is "
        "skillsbench this is forced to bench-cli unless explicitly "
        "overridden. Default: %(default)s.",
    )
    p.add_argument(
        "--task-list",
        type=Path,
        default=None,
        help="Optional path to a JSON array of task IDs to subset to. "
        "When unset, the full manifest is used (existing behaviour).",
    )
    p.add_argument(
        "--leak-policy",
        choices=("warn", "zero", "raise"),
        default="zero",
        help="Anti-leakage scanner action when a candidate references a "
        "redacted task ID, file path, magic number, or solve.sh "
        "command. warn = log only; zero = log + zero score; "
        "raise = abort run. Default: %(default)s.",
    )
    # kai-skills patch end (Phase E)

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

    # kai-skills patch (Phase E, 2026-04-29): when --task-source
    # skillsbench is selected, force the bench-cli backend (the only
    # supported backend for SkillsBench) and force --anonymize-tasks
    # on (D-5: anonymization is mandatory under SkillsBench because
    # task dir names are the leak vector). Warn loudly if the user
    # tried to disable anonymization.
    if args.task_source == "skillsbench":
        if args.agent_backend == "hermes":
            args.agent_backend = "bench-cli"
        if not args.anonymize_tasks:
            print(
                "WARNING: --task-source skillsbench forces "
                "--anonymize-tasks on (D-5 mandatory).",
                file=sys.stderr,
            )
            args.anonymize_tasks = True
        # Bug 12: insist on an explicit --inner-model for SkillsBench
        # runs. Without this guard, a future default-bump in
        # DEFAULT_INNER_MODEL would silently change which model
        # evolves against the leaderboard. ``--force-synthetic``
        # bypasses (no API call to make).
        if not args.force_synthetic and not args.inner_model:
            print(
                "ERROR: --task-source=skillsbench requires explicit "
                "--inner-model (e.g. --inner-model claude-haiku-4-5). "
                "This guard prevents silent inner-model drift on "
                "leaderboard runs.",
                file=sys.stderr,
            )
            return 2

    # Resolve the inner-model default if the user didn't pass one (and
    # we didn't bail above). Done AFTER the SkillsBench guard so the
    # error path above doesn't get masked by the default fill.
    if args.inner_model is None:
        args.inner_model = DEFAULT_INNER_MODEL

    # B5: validate that the manifest actually carries the requested
    # skillsbench task IDs before kicking off evolution. A fresh clone
    # ships ``manifest.json`` with 0 skillsbench rows; without this
    # check evolve.py would happily start a 24h run that scores 0/0
    # every generation.
    if args.task_source == "skillsbench" and args.task_list:
        try:
            import json as _json

            ids = _json.loads(Path(args.task_list).read_text())
        except Exception as e:
            print(
                f"ERROR: cannot read --task-list {args.task_list}: {e}",
                file=sys.stderr,
            )
            return 1
        from skill_evolve.benchmark import load_subset as _load_subset

        _hydrated = _load_subset(
            offline_only=True, sources=["skillsbench"], task_ids=ids
        )
        if len(_hydrated) != len(ids):
            print(
                f"ERROR: skillsbench task hydration mismatch — task_list has "
                f"{len(ids)} ids but load_subset returned {len(_hydrated)}. "
                "Likely cause: manifest.json missing skillsbench entries. "
                "Run: uv run python scripts/expand_skillsbench_manifest.py",
                file=sys.stderr,
            )
            return 1

    args.out.mkdir(parents=True, exist_ok=True)
    _cost_warning(args)

    evaluator = SkillFolderEvaluator(
        force_synthetic=args.force_synthetic,
        verify=not args.no_verify,
        cascade=True,
        max_workers=args.max_workers,
        model=args.inner_model if not args.force_synthetic else None,
        repeats=args.repeats,
        # kai-skills patch (2026-04-27)
        anonymize_tasks=args.anonymize_tasks,
        # kai-skills patch (Phase E, 2026-04-29)
        task_source=args.task_source,
        agent_backend=args.agent_backend,
        task_list=args.task_list,
        leak_policy=args.leak_policy,
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
