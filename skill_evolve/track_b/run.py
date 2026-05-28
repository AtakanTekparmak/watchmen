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
    # kai-skills patch (Group E, 2026-05-28): default flipped 0 -> None per
    # plan section 4b replicability paragraph. Unseeded by default (opt-in
    # seeding for multi-roll replicability protocol). When set, the controller
    # threads the seed into the proposer LLM client's seed= kwarg in addition
    # to the existing RNG seeding.
    p.add_argument("--rng-seed", type=int, default=None)
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
        "--agent",
        type=str,
        default="claude-code",
        choices=("claude-code", "gemini"),
        help="Inner agent harness for SkillsBench evals. Default: claude-code.",
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

    # kai-skills patch (Group B, 2026-05-27): --eval-source flag +
    # behavioral adapter + held-out validation split. Defaults preserve
    # the existing SkillsBench / tblite scoring path. When
    # ``--eval-source behavioral`` is selected, ``--eval-set`` must
    # point at a daycare-format ``eval_set.jsonl``. ``--validation-task-list``
    # triggers a second eval pass after a winner is accepted; the
    # validation score is recorded on the artifact (no re-acceptance
    # gate — recorded only).
    p.add_argument(
        "--eval-source",
        choices=("skillsbench", "behavioral"),
        default="skillsbench",
        help=(
            "Scoring backend. ``skillsbench`` (default) preserves the "
            "existing TBLite/SkillsBench agent-harness scoring. "
            "``behavioral`` scores candidates via a judge LLM against "
            "a daycare-format ``eval_set.jsonl`` (requires --eval-set)."
        ),
    )
    p.add_argument(
        "--eval-set",
        type=Path,
        default=None,
        help=(
            "Path to a daycare-style ``eval_set.jsonl`` (required when "
            "--eval-source=behavioral)."
        ),
    )
    p.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help=(
            "OpenRouter slug for the behavioral judge LLM. Required when "
            "--eval-source=behavioral and no stub is wired in."
        ),
    )
    p.add_argument(
        "--validation-task-list",
        type=Path,
        default=None,
        help=(
            "Optional held-out task list scored after each accepted "
            "winner. Recorded on the artifact as ``validation_score``; "
            "does NOT re-gate acceptance."
        ),
    )
    # kai-skills patch (Group F, 2026-05-28): bounded edit-budget L_t
    # scheduler. Caps the number of file ops applied per iteration,
    # mirroring SkillOpt's paper-default cosine taper from 8 down to 2.
    # See ``skill_evolve.shared.edit_budget`` for spec grammar. F is the
    # first second-pass group in §7n's append order; subsequent groups
    # (E/G/H) append their args AFTER this block.
    p.add_argument(
        "--edit-budget",
        type=str,
        default="cosine:8->2",
        help=(
            "Per-iteration L_t cap on file ops. Spec: ``constant:N``, "
            "``linear:N->M``, or ``cosine:N->M``. Default ``cosine:8->2`` "
            "matches the SkillOpt paper. Surplus ops are dropped in "
            "proposer-emit order before the smoke gate."
        ),
    )
    # kai-skills patch (Group E, 2026-05-28): strict validation gate +
    # rejected-edit buffer. Plan section 7j locks the gate semantics:
    #   * strict   (default): accept iff train_score >= parent_train AND
    #                          val_score > best_val_score_seen_so_far.
    #                          Ties on val_score REJECTED.
    #   * record:             plan_0 Group-B behavior — record val_score
    #                          on artifact; accept on train criterion.
    #   * relaxed:            accept iff train_score >= parent_train AND
    #                          val_score >= best_val_score_seen_so_far
    #                          (ties accepted).
    # --rejected-buffer-size caps the bounded ring of recent rejections
    # surfaced to the proposer prompt; --max-proposer-prompt-tokens caps
    # the rendered prompt size before any LLM call (3-stage degradation).
    p.add_argument(
        "--validation-gate",
        choices=("strict", "record", "relaxed"),
        default="strict",
        help=(
            "Acceptance gate for the held-out validation eval. "
            "strict (default): train >= parent AND val > best-seen "
            "(ties rejected). record: plan_0 Group-B behavior — accept "
            "on train criterion, record val. relaxed: train >= parent "
            "AND val >= best-seen (ties accepted)."
        ),
    )
    p.add_argument(
        "--rejected-buffer-size",
        type=int,
        default=10,
        help=(
            "Capacity of the bounded rejected-edit ring shown to the "
            "proposer prompt (default: 10)."
        ),
    )
    p.add_argument(
        "--max-proposer-prompt-tokens",
        type=int,
        default=90000,
        help=(
            "Hard pre-LLM cap on the rendered proposer prompt. When "
            "exceeded, the renderer degrades gracefully: "
            "(1) truncate rejected-buffer entries oldest-first, "
            "(2) truncate meta-skill tail oldest-iter-first, "
            "(3) truncate bundle context keeping SKILL.md + frontmatter "
            "+ first N scripts/* in list_scripts order. Default: 90000."
        ),
    )
    # kai-skills patch (Group H, 2026-05-28): success/failure minibatch
    # partition reflection (SkillOpt port). Per plan section 7m, the
    # ``partition`` mode forks the single proposer call into two parallel
    # calls (failure / success reflection) and merges via a keyed-dict
    # resolver in ``shared.reflection.merge_patches``. H is SECOND in §7n
    # Round 2 append order — these args go AFTER E's block and BEFORE G's
    # (G rebases later and appends its own flags).
    p.add_argument(
        "--reflection-mode",
        choices=("single", "partition"),
        default="partition",
        help=(
            "Proposer call shape. ``partition`` (default) runs TWO "
            "parallel proposer calls (failure-pattern + success-pattern "
            "reflection) and merges via failure-priority keyed-dict "
            "resolver. ``single`` preserves the back-compat one-call "
            "path."
        ),
    )
    p.add_argument(
        "--reflection-batch-size",
        type=int,
        default=8,
        help=(
            "Per-side minibatch cap B_m for partition-mode reflection "
            "(paper default 8). On hot_5 this just acts as a per-side "
            "cap since the eval set has 5 tasks total."
        ),
    )
    p.add_argument(
        "--reflection-success-threshold",
        type=float,
        default=0.5,
        help=(
            "Per-task score cutoff for the failure / success partition. "
            "Boundary score == threshold goes to success (``>=``). "
            "Default 0.5 matches the behavioral-judge threshold."
        ),
    )
    # kai-skills patch (Group G, 2026-05-28; plan §7l + §7n): slow-update
    # protected region + meta-skill consolidator. G is the LAST second-pass
    # group to append to this block per §7n Round 2 (H first, then G).
    #   * --slow-update-every K: consolidator fires every K iters (default 4).
    #   * --meta-skill-path: on-disk audit log path (default <out>/meta_skill.md).
    #   * --consolidator-model: separate slug for the slow-update LLM
    #     (default: mirror --outer-model so a one-model run still works).
    #   * --meta-skill-max-iters: tail-truncation cap on the audit log
    #     (default 20, matches plan §4b context-budget projection).
    #   * --persistent-failure-window: a task counts as "persistent failure"
    #     once it has failed in this many consecutive iters (default 3).
    p.add_argument(
        "--slow-update-every",
        type=int,
        default=4,
        help=(
            "Run the slow-update consolidator every K iterations (default "
            "4). Set to a value > num_generations to effectively disable."
        ),
    )
    p.add_argument(
        "--meta-skill-path",
        type=Path,
        default=None,
        help=(
            "Path to the rolling meta-skill audit log (markdown). Default "
            "<out>/meta_skill.md. The file is training-only and never "
            "shipped to the deployed bundle (plan §7l)."
        ),
    )
    p.add_argument(
        "--consolidator-model",
        type=str,
        default=None,
        help=(
            "Model slug for the slow-update consolidator LLM. Default: "
            "mirror --outer-model so a single-model run still works."
        ),
    )
    p.add_argument(
        "--meta-skill-max-iters",
        type=int,
        default=20,
        help=("Tail-truncation cap on the meta-skill audit log (default 20)."),
    )
    p.add_argument(
        "--persistent-failure-window",
        type=int,
        default=3,
        help=(
            "A task counts as persistent-failure when it has failed in "
            "this many consecutive iters (default 3)."
        ),
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p


# kai-skills patch (Group E, 2026-05-28): public alias for plan section
# E acceptance check (``from track_b.run import build_parser``).
build_parser = _build_parser


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

    # kai-skills patch (Group F, 2026-05-28): fail fast on a malformed
    # --edit-budget spec so a 12-hour run doesn't crash mid-iter when
    # the first non-zero gen tries to compute L_t.
    from skill_evolve.shared.edit_budget import parse_schedule as _parse_schedule

    try:
        _parse_schedule(args.edit_budget)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
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

    # kai-skills patch (Group B, 2026-05-27): validate behavioral inputs
    # before we start spinning up workers. Failure modes:
    #   * --eval-source behavioral but no --eval-set → ValueError mirrors
    #     skill_evolve.evaluator.evaluate (single source of truth).
    #   * --eval-source behavioral but --eval-set path missing on disk →
    #     fail fast before evolution loop launches a single judge call.
    if args.eval_source == "behavioral":
        if args.eval_set is None:
            print(
                "ERROR: --eval-set required when --eval-source behavioral",
                file=sys.stderr,
            )
            return 2
        if not Path(args.eval_set).exists():
            print(
                f"ERROR: --eval-set {args.eval_set} does not exist",
                file=sys.stderr,
            )
            return 2
        if not args.force_synthetic and not args.judge_model:
            print(
                "ERROR: --judge-model is required for live "
                "--eval-source=behavioral runs",
                file=sys.stderr,
            )
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
        # kai-skills patch (2026-04-27)
        anonymize_tasks=args.anonymize_tasks,
        # kai-skills patch (Phase E, 2026-04-29)
        task_source=args.task_source,
        agent_backend=args.agent_backend,
        task_list=args.task_list,
        leak_policy=args.leak_policy,
        # Phase E v7 patch (2026-05-05): inner agent selection
        agent=args.agent,
        # kai-skills patch (Group B, 2026-05-27): eval-source plumbing
        eval_source=args.eval_source,
        eval_set_path=args.eval_set,
        judge_model=args.judge_model,
        validation_task_list=args.validation_task_list,
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
        # kai-skills patch (Group F, 2026-05-28): thread the bounded
        # edit-budget L_t schedule spec end-to-end. Validated above; the
        # iteration loop re-parses (cached parse is acceptable since
        # parse_schedule is O(spec_len)).
        edit_budget=args.edit_budget,
        # kai-skills patch (Group B, 2026-05-27): held-out validation path.
        validation_task_list=args.validation_task_list,
        # kai-skills patch (Group E, 2026-05-28): strict gate + buffer
        # plumbing per plan section 7j.
        validation_gate=args.validation_gate,
        rejected_buffer_size=args.rejected_buffer_size,
        max_proposer_prompt_tokens=args.max_proposer_prompt_tokens,
        # kai-skills patch (Group H, 2026-05-28): partition reflection
        # plumbing per plan section 7m.
        reflection_mode=args.reflection_mode,
        reflection_batch_size=args.reflection_batch_size,
        reflection_success_threshold=args.reflection_success_threshold,
        # kai-skills patch (Group G, 2026-05-28; plan §7l): slow-update
        # consolidator + meta-skill audit log plumbing.
        slow_update_every=args.slow_update_every,
        meta_skill_path=(
            args.meta_skill_path
            if args.meta_skill_path is not None
            else args.out / "meta_skill.md"
        ),
        consolidator_model=(
            args.consolidator_model
            if args.consolidator_model is not None
            else args.outer_model
        ),
        meta_skill_max_iters=args.meta_skill_max_iters,
        persistent_failure_window=args.persistent_failure_window,
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
