"""Top-level evolution controller — fork of openevolve/controller.py.

Glues Database + Evaluator + LLM + PromptSampler + Iteration together
into a single synchronous run. Persists history + archive on exit.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from skill_evolve.shared.rejected_buffer import RejectedBuffer

from .database import (
    Program,
    ProgramDatabase,
    cell_key,
    new_program_id,
)
from .evaluator import SkillFolderEvaluator
from .folder_artifact import FolderArtifact
from .islands import seed_variants
from .iteration import GateState, IterationResult, run_iteration
from .llm_client import LLMClient
from .prompt_sampler import PromptSampler

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    num_generations: int = 30
    num_islands: int = 3
    migration_interval: int = 5
    # kai-skills patch (Group E, 2026-05-28): default flipped 0 -> None to
    # match the CLI flag's new opt-in semantics (plan section 4b
    # replicability paragraph).
    rng_seed: Optional[int] = None
    # kai-skills patch (Group F, 2026-05-28): bounded edit-budget L_t
    # schedule spec (``constant:N`` / ``linear:N->M`` / ``cosine:N->M``).
    # ``None`` disables clipping (back-compat for callers that haven't
    # opted in to the Group F flag).
    edit_budget: Optional[str] = None
    # kai-skills patch (Group B, 2026-05-27): held-out validation list
    # path. Surface on RunConfig so the Group E gate can branch on
    # acceptance without re-reading the evaluator.
    validation_task_list: Optional[Path] = None
    # kai-skills patch (Group E, 2026-05-28): strict validation gate +
    # rejected-edit buffer plumbing per plan section 7j.
    validation_gate: str = "strict"
    rejected_buffer_size: int = 10
    max_proposer_prompt_tokens: int = 90000
    # kai-skills patch (Group H, 2026-05-28): success/failure minibatch
    # partition reflection per plan section 7m. ``single`` keeps the
    # back-compat one-proposer-call path; ``partition`` runs TWO parallel
    # calls (failure / success reflection) and merges via failure-priority
    # keyed-dict resolver.
    reflection_mode: str = "single"
    reflection_batch_size: int = 8
    reflection_success_threshold: float = 0.5
    # kai-skills patch (Group G, 2026-05-28; plan §7l): slow-update
    # consolidator + meta-skill audit log plumbing.
    #   * slow_update_every: K — consolidator fires every K iters
    #     (default 4; > num_generations effectively disables).
    #   * meta_skill_path: on-disk markdown audit log
    #     (default None → run.py supplies <out>/meta_skill.md).
    #   * consolidator_model: separate slug for the slow-update LLM
    #     (default None → falls back to outer/proposer model).
    #   * meta_skill_max_iters: tail-truncation cap (default 20).
    #   * persistent_failure_window: a task counts as persistent failure
    #     after this many consecutive failed iters (default 3).
    slow_update_every: int = 4
    meta_skill_path: Optional[Path] = None
    consolidator_model: Optional[str] = None
    meta_skill_max_iters: int = 20
    persistent_failure_window: int = 3
    # max_iters mirrors num_generations for the edit-budget schedule
    # normalization; kept separate so future callers can clip schedule
    # endpoints independently of the run length.
    max_iters: Optional[int] = None


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
        include_exts={
            ".md",
            ".sh",
            ".py",
            ".jq",
            ".json",
            ".yaml",
            ".yml",
            ".txt",
            ".xml",
        },
    )
    # kai-skills patch (2026-04-27): if the evaluator was configured with
    # --anonymize-tasks, scrub verbatim manifest task names from the seed
    # *.md files BEFORE we evaluate-and-archive. Critical: even if the
    # evaluator's prompt-facing artifacts are clean, the K2.6 outer LLM
    # still sees the parent SKILL.md verbatim (via render_folder), and a
    # seed with task names baked into its description would propagate
    # those names into every child patch.
    #
    # kai-skills patch (Phase E, 2026-04-29): the anonymizer module is
    # now dispatched on ``evaluator.task_source``: in-file (tblite) vs
    # ``skill_evolve.benchmark.skillsbench_anonymize`` (skillsbench).
    # The Controller installs the chosen module on the evaluator and
    # builds the id_map ONCE here so the chokepoints below + downstream
    # iteration.py call through ``evaluator.anonymizer.<fn>`` rather
    # than re-importing.
    if getattr(evaluator, "anonymize_tasks", False):
        _install_anonymizer(evaluator)
        tid_map = evaluator.task_id_map()
        replaced, detail = evaluator.anonymizer.sanitize_artifact(
            base_artifact, tid_map
        )
        if replaced:
            logger.warning(
                "anonymize_tasks: sanitized %d md file(s) in seed; "
                "rewrote names in: %s",
                replaced,
                ", ".join(f"{p}({len(h)})" for p, h in detail),
            )
        else:
            logger.info("anonymize_tasks: seed clean (no task names found)")
    # kai-skills patch end
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
    # kai-skills patch (2026-04-24): after each iteration, snapshot the
    # global best-so-far to disk so a mid-run kill preserves the winning
    # folder. Prior behaviour only wrote best/ at end-of-run, so killing
    # during gen N lost the artifact bytes (history.jsonl kept only the
    # mutation recipe). Checkpoint dir: out_dir/best_so_far/.
    t_start = time.monotonic()
    best_so_far_fitness: float = float("-inf")
    best_so_far_dir = out_dir / "best_so_far"
    # kai-skills patch (Group E, 2026-05-28): instantiate the per-run
    # gate state + rejected-edit buffer ONCE so the strict gate can
    # compare each candidate against the cross-iter ``best_val_seen``.
    # Buffer is persisted to ``out_dir/rejected_buffer.jsonl`` after
    # every rejection via the iteration helper (crash-safe).
    gate_state = GateState()
    rejected_buffer = RejectedBuffer(capacity=config.rejected_buffer_size)
    rejected_buffer_path = out_dir / "rejected_buffer.jsonl"
    # kai-skills patch (Group G, 2026-05-28; plan §7l): instantiate the
    # per-run meta-skill audit log + persistent-failure tracker ONCE so
    # the consolidator path can append cross-iter lessons without
    # re-reading the file on every fire. The log is bounded to
    # ``meta_skill_max_iters`` entries via tail-truncation on append.
    from skill_evolve.shared.meta_skill import MetaSkill as _MetaSkill

    meta_skill_path: Path = (
        Path(config.meta_skill_path)
        if config.meta_skill_path is not None
        else (out_dir / "meta_skill.md")
    )
    meta_skill = _MetaSkill.from_markdown_file(
        meta_skill_path, max_entries=config.meta_skill_max_iters
    )
    # Per-task failure-streak counters for the persistent-failure window.
    # task_id -> int (consecutive failed iters since last pass).
    persistent_failure_streaks: Dict[str, int] = {}
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
                edit_budget=config.edit_budget,
                max_iters=(
                    config.max_iters
                    if config.max_iters is not None
                    else config.num_generations
                ),
                validation_gate=config.validation_gate,
                rejected_buffer=rejected_buffer,
                gate_state=gate_state,
                rejected_buffer_path=rejected_buffer_path,
                # kai-skills patch (Group H, 2026-05-28): partition
                # reflection plumbing per plan section 7m. The artifact
                # dir routes the three reflection JSON files to a
                # per-run subdir so post-hoc inspectors can read them.
                reflection_mode=config.reflection_mode,
                reflection_success_threshold=(config.reflection_success_threshold),
                reflection_artifact_dir=(out_dir / "reflection"),
                # kai-skills patch (Group G, 2026-05-28; plan §7l):
                # slow-update consolidator plumbing. The iteration loop
                # owns the every-K-iters branch on top of the regular
                # fast-edit path.
                slow_update_every=config.slow_update_every,
                meta_skill=meta_skill,
                meta_skill_path=meta_skill_path,
                consolidator_model=config.consolidator_model,
                persistent_failure_streaks=persistent_failure_streaks,
                persistent_failure_window=config.persistent_failure_window,
            )
        except LookupError as exc:
            logger.warning("iter %d: island %d empty (%s); skipping", gen, island, exc)
            continue
        _append_history(history_path, res)

        # kai-skills patch: incremental best-so-far checkpoint.
        try:
            current_best = db.best()
            if (
                current_best is not None
                and current_best.fitness() > best_so_far_fitness
            ):
                if best_so_far_dir.exists():
                    shutil.rmtree(best_so_far_dir)
                # kai-skills patch (Group G, 2026-05-28; plan §7l):
                # best_so_far is a deployment-side snapshot — strip
                # ``meta_skill.md`` and other training-only artifacts.
                current_best.artifact.write_to(best_so_far_dir, deployment=True)
                (out_dir / "best_so_far_meta.json").write_text(
                    json.dumps(
                        {
                            "id": current_best.id,
                            "generation": current_best.generation,
                            "iteration_found": current_best.iteration_found,
                            "metrics": current_best.metrics,
                            "fitness": current_best.fitness(),
                            "written_at_gen": gen,
                        },
                        indent=2,
                        default=str,
                    ),
                    encoding="utf-8",
                )
                logger.info(
                    "best_so_far: gen=%d fitness=%.4f id=%s -> %s",
                    gen,
                    current_best.fitness(),
                    current_best.id[:8],
                    best_so_far_dir,
                )
                best_so_far_fitness = current_best.fitness()
        except Exception as exc:  # pragma: no cover — checkpoint must not crash run
            logger.warning("best_so_far checkpoint failed at gen %d: %s", gen, exc)

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
            # kai-skills patch (2026-04-25): removed redundant local
            # `import shutil` here. Having it inside the function turned
            # `shutil` into a local for the whole scope (Python rule),
            # which broke the earlier best_so_far checkpoint at gen N>1
            # with "cannot access local variable 'shutil'". The module-
            # level import at top of the file is the only one needed.
            shutil.rmtree(best_dir)
        # kai-skills patch (Group G, 2026-05-28; plan §7l): best/ is the
        # deployment artifact — strip ``meta_skill.md`` and other
        # training-only files via deployment=True.
        best.artifact.write_to(best_dir, deployment=True)
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


# kai-skills patch (Phase E, 2026-04-29): anonymizer dispatch + R-12
# anti-leakage scanner. Both run at Controller.__init__ scope, called
# once per run from ``run_evolution``. The dispatched module is stored
# on the evaluator and re-used in iteration.py / evaluator chokepoints
# so per-call import dispatch is centralized here.


def _install_anonymizer(evaluator: SkillFolderEvaluator) -> None:
    """Build the id_map and install the matching anonymizer module on
    the evaluator. Dispatch keys off ``evaluator.task_source``.

    For tblite (default), reuses the existing in-file functions at
    :mod:`skill_evolve.track_b.openevolve_skills.evaluator` — both are
    callable through a tiny adapter object that exposes the same
    ``sanitize_artifact`` / ``find_leaked_names`` / ``sanitize_text``
    surface.

    For skillsbench, imports
    :mod:`skill_evolve.benchmark.skillsbench_anonymize`, hydrates the
    optional ``--task-list`` JSON to scope the id_map domain to the
    20-task subset (or whatever the user passed), and stores the
    hydrated records on the evaluator so the anti-leak scanner can
    later read each task's ``environment/`` paths and magic numbers.
    """
    if evaluator.task_source == "skillsbench":
        from skill_evolve.benchmark.skillsbench_anonymize import (
            build_skillsbench_id_map,
            find_leaked_skillsbench_names,
            sanitize_artifact_skillsbench,
            sanitize_text_skillsbench,
        )

        records = _load_skillsbench_task_records(evaluator.task_list)
        evaluator._task_records = records
        id_map = build_skillsbench_id_map(records)
        evaluator.set_task_id_map(id_map)

        def _sanitize_artifact_adapter(art, mapping):
            _, detail = sanitize_artifact_skillsbench(art, mapping)
            return len(detail), detail

        evaluator.anonymizer = _AnonymizerModule(
            sanitize_text=sanitize_text_skillsbench,
            sanitize_artifact=_sanitize_artifact_adapter,
            find_leaked_names=find_leaked_skillsbench_names,
        )
    else:
        # tblite path — reuse the in-file helpers untouched. The wrapped
        # ``sanitize_artifact`` already returns ``(replaced, detail)``.
        from .evaluator import (
            find_leaked_names as _find,
            sanitize_artifact as _sanitize_artifact,
            sanitize_text as _sanitize_text,
        )

        evaluator.anonymizer = _AnonymizerModule(
            sanitize_text=_sanitize_text,
            sanitize_artifact=_sanitize_artifact,
            find_leaked_names=_find,
        )


def _load_skillsbench_task_records(
    task_list: Optional[Path],
) -> List[Any]:
    """Hydrate the SkillsBench task records used to build the id_map.

    When ``task_list`` is provided, hydrates ONLY those tasks. When
    None, reads the full vendor directory listing. Each record is a
    :class:`skill_evolve.benchmark.load.Task` with ``task_id`` set to
    ``skillsbench/<dir_name>``.
    """
    from skill_evolve.benchmark.skillsbench_loader import hydrate_one

    vendor_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "benchmark"
        / "vendor"
        / "skillsbench"
        / "tasks"
    )
    if task_list is not None and Path(task_list).exists():
        ids = json.loads(Path(task_list).read_text(encoding="utf-8"))
        names = [(tid.split("/", 1)[-1] if "/" in tid else tid) for tid in ids]
    else:
        names = (
            sorted(
                p.name
                for p in vendor_dir.iterdir()
                if p.is_dir() and (p / "task.toml").exists()
            )
            if vendor_dir.is_dir()
            else []
        )
    records: List[Any] = []
    for name in names:
        td = vendor_dir / name
        if not (td / "task.toml").exists():
            logger.warning("skillsbench task missing: %s", td)
            continue
        try:
            records.append(hydrate_one(td))
        except Exception as exc:  # pragma: no cover — best-effort hydration
            logger.warning("skillsbench hydrate failed for %s: %s", name, exc)
    return records


@dataclass
class _AnonymizerModule:
    """Tiny duck-typed bundle holding the anonymizer entrypoints.

    Stored on ``SkillFolderEvaluator.anonymizer`` so chokepoints can
    call ``evaluator.anonymizer.sanitize_text(...)`` without re-doing
    the per-source dispatch on every call. ``sanitize_artifact`` is
    expected to return ``(replaced_count, detail)``.
    """

    sanitize_text: Any
    sanitize_artifact: Any
    find_leaked_names: Any


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
    # kai-skills patch (Group F, 2026-05-28): surface edit-budget
    # bookkeeping (parsed/applied/lt/clipped) on the iteration history
    # row when the iteration carried it. Omitted when None so legacy
    # rows are byte-identical.
    if res.edit_budget is not None:
        payload["edit_budget"] = res.edit_budget
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
