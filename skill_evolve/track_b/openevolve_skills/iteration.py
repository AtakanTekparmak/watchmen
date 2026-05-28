"""Single-iteration loop — fork of openevolve/iteration.py.

A Track-B iteration:

1. Pick an island (round-robin — simple, stable, good enough for 3
   islands × 30 generations).
2. Sample a parent uniformly from that island's MAP-Elites archive.
3. Sample a second "inspiration" program (same island, different from
   parent if possible).
4. Build a prompt (:mod:`.prompt_sampler`).
5. Ask the LLM for a patch; feed it through :mod:`.patch_parser.mutate`.
6. Evaluate the child; place it in the island archive.
7. Periodically migrate (every ``migration_interval`` generations).

Everything is synchronous — we intentionally don't replicate
openevolve's ProcessPoolExecutor worker layer, because the evaluator
itself subprocesses out to ``run_agent.py`` per task. Adding another
process layer here would be fork-overhead without benefit.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

from skill_evolve.shared.edit_budget import (
    ScheduleSpec,
    clip_ops,
    compute_lt,
)
from skill_evolve.shared.reflection import (
    merge_patches,
    partition_trajectories,
)
from skill_evolve.shared.rejected_buffer import RejectedBuffer, RejectedEdit

from .database import Program, ProgramDatabase, new_program_id
from .evaluator import SkillFolderEvaluator
from .folder_artifact import FolderArtifactError
from .llm_client import LLMClient, SyntheticLLM
from .patch_parser import (
    PatchParseError,
    apply_patch,
    mutate,
    parse_patch,
)
from .prompt_sampler import PromptSampler

logger = logging.getLogger(__name__)


# kai-skills patch (Group E, 2026-05-28): mutable per-run acceptance
# bookkeeping for the validation gate. The controller owns one
# ``GateState`` instance per run and threads it (alongside the optional
# ``RejectedBuffer``) into every ``run_iteration`` call so the strict
# gate can compare the candidate's val_score against the best-seen.
@dataclass
class GateState:
    """Run-level state required by the strict / relaxed validation gates.

    Per plan section 7j ``best_val_score_seen_so_far`` starts at
    ``-inf`` so the first non-rejected val_score is always strictly
    greater (and thus accepted) under ``strict`` mode.
    """

    best_val_score_seen_so_far: float = float("-inf")


ValidationGate = Literal["strict", "record", "relaxed"]
_VALID_GATE_MODES: tuple[str, ...] = ("strict", "record", "relaxed")


# kai-skills patch (Group H, 2026-05-28): reflection-mode plumbing per
# plan section 7m. ``single`` preserves the back-compat one-proposer-call
# path; ``partition`` runs two parallel proposer calls (one for failures,
# one for successes) and merges via the keyed-dict resolver in
# ``shared.reflection``.
ReflectionMode = Literal["single", "partition"]
_VALID_REFLECTION_MODES: tuple[str, ...] = ("single", "partition")


def _persist_rejected_buffer(
    buffer: Optional[RejectedBuffer],
    rejected_buffer_path: Optional[Path],
) -> None:
    """Best-effort JSONL persist of the buffer after every rejection.

    Errors are logged and swallowed — a crashed run still has the
    in-memory ring; persistence is a recovery aid, never load-bearing.
    """
    if buffer is None or rejected_buffer_path is None:
        return
    try:
        buffer.to_jsonl(rejected_buffer_path)
    except Exception as exc:  # pragma: no cover — persist must not crash run
        logger.warning(
            "rejected_buffer: persist failed at %s: %s",
            rejected_buffer_path,
            exc,
        )


def _push_rejection(
    *,
    buffer: Optional[RejectedBuffer],
    rejected_buffer_path: Optional[Path],
    patch_text: str,
    delta_train: float,
    delta_val: float,
    reason: str,
    iteration: int,
) -> None:
    """Push a rejection entry to the buffer and persist immediately."""
    if buffer is None:
        return
    buffer.push(
        RejectedEdit(
            patch_text=patch_text,
            delta_train=delta_train,
            delta_val=delta_val,
            rejection_reason=reason,
            iteration=iteration,
        )
    )
    _persist_rejected_buffer(buffer, rejected_buffer_path)


def _run_smoke_gate(child_artifact) -> Optional[str]:
    """Materialize ``child_artifact`` to a tmp dir, run validate_scripts
    + token cap. Returns a short reason string on rejection, None on
    pass.

    Lives at module level so tests can monkey-patch it.
    """
    import tempfile
    from pathlib import Path as _P

    from skill_evolve.shared.bundle_ops import (
        MAX_BUNDLE_TOKENS,
        bundle_tokens,
        validate_scripts,
    )

    with tempfile.TemporaryDirectory(prefix="track_b_smoke_") as tmp:
        tmp_dir = _P(tmp)
        # FolderArtifact has a files dict — write each entry to disk.
        for relpath, content in child_artifact.files.items():
            target = tmp_dir / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        smoke = validate_scripts(tmp_dir)
        if not smoke.ok:
            # Concatenate the first ~3 failures so the artifact's note
            # carries useful detail without exploding.
            head = "; ".join(f"{p}: {m}" for p, m in smoke.failures[:3])
            extra = (
                f" (+{len(smoke.failures) - 3} more)" if len(smoke.failures) > 3 else ""
            )
            return f"smoke_failed: {head}{extra}"

        tokens = bundle_tokens(tmp_dir)
        if tokens > MAX_BUNDLE_TOKENS:
            return f"token_cap_exceeded: {tokens} > {MAX_BUNDLE_TOKENS}"

    return None


# kai-skills patch (Group H, 2026-05-28): partition reflection helpers.
# These live at module level so tests can monkey-patch them. The two
# parallel proposer calls fan out via ThreadPoolExecutor(max_workers=2)
# per plan section 7m. NOTE: paper divergence (§7l, point 1): we use a
# deterministic keyed-dict resolver in merge_patches() rather than the
# paper's hierarchical LLM merge — see ``shared.reflection``.
def _parse_parent_per_task(parent) -> list:
    """Extract the parent's per-task list from its eval_artifacts.

    The evaluator stores ``per_task`` as a JSON string under
    ``eval_artifacts["per_task"]``; we decode it lazily here so the
    partition path can read it without re-evaluating. Returns an empty
    list when the field is missing or malformed (caller falls back to
    single-mode per §7l empty-side handling).
    """
    import json as _json

    raw = ""
    eval_arts = getattr(parent, "eval_artifacts", None)
    if eval_arts is not None:
        raw = eval_arts.get("per_task", "") or ""
    if not raw:
        return []
    try:
        data = _json.loads(raw)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    return data


def _render_trajectory_block(items: list, *, max_items: int = 8) -> str:
    """Render a list of per-task dicts as a compact trajectory block.

    Used by the partition-mode proposer prompts to ground the failure /
    success reflection. Caps at ``max_items`` per side (matches the
    ``--reflection-batch-size B_m`` default of 8).
    """
    import json as _json

    if not items:
        return "(none)"
    head = items[:max_items]
    lines: list[str] = []
    for i, item in enumerate(head, 1):
        task_id = item.get("task_id", "<unknown>")
        verifier = item.get("verifier_status", "")
        last_msg = item.get("last_msg", "") or ""
        last_msg = last_msg.replace("\n", " ")[:200]
        lines.append(
            f"{i}. task_id={task_id} verifier={verifier} "
            f"last_msg={_json.dumps(last_msg)}"
        )
    if len(items) > max_items:
        lines.append(f"... (+{len(items) - max_items} more)")
    return "\n".join(lines)


def _run_partition_reflection(
    *,
    parent,
    llm: LLMClient,
    base_prompt,
    success_threshold: float,
    max_items_per_side: int = 8,
):
    """Run the two parallel failure / success proposer calls and parse.

    Returns ``None`` if both partitions are empty (fall through to
    single-mode per §7l). Otherwise returns
    ``(failure_response, success_response, failure_ops, success_ops, merge_stats)``
    where each ``*_response`` may be the empty string when the
    corresponding side was empty / errored, and the parsed ``*_ops``
    lists are passed to :func:`merge_patches` by the caller.
    """
    import concurrent.futures as _cf

    # Lazy imports to avoid circular references at module load.
    from skill_evolve.track_a.prompts import (
        render_failure_reflection_prompt,
        render_success_reflection_prompt,
    )

    per_task = _parse_parent_per_task(parent)
    partition = partition_trajectories(per_task, threshold=success_threshold)

    # §7l empty-side handling: BOTH empty -> caller falls back to
    # single-mode. ONE empty -> run only the surviving side.
    if not partition.failure_items and not partition.success_items:
        logger.warning(
            "partition_both_empty: parent has no per-task data; "
            "falling back to single-mode at proposer call"
        )
        return None

    failure_trajectories = _render_trajectory_block(
        partition.failure_items, max_items=max_items_per_side
    )
    success_trajectories = _render_trajectory_block(
        partition.success_items, max_items=max_items_per_side
    )

    failure_system = render_failure_reflection_prompt(
        trajectories=failure_trajectories,
    )
    success_system = render_success_reflection_prompt(
        trajectories=success_trajectories,
    )

    def _call_failure() -> str:
        if not partition.failure_items:
            return ""
        try:
            return llm.generate(system=failure_system, user=base_prompt.user)
        except Exception as exc:  # pragma: no cover — network failures
            logger.warning("partition failure-side LLM failed: %s", exc)
            return ""

    def _call_success() -> str:
        if not partition.success_items:
            return ""
        try:
            return llm.generate(system=success_system, user=base_prompt.user)
        except Exception as exc:  # pragma: no cover — network failures
            logger.warning("partition success-side LLM failed: %s", exc)
            return ""

    with _cf.ThreadPoolExecutor(max_workers=2) as pool:
        failure_future = pool.submit(_call_failure)
        success_future = pool.submit(_call_success)
        failure_response = failure_future.result()
        success_response = success_future.result()

    # Parse each side. An empty / unparseable response yields an empty op
    # list — merge_patches handles the empty-side case per §7l.
    def _safe_parse(text: str) -> list:
        if not text:
            return []
        try:
            return parse_patch(text)
        except PatchParseError as exc:
            logger.warning("partition side parse failed: %s", exc)
            return []

    failure_ops = _safe_parse(failure_response)
    success_ops = _safe_parse(success_response)

    # Caller does the merge + clip. We expose only the raw stats here
    # so the iteration loop can surface them on the artifact.
    merge_stats = {
        "failure_count": len(failure_ops),
        "success_count": len(success_ops),
        "collisions": [],
    }
    return (
        failure_response,
        success_response,
        failure_ops,
        success_ops,
        merge_stats,
    )


def _persist_reflection_artifacts(
    artifact_dir: Optional[Path],
    *,
    generation: int,
    failure_response: str,
    success_response: str,
    failure_ops: list,
    success_ops: list,
    merge_stats: dict,
) -> None:
    """Persist per-side raw + parsed ops + merge stats to disk.

    Writes three JSON files per iteration:
        * reflection_failure.json
        * reflection_success.json
        * reflection_merge.json (with collisions list)

    No-op when ``artifact_dir`` is None. Best-effort: errors are logged
    and swallowed so a crashed write never kills the run.
    """
    if artifact_dir is None:
        return
    import json as _json

    def _op_to_dict(op) -> dict:
        return {
            "op": getattr(op, "op", "<unknown>"),
            "path": getattr(op, "path", ""),
            "content_head": ((getattr(op, "content", "") or "")[:500]),
        }

    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"iter_{generation:04d}_"
        (artifact_dir / f"{prefix}reflection_failure.json").write_text(
            _json.dumps(
                {
                    "response": failure_response,
                    "ops": [_op_to_dict(o) for o in failure_ops],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (artifact_dir / f"{prefix}reflection_success.json").write_text(
            _json.dumps(
                {
                    "response": success_response,
                    "ops": [_op_to_dict(o) for o in success_ops],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (artifact_dir / f"{prefix}reflection_merge.json").write_text(
            _json.dumps(merge_stats, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as exc:  # pragma: no cover — persist must not crash run
        logger.warning(
            "reflection artifacts persist failed at %s: %s",
            artifact_dir,
            exc,
        )


# ─── Slow-update consolidator helpers (Group G, plan §7l) ────────────────
#
# The consolidator path fires every K iters (``--slow-update-every K``,
# default 4) IN ADDITION to the regular fast-edit path. It emits a single
# EDIT_FILE SKILL.md sentinel block whose body REPLACES only the slow-
# update fenced region. The iteration loop double-guards via
# ``shared.slow_update.extract`` + ``replace`` so even if the
# consolidator's body lies about out-of-fence content, only in-fence
# bytes are taken. Per plan §7l "consolidator I/O".
#
# Paper-divergences documented at the relevant sites:
#   * Input is simplified — no A-vs-B skill comparison (plan §7l).
#   * meta_skill.md is a per-iter markdown audit log (NOT JSONL,
#     NOT a rolling coach memo) (plan §7l).


def _read_best_skill_md(database) -> str:
    """Return the current best SKILL.md text (bundle root or any nested
    ``<skill>/SKILL.md``). Used by the consolidator prompt.

    Falls back to an empty string when the database is empty or has no
    SKILL.md path. The consolidator prompt's slot has a literal
    ``(empty)`` fallback so the rendered prompt remains valid.
    """
    best = database.best()
    if best is None:
        return ""
    files = getattr(best.artifact, "files", None) or {}
    # Prefer bundle-root SKILL.md, then any nested <skill>/SKILL.md.
    if "SKILL.md" in files:
        return files["SKILL.md"]
    for rel, content in files.items():
        parts = rel.split("/")
        if parts[-1] == "SKILL.md":
            return content
    return ""


def _render_edit_history(
    rejected_buffer: Optional[RejectedBuffer],
    recent_accepted: Optional[list] = None,
) -> str:
    """Render the consolidator prompt's ``{edit_history}`` slot.

    Concatenates the recent accepted edits (most-recent first) with the
    live RejectedBuffer render. Both are optional; renders ``(no recent
    edits)`` when nothing is available.
    """
    blocks: list[str] = []
    if recent_accepted:
        blocks.append("### recent accepted edits\n")
        for i, entry in enumerate(recent_accepted[-10:][::-1]):
            blocks.append(
                f"- iter {entry.get('iteration', '?')}: {entry.get('summary', 'n/a')}"
            )
    if rejected_buffer is not None and len(rejected_buffer) > 0:
        blocks.append("\n### recently rejected patches\n")
        blocks.append(rejected_buffer.render_for_prompt())
    if not blocks:
        return "(no recent edits)"
    return "\n".join(blocks)


def _render_persistent_failures(streaks: Dict[str, int], window: int) -> str:
    """Render the consolidator prompt's ``{persistent_failures}`` slot.

    A task is "persistent" once its streak >= ``window``. Returns
    ``(none)`` when no task meets the threshold (matches the rendered
    slot's fallback literal in ``render_consolidator_prompt``).
    """
    persistent = [tid for tid, n in streaks.items() if n >= window]
    if not persistent:
        return "(none)"
    return "\n".join(
        f"- {tid} (failed {streaks[tid]} consecutive iters)"
        for tid in sorted(persistent)
    )


def _apply_validation_gate(
    *,
    gate_mode: ValidationGate,
    gate_state: GateState,
    has_val_task_list: bool,
    train_no_regression: bool,
    val_score: Optional[float],
    reason_prefix: str = "",
) -> Optional[str]:
    """Apply E's strict/relaxed validation-gate decision logic.

    Returns ``None`` on accept; a rejection-reason string on reject.
    Mutates ``gate_state.best_val_score_seen_so_far`` in place when the
    candidate's ``val_score`` advances the best-seen.

    ``reason_prefix`` is prepended to the canonical reason tokens
    (``train_no_improve`` / ``val_not_strict_gt`` / ``val_tie``) so the
    consolidator path can surface its rejections with a distinct prefix
    (e.g. ``consolidator_train_regression``).

    The fast-edit ``run_iteration`` gate site (~line 1319) and the
    consolidator path BOTH call this helper so the strict-gate semantics
    stay in exactly one place. Behavior under ``record`` is no-gate
    (returns ``None``); callers under ``record`` mode still get the
    legacy plan_0 Group-B "accept on train criterion only" semantics by
    short-circuiting before calling this helper.
    """
    if gate_mode not in ("strict", "relaxed"):
        return None

    # The fast-edit path uses the bare canonical token; the consolidator
    # uses a prefixed form. Centralize the mapping here.
    if reason_prefix:
        train_token = f"{reason_prefix}train_regression"
        not_gt_token = f"{reason_prefix}val_not_strict_gt"
        tie_token = f"{reason_prefix}val_tie"
    else:
        train_token = "train_no_improve"
        not_gt_token = "val_not_strict_gt"
        tie_token = "val_tie"

    if not train_no_regression:
        return train_token
    if has_val_task_list and val_score is None:
        # Validation eval was supposed to run but failed (logged
        # upstream) — under strict we cannot prove improvement, so
        # treat as not-strict-gt. Under relaxed we likewise refuse
        # to accept blind.
        return not_gt_token
    if has_val_task_list and val_score is not None:
        best_val = gate_state.best_val_score_seen_so_far
        if gate_mode == "strict":
            if val_score > best_val:
                gate_state.best_val_score_seen_so_far = val_score
                return None
            if val_score == best_val:
                return tie_token
            return not_gt_token
        # relaxed
        if val_score >= best_val:
            if val_score > best_val:
                gate_state.best_val_score_seen_so_far = val_score
            return None
        return not_gt_token
    # No validation set configured — strict/relaxed degenerate to
    # "train criterion only", which we have already satisfied.
    return None


def _run_consolidator_iteration(
    *,
    generation: int,
    database,
    llm: LLMClient,
    rejected_buffer: Optional[RejectedBuffer],
    meta_skill,
    meta_skill_path: Optional[Path],
    persistent_failure_streaks: Dict[str, int],
    persistent_failure_window: int,
    consolidator_model: Optional[str],
    evaluator: Optional[SkillFolderEvaluator] = None,
    validation_gate_mode: ValidationGate = "record",
    gate_state: Optional[GateState] = None,
    parent_train_score: Optional[float] = None,
    rejected_buffer_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Fire one consolidator pass.

    Returns a metadata dict that becomes ``IterationResult.consolidator``.
    The consolidator's effect is bounded: it emits a single
    ``EDIT_FILE SKILL.md`` sentinel block; the iteration loop extracts
    only the in-fence content and applies it to the parent's SKILL.md
    via ``replace_slow_update_field``.

    Plan §G.6 + §7l mandate: consolidator outputs go through the SAME
    strict validation gate as fast edits (E's gate). When ``evaluator``
    and ``validation_gate_mode`` in ``{"strict", "relaxed"}`` are
    supplied, this helper materializes the candidate bundle to a temp
    dir, calls ``evaluator.evaluate_artifact`` (and
    ``evaluator.evaluate_validation`` when train did not regress), then
    applies ``_apply_validation_gate``. On reject we push to
    ``RejectedBuffer`` with reason ``consolidator_train_regression`` /
    ``consolidator_val_not_strict_gt`` / ``consolidator_val_tie`` and
    leave the database's archive entry untouched (no in-place mutation,
    no meta-skill append).

    On parse error, fence violation, empty output, or LLM error: nothing
    is mutated; no meta-skill entry is appended.

    When ``evaluator`` is ``None`` or ``validation_gate_mode == "record"``
    the gate is skipped and the candidate is accepted unconditionally —
    this preserves the pre-fix back-compat behavior for callers
    (tests, controllers) that have not been threaded with the gate
    plumbing.

    All errors are caught and logged; the consolidator path MUST NOT
    crash the regular evolution loop.
    """
    from skill_evolve.shared.slow_update import (
        extract_slow_update_field,
        inject_slow_update_field,
        has_slow_update_field,
        replace_slow_update_field,
    )
    from skill_evolve.shared.meta_skill import MetaSkillEntry
    from skill_evolve.shared.patch_parser import (
        SentinelParseError,
        parse_sentinel_blocks,
    )
    from skill_evolve.track_a.prompts import render_consolidator_prompt

    meta: Dict[str, Any] = {
        "fired": True,
        "accepted": False,
        "reason": "",
        "meta_skill_appended": False,
    }

    best = database.best()
    if best is None:
        meta["reason"] = "no_best_program"
        return meta

    current_skill = _read_best_skill_md(database)
    # Ensure parent has a fence so the consolidator has something to
    # rewrite. If absent, inject an empty one on a working copy so the
    # consolidator can populate it on first fire.
    parent_skill_md = current_skill
    if not has_slow_update_field(parent_skill_md):
        parent_skill_md = inject_slow_update_field(parent_skill_md, content="")

    edit_history = _render_edit_history(rejected_buffer, recent_accepted=None)
    persistent_failures = _render_persistent_failures(
        persistent_failure_streaks, persistent_failure_window
    )
    system_prompt = render_consolidator_prompt(
        best_skill=parent_skill_md,
        edit_history=edit_history,
        persistent_failures=persistent_failures,
    )

    # Call the consolidator LLM. We reuse the same LLMClient for now
    # (consolidator_model is an opt-in override for the future; the live
    # LLMClient holds a single model handle so a true separate-model
    # call would require client construction at controller scope — out
    # of scope for the initial G implementation).
    _ = consolidator_model  # reserved for future multi-client wiring.
    user_msg = (
        "Emit exactly ONE EDIT_FILE SKILL.md sentinel block per the "
        "OUTPUT CONTRACT. Keep all bytes outside the SLOW_UPDATE fence "
        "byte-identical to the current best."
    )
    try:
        response = llm.generate(system=system_prompt, user=user_msg)
    except Exception as exc:  # pragma: no cover — network-level failures
        logger.warning("consolidator: LLM generate() failed: %s", exc)
        meta["reason"] = f"llm_error: {exc}"
        return meta

    # Parse the response — the consolidator MUST emit exactly one
    # EDIT_FILE on SKILL.md. Any other op shape is rejected.
    try:
        ops = parse_sentinel_blocks(
            response,
            existing_paths={"SKILL.md"},
            strict=False,
            # parent_skill_md=None here: we explicitly want the fast-
            # proposer fence-overlap check OFF for the consolidator
            # because the consolidator's whole job is to mutate the
            # fenced region. The double-guard (extract + replace below)
            # enforces the OUT-of-fence preservation independently.
            parent_skill_md=None,
        )
    except SentinelParseError as exc:
        logger.warning("consolidator: parse error: %s", exc)
        meta["reason"] = f"parse_error: {exc}"
        return meta

    # Filter to EDIT_FILE on a SKILL.md.
    edit_ops = [
        op
        for op in ops
        if op.op == "EDIT_FILE" and op.path.split("/")[-1] == "SKILL.md"
    ]
    if len(edit_ops) != 1:
        meta["reason"] = (
            f"consolidator_invalid_op_shape: expected exactly 1 "
            f"EDIT_FILE on SKILL.md, got {len(ops)} ops "
            f"({len(edit_ops)} matched)"
        )
        logger.warning("consolidator: %s", meta["reason"])
        return meta

    proposed_new_skill_md = edit_ops[0].content or ""

    # Double-guard: extract the proposed in-fence content; if the
    # consolidator's body has no fence we cannot recover the new content.
    try:
        new_in_fence = extract_slow_update_field(proposed_new_skill_md)
    except ValueError as exc:
        meta["reason"] = f"fence_extract_failed: {exc}"
        logger.warning("consolidator: %s", meta["reason"])
        return meta

    if not new_in_fence.strip():
        meta["reason"] = "empty_in_fence_content"
        logger.warning("consolidator: %s", meta["reason"])
        return meta

    # Apply ONLY to the fenced region of the actual current SKILL.md
    # (NOT proposed_new_skill_md) so out-of-fence content is preserved
    # verbatim from the current best.
    try:
        updated_skill = replace_slow_update_field(parent_skill_md, new_in_fence)
    except ValueError as exc:
        meta["reason"] = f"fence_replace_failed: {exc}"
        logger.warning("consolidator: %s", meta["reason"])
        return meta

    # Decide which path to update — bundle root SKILL.md takes
    # precedence; else any nested <skill>/SKILL.md.
    best_files = dict(best.artifact.files)
    target_key: Optional[str] = None
    if "SKILL.md" in best_files:
        target_key = "SKILL.md"
    else:
        for k in best_files:
            if k.split("/")[-1] == "SKILL.md":
                target_key = k
                break
    if target_key is None:
        meta["reason"] = "no_skill_md_in_best_artifact"
        return meta

    # ── Strict validation gate (plan §G.6 + §7l) ──────────────────────
    # Build a CANDIDATE artifact with the updated SKILL.md and run it
    # through the same gate logic as fast edits. We only run the gate
    # under strict / relaxed modes AND when an evaluator is supplied;
    # ``record`` mode and missing-evaluator both bypass the gate (back-
    # compat for callers / tests that have not been threaded).
    gate_active = (
        evaluator is not None
        and validation_gate_mode in ("strict", "relaxed")
        and gate_state is not None
    )
    if gate_active:
        assert evaluator is not None  # for type-checker
        assert gate_state is not None
        candidate_files = dict(best_files)
        candidate_files[target_key] = updated_skill
        try:
            candidate_artifact = type(best.artifact)(files=candidate_files)
            candidate_artifact.validate()
        except FolderArtifactError as exc:
            meta["reason"] = f"consolidator_candidate_invalid: {exc}"
            logger.warning("consolidator: %s", meta["reason"])
            return meta

        # Train-side eval — the consolidator candidate must not regress
        # against the parent's train composite.
        parent_train = (
            parent_train_score
            if parent_train_score is not None
            else float(best.metrics.get("composite", 0.0))
        )
        try:
            cand_eval = evaluator.evaluate_artifact(candidate_artifact, program_id="")
        except Exception as exc:  # pragma: no cover — evaluator failures
            logger.warning("consolidator: evaluator crashed: %s", exc)
            meta["reason"] = f"consolidator_eval_error: {exc}"
            return meta

        cand_train = float(cand_eval.metrics.get("composite", 0.0))
        train_delta = cand_train - parent_train
        train_no_regression = train_delta >= 0.0

        # Val-side eval — mirrors run_iteration's gate-site behavior:
        # evaluate val only when train did not regress (avoids a wasted
        # val rollout on a clearly-rejected candidate).
        has_val_task_list = getattr(evaluator, "validation_task_list", None) is not None
        cand_val_score: Optional[float] = None
        if has_val_task_list and train_no_regression:
            try:
                val_metrics_raw = evaluator.evaluate_validation(
                    candidate_artifact, program_id=""
                )
                if val_metrics_raw:
                    val_metrics = {
                        k: (0.0 if v is None else float(v))
                        for k, v in val_metrics_raw.items()
                    }
                    cand_val_score = val_metrics.get(
                        "validation_composite",
                        val_metrics.get("validation_score"),
                    )
            except Exception as exc:  # pragma: no cover — never fail iter on val
                logger.warning("consolidator: validation eval failed: %s", exc)

        rejection_reason = _apply_validation_gate(
            gate_mode=validation_gate_mode,
            gate_state=gate_state,
            has_val_task_list=has_val_task_list,
            train_no_regression=train_no_regression,
            val_score=cand_val_score,
            reason_prefix="consolidator_",
        )
        if rejection_reason is not None:
            delta_val = (
                (cand_val_score - gate_state.best_val_score_seen_so_far)
                if cand_val_score is not None
                else 0.0
            )
            _push_rejection(
                buffer=rejected_buffer,
                rejected_buffer_path=rejected_buffer_path,
                patch_text=response,
                delta_train=train_delta,
                delta_val=delta_val,
                reason=rejection_reason,
                iteration=generation,
            )
            meta["reason"] = rejection_reason
            logger.info(
                "consolidator: gate rejected (%s) train_delta=%+.4f val=%s",
                rejection_reason,
                train_delta,
                f"{cand_val_score:.4f}" if cand_val_score is not None else "n/a",
            )
            # DO NOT mutate best.artifact.files; DO NOT append meta_skill.
            return meta

    # Gate passed (or bypassed). Write the update onto the current
    # best's artifact in-place. The consolidator's edits are a
    # background channel; the existing database / archive ordering by
    # fitness still holds.
    best_files[target_key] = updated_skill
    # FolderArtifact is a frozen-ish dataclass with a normalizer in
    # __post_init__ — mutating ``files`` directly is supported because
    # the constructor only normalizes on instantiation.
    best.artifact.files = best_files

    # Append a meta-skill entry summarizing this consolidator fire and
    # persist to disk.
    try:
        entry = MetaSkillEntry(
            iteration=generation,
            patch_summary=(
                f"slow-update consolidator: {len(new_in_fence)} chars "
                f"written to slow_update field"
            ),
            lessons=[],
            failures_observed=[
                tid
                for tid, n in persistent_failure_streaks.items()
                if n >= persistent_failure_window
            ],
        )
        meta_skill.append(entry)
        if meta_skill_path is not None:
            meta_skill.to_markdown_file(meta_skill_path)
            meta["meta_skill_appended"] = True
    except Exception as exc:  # pragma: no cover — persistence is best-effort
        logger.warning("consolidator: meta-skill persist failed: %s", exc)

    meta["accepted"] = True
    meta["reason"] = "accepted"
    return meta


def _update_persistent_failure_streaks(
    streaks: Dict[str, int],
    eval_artifacts: Optional[Dict[str, str]],
    *,
    threshold: float = 0.5,
) -> None:
    """Update per-task consecutive-failure counters from the child eval.

    A task counts as "failed" when its per-task score is below
    ``threshold`` (matches Group H's reflection_success_threshold
    default). Tasks not in the current eval are left alone — we only
    increment counters for tasks we actually saw.
    """
    if not eval_artifacts:
        return
    import json as _json

    raw = eval_artifacts.get("per_task", "") if eval_artifacts else ""
    if not raw:
        return
    try:
        per_task = _json.loads(raw)
    except (ValueError, TypeError):
        return
    if not isinstance(per_task, list):
        return
    for item in per_task:
        if not isinstance(item, dict):
            continue
        tid = item.get("task_id") or item.get("id")
        if not tid:
            continue
        score = item.get("score")
        try:
            score_f = float(score) if score is not None else 0.0
        except (TypeError, ValueError):
            score_f = 0.0
        if score_f < threshold:
            streaks[tid] = streaks.get(tid, 0) + 1
        else:
            streaks[tid] = 0


@dataclass
class IterationResult:
    generation: int
    island: int
    parent_id: str
    child_id: Optional[str]
    op_type: (
        str  # "patch" | "rewrite" | "parse_error" | "validate_error" | "eval_error"
    )
    score_delta: float
    cell: Optional[tuple] = None
    notes: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    # kai-skills patch (Group F, 2026-05-28): edit-budget bookkeeping.
    # parsed_op_count == count emitted by the proposer; applied_op_count ==
    # count surviving clip_ops(parsed, L_t). ``clipped`` is True iff the
    # apply path dropped any ops. ``lt`` records the L_t value used.
    edit_budget: Optional[Dict[str, Any]] = None
    # kai-skills patch (Group H, 2026-05-28): success/failure reflection
    # bookkeeping. Records per-side parsed op counts, collisions, and the
    # merged op count when --reflection-mode partition is active. ``None``
    # in single-mode (back-compat).
    reflection: Optional[Dict[str, Any]] = None
    # kai-skills patch (Group G, 2026-05-28; plan §7l): slow-update
    # consolidator bookkeeping. Populated on iters where the consolidator
    # path fired (every K iters). Fields: ``fired`` (bool), ``accepted``
    # (bool), ``reason`` (str — accept / parse_error / fence_violation /
    # gate_rejected), ``meta_skill_appended`` (bool).
    consolidator: Optional[Dict[str, Any]] = None


def run_iteration(
    generation: int,
    database: ProgramDatabase,
    evaluator: SkillFolderEvaluator,
    llm: LLMClient,
    prompt_sampler: PromptSampler,
    *,
    island: int,
    edit_budget: Optional[Union[str, ScheduleSpec]] = None,
    max_iters: Optional[int] = None,
    validation_gate: ValidationGate = "record",
    rejected_buffer: Optional[RejectedBuffer] = None,
    gate_state: Optional[GateState] = None,
    rejected_buffer_path: Optional[Path] = None,
    reflection_mode: ReflectionMode = "single",
    reflection_success_threshold: float = 0.5,
    reflection_artifact_dir: Optional[Path] = None,
    # kai-skills patch (Group G, 2026-05-28; plan §7l): slow-update
    # consolidator + meta-skill audit log. The consolidator path runs
    # every ``slow_update_every`` iters IN ADDITION to (not instead of)
    # the regular fast-edit path.
    slow_update_every: int = 0,
    meta_skill: Optional[Any] = None,  # MetaSkill — typed loosely to keep import lazy.
    meta_skill_path: Optional[Path] = None,
    consolidator_model: Optional[str] = None,
    persistent_failure_streaks: Optional[Dict[str, int]] = None,
    persistent_failure_window: int = 3,
) -> IterationResult:
    """Run one evolution iteration on ``island``.

    Args:
        edit_budget: schedule spec for the per-iteration L_t cap
            (``constant:N`` / ``linear:N->M`` / ``cosine:N->M``) or a
            pre-parsed :class:`ScheduleSpec`. ``None`` disables clipping
            (back-compat for callers that haven't been threaded with the
            Group F flag).
        max_iters: total iteration count used to normalize ``iter_n`` in
            cosine / linear schedules. Defaults to ``generation`` when
            ``edit_budget`` is set and ``max_iters`` is not supplied
            (degenerate — every iter sits at t=1.0 so cosine endpoint
            reduces to ``end``).
        validation_gate: per plan section 7j. ``strict`` (CLI default)
            rejects child when ``train < parent_train`` OR
            ``val_score <= best_val_score_seen_so_far`` (ties rejected).
            ``relaxed`` accepts ties. ``record`` (function-default for
            back-compat) preserves the plan_0 Group-B behavior — accept
            on train criterion only and record val_score on artifact.
        rejected_buffer: optional run-level :class:`RejectedBuffer`.
            When supplied, every rejection (parse, smoke, train-noimp,
            val gate) pushes a :class:`RejectedEdit`.
        gate_state: optional run-level :class:`GateState`. Required for
            ``strict`` / ``relaxed`` modes (so the per-run
            ``best_val_score_seen_so_far`` can be tracked across
            iterations). Auto-instantiated locally if missing — but the
            state will then not persist across iters; the controller is
            responsible for threading a shared instance.
        rejected_buffer_path: optional path to persist the buffer JSONL
            after every rejection (crash-safe). No-op when ``None``.
        reflection_mode: per plan section 7m. ``single`` (default for
            back-compat) keeps the existing one-proposer-call path.
            ``partition`` runs TWO parallel proposer calls (failure /
            success reflection) and merges via
            ``shared.reflection.merge_patches``.
        reflection_success_threshold: per-task score threshold used to
            split the parent's evaluation trajectories into failure /
            success partitions under ``partition`` mode. Boundary score
            ``== threshold`` goes to the success side (``>=``).
        reflection_artifact_dir: optional directory to persist the
            per-side reflection JSON (failure / success / merge stats).
            ``None`` disables persistence; the in-memory ``reflection``
            field on :class:`IterationResult` still captures the stats.
    """
    if validation_gate not in _VALID_GATE_MODES:
        raise ValueError(
            f"validation_gate must be one of {_VALID_GATE_MODES}; "
            f"got {validation_gate!r}"
        )
    if reflection_mode not in _VALID_REFLECTION_MODES:
        raise ValueError(
            f"reflection_mode must be one of {_VALID_REFLECTION_MODES}; "
            f"got {reflection_mode!r}"
        )
    # Local gate-state fallback so callers that pass strict / relaxed
    # without threading state get sane behavior on a single iter (best
    # remains -inf the whole call). Controllers MUST supply the shared
    # instance to get cross-iter strictness.
    if gate_state is None:
        gate_state = GateState()
    t0 = time.monotonic()

    # kai-skills patch (Group G, 2026-05-28; plan §7l + §7n): top-of-loop
    # consolidator branch. Fires every ``slow_update_every`` iters when
    # iter_n > 0 (skip iter 0 — no parent yet). Runs IN ADDITION to the
    # regular fast-edit path so the iteration still produces a candidate
    # via the H/F/E pipeline below. DO NOT touch E's gate site (~533),
    # F's clip site (~309), or H's proposer call site (lower in body).
    consolidator_meta: Optional[Dict[str, Any]] = None
    if (
        slow_update_every > 0
        and generation > 0
        and generation % slow_update_every == 0
        and meta_skill is not None
        and persistent_failure_streaks is not None
    ):
        try:
            consolidator_meta = _run_consolidator_iteration(
                generation=generation,
                database=database,
                llm=llm,
                rejected_buffer=rejected_buffer,
                meta_skill=meta_skill,
                meta_skill_path=meta_skill_path,
                persistent_failure_streaks=persistent_failure_streaks,
                persistent_failure_window=persistent_failure_window,
                consolidator_model=consolidator_model,
                # kai-skills patch (Group G fix-round, 2026-05-28; plan
                # §G.6 + §7l): the consolidator's output must traverse
                # E's strict validation gate. Thread evaluator + gate
                # state + mode + rejected-buffer-persist path so the
                # consolidator can materialize, evaluate, gate, and
                # surface rejections through the same RejectedBuffer
                # the fast-edit path uses.
                evaluator=evaluator,
                validation_gate_mode=validation_gate,
                gate_state=gate_state,
                parent_train_score=None,  # helper falls back to db.best().metrics
                rejected_buffer_path=rejected_buffer_path,
            )
            logger.info(
                "consolidator fired at iter %d: accepted=%s reason=%s",
                generation,
                consolidator_meta.get("accepted"),
                consolidator_meta.get("reason"),
            )
        except Exception as exc:  # pragma: no cover — consolidator must not crash run
            logger.warning(
                "consolidator: unexpected error at iter %d: %s", generation, exc
            )
            consolidator_meta = {
                "fired": True,
                "accepted": False,
                "reason": f"unexpected_error: {exc}",
                "meta_skill_appended": False,
            }

    # 1. Sample parent + inspiration from the island's archive.
    parent = database.sample_parent(island)
    inspiration = database.sample_inspiration(island, exclude=parent.id)

    # 2. Build prompt. If our LLM is synthetic, feed it the parent directly
    #    so it can emit a deterministic valid patch.
    if isinstance(llm, SyntheticLLM):
        llm.set_parent(parent.artifact)
    prompt = prompt_sampler.build(parent, inspiration=inspiration)

    # 3. Ask the LLM.
    #
    # kai-skills patch (Group H, 2026-05-28): under ``--reflection-mode
    # partition`` the single proposer call splits into TWO parallel
    # proposer calls (failure / success reflection) whose op lists merge
    # via the keyed-dict resolver in ``shared.reflection``. The legacy
    # ``single`` path is preserved verbatim for back-compat. See plan
    # section 7m. NOTE: H owns this site; do NOT touch G's top-of-loop
    # consolidator branch, E's gate site (~533), or F's clip site (~309).
    edit_budget_meta: Optional[Dict[str, Any]] = None
    reflection_meta: Optional[Dict[str, Any]] = None
    response: str = ""  # populated below; carried through to buffer pushes.

    if reflection_mode == "partition":
        partition_result = _run_partition_reflection(
            parent=parent,
            llm=llm,
            base_prompt=prompt,
            success_threshold=reflection_success_threshold,
        )
        if partition_result is None:
            # Both sides empty (no per-task data on the parent yet — e.g.
            # generation 1 before any meaningful eval, or a fully blank
            # eval result) — fall through to single-mode behavior so the
            # iteration still produces a candidate. Per §7l empty-side
            # handling.
            logger.info(
                "partition_both_empty: falling back to single-mode "
                "proposer call at iter %d",
                generation,
            )
            try:
                response = llm.generate(system=prompt.system, user=prompt.user)
            except Exception as exc:  # pragma: no cover — network failures
                logger.warning("LLM generate() failed: %s", exc)
                return IterationResult(
                    generation=generation,
                    island=island,
                    parent_id=parent.id,
                    child_id=None,
                    op_type="eval_error",
                    score_delta=0.0,
                    notes=f"llm_error: {exc}",
                    consolidator=consolidator_meta,
                )
            partition_ops: Optional[list] = None
        else:
            (
                failure_response,
                success_response,
                failure_ops,
                success_ops,
                merge_stats,
            ) = partition_result
            reflection_meta = {
                "mode": "partition",
                "failure_count": merge_stats["failure_count"],
                "success_count": merge_stats["success_count"],
                "collisions": merge_stats["collisions"],
                "merged_count_pre_clip": (
                    merge_stats["failure_count"]
                    + merge_stats["success_count"]
                    - len(merge_stats["collisions"])
                ),
                "threshold": reflection_success_threshold,
            }
            # Apply L_t clip via merge_patches' lt arg. We compute lt up
            # front so the merge respects F's bounded-edit-budget
            # contract; the legacy single-path clip below sees no ops
            # this iter.
            effective_max_iters = max_iters if max_iters is not None else generation
            if edit_budget is not None:
                lt_value = compute_lt(edit_budget, generation, effective_max_iters)
            else:
                lt_value = None
            merged_ops, final_stats = merge_patches(
                failure_ops, success_ops, lt=lt_value
            )
            reflection_meta.update(
                {
                    "merged_count": final_stats["merged_count"],
                    "lt": lt_value,
                    "clipped": final_stats["clipped"],
                    "collisions": final_stats["collisions"],
                }
            )
            # Update the merge_stats now that we have collision data +
            # final merged count post-clip, then persist all three
            # reflection artifacts (raw failure / raw success / merge).
            merge_stats.update(final_stats)
            _persist_reflection_artifacts(
                reflection_artifact_dir,
                generation=generation,
                failure_response=failure_response,
                success_response=success_response,
                failure_ops=failure_ops,
                success_ops=success_ops,
                merge_stats=merge_stats,
            )
            partition_ops = merged_ops
            # Carry through a compact joined response string so the
            # rejected-buffer entry still has a textual handle on what
            # was proposed. Truncated to keep the buffer small.
            response = (
                "[partition]\n"
                "--- failure ---\n"
                + (failure_response[:2000] if failure_response else "")
                + "\n--- success ---\n"
                + (success_response[:2000] if success_response else "")
            )
    else:
        # ``single`` (back-compat) path — original single proposer call.
        try:
            response = llm.generate(system=prompt.system, user=prompt.user)
        except Exception as exc:  # pragma: no cover — network failures
            logger.warning("LLM generate() failed: %s", exc)
            return IterationResult(
                generation=generation,
                island=island,
                parent_id=parent.id,
                child_id=None,
                op_type="eval_error",
                score_delta=0.0,
                notes=f"llm_error: {exc}",
                consolidator=consolidator_meta,
            )
        partition_ops = None

    # 4. Parse + (optionally) clip + apply.
    #
    # kai-skills patch (Group F, 2026-05-28): the parse and apply phases
    # are split here so the bounded edit budget L_t clip lives BETWEEN
    # them — clipping after-apply would already have mutated the bundle.
    # When ``edit_budget`` is None we route through the legacy ``mutate``
    # helper unchanged (back-compat: callers that haven't been threaded
    # with the Group F flag see identical behavior).
    #
    # kai-skills patch (Group H, 2026-05-28): when ``partition_ops`` is
    # populated (partition reflection branch above) we skip the parse
    # step entirely — the ops list is already the merged + clipped
    # output of the two parallel proposer calls. Apply path is shared.
    try:
        if partition_ops is not None:
            if not partition_ops:
                raise PatchParseError(
                    "partition reflection produced no ops after merge"
                )
            edit_budget_meta = {
                "parsed_op_count": (
                    (reflection_meta or {}).get("failure_count", 0)
                    + (reflection_meta or {}).get("success_count", 0)
                ),
                "applied_op_count": len(partition_ops),
                "lt": (reflection_meta or {}).get("lt"),
                "clipped": (reflection_meta or {}).get("clipped", False),
            }
            child_artifact = apply_patch(parent.artifact, partition_ops)
            child_artifact.validate()
            if child_artifact.num_skills() == 0:
                raise PatchParseError(
                    "after applying patch: 0 skills (need >=1 SKILL.md subfolder)"
                )
        elif edit_budget is None:
            child_artifact = mutate(parent.artifact, response)
        else:
            parsed_ops = parse_patch(response)
            if not parsed_ops:
                raise PatchParseError("patch contained no operations")
            parsed_count = len(parsed_ops)
            # max_iters defaults to ``generation`` so a caller that
            # threads --edit-budget but forgets --max-iters still gets a
            # sensible cosine endpoint (t=1.0 -> ``end``).
            effective_max_iters = max_iters if max_iters is not None else generation
            lt = compute_lt(edit_budget, generation, effective_max_iters)
            applied_ops = clip_ops(parsed_ops, lt)
            applied_count = len(applied_ops)
            clipped = applied_count < parsed_count
            if clipped:
                logger.warning(
                    "edit_budget: clipped %d -> %d ops at iter %d",
                    parsed_count,
                    applied_count,
                    generation,
                )
            edit_budget_meta = {
                "parsed_op_count": parsed_count,
                "applied_op_count": applied_count,
                "lt": lt,
                "clipped": clipped,
            }
            child_artifact = apply_patch(parent.artifact, applied_ops)
            child_artifact.validate()
            if child_artifact.num_skills() == 0:
                raise PatchParseError(
                    "after applying patch: 0 skills (need >=1 SKILL.md subfolder)"
                )
    except (PatchParseError, FolderArtifactError) as exc:
        logger.warning("patch rejected: %s", exc)
        _push_rejection(
            buffer=rejected_buffer,
            rejected_buffer_path=rejected_buffer_path,
            patch_text=response,
            delta_train=0.0,
            delta_val=0.0,
            reason="parse_error",
            iteration=generation,
        )
        return IterationResult(
            generation=generation,
            island=island,
            parent_id=parent.id,
            child_id=None,
            op_type="parse_error",
            score_delta=0.0,
            notes=str(exc),
            edit_budget=edit_budget_meta,
            reflection=reflection_meta,
            consolidator=consolidator_meta,
        )

    # kai-skills patch (2026-04-27): validate-on-write guard against
    # task-name leakage. When the evaluator is in anonymize mode, the
    # outer LLM should never see real task IDs — but a misbehaving model
    # could still emit them by guessing. Reject any patch that
    # reintroduces a redacted name into the *.md prose layer (the
    # router-readable surface). Scripts/refs are not scanned because they
    # legitimately reference task fixtures by name.
    if getattr(evaluator, "anonymize_tasks", False):
        # kai-skills patch (Phase E, 2026-04-29): dispatch through the
        # anonymizer module the Controller installed (in-file for
        # tblite, skillsbench_anonymize for skillsbench). Falls back to
        # the in-file functions if Controller wiring was bypassed
        # (synthetic LLM tests etc.).
        if getattr(evaluator, "anonymizer", None) is not None:
            _find_leaked_names = evaluator.anonymizer.find_leaked_names
        else:
            from .evaluator import find_leaked_names as _find_leaked_names
        leaks = _find_leaked_names(child_artifact, evaluator.task_id_map())
        if leaks:
            file_, name = leaks[0]
            msg = (
                f"patch reintroduces redacted task name `{name}` in "
                f"`{file_}`; rejected ({len(leaks)} total leak(s))"
            )
            logger.warning(msg)
            return IterationResult(
                generation=generation,
                island=island,
                parent_id=parent.id,
                child_id=None,
                op_type="parse_error",
                score_delta=0.0,
                notes=msg,
                edit_budget=edit_budget_meta,
                reflection=reflection_meta,
            )
    # kai-skills patch end

    # kai-skills patch (2026-05-27 — Group A daycare port): smoke + token
    # cap gate. Materialize the child artifact to a tmp dir, run
    # ``validate_scripts`` (py_compile / bash -n) and ``bundle_tokens``
    # against the MAX_BUNDLE_TOKENS cap, and reject without eval if
    # either fails. Saves us a benchmark call on a candidate the agent
    # cannot run anyway.
    smoke_reason = _run_smoke_gate(child_artifact)
    if smoke_reason is not None:
        logger.warning("smoke gate rejected child: %s", smoke_reason)
        # kai-skills patch (Group E, 2026-05-28): differentiate between
        # ``smoke_rejected`` (py_compile / bash -n failure) and
        # ``token_cap`` (MAX_BUNDLE_TOKENS exceeded) when surfacing the
        # rejection to the buffer per plan section 7j enum.
        _reject_reason = (
            "token_cap"
            if smoke_reason.startswith("token_cap_exceeded")
            else "smoke_rejected"
        )
        _push_rejection(
            buffer=rejected_buffer,
            rejected_buffer_path=rejected_buffer_path,
            patch_text=response,
            delta_train=0.0,
            delta_val=0.0,
            reason=_reject_reason,
            iteration=generation,
        )
        return IterationResult(
            generation=generation,
            island=island,
            parent_id=parent.id,
            child_id=None,
            op_type="parse_error",
            score_delta=0.0,
            notes=f"smoke_rejected: {smoke_reason}",
            edit_budget=edit_budget_meta,
            reflection=reflection_meta,
            consolidator=consolidator_meta,
        )

    # 5. Evaluate child.
    try:
        eval_res = evaluator.evaluate_artifact(child_artifact, program_id="")
    except Exception as exc:  # pragma: no cover
        logger.exception("evaluator crashed: %s", exc)
        return IterationResult(
            generation=generation,
            island=island,
            parent_id=parent.id,
            child_id=None,
            op_type="eval_error",
            score_delta=0.0,
            notes=f"eval_error: {exc}",
            edit_budget=edit_budget_meta,
            reflection=reflection_meta,
            consolidator=consolidator_meta,
        )

    # 6. Wrap in Program, place in archive.
    child_metadata: Dict[str, Any] = {"iteration_time_s": time.monotonic() - t0}
    if edit_budget_meta is not None:
        # Persist into the program's metadata so the archive's meta.json
        # carries the parsed/applied/lt/clipped triple for post-hoc audit.
        child_metadata["edit_budget"] = dict(edit_budget_meta)
    child = Program(
        id=new_program_id(),
        artifact=child_artifact,
        parent_id=parent.id,
        generation=parent.generation + 1,
        iteration_found=generation,
        metrics=eval_res.metrics,
        eval_artifacts=eval_res.artifacts,
        metadata=child_metadata,
    )
    database.add(child, island=island)
    database.tick_generation(island)

    score_delta = child.fitness() - parent.fitness()
    from .database import cell_key as _cell_key

    cell = _cell_key(child.artifact)

    logger.info(
        "iter %d island %d: %.4f -> %.4f (Δ=%+.4f) cell=%s",
        generation,
        island,
        parent.fitness(),
        child.fitness(),
        score_delta,
        cell,
    )

    # kai-skills patch (Group B, 2026-05-27 / Group E, 2026-05-28):
    # held-out validation eval. Plan section 7j flips the semantics from
    # Group B's record-only behavior to a strict acceptance gate. Modes:
    #   * record  — plan_0 Group-B behavior: eval val only when train
    #               passed (score_delta > 0), record on artifact; the
    #               acceptance / rejection of the child has already been
    #               decided by score_delta > 0 (record-mode preserves
    #               the existing test_validation_holdout.py contract).
    #   * strict  — accept iff train >= parent AND val > best_val_seen.
    #   * relaxed — accept iff train >= parent AND val >= best_val_seen
    #               (ties accepted).
    # ``strict`` / ``relaxed`` evaluate val whenever the train side did
    # not regress so the gate can compare against
    # ``gate_state.best_val_score_seen_so_far`` even when score_delta == 0.
    has_val_task_list = getattr(evaluator, "validation_task_list", None) is not None
    train_no_regression = score_delta >= 0.0

    val_score: Optional[float] = None
    val_metrics: Dict[str, float] = {}
    if has_val_task_list and (
        (validation_gate == "record" and score_delta > 0.0)
        or (validation_gate in ("strict", "relaxed") and train_no_regression)
    ):
        try:
            val_metrics_raw = evaluator.evaluate_validation(
                child_artifact, program_id=child.id
            )
            if val_metrics_raw:
                val_metrics = {
                    k: (0.0 if v is None else float(v))
                    for k, v in val_metrics_raw.items()
                }
                child.metrics.update(val_metrics)
                # ``validation_score`` mirrors ``validation_composite`` for
                # downstream consumers that look for the flat
                # ``validation_score`` key (e.g. tests in plan_0 D.7).
                vc = val_metrics.get("validation_composite")
                if vc is not None and "validation_score" not in child.metrics:
                    child.metrics["validation_score"] = float(vc)
                val_score = val_metrics.get(
                    "validation_composite", val_metrics.get("validation_score")
                )
                logger.info(
                    "iter %d island %d: validation_composite=%.4f (n=%d)",
                    generation,
                    island,
                    val_metrics.get("validation_composite", 0.0),
                    val_metrics.get("validation_n", 0),
                )
        except Exception as exc:  # pragma: no cover — never fail iter on val
            logger.warning(
                "iter %d island %d: validation eval failed: %s",
                generation,
                island,
                exc,
            )

    # ----- Group E strict-gate decision -------------------------------
    # Default outcome: ACCEPTED. We override and push to the rejected
    # buffer when the gate rules say so. The Program is already in the
    # database (added above); strict-gate "rejection" here is an
    # advisory recording on the iteration result + buffer push — the
    # canonical fitness-based ``db.best()`` selection still relies on
    # the per-program ``composite`` so a rejected child with strictly
    # lower fitness cannot win anyway. The buffer push is what feeds the
    # proposer's RECENT REJECTIONS section.
    rejection_reason: Optional[str] = None
    op_type_out = "patch"
    if validation_gate in ("strict", "relaxed"):
        # Centralized gate decision — same helper used by the
        # consolidator path (plan §G.6 + §7l). The bare-token form
        # (``train_no_improve`` / ``val_not_strict_gt`` / ``val_tie``)
        # is the fast-edit canonical reason set; consolidator uses
        # ``consolidator_`` prefixed variants via reason_prefix.
        rejection_reason = _apply_validation_gate(
            gate_mode=validation_gate,
            gate_state=gate_state,
            has_val_task_list=has_val_task_list,
            train_no_regression=train_no_regression,
            val_score=val_score,
            reason_prefix="",
        )

    if rejection_reason is not None:
        # Delta_val is 0.0 when val wasn't evaluated (e.g.
        # train_no_improve rejection short-circuits before val eval).
        delta_val = (
            (val_score - gate_state.best_val_score_seen_so_far)
            if val_score is not None
            else 0.0
        )
        _push_rejection(
            buffer=rejected_buffer,
            rejected_buffer_path=rejected_buffer_path,
            patch_text=response,
            delta_train=score_delta,
            delta_val=delta_val,
            reason=rejection_reason,
            iteration=generation,
        )
        op_type_out = "gate_rejected"
        logger.info(
            "iter %d island %d: gate rejected (%s) train_delta=%+.4f val=%s",
            generation,
            island,
            rejection_reason,
            score_delta,
            f"{val_score:.4f}" if val_score is not None else "n/a",
        )

    # kai-skills patch (Group G, 2026-05-28; plan §7l): update the
    # persistent-failure streaks from this iter's child eval so the
    # next consolidator fire sees up-to-date task health.
    if persistent_failure_streaks is not None:
        _update_persistent_failure_streaks(
            persistent_failure_streaks,
            child.eval_artifacts,
            threshold=reflection_success_threshold,
        )

    return IterationResult(
        generation=generation,
        island=island,
        parent_id=parent.id,
        child_id=child.id,
        op_type=op_type_out,
        score_delta=score_delta,
        cell=cell,
        metrics=child.metrics,
        edit_budget=edit_budget_meta,
        reflection=reflection_meta,
        consolidator=consolidator_meta,
        notes=(rejection_reason or ""),
    )
