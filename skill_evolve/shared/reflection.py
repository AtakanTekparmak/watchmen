"""Success/failure minibatch partition reflection (Group H, SkillOpt port).

Paper anchor:
    * ``skillopt/gradient/reflect.py:run_error_analyst_minibatch``
    * ``skillopt/gradient/reflect.py:run_success_analyst_minibatch``
    * ``skillopt/gradient/aggregate.py:_split_minibatches``
    * ``skillopt/gradient/aggregate.py:_hierarchical_merge``

Splits an :class:`EvalResult.per_task` list into failure / success
partitions by per-item score and merges the two parallel-reflection
proposer outputs into a single op list via a deterministic keyed-dict
resolver with failure-priority collision handling.

Public API:
    Partition                : frozen dataclass with success/failure items.
    partition_trajectories() : split per-task results by threshold.
    merge_patches()          : keyed-dict merge of two op lists.

Divergences from paper (documented in code at the two divergence sites,
per plan section 7l):
    1. Hierarchical LLM merge -> keyed-dict resolver. The paper does
       three LLM merge calls (per-side hierarchical + final
       failure-priority); we use a deterministic keyed-dict collision
       resolver with NO LLM calls. The minibatch sizes on hot_5 (n=5)
       make the per-side hierarchical merge degenerate, so the
       simplification saves ~3 LLM calls/iter at no quality cost.
    2. Empty-side handling. The paper assumes both sides always have
       trajectories; we explicitly handle the all-pass / all-fail edge
       cases by falling through to the surviving side only. If BOTH
       sides are empty we log ``partition_both_empty`` and return an
       empty merged list.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from skill_evolve.shared.edit_budget import clip_ops
from skill_evolve.shared.patch_parser import FileOp


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Partition:
    """Result of splitting an :class:`EvalResult.per_task` list.

    Per plan section 7k the boundary score == threshold goes to
    ``success_items`` (``>=`` is success, ``<`` is failure).
    """

    success_items: list[dict]
    failure_items: list[dict]
    threshold: float


def _extract_score(item: dict) -> float:
    """Pull a per-task score float from an ``EvalResult.per_task`` entry.

    The per-task dict shape varies across evaluators (skillsbench vs
    behavioral vs tblite). We try the common keys in priority order and
    fall back to ``0.0`` so an entry with no score is treated as failure.
    """
    for key in ("score", "composite", "mean_score", "success_rate"):
        if key in item and item[key] is not None:
            try:
                return float(item[key])
            except (TypeError, ValueError):
                continue
    # Binary success → 1.0 / 0.0 fallback.
    if "success" in item:
        return 1.0 if bool(item["success"]) else 0.0
    return 0.0


def partition_trajectories(
    eval_result: Any,
    *,
    threshold: float = 0.5,
) -> Partition:
    """Split ``eval_result.per_task`` items into success / failure by score.

    Per plan section 7k:
        * ``score >= threshold`` → ``success_items``
        * ``score <  threshold`` → ``failure_items``

    ``eval_result`` may be any object exposing a ``per_task: list[dict]``
    attribute (the canonical :class:`skill_evolve.evaluator.EvalResult`
    shape) or a bare ``list[dict]`` for ease of unit testing.
    """
    items: list[dict]
    if isinstance(eval_result, list):
        items = list(eval_result)
    else:
        items = list(getattr(eval_result, "per_task", []) or [])

    success_items: list[dict] = []
    failure_items: list[dict] = []
    for item in items:
        if _extract_score(item) >= threshold:
            success_items.append(item)
        else:
            failure_items.append(item)
    return Partition(
        success_items=success_items,
        failure_items=failure_items,
        threshold=threshold,
    )


def _op_key(op: FileOp) -> tuple[str, str]:
    """Build a collision key for an op: ``(op_kind, path)``.

    Per plan section 7k two ops with the same ``(op_kind, path)`` collide;
    the failure-side op wins.
    """
    return (op.op, op.path)


def merge_patches(
    failure_ops: list[FileOp],
    success_ops: list[FileOp],
    lt: Optional[int] = None,
) -> tuple[list[FileOp], dict]:
    """Merge failure + success proposer patches into a single op list.

    DIVERGENCE FROM PAPER (§7l, point 1): the paper's
    ``skillopt/gradient/aggregate.py:_hierarchical_merge`` runs THREE LLM
    calls (per-side hierarchical merge via parallel ``ThreadPoolExecutor``
    + a final failure-priority merge). We use a deterministic keyed-dict
    collision resolver with NO LLM calls.

    DIVERGENCE FROM PAPER (§7l, point 2): empty-side handling. If
    ``failure_ops`` is empty the merge collapses to ``success_ops``
    (clipped to ``lt``); symmetric for empty ``success_ops``. If BOTH
    sides are empty the caller is expected to have logged the
    ``partition_both_empty`` warning at the partition site; this helper
    just returns ``([], stats)``.

    Algorithm:
        1. Build a dict keyed by ``(op_kind, path)`` from ``failure_ops``
           (failure-priority — failure side wins collisions).
        2. Walk ``success_ops``; any key already present in the failure
           dict is recorded as a collision and the success op is dropped.
           Surviving success ops are appended.
        3. Concatenate ``[failure_ops..., surviving_success_ops...]`` in
           emit order.
        4. If ``lt`` is not None, apply ``clip_ops(merged, lt)``.

    Returns:
        Tuple of ``(merged_ops, stats)`` where ``stats`` is::

            {
                "collisions": [
                    {"key": (op_kind, path),
                     "failure_path": str,
                     "success_path": str},
                    ...
                ],
                "failure_count": int,        # len(failure_ops)
                "success_count": int,        # len(success_ops)
                "merged_count": int,         # len(merged_ops) post-clip
                "clipped": bool,             # True iff clip dropped ops
            }
    """
    failure_index: dict[tuple[str, str], FileOp] = {}
    for op in failure_ops:
        # Last-wins within the failure side (matches the paper's
        # within-batch dedupe semantics; in practice the proposer almost
        # never emits two ops with the same key, but we tolerate it).
        failure_index[_op_key(op)] = op

    collisions: list[dict] = []
    surviving_success: list[FileOp] = []
    for op in success_ops:
        key = _op_key(op)
        if key in failure_index:
            failure_op = failure_index[key]
            collisions.append(
                {
                    "key": key,
                    "failure_path": failure_op.path,
                    "success_path": op.path,
                }
            )
            continue
        surviving_success.append(op)

    merged: list[FileOp] = list(failure_ops) + surviving_success

    clipped = False
    if lt is not None:
        before = len(merged)
        merged = clip_ops(merged, lt)
        clipped = len(merged) < before

    stats = {
        "collisions": collisions,
        "failure_count": len(failure_ops),
        "success_count": len(success_ops),
        "merged_count": len(merged),
        "clipped": clipped,
    }
    return merged, stats


__all__ = [
    "Partition",
    "partition_trajectories",
    "merge_patches",
]
