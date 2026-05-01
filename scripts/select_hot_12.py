#!/usr/bin/env python
"""Pick a 12-task hot subset for the inner evolution loop.

Reads:
  * a baseline summary JSON (with ``per_task`` containing per-task
    with-skills mean scores over 5 trials)
  * a task-list JSON (e.g. ``subset_20.json`` or ``subset_17.json`` —
    any JSON array of fully-qualified task IDs).

Bins the input tasks by with-skills score:
  * failing  : score < 0.2
  * partial  : 0.2 <= score < 0.8
  * passing  : score >= 0.8

Picks 4 from each bin (alpha-sorted by task_id). If a bin has < 4,
pulls from the next bin in priority order: partial > failing > passing.
If 12 cannot be assembled, fails loudly.

Tolerant of two summary shapes for the per-task with-skills score:

  * flat: ``{"task_id": ..., "with_skills_score": 0.6}``
  * nested (Phase D baseline): ``{"task_id": ..., "by_condition":
    {"with-skills": {"score_mean": 0.6, ...}, ...}}``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _extract_with_skills_score(task_record: Dict[str, Any]) -> float | None:
    """Pull a single with-skills score from a per-task summary record.

    Tolerant of multiple summary shapes:
      * ``with_skills_score`` (preferred — used by older summaries)
      * ``with_skills_rate``
      * ``by_condition["with-skills"]["score_mean"]`` (Phase D baseline)
      * ``by_condition["with-skills"]["pass_rate"]`` (alt Phase D)
      * ``score`` / ``mean_score`` / ``rate``  (fallback)
    """
    by_cond = task_record.get("by_condition")
    if isinstance(by_cond, dict):
        with_skills = by_cond.get("with-skills") or by_cond.get("with_skills")
        if isinstance(with_skills, dict):
            for key in ("score_mean", "pass_rate", "score", "rate"):
                if key in with_skills and with_skills[key] is not None:
                    try:
                        return float(with_skills[key])
                    except (TypeError, ValueError):
                        continue
    for key in (
        "with_skills_score",
        "with_skills_rate",
        "with_skills",
        "mean_score",
        "score",
        "rate",
    ):
        if key in task_record and task_record[key] is not None:
            try:
                return float(task_record[key])
            except (TypeError, ValueError):
                continue
    return None


def _normalize_id(task_id: str) -> str:
    """Strip optional `skillsbench/` prefix for lookup-friendly matching."""
    return task_id.split("/", 1)[-1] if "/" in task_id else task_id


def _build_score_index(per_task: List[Dict[str, Any]]) -> Dict[str, float]:
    index: Dict[str, float] = {}
    for rec in per_task:
        tid = rec.get("task_id") or rec.get("id")
        if not tid:
            continue
        score = _extract_with_skills_score(rec)
        if score is None:
            continue
        index[tid] = score
        index.setdefault(_normalize_id(tid), score)
    return index


def _bin(score: float) -> str:
    if score < 0.2:
        return "failing"
    if score < 0.8:
        return "partial"
    return "passing"


def select_hot_12(
    summary: Dict[str, Any],
    subset_20: List[str],
    target: int = 12,
    per_bin: int = 4,
) -> List[str]:
    """Return ``target`` task IDs (default 12), 4-per-bin with overflow.

    ``subset_20`` is a JSON list of fully-qualified task IDs. The
    parameter name is kept for backwards compatibility, but the function
    accepts any size of input list (subset_17, subset_20, etc.).
    """
    per_task = summary.get("per_task") or []
    score_index = _build_score_index(per_task)

    # Collect (task_id, score) pairs for tasks present in BOTH summary and the input list.
    scored: List[Tuple[str, float]] = []
    for tid in subset_20:
        if tid in score_index:
            scored.append((tid, score_index[tid]))
        elif _normalize_id(tid) in score_index:
            scored.append((tid, score_index[_normalize_id(tid)]))
    if not scored:
        raise RuntimeError(
            "no overlap between baseline summary and the provided task list"
        )

    bins: Dict[str, List[str]] = {"failing": [], "partial": [], "passing": []}
    for tid, score in scored:
        bins[_bin(score)].append(tid)
    for k in bins:
        bins[k].sort()

    selected: List[str] = []
    # First pass: take up to per_bin from each bin (alphabetical within bin).
    primary_order = ["failing", "partial", "passing"]
    for b in primary_order:
        take = bins[b][:per_bin]
        selected.extend(take)
        bins[b] = bins[b][per_bin:]

    # Overflow: pull from leftover bins in priority order partial > failing > passing.
    overflow_order = ["partial", "failing", "passing"]
    for b in overflow_order:
        if len(selected) >= target:
            break
        while bins[b] and len(selected) < target:
            selected.append(bins[b].pop(0))

    # Trim if we somehow over-selected.
    selected = selected[:target]

    if len(selected) < target:
        raise RuntimeError(
            f"only {len(selected)} tasks available across bins; need {target}"
        )
    return selected


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Pick a 12-task hot subset for inner evolution.",
    )
    ap.add_argument("summary_json", type=Path, help="Path to baseline summary.json")
    ap.add_argument(
        "subset_json",
        type=Path,
        help="Path to a JSON array of task IDs to subset over "
        "(e.g. subset_20.json or subset_17.json).",
    )
    ap.add_argument("-o", "--out", type=Path, required=True, help="Output JSON path")
    ap.add_argument(
        "--target",
        type=int,
        default=12,
        help="Total subset size (default: %(default)s)",
    )
    ap.add_argument(
        "--per-bin",
        type=int,
        default=4,
        help="Tasks per bin before overflow (default: %(default)s)",
    )
    args = ap.parse_args(argv)

    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    subset_20 = json.loads(args.subset_json.read_text(encoding="utf-8"))

    try:
        hot = select_hot_12(
            summary,
            subset_20,
            target=args.target,
            per_bin=args.per_bin,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(hot, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(hot)} task IDs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
