#!/usr/bin/env python
"""D-14 diverse-domain sampler.

Walks ``<vendor>/tasks/``, buckets tasks by domain, and round-robins
across domain buckets in alpha order until 20 tasks are collected. The
output is a JSON array of fully-qualified task IDs (e.g.
``"skillsbench/forensics-disk-recovery"``).

Determinism: domains and tasks are alpha-sorted before selection; the
algorithm is index-deterministic with no randomness, no time-dependent
ordering. Re-running on the same vendor commit emits the identical list.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_VENDOR = _REPO_ROOT / "skill_evolve" / "benchmark" / "vendor" / "skillsbench"
_DEFAULT_OUT = _REPO_ROOT / "skill_evolve" / "skillsbench" / "subset_20.json"


def _read_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def _extract_domain(toml_data: dict) -> str:
    """Domain fallback chain: domain → tags[0] → category → "misc"."""
    if toml_data.get("domain"):
        return str(toml_data["domain"])
    metadata = toml_data.get("metadata") or {}
    if metadata.get("domain"):
        return str(metadata["domain"])
    tags = metadata.get("tags") if "tags" in metadata else toml_data.get("tags")
    if isinstance(tags, list) and tags:
        return str(tags[0])
    if metadata.get("category"):
        return str(metadata["category"])
    if toml_data.get("category"):
        return str(toml_data["category"])
    return "misc"


def select_subset(vendor_dir: Path, target_count: int = 20) -> List[str]:
    """Return ``target_count`` fully-qualified task IDs, diverse by domain.

    Round-robins across alpha-sorted domain buckets; within each bucket,
    tasks are alpha-sorted. Skips empty buckets on subsequent passes.

    Raises:
        RuntimeError: if total available tasks < ``target_count``.
    """
    tasks_root = vendor_dir / "tasks"
    if not tasks_root.exists():
        raise RuntimeError(f"vendor tasks dir not found: {tasks_root}")

    buckets: Dict[str, List[str]] = {}
    total = 0
    for task_dir in sorted(tasks_root.iterdir()):
        if not task_dir.is_dir():
            continue
        if not (task_dir / "task.toml").exists():
            continue
        try:
            toml_data = _read_toml(task_dir / "task.toml")
        except Exception:
            continue
        domain = _extract_domain(toml_data)
        buckets.setdefault(domain, []).append(task_dir.name)
        total += 1

    if total < target_count:
        raise RuntimeError(
            f"only {total} SkillsBench tasks available; need {target_count}"
        )

    # Alpha-sort within each bucket; alpha-sort the bucket order.
    domains_sorted = sorted(buckets.keys())
    for d in domains_sorted:
        buckets[d].sort()

    selected: List[str] = []
    while len(selected) < target_count:
        progressed = False
        for d in domains_sorted:
            if len(selected) >= target_count:
                break
            bucket = buckets[d]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            progressed = True
        if not progressed:
            # No bucket had anything to give — should be unreachable
            # because total >= target_count guarded above.
            break

    return [f"skillsbench/{name}" for name in selected]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Pick a 20-task diverse-domain subset for SkillsBench.",
    )
    ap.add_argument(
        "--vendor-dir",
        type=Path,
        default=_DEFAULT_VENDOR,
        help="Path to vendored skillsbench (default: %(default)s)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help="Output JSON path (default: %(default)s)",
    )
    ap.add_argument(
        "--count", type=int, default=20, help="Subset size (default: %(default)s)"
    )
    args = ap.parse_args(argv)

    try:
        subset = select_subset(args.vendor_dir, target_count=args.count)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(subset, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(subset)} task IDs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
