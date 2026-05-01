#!/usr/bin/env python
"""Append SkillsBench task entries to ``skill_evolve/benchmark/manifest.json``.

Walks ``<vendor>/tasks/*/`` and emits one manifest entry per directory
that doesn't already exist in the manifest. Idempotent — re-running
over the same vendor tree is a no-op.

Usage:
    python scripts/expand_skillsbench_manifest.py \\
        [--vendor-dir skill_evolve/benchmark/vendor/skillsbench] \\
        [--manifest skill_evolve/benchmark/manifest.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_VENDOR = _REPO_ROOT / "skill_evolve" / "benchmark" / "vendor" / "skillsbench"
_DEFAULT_MANIFEST = _REPO_ROOT / "skill_evolve" / "benchmark" / "manifest.json"


def _read_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _extract_timeout_sec(toml_data: Dict[str, Any]) -> int:
    agent = toml_data.get("agent") or {}
    if "timeout_sec" in agent:
        return int(agent["timeout_sec"])
    verifier = toml_data.get("verifier") or {}
    if "timeout_sec" in verifier:
        return int(verifier["timeout_sec"])
    if "timeout_sec" in toml_data:
        return int(toml_data["timeout_sec"])
    return 600


def _extract_domain(toml_data: Dict[str, Any]) -> str:
    """Same fallback chain as skillsbench_loader._extract_domain."""
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
    return "skillsbench"


def _extract_difficulty(toml_data: Dict[str, Any]) -> str:
    metadata = toml_data.get("metadata") or {}
    if metadata.get("difficulty"):
        return str(metadata["difficulty"])
    if toml_data.get("difficulty"):
        return str(toml_data["difficulty"])
    return "unknown"


def _build_entry(task_dir: Path) -> Dict[str, Any]:
    name = task_dir.name
    toml_data = _read_toml(task_dir / "task.toml")
    domain = _extract_domain(toml_data)
    difficulty = _extract_difficulty(toml_data)
    timeout = _extract_timeout_sec(toml_data)
    return {
        "task_id": f"skillsbench/{name}",
        "source": "skillsbench",
        "dataset_task_name": name,
        "category": domain,
        "difficulty": difficulty,
        "skill_relevance": 1.0,
        "success_check_kind": "skillsbench_test_sh",
        "timeout_s": int(timeout),
        "stage": 1,
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Expand SkillsBench tasks into manifest entries.",
    )
    ap.add_argument(
        "--vendor-dir",
        type=Path,
        default=_DEFAULT_VENDOR,
        help="Path to vendored skillsbench (default: %(default)s)",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=_DEFAULT_MANIFEST,
        help="Path to manifest.json (default: %(default)s)",
    )
    args = ap.parse_args(argv)

    tasks_root: Path = args.vendor_dir / "tasks"
    if not tasks_root.exists():
        print(f"ERROR: vendor tasks dir not found: {tasks_root}", file=sys.stderr)
        return 2
    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    existing_ids: set[str] = {e["task_id"] for e in manifest.get("tasks", [])}

    added = 0
    skipped = 0
    new_entries: List[Dict[str, Any]] = []
    for task_dir in sorted(tasks_root.iterdir()):
        if not task_dir.is_dir():
            continue
        if not (task_dir / "task.toml").exists():
            continue
        task_id = f"skillsbench/{task_dir.name}"
        if task_id in existing_ids:
            skipped += 1
            continue
        try:
            entry = _build_entry(task_dir)
        except Exception as exc:  # noqa: BLE001
            print(
                f"WARN: failed to build entry for {task_dir.name}: {exc}",
                file=sys.stderr,
            )
            continue
        new_entries.append(entry)
        added += 1

    manifest.setdefault("tasks", []).extend(new_entries)
    args.manifest.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"added {added} skillsbench entries; skipped {skipped} (already present)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
