"""Hydrate the benchmark manifest into runnable task dicts.

Each manifest entry is a thin pointer (task_id + dataset coordinates). At
load time we pull the upstream prompt + verification payload from
HuggingFace and return a list of plain dicts so the evaluator stays
serialization-friendly.

Tasks are sorted by ``stage`` (ascending) so a cascade can run cheap
ones first; ``stage`` is a hint, not a hard ordering.

Note on success checks:

  * tblite_test_sh — success iff the task's bundled ``test.sh`` exits 0
    inside the task's docker image after the agent is done. We pass the
    raw ``test_sh`` script + ``tests_tar`` payload through to whoever
    actually runs the tests (out of scope for this loader).
  * swebench_patch_tests — success iff the agent's diff applies cleanly
    against ``base_commit`` AND the upstream ``FAIL_TO_PASS`` tests now
    pass while ``PASS_TO_PASS`` tests still pass. We pass the metadata
    through unchanged; downstream harnesses (or the official SWE-bench
    runner) own the actual verification.
"""

from __future__ import annotations

import functools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"

# Datasets are loaded lazily — load_subset() can be called with
# ``offline_only=True`` to skip the HF round-trip (handy for tests).


@dataclass
class Task:
    """Hydrated task.

    The ``success_check_payload`` shape depends on ``success_check_kind``:

      tblite_test_sh:
        - docker_image: str
        - test_sh: str         (literal contents of test.sh)
        - tests_tar: str       (base64 tarball; optional)
        - environment_tar: str (base64 tarball; optional)
        - test_timeout_sec: float

      swebench_patch_tests:
        - repo: str
        - base_commit: str
        - environment_setup_commit: str
        - FAIL_TO_PASS: list[str]
        - PASS_TO_PASS: list[str]
        - reference_patch: str   (the gold patch; for diff-based scoring)
        - test_patch: str
    """

    task_id: str
    source: str
    prompt: str
    success_check_kind: str
    success_check_payload: Dict[str, Any]
    timeout_s: int
    stage: int = 1
    skill_relevance: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "source": self.source,
            "prompt": self.prompt,
            "success_check_kind": self.success_check_kind,
            "success_check_payload": self.success_check_payload,
            "timeout_s": self.timeout_s,
            "stage": self.stage,
            "skill_relevance": self.skill_relevance,
            "extra": self.extra,
        }


def _read_manifest(path: Path = _MANIFEST_PATH) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@functools.lru_cache(maxsize=2)
def _load_tblite_index() -> Dict[str, Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("NousResearch/openthoughts-tblite", split="train")
    return {row["task_name"]: row for row in ds}


@functools.lru_cache(maxsize=2)
def _load_swebench_index() -> Dict[str, Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    return {row["instance_id"]: row for row in ds}


def _hydrate_tblite(entry: Dict[str, Any]) -> Task:
    name = entry["dataset_task_name"]
    row = _load_tblite_index().get(name)
    if row is None:
        raise KeyError(f"TBLite task not found in dataset: {name}")
    payload = {
        "docker_image": row.get("docker_image"),
        "test_sh": row.get("test_sh"),
        "tests_tar": row.get("tests_tar"),
        "environment_tar": row.get("environment_tar"),
        "test_timeout_sec": float(row.get("test_timeout_sec") or 180),
        "category": row.get("category"),
        "difficulty": row.get("difficulty"),
    }
    return Task(
        task_id=entry["task_id"],
        source="tblite",
        prompt=row["instruction"],
        success_check_kind="tblite_test_sh",
        success_check_payload=payload,
        timeout_s=int(entry.get("timeout_s") or row.get("agent_timeout_sec") or 900),
        stage=int(entry.get("stage", 1)),
        skill_relevance=entry.get("skill_relevance", ""),
        extra={"dataset_task_name": name},
    )


def _format_swebench_prompt(row: Dict[str, Any]) -> str:
    """Wrap the SWE-bench problem statement so the agent has enough context.

    We deliberately do *not* leak the gold patch or the test_patch — the
    agent only sees the problem statement, the repo coordinates, and a
    note about how it will be evaluated.
    """
    return (
        f"You are working on the open-source repository `{row['repo']}` "
        f"at base commit `{row['base_commit']}`.\n\n"
        f"Problem to solve:\n{row['problem_statement']}\n\n"
        "Your task: produce a code patch (diff against the base commit) that "
        "fixes the issue. Do NOT modify any test files. Your patch will be "
        "applied on top of the base commit and the project's test suite will "
        "be run; the failing tests listed in the issue must pass and existing "
        "passing tests must keep passing."
    )


def _hydrate_swebench(entry: Dict[str, Any]) -> Task:
    iid = entry["dataset_instance_id"]
    row = _load_swebench_index().get(iid)
    if row is None:
        raise KeyError(f"SWE-bench instance not found: {iid}")

    fail_to_pass = row.get("FAIL_TO_PASS", "[]")
    pass_to_pass = row.get("PASS_TO_PASS", "[]")
    if isinstance(fail_to_pass, str):
        fail_to_pass = json.loads(fail_to_pass)
    if isinstance(pass_to_pass, str):
        pass_to_pass = json.loads(pass_to_pass)

    payload = {
        "repo": row["repo"],
        "base_commit": row["base_commit"],
        "environment_setup_commit": row.get("environment_setup_commit"),
        "FAIL_TO_PASS": fail_to_pass,
        "PASS_TO_PASS": pass_to_pass,
        "reference_patch": row.get("patch"),
        "test_patch": row.get("test_patch"),
        "version": row.get("version"),
    }
    return Task(
        task_id=entry["task_id"],
        source="swebench",
        prompt=_format_swebench_prompt(row),
        success_check_kind="swebench_patch_tests",
        success_check_payload=payload,
        timeout_s=int(entry.get("timeout_s", 1800)),
        stage=int(entry.get("stage", 3)),
        skill_relevance=entry.get("skill_relevance", ""),
        extra={"instance_id": iid},
    )


def _hydrate_skillsbench(entry: Dict[str, Any]) -> Task:
    """Hydrate a SkillsBench manifest entry into a :class:`Task`.

    Resolves ``task_dir``: prefers an explicit ``task_dir`` field on the
    entry; otherwise builds the canonical vendor path from
    ``dataset_task_name``.
    """
    from . import skillsbench_loader  # local import to avoid cycles

    if "task_dir" in entry and entry["task_dir"]:
        task_dir = Path(entry["task_dir"])
    else:
        name = entry.get("dataset_task_name") or entry["task_id"].rsplit("/", 1)[-1]
        task_dir = (
            Path(__file__).resolve().parent / "vendor" / "skillsbench" / "tasks" / name
        )
    task = skillsbench_loader.hydrate_one(task_dir)
    # Allow manifest entry to override timeout/stage/skill_relevance.
    if "timeout_s" in entry:
        task.timeout_s = int(entry["timeout_s"])
    if "stage" in entry:
        task.stage = int(entry["stage"])
    if "skill_relevance" in entry:
        task.skill_relevance = entry["skill_relevance"]
    # Preserve the manifest task_id verbatim (in case someone aliases it).
    task.task_id = entry.get("task_id", task.task_id)
    return task


def load_subset(
    *,
    offline_only: bool = False,
    sources: Optional[List[str]] = None,
    manifest_path: Optional[Path] = None,
    task_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Hydrate every entry in the manifest into a runnable task dict.

    Args:
        offline_only: if True, skip HF lookups; the returned tasks will
            have ``prompt = ""`` and an empty ``success_check_payload``.
            Useful for tests that just want to enumerate task IDs.
        sources: optional whitelist (e.g. ``["tblite"]``) for cheap dev
            iteration when SWE-bench env isn't worth standing up.
        manifest_path: override the default manifest location.
        task_ids: optional explicit allowlist of fully-qualified task
            IDs (``"skillsbench/foo"``) or bare segments (``"foo"``).
            When set, only matching manifest entries are hydrated. Used
            by the Phase E SkillsBench dispatch (run.py ``--task-list``)
            so evolution sees the curated hot-12 subset rather than the
            full 10-task tblite manifest.

    Returns:
        list[dict] sorted by ``stage`` then by ``task_id``.
    """
    manifest = _read_manifest(manifest_path or _MANIFEST_PATH)
    tasks: List[Task] = []

    # Build task_id allowlist (covers both fully-qualified and bare-segment
    # spellings so callers can pass either form). Empty set == no filter.
    id_allowlist: Optional[set[str]] = None
    if task_ids:
        id_allowlist = set()
        for tid in task_ids:
            id_allowlist.add(tid)
            if "/" in tid:
                id_allowlist.add(tid.rsplit("/", 1)[-1])

    for entry in manifest["tasks"]:
        if sources and entry["source"] not in sources:
            continue
        if id_allowlist is not None:
            entry_id = entry.get("task_id", "")
            entry_bare = entry_id.rsplit("/", 1)[-1] if "/" in entry_id else entry_id
            if entry_id not in id_allowlist and entry_bare not in id_allowlist:
                continue
        if offline_only:
            tasks.append(
                Task(
                    task_id=entry["task_id"],
                    source=entry["source"],
                    prompt="",
                    success_check_kind=entry["success_check_kind"],
                    success_check_payload={},
                    timeout_s=int(entry.get("timeout_s", 900)),
                    stage=int(entry.get("stage", 1)),
                    skill_relevance=entry.get("skill_relevance", ""),
                )
            )
            continue
        try:
            if entry["source"] == "tblite":
                tasks.append(_hydrate_tblite(entry))
            elif entry["source"] == "swebench":
                tasks.append(_hydrate_swebench(entry))
            elif entry["source"] == "skillsbench":
                tasks.append(_hydrate_skillsbench(entry))
            else:
                logger.warning(
                    "unknown source %s; skipping %s",
                    entry["source"],
                    entry.get("task_id"),
                )
        except Exception as exc:
            logger.warning("hydration failed for %s: %s", entry.get("task_id"), exc)

    tasks.sort(key=lambda t: (t.stage, t.task_id))
    return [t.to_dict() for t in tasks]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Print hydrated benchmark subset.")
    ap.add_argument(
        "--offline",
        action="store_true",
        help="skip HF lookups; print just IDs/structure",
    )
    ap.add_argument(
        "--sources", nargs="*", default=None, help="filter by source (tblite|swebench)"
    )
    args = ap.parse_args()

    tasks = load_subset(offline_only=args.offline, sources=args.sources)
    for t in tasks:
        prompt_preview = (t["prompt"] or "")[:80].replace("\n", " ")
        print(
            f"[stage={t['stage']}] {t['task_id']:48s} "
            f"timeout={t['timeout_s']}s :: {prompt_preview}"
        )
    print(f"\ntotal: {len(tasks)} tasks")
