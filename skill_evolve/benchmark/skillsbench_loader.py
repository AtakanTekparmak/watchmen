"""Hydrate a single SkillsBench task directory into a :class:`Task`.

Each SkillsBench task lives under
``vendor/skillsbench/tasks/<task_name>/`` and ships:

  * ``task.toml``          — author metadata + timeout + difficulty
  * ``instruction.md``     — the prompt body
  * ``environment/``       — Dockerfile + assets (incl. optional ``skills/``)
  * ``tests/test.sh``      — verification entrypoint
  * ``solution/``          — reference solution (used only by maintainers)

The loader's job is to fold those into the standard ``Task`` dataclass
already used by the tblite/swebench paths so the rest of the benchmark
plumbing (verifiers, evaluator, anonymizer) doesn't need to grow new
shapes per source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover - py<3.11 fallback
    import tomli as tomllib  # type: ignore

from .load import Task


def _read_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _extract_timeout_sec(toml_data: Dict[str, Any]) -> int:
    """Pick the most appropriate timeout from the task.toml.

    Preference order: ``[agent].timeout_sec`` (the agent's wall budget) →
    ``[verifier].timeout_sec`` (verification budget; usually similar) →
    top-level ``timeout_sec`` (legacy) → 600 (default).
    """
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
    """Domain fallback chain: domain → tags[0] → category → "misc"."""
    if "domain" in toml_data and toml_data["domain"]:
        return str(toml_data["domain"])
    metadata = toml_data.get("metadata") or {}
    if "domain" in metadata and metadata["domain"]:
        return str(metadata["domain"])
    tags = metadata.get("tags") if "tags" in metadata else toml_data.get("tags")
    if isinstance(tags, list) and tags:
        return str(tags[0])
    if "category" in metadata and metadata["category"]:
        return str(metadata["category"])
    if "category" in toml_data and toml_data["category"]:
        return str(toml_data["category"])
    return "misc"


def hydrate_one(task_dir: Path) -> Task:
    """Build a :class:`Task` from a SkillsBench task directory.

    Args:
        task_dir: absolute path to ``vendor/skillsbench/tasks/<name>/``.

    Returns:
        A standard :class:`skill_evolve.benchmark.load.Task` whose
        ``success_check_kind`` is ``"skillsbench_test_sh"`` and whose
        ``success_check_payload`` contains the on-disk paths the
        verifier needs (``task_dir``, ``environment_dir``, ``tests_dir``,
        ``skills_dir``, ``domain``, ``timeout_sec``).
    """
    task_dir = Path(task_dir)
    toml_path = task_dir / "task.toml"
    instruction_path = task_dir / "instruction.md"

    toml_data = _read_toml(toml_path)
    prompt = (
        instruction_path.read_text(encoding="utf-8")
        if instruction_path.exists()
        else ""
    )

    timeout_sec = _extract_timeout_sec(toml_data)
    domain = _extract_domain(toml_data)

    environment_dir = task_dir / "environment"
    tests_dir = task_dir / "tests"
    skills_dir = environment_dir / "skills"

    success_check_payload: Dict[str, Any] = {
        "task_dir": str(task_dir),
        "environment_dir": str(environment_dir),
        "tests_dir": str(tests_dir),
        "skills_dir": str(skills_dir) if skills_dir.exists() else None,
        "domain": domain,
        "timeout_sec": timeout_sec,
    }

    return Task(
        task_id=f"skillsbench/{task_dir.name}",
        source="skillsbench",
        prompt=prompt,
        success_check_kind="skillsbench_test_sh",
        success_check_payload=success_check_payload,
        timeout_s=int(timeout_sec or 600),
        stage=1,
        skill_relevance="",
        extra={"dataset_task_name": task_dir.name},
    )
