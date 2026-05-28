"""Deterministic backend stub for the mock_skillsbench fixture.

``run_task(task_id, bundle_dir, ...)`` returns ``{"score": 1.0}`` for
``mock_task_a`` and ``mock_task_b`` provided the candidate bundle has at
least one ``SKILL.md`` file; ``mock_task_c`` always scores 0.0.

The shape mirrors the minimal interface evaluator code paths consume:
a callable returning a dict with a ``score`` key. Tests register this
stub via ``monkeypatch.setattr`` so the evaluator's existing
``agent_backend`` dispatch finds it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


PASSING_TASKS = frozenset({"mock_task_a", "mock_task_b"})
FAILING_TASKS = frozenset({"mock_task_c"})


def has_skill_md(bundle_dir: Path) -> bool:
    return any(p.name == "SKILL.md" for p in Path(bundle_dir).rglob("*"))


class StubBenchBackend:
    """Deterministic stub mimicking ``BenchCliBackend.run_task``."""

    name = "stub-mock-skillsbench"

    def run_task(
        self,
        task_id: str,
        bundle_dir: Path,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        if task_id in PASSING_TASKS and has_skill_md(bundle_dir):
            return {"task_id": task_id, "score": 1.0, "passed": True}
        # Anything not in PASSING_TASKS (incl. mock_task_c) → fail.
        return {"task_id": task_id, "score": 0.0, "passed": False}
