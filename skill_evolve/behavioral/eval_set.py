"""Loader for daycare-format ``eval_set.jsonl`` files.

Each row is a JSON object with (at minimum) the keys::

    id      : str  — stable eval id
    prompt  : str  — user prompt fed to the candidate
    rubric  : str  — scoring rubric handed to the judge LLM

Optional keys handled today::

    expected_action : str | None — non-binding hint used by some adapters
    anonymized_prompt / anonymized_rubric, type, accepted, source_*  — copied
        verbatim into ``EvalItem.metadata`` so downstream consumers
        (e.g. score aggregation by ``type``) can read them.

Blank lines in the .jsonl are skipped. Missing required keys raise
``EvalSetFormatError`` with the offending line number so a malformed
fixture is easy to debug.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


class EvalSetFormatError(ValueError):
    """Raised when an ``eval_set.jsonl`` row is missing required keys.

    Carries ``line_no`` (1-indexed) and ``missing`` so the caller can
    surface a useful diagnostic without re-parsing.
    """

    def __init__(self, line_no: int, missing: List[str], *args: Any) -> None:
        msg = f"eval_set row {line_no} is missing required key(s): {', '.join(missing)}"
        super().__init__(msg, *args)
        self.line_no = line_no
        self.missing = list(missing)


@dataclass
class EvalItem:
    """One scoring row consumed by ``behavioral.adapter.score_bundle_behavioral``."""

    id: str
    prompt: str
    rubric: str
    expected_action: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default=None)


_REQUIRED_KEYS: tuple[str, ...] = ("id", "prompt", "rubric")


def load_eval_set(path: Path) -> List[EvalItem]:
    """Read a daycare-style ``eval_set.jsonl`` and return ``EvalItem`` rows.

    Blank lines are skipped. Malformed JSON raises ``EvalSetFormatError``
    pointing at the line. Missing required keys (``id``, ``prompt``,
    ``rubric``) raise ``EvalSetFormatError`` with the missing key list.
    Any extra keys are preserved in ``EvalItem.metadata`` verbatim so
    downstream aggregation by ``type`` / ``accepted`` / ``baseline_score``
    keeps working.
    """
    path = Path(path)
    items: List[EvalItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise EvalSetFormatError(
                    line_no, ["<json>"], f"json decode error: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise EvalSetFormatError(line_no, ["<object>"], "row is not an object")
            missing = [k for k in _REQUIRED_KEYS if not isinstance(row.get(k), str)]
            if missing:
                raise EvalSetFormatError(line_no, missing)
            expected_action = row.get("expected_action")
            if expected_action is not None and not isinstance(expected_action, str):
                expected_action = None
            metadata = {
                k: v
                for k, v in row.items()
                if k not in {"id", "prompt", "rubric", "expected_action"}
            }
            items.append(
                EvalItem(
                    id=row["id"],
                    prompt=row["prompt"],
                    rubric=row["rubric"],
                    expected_action=expected_action,
                    metadata=metadata or None,
                )
            )
    return items
