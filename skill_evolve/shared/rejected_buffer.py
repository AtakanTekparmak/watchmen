"""Bounded ring buffer of recently-rejected candidate patches.

Group E of plan_0.md — feeds the proposer system prompt with a
``## RECENT REJECTIONS`` block so the outer LLM can avoid re-proposing
edits that already failed the validation gate.

The buffer is FIFO-evicted (oldest first) once capacity is reached and
persists to a JSONL file (one ``RejectedEdit`` per line) after every
push so a crashed run still has the history. Plan section 7j locks the
schema; downstream Groups G + H consume the same renderer.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

RejectionReason = Literal[
    "train_no_improve",
    "val_not_strict_gt",
    "val_tie",
    "smoke_rejected",
    "token_cap",
    "parse_error",
]


@dataclass(frozen=True)
class RejectedEdit:
    """A single rejected candidate.

    Field shape is locked by plan_0.md section 7j. ``patch_text`` is the
    raw sentinel-block patch the proposer emitted; ``delta_train`` /
    ``delta_val`` are the per-axis score deltas relative to the parent
    (or 0.0 for pre-eval rejections like ``smoke_rejected`` /
    ``parse_error``).
    """

    patch_text: str
    delta_train: float
    delta_val: float
    rejection_reason: str
    iteration: int


class RejectedBuffer:
    """Bounded FIFO ring of :class:`RejectedEdit` entries.

    Single instance per run, owned by the controller and threaded into
    every ``run_iteration(...)`` call. Capacity defaults to 10
    (paper-flavored — small enough to fit in the proposer prompt budget,
    large enough to surface recurring mutation shapes).
    """

    def __init__(self, capacity: int = 10) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._capacity = capacity
        self._items: deque[RejectedEdit] = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._items)

    def push(self, edit: RejectedEdit) -> None:
        """Append ``edit``; evict the oldest entry when full."""
        self._items.append(edit)

    def recent(self, n: int) -> list[RejectedEdit]:
        """Return the most-recent ``n`` entries (newest last)."""
        if n <= 0:
            return []
        items = list(self._items)
        return items[-n:]

    def to_jsonl(self, path: Path) -> None:
        """Write the buffer to ``path`` as JSONL (one entry per line).

        Overwrites any existing file so the on-disk view always matches
        the live buffer's bounded ring (oldest entries already evicted).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for edit in self._items:
                fh.write(json.dumps(asdict(edit)) + "\n")

    @classmethod
    def from_jsonl(cls, path: Path, capacity: int = 10) -> "RejectedBuffer":
        """Rehydrate a buffer from a JSONL file produced by :meth:`to_jsonl`.

        Lines that fail to parse are skipped (warn-by-omission); the
        resulting buffer retains the last ``capacity`` valid records in
        file order.
        """
        buf = cls(capacity=capacity)
        path = Path(path)
        if not path.exists():
            return buf
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            try:
                buf.push(
                    RejectedEdit(
                        patch_text=str(obj["patch_text"]),
                        delta_train=float(obj["delta_train"]),
                        delta_val=float(obj["delta_val"]),
                        rejection_reason=str(obj["rejection_reason"]),
                        iteration=int(obj["iteration"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return buf

    def render_for_prompt(self) -> str:
        """Render the ``## RECENT REJECTIONS`` section body.

        Returns an empty string when the buffer is empty so the caller
        can decide whether to render the surrounding header / fallback
        text. The :func:`render_for_prompt` module helper (below) adds
        the locked ``(none yet)`` literal expected by plan section 7j.
        """
        if not self._items:
            return ""
        lines: list[str] = []
        for edit in self._items:
            head = edit.patch_text[:200]
            lines.append(
                f"- iter {edit.iteration} "
                f"reason={edit.rejection_reason} "
                f"Δtrain={edit.delta_train:+.4f} "
                f"Δval={edit.delta_val:+.4f}\n"
                f"  patch_head: {head!r}"
            )
        return "\n".join(lines)


def render_for_prompt(buffer: RejectedBuffer) -> str:
    """Markdown block for the ``## RECENT REJECTIONS`` prompt slot.

    Returns the empty string when ``buffer`` is empty. Callers that need
    the locked ``(none yet)`` fallback (per plan section 7j) substitute
    it themselves so this helper stays composable.
    """
    return buffer.render_for_prompt()
