"""Bounded edit-budget L_t scheduler (Group F, SkillOpt port).

Paper anchor:
    * ``skillopt/optimizer/scheduler.py:CosineScheduler._compute_lr``
    * ``skillopt/optimizer/clip.py:rank_and_select``

Mirrors the paper's per-iteration cap on the number of file-ops applied
to a bundle. ``compute_lt`` selects the cap based on a schedule spec
(``constant:N`` / ``linear:N->M`` / ``cosine:N->M``); ``clip_ops`` truncates
the proposer's parsed op list to that cap using the parser's own emit
order (== proposer's priority order). The ranking-LLM "rank_and_select"
fallback path described in the paper is not implemented — clipping is
deterministic.

Public API:
    ScheduleKind        : Literal["constant","linear","cosine"]
    ScheduleSpec        : frozen dataclass (kind, start, end)
    parse_schedule(s)   : str -> ScheduleSpec
    compute_lt(s, n, M) : (str | ScheduleSpec, int, int) -> int (>= 1)
    clip_ops(ops, lt)   : (list[FileOp], int) -> list[FileOp]
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal, Union

from skill_evolve.shared.patch_parser import FileOp


ScheduleKind = Literal["constant", "linear", "cosine"]


@dataclass(frozen=True)
class ScheduleSpec:
    """Parsed edit-budget schedule.

    For ``constant:N`` both ``start`` and ``end`` are ``N``. For
    ``linear:N->M`` / ``cosine:N->M`` they are the endpoints.
    """

    kind: ScheduleKind
    start: int
    end: int


_CONSTANT_RE = re.compile(r"^constant:(-?\d+)$")
_RANGE_RE = re.compile(r"^(linear|cosine):(-?\d+)->(-?\d+)$")


def parse_schedule(spec: str) -> ScheduleSpec:
    """Parse a schedule spec string into a :class:`ScheduleSpec`.

    Forms:
        ``constant:N``       -> ScheduleSpec("constant", N, N)
        ``linear:N->M``      -> ScheduleSpec("linear", N, M)
        ``cosine:N->M``      -> ScheduleSpec("cosine", N, M)

    Raises ``ValueError`` on any malformed input.
    """
    if not isinstance(spec, str) or not spec:
        raise ValueError(f"malformed schedule spec: {spec!r}")

    m = _CONSTANT_RE.match(spec)
    if m is not None:
        n = int(m.group(1))
        return ScheduleSpec(kind="constant", start=n, end=n)

    m = _RANGE_RE.match(spec)
    if m is not None:
        kind = m.group(1)
        start = int(m.group(2))
        end = int(m.group(3))
        # mypy-friendly literal cast via if/elif rather than cast()
        if kind == "linear":
            return ScheduleSpec(kind="linear", start=start, end=end)
        return ScheduleSpec(kind="cosine", start=start, end=end)

    raise ValueError(f"malformed schedule spec: {spec!r}")


def _coerce_spec(spec: Union[str, ScheduleSpec]) -> ScheduleSpec:
    if isinstance(spec, ScheduleSpec):
        return spec
    return parse_schedule(spec)


def compute_lt(
    spec: Union[str, ScheduleSpec],
    iter_n: int,
    max_iters: int,
) -> int:
    """Return the per-iteration edit-budget L_t for the given schedule.

    Semantics:
        * ``constant:N``    -> always ``N``
        * ``linear:N->M``   -> ``round(N + (M - N) * t)``, ``t = iter_n / max(1, max_iters)``
        * ``cosine:N->M``   -> ``round(M + 0.5 * (N - M) * (1 + cos(pi * t)))``
        * Degenerate ``N == M`` collapses to constant behavior.
        * Result is clamped to ``>= 1`` (the apply path cannot use L_t == 0).

    Note: Python's ``round`` uses banker's rounding (half-to-even). For
    the canonical ``cosine:8->2`` schedule used by ``--canonical`` this
    is irrelevant (midpoint lands at 5.0 exactly), but operators passing
    custom specs with half-integer midpoints should be aware.
    """
    s = _coerce_spec(spec)

    if s.kind == "constant" or s.start == s.end:
        return max(1, s.start)

    denom = max(1, max_iters)
    t = iter_n / denom

    if s.kind == "linear":
        raw = s.start + (s.end - s.start) * t
    else:  # cosine
        raw = s.end + 0.5 * (s.start - s.end) * (1.0 + math.cos(math.pi * t))

    return max(1, int(round(raw)))


# Priority order for the ranking-heuristic fallback used ONLY when the
# proposer's emit order leaves ties to resolve. clip_ops keeps the
# proposer's emit order in the dominant code path (paper's fallback
# guarantee); this ordering is reserved for tie-breaking among ops the
# proposer emitted at the same priority.
_OP_PRIORITY: dict[str, int] = {
    "ADD_FILE": 0,
    "EDIT_FILE": 1,
    "DELETE_FILE": 2,
    "REWRITE_FOLDER": 3,
}


def clip_ops(ops: list[FileOp], lt: int) -> list[FileOp]:
    """Truncate ``ops`` to at most ``lt`` entries, preserving emit order.

    Length-stable when ``len(ops) <= lt``: returns the input list reference
    unchanged so callers can fast-path on identity.

    Over budget: returns ``ops[:lt]`` — the proposer's parser order doubles
    as priority order per the paper's fallback contract. The
    ``ADD > EDIT > DELETE > REWRITE`` ranking is reserved as a tie-breaker
    heuristic for future ranking-LLM integrations; the deterministic path
    here never re-sorts.
    """
    if lt < 0:
        raise ValueError(f"lt must be >= 0, got {lt}")
    if len(ops) <= lt:
        return ops
    return ops[:lt]


__all__ = [
    "ScheduleKind",
    "ScheduleSpec",
    "parse_schedule",
    "compute_lt",
    "clip_ops",
]
