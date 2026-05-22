"""Phase 4 — Baseline controls (A / B / C).

After evolution stops (budget exhaust or stall), three baselines run on
their own time (do NOT count against ``--budget``):

  - Baseline A (J1, HARD PROMOTION GATE) — weak model + empty bundle.
    If best.holdout_score ≤ baseline_a + 2ε, ``promote_blocked = True``
    is written to run.json and ``daycare promote`` refuses to copy.
  - Baseline B (M4 renamed) — weak model + 5-shot SKILL.md constructed
    mechanically from the train slice (anonymized). Soft signal: if best
    ≤ baseline_b + ε, naive distillation was enough; log, don't crash.
  - Baseline C (F1, v3) — TEACHER model + empty bundle. Establishes
    the distillation ceiling. The key metric:
        gap_closed = (best - a) / max(0.001, c - a)

All three write a per-baseline dir under ``run_dir/`` and the
``run_all_baselines`` aggregator dumps ``phase4_results.json``.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .anchor import _eval_summary_to_dict
from .anonymize import AnonymizeContext, strip
from .verifier import PartialScoringError, score_bundle


@dataclass
class BaselineResult:
    """One baseline's outcome.

    Fields:
        label: ``baseline_a`` | ``baseline_b`` | ``baseline_c``.
        holdout_score: aggregate over the holdout slice.
        n_holdout: number of holdout rows scored.
        gap_closed: only set on baseline_c — distillation-ceiling metric.
        promote_blocked: True iff baseline_a's J1 hard-block fires.
        output_dir: where this baseline's eval_summary.json lives.
    """

    label: str
    holdout_score: float
    n_holdout: int
    gap_closed: float | None
    promote_blocked: bool
    output_dir: Path


# ─── Baseline A — empty bundle (J1 hard gate) ──────────────────────────────


def run_baseline_a(
    run_dir: Path,
    holdout_evals: list[dict],
    best_bundle_dir: Path,
    model: str,
    judge_model: str,
    api_key: str,
    rollouts: int,
    epsilon: float,
) -> BaselineResult:
    """Score the weak model + empty bundle on the holdout slice.

    The bundle is a freshly-created empty temp dir (no SKILL.md, no
    scripts). The J1 hard-block fires iff
    ``best_fitness <= baseline_a + 2*epsilon``.

    Writes ``run_dir/baseline_a_empty/`` with eval_summary.json and a
    ``bundle/`` placeholder dir (kept for the directory layout's
    symmetry with the iter dirs).
    """
    out_dir = run_dir / "baseline_a_empty"
    out_dir.mkdir(parents=True, exist_ok=True)
    empty_bundle = out_dir / "bundle"
    if empty_bundle.exists():
        shutil.rmtree(empty_bundle)
    empty_bundle.mkdir(parents=True, exist_ok=True)

    try:
        summary = score_bundle(
            eval_rows=holdout_evals,
            split="holdout",
            bundle_dir=empty_bundle,
            model=model,
            judge_model=judge_model,
            api_key=api_key,
            rollouts=rollouts,
            max_workers=2,
        )
    except PartialScoringError as exc:
        print(f"[controls] baseline_a_partial: {exc}", file=sys.stderr)
        # Fall back to a degraded summary so the J1 check still has a number.
        summary = exc  # type: ignore[assignment]
        summary_score = 0.0
        n_holdout = exc.total
    else:
        summary_score = float(summary.holdout_score)
        n_holdout = int(summary.n_holdout)

    # Persist eval_summary.json (or a stub on partial-scoring).
    if isinstance(summary, PartialScoringError):
        (out_dir / "eval_summary.json").write_text(
            json.dumps(
                {
                    "holdout_score": 0.0,
                    "n_holdout": n_holdout,
                    "successful_evals": getattr(summary, "successful", 0),
                    "status": "partial_scoring",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        (out_dir / "eval_summary.json").write_text(
            json.dumps(_eval_summary_to_dict(summary), indent=2, default=str),
            encoding="utf-8",
        )

    # Read best bundle's fitness from disk so the J1 check is stable across
    # callers (some pass `best_bundle_dir` from iter_N, others from optimized/).
    best_fitness = _read_best_fitness(run_dir, best_bundle_dir)

    promote_blocked = best_fitness <= summary_score + 2.0 * epsilon

    return BaselineResult(
        label="baseline_a",
        holdout_score=summary_score,
        n_holdout=n_holdout,
        gap_closed=None,
        promote_blocked=promote_blocked,
        output_dir=out_dir,
    )


def _read_best_fitness(run_dir: Path, best_bundle_dir: Path) -> float:
    """Try to recover the best bundle's fitness from on-disk artefacts.

    Looks at ``best_bundle_dir.parent / eval_summary.json`` (per-iter
    layout) first, then ``run_dir / best_iter.json``, then 0.0 as the
    safety fallback.
    """
    candidates = [
        best_bundle_dir.parent / "eval_summary.json",
        run_dir / "best_iter.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        for key in ("fitness", "holdout_score", "best_fitness", "best_holdout_score"):
            v = data.get(key)
            if isinstance(v, (int, float)):
                return float(v)
    return 0.0


# ─── Baseline B — naive few-shot ───────────────────────────────────────────


def _build_fewshot_skill_md(train_evals: list[dict], ctx: AnonymizeContext, n: int = 5) -> str:
    """Construct a mechanical 5-shot SKILL.md from train evals.

    No procedural guidance — just (prompt, response) example pairs run
    through ``anonymize.strip`` (the spec mandates anonymization even
    for the baseline construction since the proposer never sees this
    bundle, but the principle of "no raw identifiers in skills" holds).
    """
    chosen = list(train_evals)[:n]
    parts: list[str] = ["# Naive few-shot baseline (Phase 4b)", ""]
    parts.append("_Mechanically constructed from train-slice examples. No procedural guidance — see Baseline B (M4)._")
    parts.append("")
    for i, e in enumerate(chosen, start=1):
        prompt = strip(e.get("prompt") or "", ctx)
        reference = strip(e.get("reference") or "", ctx)
        # Trim each to a sane length so the resulting SKILL.md stays
        # under the 2500-token cap even with 5 examples.
        parts.append(f"## Example {i}")
        parts.append("")
        parts.append("**Prompt:** " + prompt[:1500])
        parts.append("")
        parts.append("**Response:** " + reference[:1500])
        parts.append("")
    return "\n".join(parts)


def run_baseline_b(
    run_dir: Path,
    train_evals: list[dict],
    holdout_evals: list[dict],
    best_bundle_dir: Path,
    model: str,
    judge_model: str,
    api_key: str,
    rollouts: int,
    epsilon: float,
    ctx: AnonymizeContext,
) -> BaselineResult:
    """Score weak model + naive few-shot SKILL.md on holdout."""
    out_dir = run_dir / "baseline_b_fewshot"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / "bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True, exist_ok=True)

    skill_md = _build_fewshot_skill_md(train_evals, ctx, n=5)
    (bundle / "SKILL.md").write_text(skill_md, encoding="utf-8")

    try:
        summary = score_bundle(
            eval_rows=holdout_evals,
            split="holdout",
            bundle_dir=bundle,
            model=model,
            judge_model=judge_model,
            api_key=api_key,
            rollouts=rollouts,
            max_workers=2,
        )
        summary_score = float(summary.holdout_score)
        n_holdout = int(summary.n_holdout)
        (out_dir / "eval_summary.json").write_text(
            json.dumps(_eval_summary_to_dict(summary), indent=2, default=str),
            encoding="utf-8",
        )
    except PartialScoringError as exc:
        print(f"[controls] baseline_b_partial: {exc}", file=sys.stderr)
        summary_score = 0.0
        n_holdout = exc.total
        (out_dir / "eval_summary.json").write_text(
            json.dumps(
                {
                    "holdout_score": 0.0,
                    "n_holdout": n_holdout,
                    "successful_evals": exc.successful,
                    "status": "partial_scoring",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # Best fitness only used for log-level reporting here — Baseline B is
    # not a hard gate, just a "naive distillation suffices?" signal.
    _ = epsilon  # documented usage; not enforced as a gate.

    return BaselineResult(
        label="baseline_b",
        holdout_score=summary_score,
        n_holdout=n_holdout,
        gap_closed=None,
        promote_blocked=False,
        output_dir=out_dir,
    )


# ─── Baseline C — teacher ceiling ──────────────────────────────────────────


def run_baseline_c(
    run_dir: Path,
    holdout_evals: list[dict],
    teacher_model: str,
    judge_model: str,
    api_key: str,
    rollouts: int,
    best_holdout_score: float,
    baseline_a_score: float,
) -> BaselineResult:
    """Score TEACHER model + empty bundle on holdout. Compute gap_closed.

    gap_closed = (best - a) / max(0.001, c - a)
      - 0.0 → no distillation
      - 1.0 → full distillation
      - >1.0 → weak model EXCEEDS teacher on these evals (rare — usually
              means evals are too easy; the J1 floor should already block
              this case)
    """
    out_dir = run_dir / "baseline_c_teacher"
    out_dir.mkdir(parents=True, exist_ok=True)
    empty_bundle = out_dir / "bundle"
    if empty_bundle.exists():
        shutil.rmtree(empty_bundle)
    empty_bundle.mkdir(parents=True, exist_ok=True)

    try:
        summary = score_bundle(
            eval_rows=holdout_evals,
            split="holdout",
            bundle_dir=empty_bundle,
            model=teacher_model,
            judge_model=judge_model,
            api_key=api_key,
            rollouts=rollouts,
            max_workers=2,
        )
        teacher_score = float(summary.holdout_score)
        n_holdout = int(summary.n_holdout)
        (out_dir / "eval_summary.json").write_text(
            json.dumps(_eval_summary_to_dict(summary), indent=2, default=str),
            encoding="utf-8",
        )
    except PartialScoringError as exc:
        print(f"[controls] baseline_c_partial: {exc}", file=sys.stderr)
        teacher_score = 0.0
        n_holdout = exc.total
        (out_dir / "eval_summary.json").write_text(
            json.dumps(
                {
                    "holdout_score": 0.0,
                    "n_holdout": n_holdout,
                    "successful_evals": exc.successful,
                    "status": "partial_scoring",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    denom = max(0.001, teacher_score - baseline_a_score)
    gap_closed = (best_holdout_score - baseline_a_score) / denom

    return BaselineResult(
        label="baseline_c",
        holdout_score=teacher_score,
        n_holdout=n_holdout,
        gap_closed=gap_closed,
        promote_blocked=False,
        output_dir=out_dir,
    )


# ─── Orchestrator ──────────────────────────────────────────────────────────


def run_all_baselines(
    run_dir: Path,
    train_evals: list[dict],
    holdout_evals: list[dict],
    best_bundle_dir: Path,
    best_holdout_score: float,
    model: str,
    teacher_model: str,
    judge_model: str,
    api_key: str,
    rollouts: int,
    epsilon: float,
    ctx: AnonymizeContext,
) -> dict[str, BaselineResult]:
    """Run A → B → C in sequence; write ``run_dir/phase4_results.json``.

    Returns a dict ``{label: BaselineResult}``. The phase4_results.json
    aggregates every baseline's number plus the J1 verdict and
    gap_closed metric for downstream CLI display.
    """
    a = run_baseline_a(
        run_dir=run_dir,
        holdout_evals=holdout_evals,
        best_bundle_dir=best_bundle_dir,
        model=model,
        judge_model=judge_model,
        api_key=api_key,
        rollouts=rollouts,
        epsilon=epsilon,
    )

    b = run_baseline_b(
        run_dir=run_dir,
        train_evals=train_evals,
        holdout_evals=holdout_evals,
        best_bundle_dir=best_bundle_dir,
        model=model,
        judge_model=judge_model,
        api_key=api_key,
        rollouts=rollouts,
        epsilon=epsilon,
        ctx=ctx,
    )

    c = run_baseline_c(
        run_dir=run_dir,
        holdout_evals=holdout_evals,
        teacher_model=teacher_model,
        judge_model=judge_model,
        api_key=api_key,
        rollouts=rollouts,
        best_holdout_score=best_holdout_score,
        baseline_a_score=a.holdout_score,
    )

    results = {"baseline_a": a, "baseline_b": b, "baseline_c": c}

    payload = {
        "baseline_a": {
            "holdout_score": a.holdout_score,
            "n_holdout": a.n_holdout,
            "promote_blocked": a.promote_blocked,
            "output_dir": str(a.output_dir),
        },
        "baseline_b": {
            "holdout_score": b.holdout_score,
            "n_holdout": b.n_holdout,
            "output_dir": str(b.output_dir),
        },
        "baseline_c": {
            "holdout_score": c.holdout_score,
            "n_holdout": c.n_holdout,
            "gap_closed": c.gap_closed,
            "output_dir": str(c.output_dir),
        },
        "epsilon": epsilon,
        "best_holdout_score": best_holdout_score,
        "promote_blocked": a.promote_blocked,
        "gap_closed": c.gap_closed,
    }
    (run_dir / "phase4_results.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return results
