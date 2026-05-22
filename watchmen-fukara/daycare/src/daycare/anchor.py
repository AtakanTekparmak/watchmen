"""Phase 2 — Anchor (iter_0) baseline scoring.

Snapshots the baseline bundle to ``iter_0/bundle/``, scores it against
the holdout slice via ``verifier.score_bundle()``, and writes the two
per-iter artefacts every subsequent iter must beat:

  - ``iter_0/eval_summary.json``  — the holdout aggregate
  - ``iter_0/sampling.json``      — frozen sampling config + the
    ``injection_format_hash`` SHA256 of build_skill_system_prompt's
    output (anchor for the M6-style injection-pin)

``load_best_bundle`` is the read-side helper used by Phase 4 and the
outer evolution loop: it scans every ``iter_*/eval_summary.json`` and
returns the highest-fitness bundle's path. Falls back to iter_0 if no
iter has recorded a fitness yet (cold start / aborted run).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from .runner import build_skill_system_prompt
from .verifier import EvalSummary, score_bundle


# ─── Helpers ──────────────────────────────────────────────────────────────


def _eval_summary_to_dict(summary: EvalSummary) -> dict:
    """Convert dataclass → dict, robust to nested dataclasses or extra fields."""
    try:
        return asdict(summary)
    except TypeError:
        # Defensive — if some nested field isn't a dataclass-friendly type.
        return {
            "holdout_score": float(getattr(summary, "holdout_score", 0.0)),
            "fitness": float(getattr(summary, "fitness", 0.0)),
            "tokens_skill_md": int(getattr(summary, "tokens_skill_md", 0)),
            "penalty": float(getattr(summary, "penalty", 0.0)),
            "lambda_n": float(getattr(summary, "lambda_n", 0.0)),
            "n_holdout": int(getattr(summary, "n_holdout", 0)),
            "by_type": dict(getattr(summary, "by_type", {}) or {}),
            "by_accepted": dict(getattr(summary, "by_accepted", {}) or {}),
            "by_length_quartile": dict(getattr(summary, "by_length_quartile", {}) or {}),
            "invocations_render": dict(getattr(summary, "invocations_render", {}) or {}),
            "unused_render": list(getattr(summary, "unused_render", []) or []),
            "smoke_failed": bool(getattr(summary, "smoke_failed", False)),
            "successful_evals": int(getattr(summary, "successful_evals", 0)),
        }


def _injection_format_hash(bundle_dir: Path) -> str:
    """SHA256 of the build_skill_system_prompt(bundle_dir) byte output.

    Used as the per-run injection-format pin: any deviation in subsequent
    iters means the substrate template drifted, which would invalidate
    cross-iter score comparisons.
    """
    blob = build_skill_system_prompt(bundle_dir).encode("utf-8", errors="replace")
    return hashlib.sha256(blob).hexdigest()


def _load_eval_rows(eval_set_path: Path) -> list[dict]:
    """Read the frozen eval_set.jsonl into a list of dicts."""
    rows: list[dict] = []
    if not eval_set_path.exists():
        return rows
    with eval_set_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# ─── Phase 2 — run_anchor ─────────────────────────────────────────────────


def run_anchor(
    run_dir: Path,
    bundle_dir: Path,
    eval_set_path: Path,
    model: str,
    judge_model: str,
    api_key: str,
    rollouts: int,
    seed: int,
    temperature: float,
    max_workers: int,
) -> EvalSummary:
    """Snapshot ``bundle_dir`` → ``run_dir/iter_0/bundle/``, score holdout,
    write eval_summary.json + sampling.json.

    Returns the EvalSummary so the caller can stash it on run.json or
    feed it into the evolve outer loop as ``parent_fitness``.
    """
    iter0_dir = run_dir / "iter_0"
    iter0_dir.mkdir(parents=True, exist_ok=True)

    target_bundle = iter0_dir / "bundle"
    # Snapshot (skip if the source bundle is empty / nonexistent — Baseline A
    # case where iter_0 IS the empty-bundle floor).
    if bundle_dir.exists() and any(bundle_dir.iterdir()):
        if target_bundle.exists():
            shutil.rmtree(target_bundle)
        shutil.copytree(bundle_dir, target_bundle)
    else:
        # Empty bundle: ensure the dir exists so downstream readers don't
        # blow up; build_skill_system_prompt returns "" for an empty dir.
        target_bundle.mkdir(parents=True, exist_ok=True)

    # Score the holdout split.
    eval_rows = _load_eval_rows(eval_set_path)
    summary = score_bundle(
        eval_rows=eval_rows,
        split="holdout",
        bundle_dir=target_bundle,
        model=model,
        judge_model=judge_model,
        api_key=api_key,
        rollouts=rollouts,
        max_workers=max_workers,
    )

    # Persist eval_summary.json.
    (iter0_dir / "eval_summary.json").write_text(
        json.dumps(_eval_summary_to_dict(summary), indent=2, default=str),
        encoding="utf-8",
    )

    # Persist sampling.json — model_version_pin is populated by the caller
    # (it owns the OR header capture from Phase 0 / M6). injection_format_hash
    # is computed from the snapshotted bundle's prompt.
    sampling = {
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "n_per_eval": rollouts,
        "model_version_pin": "unknown",
        "injection_format_hash": _injection_format_hash(target_bundle),
    }
    (iter0_dir / "sampling.json").write_text(json.dumps(sampling, indent=2), encoding="utf-8")

    return summary


# ─── load_best_bundle ─────────────────────────────────────────────────────


def load_best_bundle(run_dir: Path) -> tuple[Path, float]:
    """Scan ``run_dir/iter_*/eval_summary.json`` and return the bundle with the
    highest recorded fitness.

    Iteration semantics:
      - For iter_N (N ≥ 1), look at ``iter_N/bundle/`` (the promoted winner).
      - For iter_0, the bundle lives at ``iter_0/bundle/`` regardless.
      - If no iter has a recorded fitness, fall back to iter_0's bundle and
        whatever fitness it recorded (or 0.0).
    """
    best_path: Path | None = None
    best_fitness = float("-inf")

    if not run_dir.exists():
        # Caller will likely raise downstream; return a stable empty pair.
        return run_dir / "iter_0" / "bundle", 0.0

    for child in sorted(run_dir.iterdir()):
        if not child.is_dir() or not child.name.startswith("iter_"):
            continue
        summary_path = child / "eval_summary.json"
        if not summary_path.exists():
            continue
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        # Use fitness if present; fall back to holdout_score.
        fitness = data.get("fitness")
        if not isinstance(fitness, (int, float)):
            fitness = data.get("holdout_score")
        if not isinstance(fitness, (int, float)):
            continue
        bundle = child / "bundle"
        if not bundle.exists():
            continue
        if float(fitness) > best_fitness:
            best_fitness = float(fitness)
            best_path = bundle

    if best_path is None:
        # Fallback to iter_0/bundle (always written by run_anchor).
        return run_dir / "iter_0" / "bundle", 0.0

    return best_path, best_fitness
