"""Phase 5 — Finalize: optimized bundle + scratchpad summary + promote.

Three responsibilities:

  - ``write_optimized``: copy the best bundle into ``run_dir/optimized/<slug>/``.
  - ``write_summary``: append a ``## Final summary`` block to scratchpad.md
    summarising per-iter scores, baseline A/B/C, gap_closed, J1 verdict,
    and a recommended action.
  - ``promote``: copy ``optimized/<slug>/`` into
    ``watchmen_home/bundles/<project>/skills/<slug>/`` and update the
    curation log / _pinned.json (flat array per W4) / _manifest.json.

J1 enforcement: ``promote`` refuses (raises RuntimeError) if
``phase4_results.json.promote_blocked`` is True. Callers must un-block
manually (by running more iters or overriding via the CLI flag).
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


# ─── write_optimized ──────────────────────────────────────────────────────


def write_optimized(run_dir: Path, best_bundle_dir: Path, slug: str) -> Path:
    """Copy ``best_bundle_dir`` → ``run_dir/optimized/<slug>/``.

    Returns the destination path. Overwrites any pre-existing optimized
    dir for this slug (re-runs of Phase 5 are idempotent).
    """
    out_root = run_dir / "optimized"
    out_root.mkdir(parents=True, exist_ok=True)

    dest = out_root / slug
    if dest.exists():
        shutil.rmtree(dest)

    if best_bundle_dir.exists():
        shutil.copytree(best_bundle_dir, dest)
    else:
        # Defensive — if the source bundle vanished (very rare; tests),
        # write an empty dir so callers can still reference the path.
        dest.mkdir(parents=True, exist_ok=True)

    return dest


# ─── write_summary ────────────────────────────────────────────────────────


def _read_metrics(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        return {"iters": []}
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"iters": []}
    if not isinstance(data, dict):
        return {"iters": []}
    return data


def _recommend_action(promote_blocked: bool, gap_closed: float | None) -> str:
    """Map the (J1, gap_closed) tuple to the spec §4c verdict table."""
    if promote_blocked:
        return "discard"  # Fail A — bug or judge gaming.
    if gap_closed is None:
        return "review"
    if gap_closed >= 0.3:
        return "adopt"
    if gap_closed >= 0.1:
        return "review"
    return "discard"


def write_summary(
    run_dir: Path,
    metrics_path: Path,
    baseline_results: dict,
) -> None:
    """Append ``## Final summary`` to ``run_dir/scratchpad.md``.

    The block contains:
      - per-iter table (iter, holdout_score, fitness, Δ, tokens)
      - baselines A/B/C scores
      - gap_closed metric
      - J1 promote_blocked verdict
      - recommended action (adopt / review / discard)
    """
    metrics = _read_metrics(metrics_path)
    iters: list[dict] = list(metrics.get("iters") or [])

    a = baseline_results.get("baseline_a")
    b = baseline_results.get("baseline_b")
    c = baseline_results.get("baseline_c")

    a_score = getattr(a, "holdout_score", 0.0) if a else 0.0
    b_score = getattr(b, "holdout_score", 0.0) if b else 0.0
    c_score = getattr(c, "holdout_score", 0.0) if c else 0.0
    gap_closed = getattr(c, "gap_closed", None) if c else None
    promote_blocked = bool(getattr(a, "promote_blocked", False)) if a else False

    lines: list[str] = []
    lines.append("")
    lines.append("## Final summary")
    lines.append("")
    lines.append(f"_Generated at {datetime.now(timezone.utc).isoformat()}_")
    lines.append("")

    # Per-iter table.
    lines.append("### Per-iter scores")
    lines.append("")
    lines.append("| Iter | Holdout | Fitness | Δ fitness | Tokens |")
    lines.append("|---:|---:|---:|---:|---:|")
    prev_fitness: float | None = None
    for it in iters:
        i = it.get("iter")
        h = float(it.get("holdout_score") or 0.0)
        f = float(it.get("fitness") or 0.0)
        delta = (f - prev_fitness) if prev_fitness is not None else 0.0
        tokens = int(it.get("tokens") or 0)
        lines.append(f"| {i} | {h:.4f} | {f:.4f} | {delta:+.4f} | {tokens} |")
        prev_fitness = f
    if not iters:
        lines.append("| - | - | - | - | - |")
    lines.append("")

    # Baselines.
    lines.append("### Baselines (Phase 4)")
    lines.append("")
    lines.append("| Baseline | Score | Notes |")
    lines.append("|---|---:|---|")
    lines.append(f"| A — empty bundle (floor, J1 gate) | {a_score:.4f} | hard promotion gate |")
    lines.append(f"| B — naive few-shot | {b_score:.4f} | soft signal |")
    lines.append(
        f"| C — teacher + empty (ceiling) | {c_score:.4f} | "
        f"gap_closed={'-' if gap_closed is None else f'{gap_closed:.4f}'} |"
    )
    lines.append("")

    # Verdict.
    action = _recommend_action(promote_blocked, gap_closed)
    lines.append("### Verdict")
    lines.append("")
    lines.append(f"- promote_blocked: **{promote_blocked}**")
    lines.append(f"- gap_closed: **{'-' if gap_closed is None else f'{gap_closed:.4f}'}**")
    lines.append(f"- recommended action: **{action}**")
    lines.append("")

    scratchpad = run_dir / "scratchpad.md"
    scratchpad.parent.mkdir(parents=True, exist_ok=True)
    with scratchpad.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ─── promote ──────────────────────────────────────────────────────────────


def _load_phase4(run_dir: Path) -> dict:
    p = run_dir / "phase4_results.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _append_curation_log(curation_log_path: Path, run_dir: Path, slug: str) -> None:
    """Append an evolution-provenance block to ``_curation_log.md``."""
    phase4 = _load_phase4(run_dir)
    a = phase4.get("baseline_a") or {}
    c = phase4.get("baseline_c") or {}
    gap = phase4.get("gap_closed")
    best = phase4.get("best_holdout_score")

    lines: list[str] = []
    lines.append("")
    lines.append(f"## daycare promote · {slug}")
    lines.append("")
    lines.append(f"- run: `{run_dir}`")
    lines.append(f"- promoted_at: {datetime.now(timezone.utc).isoformat()}")
    if isinstance(best, (int, float)):
        lines.append(f"- best_holdout_score: {best:.4f}")
    if isinstance(a.get("holdout_score"), (int, float)):
        lines.append(f"- baseline_a (floor): {a['holdout_score']:.4f}")
    if isinstance(c.get("holdout_score"), (int, float)):
        lines.append(f"- baseline_c (ceiling): {c['holdout_score']:.4f}")
    if isinstance(gap, (int, float)):
        lines.append(f"- gap_closed: {gap:.4f}")
    lines.append("")

    curation_log_path.parent.mkdir(parents=True, exist_ok=True)
    with curation_log_path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _update_pinned_json(pinned_path: Path, slug: str) -> None:
    """W4 — _pinned.json is a flat JSON array of slug strings.

    Create with ``[slug]`` if missing; otherwise append (preserving order)
    and dedupe.
    """
    pinned_path.parent.mkdir(parents=True, exist_ok=True)
    if pinned_path.exists():
        try:
            data = json.loads(pinned_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = []
    else:
        data = []
    if not isinstance(data, list):
        # Defensive — if a previous run wrote a dict, replace with a list.
        data = []
    if slug not in data:
        data.append(slug)
    pinned_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _update_manifest(manifest_path: Path, slug: str, skill_dir: Path) -> None:
    """Update _manifest.json mtimes for every file in the promoted skill dir.

    The schema is intentionally lenient — if the manifest file doesn't
    exist or has an unexpected shape we replace it with a flat dict of
    ``{relative_path: mtime}``.
    """
    if not skill_dir.exists():
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            manifest = {}
    else:
        manifest = {}
    if not isinstance(manifest, dict):
        manifest = {}

    bundle_root = skill_dir.parent.parent  # bundles/<project>/
    for p in skill_dir.rglob("*"):
        if not p.is_file():
            continue
        try:
            rel = str(p.relative_to(bundle_root))
            manifest[rel] = p.stat().st_mtime
        except (OSError, ValueError):
            continue

    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def promote(
    run_dir: Path,
    slug: str,
    project: str,
    watchmen_home: Path,
) -> None:
    """Copy the optimized bundle into the watchmen home and pin it.

    Steps:
      1. Read ``run_dir/phase4_results.json`` — refuse if
         ``promote_blocked`` is True (J1 hard gate).
      2. ``shutil.copytree`` from ``run_dir/optimized/<slug>/`` to
         ``watchmen_home/bundles/<project>/skills/<slug>/``.
      3. Append a provenance block to
         ``watchmen_home/bundles/<project>/_curation_log.md``.
      4. Update ``_pinned.json`` (flat array, W4).
      5. Update ``_manifest.json`` mtimes for the promoted files.
    """
    phase4 = _load_phase4(run_dir)
    if phase4.get("promote_blocked"):
        raise RuntimeError(
            f"promote blocked by Baseline A (J1): best_holdout_score "
            f"({phase4.get('best_holdout_score', '?')}) ≤ "
            f"baseline_a ({phase4.get('baseline_a', {}).get('holdout_score', '?')}) + 2ε. "
            f"Run more iters or use a richer skill before promoting."
        )

    src = run_dir / "optimized" / slug
    if not src.exists():
        raise RuntimeError(f"optimized bundle missing: {src}")

    bundle_root = watchmen_home / "bundles" / project
    dest = bundle_root / "skills" / slug
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, dirs_exist_ok=True)

    _append_curation_log(bundle_root / "_curation_log.md", run_dir, slug)
    _update_pinned_json(bundle_root / "_pinned.json", slug)
    _update_manifest(bundle_root / "_manifest.json", slug, dest)
