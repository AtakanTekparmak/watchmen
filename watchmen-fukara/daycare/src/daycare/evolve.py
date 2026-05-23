"""Phase 3 — Evolution loop (3a → 3b → 3b.5 → 3c → 3d → 3e).

Each iter performs the full cycle once. All load-bearing state is on disk
(weakness_report.md, mutation_log.md, history.jsonl, metrics.json, the
iter_N/bundle/ snapshot) so the proposer's context is rebuilt fresh every
iter — there is no in-memory carry-over between iters (3e context reset).

Sub-phases:

  3a  build_weakness_report   — score train slice, cluster failures, build
                                length-quartile + invocations/unused tables.
  3b  propose_candidates      — K=6 multi-turn watchmen.Agent invocations
                                with the 10-tool spec; sentinel-block patch
                                emitted via the terminal finish_candidate tool.
  3b.5 leak scan              — leak_scanner.scan + enforce_policy on each
                                surviving candidate (handled inside propose).
  3c  score_candidates        — smoke-3 short-circuit + full holdout scoring
                                with fitness penalty.
  3d  accept_or_discard       — promote winner if fitness > parent + epsilon,
                                else parent unchanged. Stall detector after 3
                                consecutive no-improvement iters.
  3e  context reset           — implicit; each iter calls build_weakness_report
                                fresh from disk.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from .anchor import _eval_summary_to_dict, load_best_bundle
from .leak_scanner import enforce_policy, scan as leak_scan
from .mutator import (
    hash_bundle,
    parse_sentinel_blocks,
    parse_and_apply,
    validate_scripts,
)
from .runner import build_skill_system_prompt, run_rollout_subprocess
from .verifier import (
    EvalSummary,
    PartialScoringError,
    score_bundle,
    score_single,
    smoke_3_check,
)

if TYPE_CHECKING:  # pragma: no cover
    from .watchdog import Watchdog


# ─── Constants ────────────────────────────────────────────────────────────


MAX_SKILL_TOKENS = 3000
TEMPERATURE_SCHEDULE = [0.3, 0.6, 0.9, 0.3, 0.6, 0.9]


# Cluster round-robin: c0,c3 → cluster_1; c1,c4 → cluster_2; c2,c5 → cluster_3.
def _target_cluster_for(slot: int) -> str:
    return f"cluster_{(slot % 3) + 1}"


# ─── Token counter (W5) ───────────────────────────────────────────────────


def _count_tokens(text: str) -> int:
    """tiktoken cl100k_base count, with a ~4-chars-per-token fallback."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except Exception:
        return max(0, len(text or "") // 4)


def _skill_md_tokens(bundle_dir: Path) -> int:
    """Token count of SKILL.md inside ``bundle_dir`` (0 if missing)."""
    p = bundle_dir / "SKILL.md"
    if not p.exists():
        return 0
    try:
        return _count_tokens(p.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return 0


# ─── history.jsonl + mutation_log.md helpers ──────────────────────────────


def _append_history(
    run_dir: Path,
    iter_n: int,
    candidate: int,
    outcome: str,
    fitness: float | None,
    reasoning_preview: str = "",
) -> None:
    """Append one row to history.jsonl (K4)."""
    row = {
        "iter": iter_n,
        "candidate": candidate,
        "outcome": outcome,
        "fitness": fitness,
        "ts": datetime.now(timezone.utc).isoformat(),
        "reasoning_preview": (reasoning_preview or "")[:80],
    }
    path = run_dir / "history.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def _load_eval_rows(eval_set_path: Path) -> list[dict]:
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


# ─── Phase 3a — weakness report ───────────────────────────────────────────


_SKILL_INVOCATION_RE = re.compile(
    r"(?:<skill>|<Skill\b|<tool_use\b[^>]*name=\"Skill\")",
    re.IGNORECASE,
)
_SKILL_SLUG_IN_BLOCK_RE = re.compile(r"<skill[^>]*>\s*([A-Za-z0-9_\-]+)", re.IGNORECASE)


def _count_invocations(
    eval_rows: list[dict],
    bundle_dir: Path,
    model: str,
    api_key: str,
    rollouts: int,
    seed: int,
    max_workers: int = 4,
) -> tuple[Counter, int]:
    """Run the bundle once over the train slice, return (Counter of slugs, total_rollouts).

    Parallelized with ThreadPoolExecutor to avoid the sequential bottleneck.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    skill_prompt = build_skill_system_prompt(bundle_dir)
    counts: Counter = Counter()
    total = 0

    def _one(idx_row: tuple[int, dict]):
        idx, row = idx_row
        prompt = row.get("anonymized_prompt") or row.get("prompt") or ""
        return run_rollout_subprocess(
            prompt=prompt,
            skill_system_prompt=skill_prompt,
            model=model,
            api_key=api_key,
            seed=seed + idx,
            temperature=0.7,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_one, (idx, row)): idx for idx, row in enumerate(eval_rows)}
        for fut in as_completed(futures):
            rollout = fut.result()
            total += 1
            if rollout.error is not None or not rollout.completion:
                continue
            completion = rollout.completion
            if not _SKILL_INVOCATION_RE.search(completion):
                continue
            # Pull out the slug if present; otherwise count as "<unknown>".
            m = _SKILL_SLUG_IN_BLOCK_RE.search(completion)
            slug = m.group(1) if m else "<unknown>"
            counts[slug] += 1
    return counts, total


def _cluster_failures(
    failing_rows: list[dict],
    judge_model: str,
    api_key: str,
) -> list[dict]:
    """Cluster failing evals into 3–6 modes via a single judge call.

    Returns a list of {name, severity, mode, examples} dicts. On parse
    failure, falls back to grouping by ``type`` field.
    """
    if not failing_rows:
        return []

    # Compose a compact payload — the judge gets paraphrase-able prompt
    # snippets + the eval type + the rubric so it can spot common modes.
    payload_rows = []
    for r in failing_rows[:30]:
        payload_rows.append(
            {
                "id": r.get("id", ""),
                "type": r.get("type") or "",
                "anonymized_prompt": (r.get("anonymized_prompt") or r.get("prompt") or "")[:400],
                "anonymized_rubric": (r.get("anonymized_rubric") or r.get("rubric") or "")[:200],
            }
        )

    system = (
        "Cluster the failing evals into 3-6 named failure modes. "
        "For each cluster: name (snake_case), severity (count of evals), "
        'primary failure mode in {"truncation","reasoning","format"}, '
        "and 2-3 paraphrased example prompts (≤120 chars each, ANONYMIZED).\n"
        'Output JSON: {"clusters": [{"name": "...", "severity": <int>, '
        '"mode": "truncation|reasoning|format", "examples": ["...", ...]}, ...]}'
    )
    user_payload = json.dumps({"failing_evals": payload_rows}, ensure_ascii=False)
    body = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_payload},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(timeout=90.0) as client:
            r = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=body,
                headers=headers,
            )
            r.raise_for_status()
            data = r.json()
        text = data["choices"][0]["message"]["content"] or ""
    except Exception as exc:  # noqa: BLE001
        print(f"[evolve] cluster_judge_error: {exc}", file=sys.stderr)
        return _fallback_cluster_by_type(failing_rows)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Single regex recovery pass.
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return _fallback_cluster_by_type(failing_rows)
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return _fallback_cluster_by_type(failing_rows)

    clusters = parsed.get("clusters") if isinstance(parsed, dict) else None
    if not isinstance(clusters, list) or not clusters:
        return _fallback_cluster_by_type(failing_rows)

    # Defensive normalisation.
    out: list[dict] = []
    for c in clusters[:6]:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name") or "unnamed")
        severity = int(c.get("severity") or 0)
        mode = str(c.get("mode") or "reasoning")
        if mode not in ("truncation", "reasoning", "format"):
            mode = "reasoning"
        examples = [str(x)[:120] for x in (c.get("examples") or [])][:3]
        out.append({"name": name, "severity": severity, "mode": mode, "examples": examples})
    return out or _fallback_cluster_by_type(failing_rows)


def _fallback_cluster_by_type(rows: list[dict]) -> list[dict]:
    """If the judge call failed, group by eval ``type`` so we still have
    SOMETHING for the proposer to target."""
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r.get("type") or "unknown", []).append(r)
    clusters: list[dict] = []
    for t, items in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        clusters.append(
            {
                "name": f"{t}_failures",
                "severity": len(items),
                "mode": "reasoning",
                "examples": [(r.get("anonymized_prompt") or r.get("prompt") or "")[:120] for r in items[:3]],
            }
        )
    return clusters[:6]


def _length_quartile_bounds(token_counts: list[int]) -> list[int]:
    """Compute [q1, q2, q3] inclusive upper bounds of the four quartiles."""
    if not token_counts:
        return [0, 0, 0]
    s = sorted(token_counts)
    n = len(s)

    def q(p: float) -> int:
        idx = max(0, min(n - 1, int(p * n)))
        return s[idx]

    return [q(0.25), q(0.50), q(0.75)]


def _quartile_of(value: int, bounds: list[float]) -> str:
    if not bounds or len(bounds) < 3:
        return "q1"
    if value <= bounds[0]:
        return "q1"
    if value <= bounds[1]:
        return "q2"
    if value <= bounds[2]:
        return "q3"
    return "q4"


def build_weakness_report(
    run_dir: Path,
    iter_n: int,
    train_evals: list[dict],
    best_bundle_dir: Path,
    model: str,
    judge_model: str,
    api_key: str,
    rollouts: int,
    baseline_len_quartiles: list[float],
    max_workers: int = 4,
) -> Path:
    """Phase 3a — produce ``run_dir/iter_{n}/weakness_report.md``.

    Layout (per spec §3a):
      - failure clusters (3-6, named, with 2-3 anonymized example prompts)
      - length-quartile table (Q1-Q4 mean score + dominant failure mode)
      - invocations_render (skill XML hit counts across train rollouts)
      - unused_render (skills bundled but never invoked)

    Implementation notes:
      - Train scoring uses verifier.score_bundle so the by_length_quartile
        block is populated for the table.
      - Bottom-quartile rows by per-eval score feed the clustering judge call.
      - invocations/unused come from a lightweight 1-rollout train pass that
        parses <skill>...</skill> blocks out of the model output.
    """
    iter_dir = run_dir / f"iter_{iter_n}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # 1. Train-slice scoring (real rollouts so by_length_quartile is meaningful).
    # We mark rows split="train" already; score_bundle filters by split.
    # Re-stamp split just in case (defensive — some callers may pass a flat slice).
    for r in train_evals:
        if r.get("split") != "train":
            r["split"] = "train"

    try:
        train_summary = score_bundle(
            eval_rows=train_evals,
            split="train",
            bundle_dir=best_bundle_dir,
            model=model,
            judge_model=judge_model,
            api_key=api_key,
            rollouts=rollouts,
            max_workers=2,
            length_quartile_bounds=[int(b) for b in baseline_len_quartiles[:3]] if baseline_len_quartiles else None,
        )
    except PartialScoringError as exc:
        print(f"[evolve] weakness_train_partial: {exc}", file=sys.stderr)
        # Fall through with an empty-ish summary; the clustering will fall
        # back to type-grouping.
        train_summary = EvalSummary(
            holdout_score=0.0,
            fitness=0.0,
            tokens_skill_md=_skill_md_tokens(best_bundle_dir),
            penalty=0.0,
            lambda_n=0.0,
            n_holdout=len(train_evals),
        )

    # 2. Per-eval scores (parallelized) — 1 rollout per train eval for bottom-quartile ranking.
    per_eval_scores: list[tuple[float, dict]] = []
    skill_prompt = build_skill_system_prompt(best_bundle_dir)

    def _score_one_train(idx_row: tuple[int, dict]) -> tuple[float, dict]:
        idx, row = idx_row
        prompt = row.get("anonymized_prompt") or row.get("prompt") or ""
        raw_prompt = row.get("prompt") or ""
        rubric = row.get("rubric") or ""
        rollout = run_rollout_subprocess(
            prompt=prompt,
            skill_system_prompt=skill_prompt,
            model=model,
            api_key=api_key,
            seed=10_000 + idx,
            temperature=0.7,
        )
        if rollout.error is not None or not rollout.completion:
            return (0.0, row)
        s = score_single(
            prompt=raw_prompt,
            rubric=rubric,
            completion=rollout.completion,
            judge_model=judge_model,
            api_key=api_key,
        )
        return (float(s) if s is not None else 0.0, row)

    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_score_one_train, (i, r)): i for i, r in enumerate(train_evals)}
        for fut in _as_completed(futs):
            per_eval_scores.append(fut.result())

    # Bottom-quartile rows by score.
    per_eval_scores.sort(key=lambda kv: kv[0])
    n_quartile = max(1, len(per_eval_scores) // 4)
    failing_rows = [r for _, r in per_eval_scores[:n_quartile]]

    # 3. Cluster failure modes via judge call.
    clusters = _cluster_failures(failing_rows, judge_model, api_key)

    # 4. Length-quartile table — we already have means from train_summary.by_length_quartile.
    # Map each cluster mode in by counting how many rows in each quartile match each
    # cluster's mode tag (we use cluster severity-weighted mode as fallback).
    lq_means = train_summary.by_length_quartile or {}
    quartile_counts: dict[str, int] = {"q1": 0, "q2": 0, "q3": 0, "q4": 0}
    quartile_mode: dict[str, str] = {"q1": "-", "q2": "-", "q3": "-", "q4": "-"}
    for row in train_evals:
        ql = _quartile_of(
            int(row.get("baseline_completion_len_tokens") or 0),
            list(baseline_len_quartiles),
        )
        quartile_counts[ql] = quartile_counts.get(ql, 0) + 1
    # Dominant failure mode per quartile: pick the cluster mode whose row
    # set has the most overlap with that quartile (cheap heuristic: assign
    # the cluster mode by row position in failing_rows).
    # Simpler approach: tag each failing row with the strongest cluster mode
    # (we use the first cluster as proxy) and count per-quartile.
    failing_quartile_modes: dict[str, Counter] = {q: Counter() for q in ("q1", "q2", "q3", "q4")}
    dominant_mode = clusters[0]["mode"] if clusters else "reasoning"
    for row in failing_rows:
        ql = _quartile_of(
            int(row.get("baseline_completion_len_tokens") or 0),
            list(baseline_len_quartiles),
        )
        failing_quartile_modes[ql][dominant_mode] += 1
    for q in quartile_mode:
        if failing_quartile_modes[q]:
            quartile_mode[q] = failing_quartile_modes[q].most_common(1)[0][0]

    # 5. Invocations / unused render — lightweight train pass.
    invocation_counts, total_train = _count_invocations(
        train_evals, best_bundle_dir, model, api_key, rollouts=1, seed=20_000, max_workers=max_workers
    )
    bundled_slugs: set[str] = set()
    # The bundle dir IS one skill (per spec the bundle is the SKILL.md+scripts
    # for a single slug). The slug is the bundle_dir.parent.name's child name —
    # we approximate by using the parent dir name OR the bundle dir name.
    candidate_slug = best_bundle_dir.name
    if candidate_slug == "bundle":
        # When called from iter_N/bundle/ the slug is unknown — leave empty.
        pass
    else:
        bundled_slugs.add(candidate_slug)
    unused = sorted(bundled_slugs - set(invocation_counts.keys()))

    # 6. Render markdown.
    lines: list[str] = []
    lines.append(f"# Weakness report — iter {iter_n}")
    lines.append("")
    lines.append(f"Train slice: {len(train_evals)} evals (holdout untouched).")
    lines.append(f"Bottom-quartile failing evals: {len(failing_rows)}")
    lines.append("")

    # Clusters.
    lines.append("## Failure clusters")
    lines.append("")
    if not clusters:
        lines.append("_No failure clusters extracted._")
    else:
        for c in clusters:
            lines.append(f"### {c['name']}  · severity={c['severity']}  · mode={c['mode']}")
            for ex in c.get("examples", []):
                lines.append(f"- `{ex}`")
            lines.append("")
    lines.append("")

    # Length-quartile table.
    lines.append("## Length-quartile breakdown")
    lines.append("")
    lines.append("| Quartile | n train evals | mean score | dominant failure mode |")
    lines.append("|---|---:|---:|---|")
    for q in ("q1", "q2", "q3", "q4"):
        mean = lq_means.get(q, 0.0)
        lines.append(f"| {q.upper()} | {quartile_counts.get(q, 0)} | {mean:.3f} | {quartile_mode.get(q, '-')} |")
    lines.append("")

    # Invocations render.
    lines.append("## invocations_render")
    lines.append("")
    if invocation_counts:
        for slug, n in invocation_counts.most_common():
            lines.append(f"- `{slug}`: invoked in {n}/{total_train} train rollouts at iter_{iter_n - 1}")
    else:
        lines.append("_No skill invocations detected across the train pass._")
    lines.append("")

    # Unused render.
    lines.append("## unused_render")
    lines.append("")
    if unused:
        for slug in unused:
            lines.append(f"- `{slug}`: bundled but never invoked")
    else:
        lines.append("_No unused skills (or bundle-slug introspection unavailable)._")
    lines.append("")

    out_path = iter_dir / "weakness_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ─── Phase 3b — propose_candidates ────────────────────────────────────────


_PROPOSER_SYSTEM_PROMPT = """You are a skill-bundle mutator. Your job is to emit ONE mutation
to the current best bundle, targeting the specified failure cluster.

IMPORTANT WORKFLOW — follow exactly:
1. Call read_weakness_report() — understand the failure clusters.
2. Call read_parent_bundle_file("SKILL.md") — read the current skill.
3. Write your mutation targeting the assigned failure cluster.
4. If your mutation includes ANY script file changes, call lint_script(path, content)
   on EACH modified script BEFORE calling finish_candidate. Fix any syntax errors.
5. Call finish_candidate() with the validated patch.

HARD LIMIT: By your 6th tool call, you MUST have already decided your mutation.
Your 7th tool call MUST be finish_candidate() with a non-empty patch_text.
If you have not emitted a patch by tool call 7, call finish_candidate() immediately
with whatever SKILL.md edit you have — even a small targeted addition is better than empty_patch.

You have at most 8 tool calls. An empty patch_text is REJECTED.
PREFER mutating SKILL.md only (add missing details, fix wrong values, expand guidance).
Only modify scripts if absolutely necessary — they are complex Python and syntax errors will discard your candidate.

The mutation must target the specific failure cluster you were assigned.
Add concrete details the model is missing: exact flag names, exact thresholds,
exact field names, exact command patterns — things only this skill can supply.

Constraints on emitted mutations (K8 — script discipline):
- No `pip install` calls — only Python stdlib + libraries already declared in
  `requirements.txt` of the bundle.
- Each script file ≤ 150 lines.
- All CLI arguments via `argparse`. No hardcoded paths.
- Every Python script: `python -m py_compile <file>` must pass.
- Every bash script: `bash -n <file>` must pass.
- SKILL.md must stay under MAX_SKILL_TOKENS = 2500 (tiktoken cl100k_base).
- Do not include literal session IDs, user names, absolute home paths, or any
  project identifier from the eval set. Generic guidance only.

Sentinel-block mutation format (the ONLY accepted format):

    <<<ADD_FILE path/relative/to/bundle>>>
    ... full file content ...
    <<<END_FILE>>>

    <<<EDIT_FILE path/relative/to/bundle>>>
    ... FULL rewritten body (NOT a diff) ...
    <<<END_FILE>>>

    <<<DELETE_FILE path/relative/to/bundle>>>

    <<<REWRITE_FOLDER scripts>>>
    --- file: scripts/foo.py
    ... content ...
    --- file: scripts/bar.sh
    ... content ...
    <<<END_REWRITE>>>

You may NOT read any held-out slice data. Use the available tools to read the
weakness report, parent bundle, prior mutation log, and peer skills.

Use `validate_sentinel_patch` to dry-run your patch before emitting it.
Use `lint_script` to validate any script body. Use `count_skill_tokens` to
stay under 2500 on SKILL.md.

When ready, call `finish_candidate(patch_text, target_cluster, reasoning)`.
Reasoning ≤ 200 words.
"""


def _make_proposer_tools(
    weakness_report_path: Path,
    parent_bundle_dir: Path,
    bundles_root: Path | None,
    mutation_log_path: Path | None,
    history_path: Path | None,
) -> tuple[list[dict], dict]:
    """Build the (tool_specs, tool_handlers) pair for one proposer Agent.

    The 10 tools per spec §"Proposer agent tool inventory":
      read_weakness_report, list_parent_bundle_files, read_parent_bundle_file,
      read_peer_skill, read_mutation_log, read_history_aggregates,
      validate_sentinel_patch, lint_script, count_skill_tokens, finish_candidate.

    All read tools are path-isolated — they cannot access ``held_out_log/``.
    """

    # ── Handlers ──────────────────────────────────────────────────────────

    def read_weakness_report() -> str:
        if not weakness_report_path.exists():
            return "(no weakness report on disk)"
        try:
            return weakness_report_path.read_text(encoding="utf-8")[:30000]
        except OSError as exc:
            return f"ERROR: {exc}"

    def list_parent_bundle_files() -> str:
        if not parent_bundle_dir.exists():
            return "(parent bundle missing)"
        files = [str(p.relative_to(parent_bundle_dir)) for p in sorted(parent_bundle_dir.rglob("*")) if p.is_file()]
        return "\n".join(files) if files else "(empty bundle)"

    def read_parent_bundle_file(path: str) -> str:
        # Resolve safely — refuse anything outside parent_bundle_dir.
        try:
            target = (parent_bundle_dir / path).resolve()
            base = parent_bundle_dir.resolve()
            if not str(target).startswith(str(base)):
                return "ERROR: path escapes parent bundle"
            if not target.exists() or not target.is_file():
                return f"ERROR: not a file: {path}"
            return target.read_text(encoding="utf-8", errors="replace")[:30000]
        except (OSError, ValueError) as exc:
            return f"ERROR: {exc}"

    def read_peer_skill(slug: str, path: str) -> str:
        if not bundles_root or not bundles_root.exists():
            return "ERROR: no bundles root configured"
        try:
            target = (bundles_root / slug / path).resolve()
            base = bundles_root.resolve()
            if not str(target).startswith(str(base)):
                return "ERROR: path escapes bundles root"
            if not target.exists() or not target.is_file():
                return f"ERROR: not a file: {slug}/{path}"
            return target.read_text(encoding="utf-8", errors="replace")[:30000]
        except (OSError, ValueError) as exc:
            return f"ERROR: {exc}"

    def read_mutation_log() -> str:
        if not mutation_log_path or not mutation_log_path.exists():
            return "(no prior mutation log)"
        try:
            return mutation_log_path.read_text(encoding="utf-8")[:20000]
        except OSError as exc:
            return f"ERROR: {exc}"

    def read_history_aggregates() -> str:
        if not history_path or not history_path.exists():
            return "(no history)"
        try:
            outcomes: Counter = Counter()
            with history_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    outcome = row.get("outcome")
                    if isinstance(outcome, str):
                        outcomes[outcome] += 1
            return json.dumps(outcomes, indent=2)
        except OSError as exc:
            return f"ERROR: {exc}"

    def validate_sentinel_patch(patch_text: str) -> str:
        # Build existing_paths from parent_bundle_dir.
        existing = {p.relative_to(parent_bundle_dir).as_posix() for p in parent_bundle_dir.rglob("*") if p.is_file()}
        try:
            ops = parse_sentinel_blocks(patch_text, existing_paths=existing)
            return f"OK: parsed {len(ops)} ops"
        except ValueError as exc:
            return f"ERROR: {exc}"

    def lint_script(path: str, content: str) -> str:
        # Pick suffix from path.
        suffix = ".py" if path.endswith(".py") else (".sh" if path.endswith(".sh") else None)
        if suffix is None:
            return "OK: non-script path"
        with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8") as fh:
            fh.write(content)
            tmp = fh.name
        try:
            if suffix == ".py":
                proc = subprocess.run(
                    ["python", "-m", "py_compile", tmp],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            else:
                proc = subprocess.run(
                    ["bash", "-n", tmp],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            if proc.returncode == 0:
                return "OK"
            return f"ERROR: {(proc.stderr or proc.stdout)[:300]}"
        except (subprocess.SubprocessError, OSError) as exc:
            return f"ERROR: {exc}"
        finally:
            try:
                Path(tmp).unlink(missing_ok=True)
            except Exception:
                pass

    def count_skill_tokens(content: str) -> str:
        return str(_count_tokens(content))

    def finish_candidate(patch_text: str, target_cluster: str, reasoning: str) -> str:
        # Terminal — agent loop captures the args; this handler only echoes.
        return "ok"

    handlers = {
        "read_weakness_report": read_weakness_report,
        "list_parent_bundle_files": list_parent_bundle_files,
        "read_parent_bundle_file": read_parent_bundle_file,
        "read_peer_skill": read_peer_skill,
        "read_mutation_log": read_mutation_log,
        "read_history_aggregates": read_history_aggregates,
        "validate_sentinel_patch": validate_sentinel_patch,
        "lint_script": lint_script,
        "count_skill_tokens": count_skill_tokens,
        "finish_candidate": finish_candidate,
    }

    # ── Specs (OpenAI tool-calling schema) ────────────────────────────────

    specs = [
        {
            "type": "function",
            "function": {
                "name": "read_weakness_report",
                "description": "Read the iter's weakness_report.md in full.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_parent_bundle_files",
                "description": "Enumerate the current best bundle's files.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_parent_bundle_file",
                "description": "Read SKILL.md or any scripts/* from the parent bundle.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "relative path within bundle"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_peer_skill",
                "description": "Read SKILL.md or a script from a peer skill (other slug in same project bundle).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "slug": {"type": "string"},
                        "path": {"type": "string"},
                    },
                    "required": ["slug", "path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_mutation_log",
                "description": "Read the prior-iter mutation_log.md (anonymized).",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_history_aggregates",
                "description": "Return per-outcome counts from history.jsonl (not raw rows).",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "validate_sentinel_patch",
                "description": "Dry-run parse the sentinel-block patch and return errors (if any).",
                "parameters": {
                    "type": "object",
                    "properties": {"patch_text": {"type": "string"}},
                    "required": ["patch_text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lint_script",
                "description": "py_compile / bash -n a candidate script body. Pass file extension via path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "count_skill_tokens",
                "description": "tiktoken cl100k_base count of a string. Use to stay under 2500.",
                "parameters": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                    "required": ["content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finish_candidate",
                "description": "TERMINAL. Emit the sentinel-block patch, target cluster, and reasoning (≤200 words).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patch_text": {"type": "string"},
                        "target_cluster": {"type": "string"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["patch_text", "target_cluster", "reasoning"],
                },
            },
        },
    ]

    return specs, handlers


def _save_transcript(messages: list[dict], out_path: Path) -> None:
    """Write the proposer's full message history as JSONL (one message per line)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for m in messages:
            try:
                fh.write(json.dumps(m, default=str) + "\n")
            except (TypeError, ValueError):
                # Drop any non-serialisable message.
                continue


def _run_one_proposer(
    slot: int,
    weakness_report_path: Path,
    parent_bundle_dir: Path,
    bundles_root: Path | None,
    mutation_log_path: Path | None,
    history_path: Path | None,
    proposer_model: str,
    api_key: str,
    target_cluster: str,
    temperature: float,
    candidate_dir_root: Path,
) -> tuple[int, dict, list[dict]]:
    """Run one watchmen.Agent and return (slot, terminal_args, messages)."""
    # Import here so module import doesn't pay watchmen.Agent's HTTP-client cost
    # for callers that only want anchor.run_anchor.
    from watchmen.agent import Agent

    specs, handlers = _make_proposer_tools(
        weakness_report_path=weakness_report_path,
        parent_bundle_dir=parent_bundle_dir,
        bundles_root=bundles_root,
        mutation_log_path=mutation_log_path,
        history_path=history_path,
    )

    agent = Agent(
        name=f"proposer_c{slot}",
        model=proposer_model,
        system_prompt=_PROPOSER_SYSTEM_PROMPT,
        tool_specs=specs,
        tool_handlers=handlers,
        terminal_tool="finish_candidate",
        api_key=api_key,
    )

    # Multi-turn instruction: weakness context first, then bundle content,
    # then mutation_log, then emit instruction with target cluster. We pack
    # this into one user message — the agent's tools cover the per-context
    # access; the kickoff message just sets the target.
    user_msg = (
        f"Target cluster: {target_cluster}\n"
        f"Temperature: {temperature:.2f}\n\n"
        "Step 1: call read_weakness_report() to load the current iter's report.\n"
        "Step 2: call list_parent_bundle_files() + read_parent_bundle_file() to inspect the parent bundle.\n"
        "Step 3: call read_mutation_log() and read_history_aggregates() to learn from prior iters.\n"
        "Step 4: draft the sentinel-block patch. Validate with validate_sentinel_patch() and "
        "lint_script() / count_skill_tokens() as needed.\n"
        f'Step 5: call finish_candidate(patch_text, target_cluster="{target_cluster}", reasoning="...").'
    )

    try:
        # max_iter=8 per spec FM bookkeeping. Temperature is set on the agent
        # via the model — watchmen.Agent doesn't take a temperature kwarg
        # directly; the value flows through any provider that honours it on
        # subsequent translate_request calls. We pass it as a free-floating
        # marker in the kickoff message for now (the proposer model's own
        # sampling temperature is fixed at the provider default — this is a
        # known limitation of watchmen.Agent's signature).
        terminal_args, messages = agent.run(user_msg, max_iter=16)
    except Exception as exc:  # noqa: BLE001 — proposer faults shouldn't crash run
        print(f"[evolve] proposer_c{slot}_error: {exc}", file=sys.stderr)
        terminal_args = {}
        messages = [{"role": "system", "content": "(proposer crashed)"}]

    # Persist the transcript regardless of outcome.
    transcript_path = candidate_dir_root / f"c{slot}" / "proposer_transcript.jsonl"
    _save_transcript(messages, transcript_path)

    return slot, terminal_args, messages


def propose_candidates(
    run_dir: Path,
    iter_n: int,
    K: int,
    weakness_report_path: Path,
    best_bundle_dir: Path,
    proposer_model: str,
    api_key: str,
    seed: int,
    max_workers: int,
    fingerprints: set[str],
    eval_set_path: Path,
    leak_policy: str = "zero",
    bundles_root: Path | None = None,
) -> list[Path]:
    """Phase 3b — run K=6 multi-turn proposer agents, parse their patches,
    validate, and return the list of surviving candidate bundle dirs.

    Surviving = parse_sentinel_blocks succeeded + scripts compile + SKILL.md
    under MAX_SKILL_TOKENS + leak scan passed (per ``leak_policy``).

    All outcomes (success or rejection reason) are appended to
    ``run_dir/history.jsonl`` per K4.
    """
    iter_dir = run_dir / f"iter_{iter_n}"
    candidates_root = iter_dir / "candidates"
    candidates_root.mkdir(parents=True, exist_ok=True)

    # Prior iter's mutation log feeds the proposer's context tool.
    prior_mutation_log = None
    if iter_n > 1:
        candidate = run_dir / f"iter_{iter_n - 1}" / "mutation_log.md"
        if candidate.exists():
            prior_mutation_log = candidate

    history_path = run_dir / "history.jsonl"

    # Run K proposers in parallel, capped by max_workers (typically 2 for OR cap).
    futures = []
    results: dict[int, tuple[dict, list[dict]]] = {}
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        for slot in range(K):
            cluster = _target_cluster_for(slot)
            temperature = TEMPERATURE_SCHEDULE[slot % len(TEMPERATURE_SCHEDULE)]
            futures.append(
                pool.submit(
                    _run_one_proposer,
                    slot=slot,
                    weakness_report_path=weakness_report_path,
                    parent_bundle_dir=best_bundle_dir,
                    bundles_root=bundles_root,
                    mutation_log_path=prior_mutation_log,
                    history_path=history_path if history_path.exists() else None,
                    proposer_model=proposer_model,
                    api_key=api_key,
                    target_cluster=cluster,
                    temperature=temperature,
                    candidate_dir_root=candidates_root,
                )
            )
        for fut in as_completed(futures):
            try:
                slot, terminal_args, messages = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"[evolve] proposer_future_error: {exc}", file=sys.stderr)
                continue
            results[slot] = (terminal_args, messages)

    surviving: list[Path] = []

    for slot in range(K):
        slot_dir = candidates_root / f"c{slot}"
        slot_dir.mkdir(parents=True, exist_ok=True)
        if slot not in results:
            _append_history(run_dir, iter_n, slot, "parse_error", None, "no terminal call")
            continue

        terminal_args, _messages = results[slot]
        patch_text = (terminal_args.get("patch_text") or "").strip()
        target_cluster = terminal_args.get("target_cluster") or _target_cluster_for(slot)
        reasoning = terminal_args.get("reasoning") or ""

        if not patch_text:
            _append_history(run_dir, iter_n, slot, "parse_error", None, "empty_patch")
            continue

        # Persist the raw patch text for debugging (alongside the transcript).
        try:
            (slot_dir / "patch.txt").write_text(patch_text, encoding="utf-8")
            (slot_dir / "reasoning.txt").write_text(reasoning, encoding="utf-8")
            (slot_dir / "target_cluster.txt").write_text(str(target_cluster), encoding="utf-8")
        except OSError:
            pass

        # Parse + apply + shebang + validate.
        candidate_bundle = slot_dir / "bundle"
        try:
            parse_and_apply(patch_text, best_bundle_dir, candidate_bundle)
        except ValueError as exc:
            _append_history(run_dir, iter_n, slot, f"parse_error:{exc}", None, reasoning[:80])
            # Clean up any half-applied bundle.
            if candidate_bundle.exists():
                shutil.rmtree(candidate_bundle, ignore_errors=True)
            continue
        except Exception as exc:  # noqa: BLE001
            _append_history(run_dir, iter_n, slot, f"parse_error:{exc}", None, reasoning[:80])
            if candidate_bundle.exists():
                shutil.rmtree(candidate_bundle, ignore_errors=True)
            continue

        # validate_scripts is run inside parse_and_apply, but we want to log
        # the error list explicitly so re-run it (it's cheap).
        script_errors = validate_scripts(candidate_bundle, parent_bundle_dir=best_bundle_dir)
        if script_errors:
            _append_history(
                run_dir,
                iter_n,
                slot,
                "validate_error:" + ";".join(script_errors)[:60],
                None,
                reasoning[:80],
            )
            shutil.rmtree(candidate_bundle, ignore_errors=True)
            continue

        # SKILL.md token cap.
        tokens = _skill_md_tokens(candidate_bundle)
        if tokens > MAX_SKILL_TOKENS:
            _append_history(
                run_dir,
                iter_n,
                slot,
                f"validate_error:skill_too_large:{tokens}",
                None,
                reasoning[:80],
            )
            shutil.rmtree(candidate_bundle, ignore_errors=True)
            continue

        # Leak scan (Phase 3b.5).
        leak_log_path = run_dir / "leak_log.md"
        leaks = leak_scan(candidate_bundle, fingerprints)
        if enforce_policy(leaks, leak_policy, leak_log_path=leak_log_path):
            _append_history(
                run_dir,
                iter_n,
                slot,
                f"validate_error:leak:{len(leaks)}",
                None,
                reasoning[:80],
            )
            shutil.rmtree(candidate_bundle, ignore_errors=True)
            continue

        # Survived everything — candidate is ready for scoring.
        surviving.append(candidate_bundle)

    return surviving


# ─── Phase 3c — score_candidates ──────────────────────────────────────────


def score_candidates(
    candidate_dirs: list[Path],
    holdout_evals: list[dict],
    model: str,
    judge_model: str,
    api_key: str,
    rollouts: int,
    max_workers: int,
    lambda_n: float,
    parent_tokens: int,
    iter_n: int,
    run_dir: Path,
    seed: int = 42,
) -> list[tuple[Path, EvalSummary, float]]:
    """Phase 3c — smoke-3 short-circuit + full holdout + fitness penalty.

    Returns a list of (candidate_dir, eval_summary, fitness), sorted by
    fitness desc. Candidates that smoke-fail are still included with
    fitness=0.0 and ``smoke_failed=True`` on the summary so the caller can
    log them in mutation_log.md, but they never win.
    """
    scored: list[tuple[Path, EvalSummary, float]] = []

    for cand in candidate_dirs:
        # Smoke-3 first (K6).
        try:
            survived = smoke_3_check(
                holdout_rows=[r for r in holdout_evals if r.get("split") == "holdout"] or holdout_evals,
                bundle_dir=cand,
                model=model,
                api_key=api_key,
                seed=seed,
                judge_model=judge_model,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[evolve] smoke3_error c={cand.name}: {exc}", file=sys.stderr)
            survived = True  # don't penalise on infra error

        slot = _slot_from_candidate_path(cand)

        if not survived:
            summary = EvalSummary(
                holdout_score=0.0,
                fitness=0.0,
                tokens_skill_md=_skill_md_tokens(cand),
                penalty=0.0,
                lambda_n=lambda_n,
                n_holdout=0,
                smoke_failed=True,
            )
            _write_candidate_summary(cand, summary)
            _append_history(run_dir, iter_n, slot, "smoke_failed", 0.0, "")
            scored.append((cand, summary, 0.0))
            continue

        # Full holdout scoring.
        try:
            summary = score_bundle(
                eval_rows=holdout_evals,
                split="holdout",
                bundle_dir=cand,
                model=model,
                judge_model=judge_model,
                api_key=api_key,
                rollouts=rollouts,
                max_workers=max_workers,
            )
        except PartialScoringError as exc:
            print(f"[evolve] partial_scoring c={cand.name}: {exc}", file=sys.stderr)
            _append_history(
                run_dir,
                iter_n,
                slot,
                f"eval_error:partial:{exc.successful}/{exc.total}",
                None,
                "",
            )
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"[evolve] score_error c={cand.name}: {exc}", file=sys.stderr)
            _append_history(run_dir, iter_n, slot, f"eval_error:{exc}", None, "")
            continue

        tokens = _skill_md_tokens(cand)
        delta_tokens = max(0, tokens - parent_tokens)
        penalty = min(0.05, float(lambda_n) * float(delta_tokens))
        fitness = float(summary.holdout_score) - penalty

        summary.fitness = fitness
        summary.tokens_skill_md = tokens
        summary.penalty = penalty
        summary.lambda_n = lambda_n

        _write_candidate_summary(cand, summary)
        scored.append((cand, summary, fitness))

    scored.sort(key=lambda t: t[2], reverse=True)
    return scored


def _slot_from_candidate_path(cand: Path) -> int:
    """Recover the slot index from a path like .../candidates/c3/bundle/."""
    parent = cand.parent.name  # e.g. "c3"
    if parent.startswith("c") and parent[1:].isdigit():
        return int(parent[1:])
    return -1


def _write_candidate_summary(candidate_bundle_dir: Path, summary: EvalSummary) -> None:
    """Write candidate's eval_summary.json next to its bundle dir."""
    # Candidate root = candidate_bundle_dir.parent (the c{slot}/ dir).
    candidate_root = candidate_bundle_dir.parent
    candidate_root.mkdir(parents=True, exist_ok=True)
    (candidate_root / "eval_summary.json").write_text(
        json.dumps(_eval_summary_to_dict(summary), indent=2, default=str),
        encoding="utf-8",
    )


# ─── Phase 3d — accept_or_discard ─────────────────────────────────────────


def accept_or_discard(
    scored: list[tuple[Path, EvalSummary, float]],
    parent_bundle_dir: Path,
    parent_fitness: float,
    epsilon: float,
    iter_n: int,
    run_dir: Path,
    stall_counter: int,
) -> tuple[Path, float, int]:
    """Phase 3d — promote winner (if fitness > parent + ε) or keep parent.

    Writes ``iter_N/mutation_log.md`` summarising every candidate's
    outcome. Returns ``(winning_bundle_dir, winning_fitness, new_stall_counter)``.

    If no scored candidates at all (all parse_error / validate_error /
    smoke_failed), parent is kept and stall_counter increments.
    """
    iter_dir = run_dir / f"iter_{iter_n}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    parent_hash = hash_bundle(parent_bundle_dir)

    if not scored:
        # No scorable candidates — parent unchanged, stall++.
        _write_mutation_log(
            iter_dir / "mutation_log.md",
            parent_hash=parent_hash,
            scored=[],
            winner=None,
            delta=0.0,
            verdict_by_slot={},
        )
        return parent_bundle_dir, parent_fitness, stall_counter + 1

    winner_cand, winner_summary, winner_fitness = scored[0]
    winner_slot = _slot_from_candidate_path(winner_cand)

    verdict_by_slot: dict[int, str] = {}
    promoted = False

    if winner_fitness > parent_fitness + epsilon:
        # Promote: copy winner → iter_N/bundle/.
        target = iter_dir / "bundle"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(winner_cand, target)
        # Also persist the iter-level eval_summary.json.
        (iter_dir / "eval_summary.json").write_text(
            json.dumps(_eval_summary_to_dict(winner_summary), indent=2, default=str),
            encoding="utf-8",
        )
        verdict_by_slot[winner_slot] = "accepted"
        _append_history(run_dir, iter_n, winner_slot, "promoted", winner_fitness, "")
        new_stall = 0
        chosen_path = target
        chosen_fitness = winner_fitness
        promoted = True
    else:
        verdict_by_slot[winner_slot] = "rejected_by_fitness"
        _append_history(run_dir, iter_n, winner_slot, "rejected_by_fitness", winner_fitness, "")
        new_stall = stall_counter + 1
        chosen_path = parent_bundle_dir
        chosen_fitness = parent_fitness

    # Log everyone else.
    for cand, summary, fitness in scored[1:]:
        slot = _slot_from_candidate_path(cand)
        if summary.smoke_failed:
            verdict_by_slot[slot] = "smoke_failed"
        else:
            verdict_by_slot[slot] = "rejected_by_fitness"
            _append_history(run_dir, iter_n, slot, "rejected_by_fitness", fitness, "")

    _write_mutation_log(
        iter_dir / "mutation_log.md",
        parent_hash=parent_hash,
        scored=scored,
        winner=(winner_cand, winner_summary, winner_fitness) if promoted else None,
        delta=(winner_fitness - parent_fitness) if promoted else 0.0,
        verdict_by_slot=verdict_by_slot,
    )

    return chosen_path, chosen_fitness, new_stall


def _write_mutation_log(
    path: Path,
    parent_hash: str,
    scored: list[tuple[Path, EvalSummary, float]],
    winner: tuple[Path, EvalSummary, float] | None,
    delta: float,
    verdict_by_slot: dict[int, str],
) -> None:
    """Render iter_N/mutation_log.md per spec §3d."""
    lines: list[str] = []
    lines.append("# Mutation log")
    lines.append("")
    lines.append(f"Parent bundle hash: `{parent_hash}`")
    lines.append("")
    if winner is not None:
        wcand, wsum, wfit = winner
        lines.append(f"Winner: `{wcand}` · fitness={wfit:.4f} · Δ={delta:+.4f}")
    else:
        lines.append("Winner: _none_ (parent unchanged)")
    lines.append("")
    lines.append("## Candidates")
    lines.append("")
    lines.append("| slot | fitness | holdout | tokens | penalty | verdict |")
    lines.append("|---:|---:|---:|---:|---:|---|")
    for cand, summary, fitness in scored:
        slot = _slot_from_candidate_path(cand)
        verdict = verdict_by_slot.get(slot, "?")
        lines.append(
            f"| c{slot} | {fitness:.4f} | {summary.holdout_score:.4f} | "
            f"{summary.tokens_skill_md} | {summary.penalty:.4f} | {verdict} |"
        )
    lines.append("")

    # Reasoning previews.
    lines.append("## Reasoning previews")
    lines.append("")
    for cand, _summary, _fitness in scored:
        slot = _slot_from_candidate_path(cand)
        reasoning_path = cand.parent / "reasoning.txt"
        preview = ""
        if reasoning_path.exists():
            try:
                preview = reasoning_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                preview = ""
        lines.append(f"### c{slot}")
        lines.append("")
        lines.append((preview or "_no reasoning recorded_")[:600])
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ─── Phase 3e — run_evolution outer loop ──────────────────────────────────


def _append_metrics(
    run_dir: Path,
    iter_n: int,
    holdout_score: float,
    fitness: float,
    tokens: int,
    by_length_quartile: dict,
    status: str,
) -> None:
    """Append one iter's metrics row to metrics.json (creating the file if needed)."""
    path = run_dir / "metrics.json"
    metrics: dict
    if path.exists():
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(metrics, dict):
                metrics = {"iters": []}
        except (json.JSONDecodeError, OSError):
            metrics = {"iters": []}
    else:
        metrics = {"iters": []}

    iters = metrics.setdefault("iters", [])
    iters.append(
        {
            "iter": iter_n,
            "holdout_score": holdout_score,
            "fitness": fitness,
            "tokens": tokens,
            "by_length_quartile": by_length_quartile or {},
            "status": status,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
    )
    path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")


def run_evolution(
    run_dir: Path,
    iter_0_summary: EvalSummary,
    train_evals: list[dict],
    holdout_evals: list[dict],
    eval_set_path: Path,
    best_bundle_dir: Path,
    model: str,
    judge_model: str,
    proposer_model: str,
    api_key: str,
    rollouts: int,
    max_workers: int,
    max_iters: int,
    lambda_init: float,
    lambda_cap: float,
    seed: int,
    fingerprints: set[str],
    watchdog: "Watchdog",
    leak_policy: str = "zero",
    bundles_root: Path | None = None,
    K: int = 6,
) -> Path:
    """Phase 3e outer loop.

    Drives iters 1..max_iters of:
      3a  build_weakness_report
      3b  propose_candidates (incl. 3b.5 leak scan)
      3c  score_candidates
      3d  accept_or_discard

    Termination conditions (in priority order):
      - watchdog.should_stop() True at iter boundary → break (status=budget_exhausted)
      - stall_counter ≥ 3 → break (status=stalled)
      - iter_n > max_iters → break (status=max_iters_reached)

    Returns the path to the best bundle (via load_best_bundle).
    """
    parent_bundle = best_bundle_dir
    parent_fitness = float(iter_0_summary.holdout_score)
    parent_tokens = _skill_md_tokens(best_bundle_dir)
    epsilon = max(0.01, 1.0 / max(1, len(holdout_evals)))

    # Compute frozen length-quartile bounds off iter_0's train baseline_completion_len_tokens.
    train_lens = [int(r.get("baseline_completion_len_tokens") or 0) for r in train_evals]
    baseline_len_quartiles = [float(b) for b in _length_quartile_bounds(train_lens)]

    stall_counter = 0
    status = "running"

    for iter_n in range(1, max_iters + 1):
        if watchdog is not None and watchdog.should_stop():
            status = "budget_exhausted"
            break

        # λ_N annealing per M5.
        lambda_n = min(lambda_cap, lambda_init * (1.0 + 0.5 * iter_n))

        try:
            # 3a — weakness analysis.
            weakness_path = build_weakness_report(
                run_dir=run_dir,
                iter_n=iter_n,
                train_evals=train_evals,
                best_bundle_dir=parent_bundle,
                model=model,
                judge_model=judge_model,
                api_key=api_key,
                rollouts=rollouts,
                baseline_len_quartiles=baseline_len_quartiles,
                max_workers=max_workers,
            )

            # 3b + 3b.5 — propose K candidates.
            candidate_dirs = propose_candidates(
                run_dir=run_dir,
                iter_n=iter_n,
                K=K,
                weakness_report_path=weakness_path,
                best_bundle_dir=parent_bundle,
                proposer_model=proposer_model,
                api_key=api_key,
                seed=seed + iter_n,
                max_workers=max_workers,
                fingerprints=fingerprints,
                eval_set_path=eval_set_path,
                leak_policy=leak_policy,
                bundles_root=bundles_root,
            )

            # 3c — score.
            scored = score_candidates(
                candidate_dirs=candidate_dirs,
                holdout_evals=holdout_evals,
                model=model,
                judge_model=judge_model,
                api_key=api_key,
                rollouts=rollouts,
                max_workers=max_workers,
                lambda_n=lambda_n,
                parent_tokens=parent_tokens,
                iter_n=iter_n,
                run_dir=run_dir,
                seed=seed + iter_n,
            )

            # 3d — accept or discard.
            parent_bundle, parent_fitness, stall_counter = accept_or_discard(
                scored=scored,
                parent_bundle_dir=parent_bundle,
                parent_fitness=parent_fitness,
                epsilon=epsilon,
                iter_n=iter_n,
                run_dir=run_dir,
                stall_counter=stall_counter,
            )
            parent_tokens = _skill_md_tokens(parent_bundle)
        except Exception as exc:  # noqa: BLE001 — never let one iter kill the run
            import traceback
            print(f"[evolve] iter_{iter_n} crashed: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            stall_counter += 1

        # Metrics row.
        # Try to read this iter's eval_summary.json if a winner was written.
        iter_summary_path = run_dir / f"iter_{iter_n}" / "eval_summary.json"
        iter_holdout = parent_fitness  # default — parent unchanged
        iter_lq: dict = {}
        if iter_summary_path.exists():
            try:
                summary_data = json.loads(iter_summary_path.read_text(encoding="utf-8"))
                iter_holdout = float(summary_data.get("holdout_score", parent_fitness))
                iter_lq = summary_data.get("by_length_quartile") or {}
            except (json.JSONDecodeError, OSError):
                pass

        _append_metrics(
            run_dir,
            iter_n=iter_n,
            holdout_score=iter_holdout,
            fitness=parent_fitness,
            tokens=parent_tokens,
            by_length_quartile=iter_lq,
            status="promoted" if stall_counter == 0 else "no_improvement",
        )

        if stall_counter >= 3:
            status = "stalled"
            break
    else:
        # Loop ran to completion.
        status = "max_iters_reached"

    # Final metrics dump — append a top-level status field for the controller.
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if not isinstance(metrics, dict):
                metrics = {"iters": []}
        except (json.JSONDecodeError, OSError):
            metrics = {"iters": []}
    else:
        metrics = {"iters": []}
    metrics["final_status"] = status
    metrics["epsilon"] = epsilon
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    # Return the best bundle from disk.
    best_path, _best_fitness = load_best_bundle(run_dir)
    return best_path
