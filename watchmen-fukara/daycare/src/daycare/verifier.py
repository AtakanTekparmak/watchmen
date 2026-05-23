"""LLM-judge scoring loop + rollout aggregation (Phase 2/3c).

Three responsibilities:
  - ``score_single``: one judge call against the §1d scoring prompt.
  - ``aggregate_rollouts`` / ``smoke_3_check`` / ``score_bundle``: rollout
    orchestration with parallelism (max_workers) and the W9 / failure-mode #3
    aggregation rules baked in.

Everything is stateless. Each call to OpenRouter is its own HTTP request
through httpx — we don't pool a Provider here because rollouts already
go via ``runner.run_rollout_subprocess`` (subprocess-isolated), and the
judge calls are short single-shot exchanges where pooling overhead is
not worth the complexity.
"""

from __future__ import annotations

import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .runner import RunResult, build_skill_system_prompt, run_rollout_subprocess


# ─── Public types ──────────────────────────────────────────────────────────


@dataclass
class EvalSummary:
    """Per-bundle scoring summary (the row written to ``eval_summary.json``).

    Fields mirror Phase 3c's per-candidate output spec, plus the extra
    bookkeeping the controller needs (successful_evals, smoke_failed).
    """

    holdout_score: float
    fitness: float
    tokens_skill_md: int
    penalty: float
    lambda_n: float
    n_holdout: int
    by_type: dict = field(default_factory=dict)
    by_accepted: dict = field(default_factory=dict)
    by_length_quartile: dict = field(default_factory=dict)
    invocations_render: dict = field(default_factory=dict)
    unused_render: list = field(default_factory=list)
    smoke_failed: bool = False
    successful_evals: int = 0


class PartialScoringError(Exception):
    """Raised when too many evals errored to trust the aggregate (FM#3).

    Spec rule: if ``successful_evals / n_holdout < 0.90``, the candidate
    is marked ``eval_error`` and excluded from winner selection — NOT
    scored as 0.0 (which would silently bias the leaderboard).
    """

    def __init__(self, message: str, successful: int, total: int):
        super().__init__(message)
        self.successful = successful
        self.total = total


# ─── Judge call (§1d) ──────────────────────────────────────────────────────


_JUDGE_SYSTEM_PROMPT = (
    "Score the candidate response 0.0–1.0 using the rubric below.\n"
    'Output JSON: {"score": <float 0-1>, "reasoning": "≤60 words"}\n'
    "Do NOT compare to any reference. Judge only what the candidate text says."
)

# Greedy regex: first { to last } — handles Claude Haiku's markdown-fenced JSON
# and reasoning strings that contain inner } characters.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

# Strip markdown code fences (```json ... ``` or ``` ... ```)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_judge_response(text: str) -> float | None:
    """Pull a float score out of the judge response, with fallback passes.

    Strategy:
      1. Strip markdown code fences; try json.loads on the inner content.
      2. Direct json.loads of the full body.
      3. Greedy-regex out the outermost {…} block and json.loads that.
    Returns the clamped score, or None if all attempts fail.
    """
    candidates: list[str] = []

    # Pass 1: strip markdown code fences (Claude Haiku wraps JSON in ```json...```)
    fence_m = _CODE_FENCE_RE.search(text)
    if fence_m:
        candidates.append(fence_m.group(1))

    # Pass 2: raw full text
    candidates.append(text)

    # Pass 3: greedy outer-brace extraction (first { to last })
    brace_m = _JSON_OBJECT_RE.search(text)
    if brace_m:
        candidates.append(brace_m.group(0))

    for blob in candidates:
        try:
            data = json.loads(blob)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict):
            continue
        score = data.get("score")
        if isinstance(score, (int, float)):
            return max(0.0, min(1.0, float(score)))
    return None


def score_single(
    prompt: str,
    rubric: str,
    completion: str,
    judge_model: str,
    api_key: str,
) -> float | None:
    """Make one OR judge call, return the parsed score (or None on parse fail).

    Uses the §1d "calibration scoring prompt". Temperature 0. Requests
    ``response_format: json_object`` on the off-chance the underlying model
    honours it (DeepSeek-v4-pro does; Qwen3 partially does). On parse
    failure logs a single line to stderr — callers decide whether to
    retry or skip-this-rollout (M8 / failure-mode #1 says: don't bias by
    treating parse-fail as 0.0).
    """
    body = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (f"prompt: {prompt}\nrubric: {rubric}\ncandidate: {completion[:2000]}"),
            },
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=body,
                headers=headers,
            )
            r.raise_for_status()
            data = r.json()
    except Exception as exc:  # noqa: BLE001 — caller handles None
        print(f"[verifier] judge_http_error: {exc}", file=sys.stderr)
        return None

    try:
        text = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        print(f"[verifier] judge_shape_error: {exc}", file=sys.stderr)
        return None

    score = _parse_judge_response(text)
    if score is None:
        print(
            f"[verifier] judge_parse_error: text_head={text[:200]!r}",
            file=sys.stderr,
        )
    return score


# ─── Rollout aggregation (W9) ──────────────────────────────────────────────


def aggregate_rollouts(results: list[RunResult | None]) -> float:
    """Mean of valid (non-None, non-error) rollout scores.

    Full failure (all None / all errors) → 0.0 per W9 step 2. Partial
    failures aggregate over the survivors — this is the rollout-level
    rule, distinct from the eval-level 90% threshold checked in
    ``score_bundle``.
    """
    if not results:
        return 0.0
    valid_scores: list[float] = []
    for r in results:
        if r is None:
            continue
        if r.error is not None:
            continue
        valid_scores.append(r.score)
    if not valid_scores:
        return 0.0
    return sum(valid_scores) / len(valid_scores)


# ─── Smoke-3 short-circuit (K6) ────────────────────────────────────────────


def smoke_3_check(
    holdout_rows: list[dict],
    bundle_dir: Path,
    model: str,
    api_key: str,
    seed: int,
    judge_model: str | None = None,
) -> bool:
    """Run 1 rollout × 3 random holdout evals. Return False iff all 3 → 0.0.

    The judge-model argument is optional purely so older call sites that
    haven't been migrated don't crash; if omitted we fall back to ``model``
    for the judge call (degraded but functional). The spec contract is
    "3 random evals at score==0.0 → skip", so all we need is some judge to
    score them.
    """
    if not holdout_rows:
        return True

    rng = random.Random(seed)
    picks = rng.sample(holdout_rows, k=min(3, len(holdout_rows)))

    skill_prompt = build_skill_system_prompt(bundle_dir)
    judge = judge_model or model

    for idx, row in enumerate(picks):
        rollout = run_rollout_subprocess(
            prompt=row.get("anonymized_prompt") or row.get("prompt") or "",
            skill_system_prompt=skill_prompt,
            model=model,
            api_key=api_key,
            seed=seed + idx,
            temperature=0.7,
        )
        if rollout.error is not None:
            # Treat infra error as "not 0.0" — we can't claim the bundle
            # smoke-failed if we never got a chance to score it. Better
            # to fall through to full scoring and have W9/FM#3 catch the
            # systemic failure there.
            return True

        score = score_single(
            prompt=row.get("prompt") or "",
            rubric=row.get("rubric") or "",
            completion=rollout.completion,
            judge_model=judge,
            api_key=api_key,
        )
        if score is None:
            return True  # parse fail → don't declare smoke_failed
        if score > 0.0:
            return True

    return False


# ─── Full bundle scoring (Phase 3c) ────────────────────────────────────────


def _length_quartile_of(token_count: int, bounds: list[int]) -> str:
    """Bucket a token count into q1..q4 given pre-computed quartile bounds.

    ``bounds`` is a 3-element list ``[q1_top, q2_top, q3_top]`` (the iter_0
    frozen split per M3). Anything ≥ q3_top falls into q4.
    """
    if not bounds or len(bounds) < 3:
        return "q1"
    if token_count <= bounds[0]:
        return "q1"
    if token_count <= bounds[1]:
        return "q2"
    if token_count <= bounds[2]:
        return "q3"
    return "q4"


def _score_one_eval(
    row: dict,
    bundle_dir: Path,
    skill_prompt: str,
    model: str,
    judge_model: str,
    api_key: str,
    rollouts: int,
    seed: int,
) -> tuple[str, float | None, list[RunResult]]:
    """Score a single eval row with N rollouts → (eval_id, mean_score, results).

    mean_score is None iff every rollout errored or every judge call
    returned None (FM#3 — caller treats None as "exclude this eval").
    """
    prompt = row.get("anonymized_prompt") or row.get("prompt") or ""
    raw_prompt = row.get("prompt") or ""
    rubric = row.get("rubric") or ""

    results: list[RunResult] = []
    for k in range(rollouts):
        rollout = run_rollout_subprocess(
            prompt=prompt,
            skill_system_prompt=skill_prompt,
            model=model,
            api_key=api_key,
            seed=seed + k,
            temperature=0.7,
        )
        if rollout.error is None and rollout.completion:
            score = score_single(
                prompt=raw_prompt,
                rubric=rubric,
                completion=rollout.completion,
                judge_model=judge_model,
                api_key=api_key,
            )
            if score is not None:
                rollout.score = score
            else:
                rollout.error = "judge_parse_error"
        results.append(rollout)

    valid = [r.score for r in results if r.error is None]
    if not valid:
        return row.get("id", ""), None, results

    return row.get("id", ""), sum(valid) / len(valid), results


def score_bundle(
    eval_rows: list[dict],
    split: str,
    bundle_dir: Path,
    model: str,
    judge_model: str,
    api_key: str,
    rollouts: int,
    max_workers: int,
    n_rollouts_calibration: int = 5,
    length_quartile_bounds: list[int] | None = None,
) -> EvalSummary:
    """Score ``bundle_dir`` against every row in ``eval_rows`` with split=split.

    Aggregation order (W9):
      1. Per eval: mean over rollouts; None if all error.
      2. Holdout score = mean over evals (excluding Nones).
      3. ``by_type`` / ``by_accepted`` / ``by_length_quartile`` use the same
         exclude-None rule.

    Failure mode #3: assert successful_evals / n_holdout ≥ 0.90 — else
    raise PartialScoringError. Caller logs ``eval_error`` to history.jsonl.

    Note on fields not computed here:
      - ``fitness``, ``tokens_skill_md``, ``penalty``, ``lambda_n``: filled in
        by the controller (it knows λ_N and parent token count).
      - ``invocations_render`` / ``unused_render``: filled in by Phase 3a's
        weakness analyzer when it parses completions for skill XML blocks.
        We leave them empty here — score_bundle's job is just the numeric
        aggregate.
    """
    rows = [r for r in eval_rows if (r.get("split") == split or not split)]
    n_total = len(rows)
    if n_total == 0:
        return EvalSummary(
            holdout_score=0.0,
            fitness=0.0,
            tokens_skill_md=0,
            penalty=0.0,
            lambda_n=0.0,
            n_holdout=0,
            successful_evals=0,
        )

    skill_prompt = build_skill_system_prompt(bundle_dir)

    # Parallelize across evals; rollouts within an eval stay serial so
    # we don't blow past max_workers when k=5.
    eval_means: dict[str, float | None] = {}
    eval_results: dict[str, list[RunResult]] = {}

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = []
        for idx, row in enumerate(rows):
            futures.append(
                pool.submit(
                    _score_one_eval,
                    row,
                    bundle_dir,
                    skill_prompt,
                    model,
                    judge_model,
                    api_key,
                    rollouts,
                    1000 * (idx + 1),
                )
            )
        for fut in as_completed(futures):
            try:
                eid, mean, results = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"[verifier] eval_executor_error: {exc}", file=sys.stderr)
                continue
            eval_means[eid] = mean
            eval_results[eid] = results

    successful = sum(1 for v in eval_means.values() if v is not None)

    # FM#3 — partial-scoring gate.
    if n_total > 0 and (successful / n_total) < 0.90:
        raise PartialScoringError(
            f"only {successful}/{n_total} evals scored (need ≥90%)",
            successful=successful,
            total=n_total,
        )

    # Top-line holdout score (mean of per-eval means, excluding Nones).
    per_eval_scores = [v for v in eval_means.values() if v is not None]
    holdout_score = sum(per_eval_scores) / len(per_eval_scores) if per_eval_scores else 0.0

    # Grouped aggregates.
    by_type: dict[str, list[float]] = {}
    by_accepted: dict[str, list[float]] = {}
    by_lq: dict[str, list[float]] = {"q1": [], "q2": [], "q3": [], "q4": []}

    bounds = length_quartile_bounds or []
    for row in rows:
        eid = row.get("id", "")
        mean = eval_means.get(eid)
        if mean is None:
            continue
        t = row.get("type") or "unknown"
        by_type.setdefault(t, []).append(mean)
        a_key = "true" if row.get("accepted") else "false"
        by_accepted.setdefault(a_key, []).append(mean)
        if bounds:
            lq = _length_quartile_of(int(row.get("baseline_completion_len_tokens") or 0), bounds)
            by_lq[lq].append(mean)

    by_type_mean = {k: (sum(v) / len(v) if v else 0.0) for k, v in by_type.items()}
    by_accepted_mean = {k: (sum(v) / len(v) if v else 0.0) for k, v in by_accepted.items()}
    by_lq_mean = {k: (sum(v) / len(v) if v else 0.0) for k, v in by_lq.items()}

    return EvalSummary(
        holdout_score=holdout_score,
        fitness=holdout_score,  # caller subtracts penalty for true fitness
        tokens_skill_md=0,
        penalty=0.0,
        lambda_n=0.0,
        n_holdout=n_total,
        by_type=by_type_mean,
        by_accepted=by_accepted_mean,
        by_length_quartile=by_lq_mean,
        invocations_render={},
        unused_render=[],
        smoke_failed=False,
        successful_evals=successful,
    )
