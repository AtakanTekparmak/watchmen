"""Score a candidate bundle against a daycare-style ``eval_set.jsonl``.

This is the ``--eval-source behavioral`` plumbing. The judge LLM scores
each item in [0, 1] and we return a ``skill_evolve.evaluator.EvalResult``
so the rest of the evolution pipeline reads the same shape it would for
SkillsBench scoring.

Threshold convention: an item with ``score >= 0.5`` counts toward
``success_rate``; the cutoff matches daycare's verifier defaults. The
returned ``EvalResult.composite`` mirrors ``mean_score`` (a continuous
[0, 1] aggregate) — we do NOT subtract a tool-call overhead term here
because behavioral scoring does not run an agent harness that produces
tool-call counts. Keeping composite == mean_score gives the rest of the
selector pipeline a single, comparable continuous fitness.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from skill_evolve.behavioral.eval_set import EvalItem, load_eval_set
from skill_evolve.evaluator import EvalResult
from skill_evolve.shared.bundle_ops import bundle_tokens, list_scripts

logger = logging.getLogger(__name__)


# Threshold above which a per-item judge score counts as "success" in
# ``EvalResult.success_rate``. Matches daycare's verifier convention
# (the §1d scoring prompt is 0-1 with 0.5 as the explicit cutoff).
SUCCESS_THRESHOLD: float = 0.5

# Judge system prompt — ported verbatim from
# ``daycare.verifier._JUDGE_SYSTEM_PROMPT`` so the behavioral adapter
# scores the same way the daycare run loop did.
_JUDGE_SYSTEM_PROMPT = (
    "Score the candidate response 0.0–1.0 using the rubric below.\n"
    'Output JSON: {"score": <float 0-1>, "reasoning": "≤60 words"}\n'
    "Do NOT compare to any reference. Judge only what the candidate text says."
)


# Regexes ported from daycare/verifier.py:_parse_judge_response.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_SCORE_RE = re.compile(r'"score"\s*:\s*([0-9]*\.?[0-9]+)')


def _parse_judge_response(text: str) -> Optional[float]:
    """Port of daycare ``verifier._parse_judge_response``.

    Tries (1) markdown-fenced JSON, (2) raw JSON, (3) greedy outer-brace
    JSON, (4) regex-extract just the ``score`` float. Returns the
    clamped [0, 1] score or None if all attempts fail.
    """
    candidates: List[str] = []

    fence_m = _CODE_FENCE_RE.search(text)
    if fence_m:
        candidates.append(fence_m.group(1))
    candidates.append(text)
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

    m = _SCORE_RE.search(text)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            pass
    return None


class _JudgeLLM(Protocol):
    """Minimal stub-friendly judge interface.

    The default implementation is ``track_b.openevolve_skills.llm_client.
    OpenRouterLLM``. Tests pass in a fake with the same ``.generate()``
    signature. Either the returned object exposes ``content`` /
    ``reasoning`` attributes (OpenAI-SDK msg) or it returns a plain
    string from ``generate(...)`` — both are accepted by ``_call_judge``.
    """

    def generate(self, *, system: str, user: str) -> str: ...


def _render_bundle(bundle_dir: Path) -> str:
    """Render the bundle for the judge prompt.

    Reads ``SKILL.md`` plus the ``scripts/`` directory layout. Uses
    ``shared.bundle_ops.list_scripts`` (which filters AppleDouble
    ``._*`` files and ``__pycache__/``) and ``bundle_tokens`` for a
    rough size annotation. Deliberately stays small — judges only
    need to see the bundle's outward shape to score the per-item
    rubric.
    """
    chunks: List[str] = []
    skill_md = bundle_dir / "SKILL.md"
    if skill_md.is_file():
        try:
            chunks.append(f"# SKILL.md\n{skill_md.read_text(encoding='utf-8')}")
        except Exception:
            chunks.append("# SKILL.md (unreadable)")
    chunks.append(f"# scripts/\n{list_scripts(bundle_dir)}")
    chunks.append(f"# bundle_tokens={bundle_tokens(bundle_dir)}")
    return "\n\n".join(chunks)


def _extract_text(response: Any) -> str:
    """Unified DeepSeek content/reasoning short-circuit.

    Mirrors daycare ``verifier.py:189``::

        text = msg.get("content") or msg.get("reasoning") or ""

    Accepts (a) a raw string (what ``OpenRouterLLM.generate`` returns
    today), (b) a dict with ``content`` / ``reasoning`` keys (raw API
    response), or (c) an object exposing ``content`` / ``reasoning``
    attributes (OpenAI SDK message). The fallback ensures DeepSeek
    thinking-mode replies (where ``content`` is None but the answer
    lives under ``reasoning``) still produce a parseable score.
    """
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return response.get("content") or response.get("reasoning") or ""
    content = getattr(response, "content", None)
    reasoning = getattr(response, "reasoning", None)
    return content or reasoning or ""


def _call_judge(
    llm: _JudgeLLM,
    *,
    item: EvalItem,
    bundle_render: str,
) -> tuple[Optional[float], str]:
    """One judge call → (score, raw_reasoning_text).

    Returns ``(None, raw_text)`` when the judge response cannot be
    parsed into a float — caller decides whether to drop the item from
    the mean or treat it as 0.
    """
    user = (
        f"prompt: {item.prompt}\n"
        f"rubric: {item.rubric}\n"
        f"candidate_bundle:\n{bundle_render[:4000]}"
    )
    try:
        raw = llm.generate(system=_JUDGE_SYSTEM_PROMPT, user=user)
    except Exception as exc:  # pragma: no cover — network-level failures
        print(f"[behavioral] judge_call_error: {exc}", file=sys.stderr)
        return None, f"judge_call_error: {exc}"
    text = _extract_text(raw)
    score = _parse_judge_response(text)
    return score, text


def _build_default_llm(judge_model: str) -> _JudgeLLM:
    """Default judge: OpenRouter via the track_b llm_client.

    Imported lazily so unit tests that pass an explicit ``llm`` stub
    don't have to install the OpenAI SDK or set OPENROUTER_API_KEY.
    """
    from skill_evolve.track_b.openevolve_skills.llm_client import OpenRouterLLM

    return OpenRouterLLM(model=judge_model)


def score_bundle_behavioral(
    bundle_dir: Path,
    eval_set_path: Path,
    judge_model: Optional[str],
    *,
    repeats: int = 1,
    llm: Optional[_JudgeLLM] = None,
) -> EvalResult:
    """Score ``bundle_dir`` against every row of ``eval_set_path``.

    Args:
        bundle_dir: candidate skill bundle directory (contains SKILL.md
            and optionally ``scripts/``).
        eval_set_path: daycare-style ``eval_set.jsonl`` (one JSON object
            per line, ``id``/``prompt``/``rubric`` required).
        judge_model: OpenRouter slug for the judge LLM. Required when
            ``llm`` is None.
        repeats: per-item judge repeats (mean of valid scores). Default 1.
        llm: optional pre-built judge client. When None we instantiate
            ``OpenRouterLLM(model=judge_model)``. Tests pass a stub.

    Returns:
        :class:`skill_evolve.evaluator.EvalResult` with continuous
        ``mean_score`` (mean of per-item judge scores), ``success_rate``
        (fraction with score >= ``SUCCESS_THRESHOLD``), and ``composite``
        equal to ``mean_score`` (no tool-call penalty since behavioral
        scoring does not run an agent harness).
    """
    bundle_dir = Path(bundle_dir).expanduser().resolve()
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"behavioral bundle dir not found: {bundle_dir}")
    eval_set_path = Path(eval_set_path).expanduser().resolve()
    if not eval_set_path.is_file():
        raise FileNotFoundError(f"behavioral eval_set not found: {eval_set_path}")

    items = load_eval_set(eval_set_path)
    n = len(items)

    if llm is None:
        if not judge_model:
            raise ValueError(
                "score_bundle_behavioral: judge_model is required when llm is None"
            )
        # Allow tests to short-circuit OpenRouter via env var.
        if os.environ.get("SKILL_EVOLVE_BEHAVIORAL_STUB") == "1":
            raise RuntimeError(
                "SKILL_EVOLVE_BEHAVIORAL_STUB=1 set but no llm passed in; "
                "construct the stub explicitly"
            )
        llm = _build_default_llm(judge_model)

    bundle_render = _render_bundle(bundle_dir)

    per_task: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    valid_scores: List[float] = []
    successes = 0

    for item in items:
        scores: List[float] = []
        last_reasoning = ""
        for _ in range(max(1, repeats)):
            s, text = _call_judge(llm, item=item, bundle_render=bundle_render)
            last_reasoning = text
            if s is not None:
                scores.append(s)
        if scores:
            mean = sum(scores) / len(scores)
            valid_scores.append(mean)
        else:
            mean = 0.0  # parse-fail counts as 0 in the aggregate so a
            # bundle that crashes every judge call doesn't silently
            # produce a missing/None composite.
        is_success = mean >= SUCCESS_THRESHOLD
        if is_success:
            successes += 1
        per_task.append(
            {
                "task_id": item.id,
                "item_id": item.id,
                "score": mean,
                "success": bool(is_success),
                "reasoning": (last_reasoning or "")[:400],
                "parsed_repeats": len(scores),
                "requested_repeats": max(1, repeats),
            }
        )
        if not is_success:
            failures.append(
                {
                    "task_id": item.id,
                    "last_msg": (last_reasoning or "")[:200],
                }
            )

    mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    success_rate = (successes / n) if n else 0.0
    composite = mean_score  # documented: behavioral has no tool-call term

    return EvalResult(
        success_rate=success_rate,
        tool_calls_per_success=0.0,  # not applicable to behavioral judge
        composite=composite,
        per_task=per_task,
        failures=failures,
        skills_folder=str(bundle_dir),
        n_tasks=n,
        cascade_truncated=False,
        synthetic=False,
        notes=f"behavioral judge={judge_model or 'stub'} threshold={SUCCESS_THRESHOLD}",
        verified_count=successes,
        unverified_count=n - successes,
        verifier_disabled=True,  # no test.sh / docker harness on this path
        mean_score=mean_score if n else None,
        scored_task_count=len(valid_scores),
    )
