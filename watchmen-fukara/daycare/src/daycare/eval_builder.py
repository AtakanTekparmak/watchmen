"""Phase 1 — eval-set construction pipeline.

This module is the most project-specific part of daycare. Everything
downstream (anchor, evolution, baselines) only sees the frozen
``eval_set.jsonl`` rows produced here.

The pipeline (and the named sub-phases from DAYCARE_SPEC.md §"Phase 1"):

  1a. Session pull  (corpus.query_sessions → parse_transcript)
  1b. Type classification + hard discards (live-infra, safety refusals)
  1c. Rubric generation (judge call, ≤150-word output)
  1d. Knowledge-gap calibration (run weak model + empty skill; keep middle band)
  1e. Stratified 50/50 split + anonymization + semantic dedup (R2)
  1f. Round-trip sanity gate (M2 — score reference ≥0.90 + structural asserts)

The judge calls use stateless OR HTTP, identical wire shape to
``verifier.score_single`` — one POST per call, temperature 0, JSON
response format requested.
"""

from __future__ import annotations

import hashlib
import json
import math
import py_compile
import random
import re
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from .anonymize import AnonymizeContext, strip
from .corpus import Turn, parse_transcript, query_sessions
from .runner import build_skill_system_prompt, run_rollout_subprocess
from .verifier import score_single


# ─── Hard discard regexes (R3 + R5) ────────────────────────────────────────


LIVE_INFRA_PATTERNS: list[str] = [
    r"ssh\s+root@",
    r"ssh\s+ubuntu@",
    r"nvidia-smi",
    r"tmux\s+list-sessions",
    r"tail\s+-f\s+/",
    r"\bwatch\s+",
]

SAFETY_REFUSAL_PATTERNS: list[str] = [
    r"I cannot assist with",
    r"I'm not able to help with",
    r"I can't help with",
    r"I cannot help with",
    r"I'm not able to assist with",
]

_LIVE_INFRA_RE = [re.compile(p, re.IGNORECASE) for p in LIVE_INFRA_PATTERNS]
_SAFETY_RE = [re.compile(p, re.IGNORECASE) for p in SAFETY_REFUSAL_PATTERNS]


# ─── Tiny judge helper (shared by classify + rubric + calibration) ─────────


def _judge_call(
    system_prompt: str,
    user_payload: str,
    model: str,
    api_key: str,
    expect_json: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 600,
) -> str | None:
    """Single OR chat completion call. Returns the assistant text or None.

    All Phase 1 judge calls share this skeleton (per spec §"Phase 1 judge
    prompt templates"): stateless, temperature 0, JSON response format
    when supported.
    """
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if expect_json:
        body["response_format"] = {"type": "json_object"}
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
        return data["choices"][0]["message"]["content"] or ""
    except Exception as exc:  # noqa: BLE001
        print(f"[eval_builder] judge_call_error: {exc}", file=sys.stderr)
        return None


_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _extract_json_obj(text: str) -> dict | None:
    """Best-effort JSON object extraction with one regex recovery pass."""
    if not text:
        return None
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    m = _JSON_OBJECT_RE.search(text)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return None


# ─── Phase 1b — classification + hard discards ─────────────────────────────


_CLASSIFY_SYSTEM = (
    "Classify the assistant's turn into exactly one of:\n"
    "  script_gen   — assistant wrote substantive shell/Python/JS code (≥3 lines)\n"
    "  skill_invoke — a named skill was invoked AND assistant_text references the skill outcome\n"
    "  procedural_qa — assistant explained a procedure, concept, or made a decision (text only)\n"
    "  discard      — confirmation, single-word, live-infra-dependent, or no substantive content\n\n"
    'Output JSON only: {"type": "...", "reason": "≤20 words"}'
)


def _has_live_infra(turn: Turn) -> bool:
    """True iff any Bash tool_call input OR the assistant text contains a
    live-infra trigger (R3 — these are unverifiable script_gen lookalikes).
    """
    bodies: list[str] = [turn.assistant_text or ""]
    for tc in turn.tool_calls or []:
        inp = tc.get("input")
        if isinstance(inp, dict):
            # Bash/Edit/Write tools have input.command / input.content.
            for key in ("command", "content", "code", "input"):
                v = inp.get(key)
                if isinstance(v, str):
                    bodies.append(v)
        elif isinstance(inp, str):
            bodies.append(inp)
    blob = "\n".join(bodies)
    return any(rx.search(blob) for rx in _LIVE_INFRA_RE)


def _has_safety_refusal(turn: Turn) -> bool:
    """True iff the assistant text matches a safety-refusal marker (R5)."""
    txt = turn.assistant_text or ""
    return any(rx.search(txt) for rx in _SAFETY_RE)


def classify_triple(turn: Turn, judge_model: str, api_key: str) -> str:
    """Return one of:
      script_gen | skill_invoke | procedural_qa |
      discard_live_infra | discard_safety_refusal | discard_short |
      discard_confirmation | discard_other

    Hard-discard checks (live-infra / safety / short / confirmation) run
    BEFORE the judge call so we don't waste tokens on un-scorable triples.
    """
    if _has_safety_refusal(turn):
        return "discard_safety_refusal"
    if _has_live_infra(turn):
        return "discard_live_infra"

    atxt = (turn.assistant_text or "").strip()
    utxt = (turn.user_text or "").strip()
    if not atxt or len(atxt) < 20:
        return "discard_short"
    # Pure confirmations like "ok", "yes", "keep monitoring".
    if len(utxt.split()) <= 2 and not turn.tool_calls:
        return "discard_confirmation"

    # Build the judge payload.
    tool_summary: list[dict] = []
    for tc in (turn.tool_calls or [])[:6]:
        inp = tc.get("input") or {}
        # input_preview = first 200 chars of the dominant input field.
        preview = ""
        if isinstance(inp, dict):
            for k in ("command", "content", "code", "file_path", "path"):
                v = inp.get(k)
                if isinstance(v, str):
                    preview = v[:200]
                    break
            if not preview:
                # Fall back to a JSON dump of the input dict.
                try:
                    preview = json.dumps(inp)[:200]
                except Exception:
                    preview = ""
        elif isinstance(inp, str):
            preview = inp[:200]
        tool_summary.append({"tool": tc.get("name", ""), "input_preview": preview})

    payload = json.dumps(
        {
            "user_turn": utxt[:2000],
            "assistant_text": atxt[:1000],
            "tool_calls": tool_summary,
            "skill_name_in_session": turn.skill_name,
        },
        ensure_ascii=False,
    )

    raw = _judge_call(_CLASSIFY_SYSTEM, payload, judge_model, api_key)
    if raw is None:
        # Retry once per spec §1b "On parse failure: retry once".
        raw = _judge_call(_CLASSIFY_SYSTEM, payload, judge_model, api_key)
    parsed = _extract_json_obj(raw or "")
    if not parsed:
        return "discard_other"
    t = parsed.get("type")
    if t == "skill_invoke" and turn.skill_name is None:
        # Hard rule: skill_invoke requires a populated skill_name.
        return "procedural_qa"
    if t in ("script_gen", "skill_invoke", "procedural_qa"):
        return t  # type: ignore[return-value]
    if t == "discard":
        return "discard_other"
    return "discard_other"


# ─── Phase 1c — rubric generation ──────────────────────────────────────────


_RUBRIC_SYSTEM = (
    "Write a scoring rubric for a weak LLM called {weak_model_name}.\n"
    "Rules:\n"
    '- Forbidden: criteria requiring "matches reference exactly", "same approach as reference",\n'
    "  live internet access, filesystem access, or domain knowledge beyond what a 30B model knows.\n"
    "- Required: concrete observable properties of the CANDIDATE TEXT ALONE (no reference comparison).\n"
    '- Format: "Score 0.0–1.0. Award 1.0 if: [criterion]. Award 0.5 if: [partial]. Award 0.0 if: [failure]."\n'
    "- Max 150 words.\n"
    'Output JSON: {{"rubric": "<text>"}}'
)


def generate_rubric(
    eval_type: str,
    prompt: str,
    reference: str,
    weak_model_name: str,
    judge_model: str,
    api_key: str,
) -> str:
    """Produce a ≤150-word rubric string. Returns "" on failure."""
    system = _RUBRIC_SYSTEM.format(weak_model_name=weak_model_name)
    user = json.dumps(
        {
            "type": eval_type,
            "anonymized_prompt": prompt[:1500],
            "anonymized_reference": reference[:500],
        },
        ensure_ascii=False,
    )
    raw = _judge_call(system, user, judge_model, api_key, max_tokens=400)
    parsed = _extract_json_obj(raw or "")
    if parsed and isinstance(parsed.get("rubric"), str):
        return parsed["rubric"].strip()
    # Fallback to using the raw text if JSON wrapping failed.
    return (raw or "").strip()[:1200]


# ─── Phase 1a — acceptance signal ──────────────────────────────────────────


_REJECTED_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^\s*no\b",
        r"^\s*wrong\b",
        r"^\s*actually\b",
        r"^\s*that's not\b",
        r"^\s*try again\b",
        r"^\s*re-?do\b",
        r"^\s*not right\b",
    ]
]

_ACCEPTED_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(it|that|the script|the code|this|the approach)\b.{0,80}[\?\.]",
    ]
]

_ACCEPT_SYSTEM = (
    "Did the user accept or reject the assistant's prior response?\n"
    'Output JSON: {"accepted": true|false, "confidence": 0.0-1.0}'
)


def detect_acceptance(
    turn: Turn,
    next_user_text: str | None,
    judge_model: str | None = None,
    api_key: str | None = None,
) -> bool | None:
    """Regex-first, LLM fallback. Returns True/False, or None if either
    confidence is too low or no signal at all (drop the eval candidate).
    """
    if next_user_text is None:
        return None
    ntxt = next_user_text.strip()
    if not ntxt:
        return None

    for rx in _REJECTED_PATTERNS:
        if rx.search(ntxt):
            return False
    for rx in _ACCEPTED_PATTERNS:
        if rx.search(ntxt):
            return True

    # LLM fallback. Only attempt if we have judge credentials.
    if not judge_model or not api_key:
        return None
    payload = json.dumps(
        {
            "assistant_response": (turn.assistant_text or "")[:300],
            "next_user_turn": ntxt[:600],
        },
        ensure_ascii=False,
    )
    raw = _judge_call(_ACCEPT_SYSTEM, payload, judge_model, api_key, max_tokens=80)
    parsed = _extract_json_obj(raw or "")
    if not parsed:
        return None
    conf = parsed.get("confidence")
    if not isinstance(conf, (int, float)) or float(conf) < 0.6:
        return None
    val = parsed.get("accepted")
    if isinstance(val, bool):
        return val
    return None


# ─── Phase 1d — knowledge-gap calibration ─────────────────────────────────


def calibrate_eval(
    prompt: str,
    rubric: str,
    bundle_dir: Path,
    weak_model: str,
    api_key: str,
    judge_model: str,
    seed: int,
    n_rollouts: int = 3,
) -> float:
    """Score (weak model + empty skill) on (prompt, rubric) over n_rollouts.

    Failure-mode #1 defense: use ``n_rollouts >= 3`` (not 1) so the
    iter_0 baseline is comparable to subsequent iters' 5-rollout averages.
    """
    skill_prompt = build_skill_system_prompt(bundle_dir)
    scores: list[float] = []
    for k in range(n_rollouts):
        rollout = run_rollout_subprocess(
            prompt=prompt,
            skill_system_prompt=skill_prompt,
            model=weak_model,
            api_key=api_key,
            seed=seed + k,
            temperature=0.7,
        )
        if rollout.error is not None or not rollout.completion:
            continue
        s = score_single(
            prompt=prompt,
            rubric=rubric,
            completion=rollout.completion,
            judge_model=judge_model,
            api_key=api_key,
        )
        if s is not None:
            scores.append(s)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


# ─── Phase 1a–1d orchestration ─────────────────────────────────────────────


def _make_eval_id(prompt: str, source_session: str) -> str:
    """Stable 12-char eval id (sha256 prefix)."""
    h = hashlib.sha256()
    h.update(prompt.encode("utf-8", errors="replace"))
    h.update(b"\x00")
    h.update(source_session.encode("utf-8", errors="replace"))
    return h.hexdigest()[:12]


def _summarize_assistant(turn: Turn) -> str:
    """Build a reference string = assistant text + a tiny tool-call summary."""
    parts = [turn.assistant_text or ""]
    for tc in (turn.tool_calls or [])[:5]:
        name = tc.get("name", "")
        inp = tc.get("input") or {}
        preview = ""
        if isinstance(inp, dict):
            for k in ("command", "content", "code", "file_path", "path"):
                v = inp.get(k)
                if isinstance(v, str):
                    preview = v[:300]
                    break
            if not preview:
                try:
                    preview = json.dumps(inp)[:300]
                except Exception:
                    preview = ""
        elif isinstance(inp, str):
            preview = inp[:300]
        parts.append(f"\n[tool:{name}] {preview}")
    return "".join(parts)


def _log_discard(log_path: Path, kind: str, detail: str) -> None:
    """Append a line to eval_extraction_log.md (creates the file if absent)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"- {kind}: {detail}\n")


def pull_and_classify(
    db_path: Path,
    source_repo: str,
    projects_json: Path,
    bundle_dir: Path,
    days: int,
    weak_model: str,
    judge_model: str,
    api_key: str,
    seed: int,
    run_dir: Path,
    max_candidates: int | None = None,
    max_workers: int = 4,
) -> list[dict]:
    """Phases 1a–1d, returning a list of pre-anonymization eval dicts.

    Each dict carries: id, type, prompt, reference, rubric, baseline_score,
    baseline_completion_len_tokens, accepted, source_session, source_skill.

    Discards are logged to ``run_dir/eval_extraction_log.md``.

    Args:
        max_candidates: if set, randomly sample this many triples before
            LLM calls (for testing/speed). None = use all.
        max_workers: thread-pool size for parallel classify + rubric calls.
    """
    log_path = run_dir / "eval_extraction_log.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_lock = threading.Lock()

    def _log(kind: str, detail: str) -> None:
        with log_lock:
            _log_discard(log_path, kind, detail)

    sessions = query_sessions(db_path, source_repo, days=days)
    candidates: list[dict] = []

    # ── Step 1: collect all (session_meta, turn_index, turn, next_user) ──────
    raw_triples: list[tuple[dict, int, Turn, str | None]] = []
    for sess in sessions:
        transcript_path = sess.get("transcript_path") or ""
        if not transcript_path:
            _log("no_transcript_path", str(sess.get("session_id", "")))
            continue
        path = Path(transcript_path)
        if not path.exists():
            _log("transcript_gone", str(sess.get("session_id", "")))
            continue
        turns = parse_transcript(path)
        if not turns:
            _log("empty_transcript", str(sess.get("session_id", "")))
            continue
        for i, turn in enumerate(turns):
            next_user = turns[i + 1].user_text if (i + 1) < len(turns) else None
            raw_triples.append((sess, i, turn, next_user))

    # Optional cap: sample before expensive LLM calls (for testing).
    if max_candidates and len(raw_triples) > max_candidates:
        rng = random.Random(seed)
        raw_triples = rng.sample(raw_triples, max_candidates)

    # ── Step 2: parallel classification (I/O-bound OR calls) ─────────────────
    def _classify_one(item: tuple[dict, int, Turn, str | None]) -> tuple[dict, int, Turn, str | None, str, bool | None]:
        sess, i, turn, next_user = item
        sid = str(sess.get("session_id", ""))
        cls = classify_triple(turn, judge_model, api_key)
        if cls.startswith("discard_"):
            _log(cls, f"session={sid} turn={i}")
            return sess, i, turn, next_user, cls, None
        accepted = detect_acceptance(turn, next_user, judge_model=judge_model, api_key=api_key)
        if accepted is None:
            _log("discard_no_acceptance_signal", f"session={sid} turn={i}")
            return sess, i, turn, next_user, "discard_no_acceptance_signal", None
        return sess, i, turn, next_user, cls, accepted

    classified: list[tuple[dict, int, Turn, str | None, str, bool | None]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_classify_one, item): item for item in raw_triples}
        for fut in as_completed(futures):
            result = fut.result()
            if not result[4].startswith("discard_"):
                classified.append(result)

    # ── Step 3: parallel rubric generation ───────────────────────────────────
    def _rubric_one(item: tuple[dict, int, Turn, str | None, str, bool | None]):
        sess, i, turn, next_user, cls, accepted = item
        sid = str(sess.get("session_id", ""))
        prompt = turn.user_text or ""
        reference = _summarize_assistant(turn)
        if not prompt.strip() or not reference.strip():
            _log("discard_empty_fields", f"session={sid} turn={i}")
            return None
        rubric = generate_rubric(cls, prompt, reference, weak_model, judge_model, api_key)
        if not rubric:
            _log("discard_rubric_empty", f"session={sid} turn={i}")
            return None
        return sess, i, turn, cls, accepted, prompt, reference, rubric

    with_rubrics = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_rubric_one, item) for item in classified]
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                with_rubrics.append(r)

    # ── Step 4: calibration (subprocess-heavy; fewer workers) ─────────────────
    def _calibrate_one(item):
        sess, i, turn, cls, accepted, prompt, reference, rubric = item
        sid = str(sess.get("session_id", ""))
        baseline_score = calibrate_eval(
            prompt=prompt,
            rubric=rubric,
            bundle_dir=bundle_dir,
            weak_model=weak_model,
            api_key=api_key,
            judge_model=judge_model,
            seed=seed + i,
            n_rollouts=3,
        )
        # Knowledge-gap filter.
        if baseline_score <= 0.0:
            _log("discard_calibration_zero", f"session={sid} turn={i}")
            return None
        if baseline_score >= 0.9:
            _log("discard_calibration_solved", f"session={sid} turn={i} score={baseline_score:.2f}")
            return None
        return sess, i, turn, cls, accepted, prompt, reference, rubric, baseline_score

    # Use fewer workers for calibration (each spawns 3 subprocesses).
    cal_workers = max(1, max_workers // 2)
    surviving = []
    with ThreadPoolExecutor(max_workers=cal_workers) as pool:
        futures = [pool.submit(_calibrate_one, item) for item in with_rubrics]
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                surviving.append(r)

    # Rebuild candidates list from surviving items.
    for item in surviving:
        sess, i, turn, cls, accepted, prompt, reference, rubric, baseline_score = item
        sid = str(sess.get("session_id", ""))
        candidates.append(
            {
                "id": _make_eval_id(prompt, sid),
                "type": cls,
                "prompt": prompt,
                "reference": reference,
                "rubric": rubric,
                "baseline_score": baseline_score,
                "baseline_completion_len_tokens": 0,  # filled at anonymize time
                "accepted": bool(accepted),
                "source_session": sid,
                "source_skill": turn.skill_name,
            }
        )

    return candidates


# ─── Phase 1e — semantic dedup + split + anonymization ─────────────────────


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity for two equal-length float vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts with BAAI/bge-small-en-v1.5 via fastembed.

    Import is local so the module imports cheaply when only Phase 1a is used.
    """
    from fastembed import TextEmbedding

    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    out: list[list[float]] = []
    for vec in model.embed(texts):
        # fastembed yields numpy arrays; convert to plain lists for sanity.
        out.append([float(x) for x in vec])
    return out


def semantic_dedup(
    evals: list[dict],
    threshold: float = 0.92,
    max_per_cluster: int = 5,
) -> list[dict]:
    """Cluster on cosine(anonymized_prompt) ≥ threshold, cap at max_per_cluster
    per cluster (keeping the items that maximise baseline_score variance).

    R2 mandate: after dedup, we must have ≥30 distinct clusters or the
    corpus lacks distillable surface — raise ValueError.
    """
    if not evals:
        raise ValueError("insufficient_distillable_surface")

    prompts = [e.get("anonymized_prompt") or e.get("prompt") or "" for e in evals]
    embeddings = _embed_texts(prompts)

    # Greedy single-link clustering. Each eval is assigned to the first
    # existing cluster whose centroid has cosine ≥ threshold; else starts
    # a new cluster. Centroid = mean of member vectors (recomputed on add).
    clusters: list[dict] = []  # {centroid: list[float], members: list[int]}

    for idx, vec in enumerate(embeddings):
        placed = False
        for cl in clusters:
            if _cosine(cl["centroid"], vec) >= threshold:
                cl["members"].append(idx)
                # Update centroid as running mean.
                n = len(cl["members"])
                cl["centroid"] = [(cl["centroid"][k] * (n - 1) + vec[k]) / n for k in range(len(vec))]
                placed = True
                break
        if not placed:
            clusters.append({"centroid": list(vec), "members": [idx]})

    if len(clusters) < 30:
        raise ValueError("insufficient_distillable_surface")

    # Cap each cluster — keep the max_per_cluster items with the widest
    # baseline_score spread (preserves the most informative range per R2).
    kept_indices: list[int] = []
    for cl in clusters:
        members = cl["members"]
        if len(members) <= max_per_cluster:
            kept_indices.extend(members)
            continue
        scored = [(evals[i].get("baseline_score") or 0.0, i) for i in members]
        # Pick min, max, and fill the remaining slots from the middle.
        scored.sort()
        picked: list[int] = []
        # Always keep min and max.
        picked.append(scored[0][1])
        picked.append(scored[-1][1])
        # Fill from evenly spaced indices in the middle.
        mid = scored[1:-1]
        if mid and (max_per_cluster - 2) > 0:
            step = max(1, len(mid) // (max_per_cluster - 2))
            for i in range(0, len(mid), step):
                if len(picked) >= max_per_cluster:
                    break
                picked.append(mid[i][1])
        kept_indices.extend(picked[:max_per_cluster])

    kept_set = set(kept_indices)
    return [e for i, e in enumerate(evals) if i in kept_set]


def _count_tokens(text: str) -> int:
    """tiktoken cl100k_base token count (W5)."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except Exception:
        # Approximate fallback — ~4 chars/token. Used only when tiktoken
        # isn't installed in the env; tests don't depend on the exact count.
        return max(0, len(text or "") // 4)


def _stratified_split(evals: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    """50/50 split stratified by (type, accepted).

    Failure-mode #2 defense: assert n_holdout_per_type ≥ 3 for every
    present type, else raise ValueError("insufficient_stratification").
    """
    rng = random.Random(seed)

    # Group by stratum.
    by_stratum: dict[tuple, list[dict]] = {}
    for e in evals:
        key = (e.get("type"), bool(e.get("accepted")))
        by_stratum.setdefault(key, []).append(e)

    train: list[dict] = []
    holdout: list[dict] = []
    for key, items in by_stratum.items():
        shuffled = list(items)
        rng.shuffle(shuffled)
        cut = len(shuffled) // 2
        # Round-robin the remainder: if odd, give the extra to train so
        # holdout never overflows train.
        holdout.extend(shuffled[:cut])
        train.extend(shuffled[cut:])

    # Stamp split field.
    for e in train:
        e["split"] = "train"
    for e in holdout:
        e["split"] = "holdout"

    # FM#2: per-type holdout floor.
    holdout_by_type: dict[str, int] = {}
    for e in holdout:
        t = e.get("type") or "unknown"
        holdout_by_type[t] = holdout_by_type.get(t, 0) + 1
    for t, n in holdout_by_type.items():
        if n < 3:
            raise ValueError(f"insufficient_stratification: type={t} has only {n} holdout evals")

    return train, holdout


def build_eval_set(
    raw_evals: list[dict],
    seed: int,
    ctx: AnonymizeContext,
    run_dir: Path,
) -> tuple[list[dict], list[dict]]:
    """Phase 1e — anonymize → dedup → stratified split → write eval_set.jsonl.

    Returns (train, holdout). All rows in both slices have the anonymized_*
    fields populated; the proposer reads only those.
    """
    # Anonymize fields. The proposer reads only the anonymized_* keys, but
    # the judge needs the raw forms for scoring.
    for e in raw_evals:
        e["anonymized_prompt"] = strip(e.get("prompt") or "", ctx)
        e["anonymized_reference"] = strip(e.get("reference") or "", ctx)
        e["anonymized_rubric"] = strip(e.get("rubric") or "", ctx)
        # baseline completion token count is computed off the reference as
        # a proxy (we don't store the iter_0 completion here — that lives
        # in the anchor's eval_summary). M3 quartile bounds are stable
        # across iters because the reference text never changes.
        e["baseline_completion_len_tokens"] = _count_tokens(e.get("reference") or "")

    # Semantic dedup gate (R2).
    deduped = semantic_dedup(raw_evals)

    # Stratified split with FM#2 floor.
    train, holdout = _stratified_split(deduped, seed)

    # Persist to disk.
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "eval_set.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for e in train + holdout:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    return train, holdout


# ─── Phase 1f — round-trip sanity gate ─────────────────────────────────────


_SHELL_FENCE_RE = re.compile(r"```(?:bash|sh|shell)\n(.*?)```", re.DOTALL)
_PY_FENCE_RE = re.compile(r"```(?:python|py)\n(.*?)```", re.DOTALL)


def _extract_script_blobs(reference: str) -> list[tuple[str, str]]:
    """Pull (kind, body) pairs out of a reference string.

    Kind is "py" or "sh". Looks at fenced code blocks and the canonical
    [tool:Bash]/[tool:Write] tool-summary suffixes we synthesise in
    _summarize_assistant.
    """
    blobs: list[tuple[str, str]] = []
    for m in _PY_FENCE_RE.finditer(reference):
        blobs.append(("py", m.group(1)))
    for m in _SHELL_FENCE_RE.finditer(reference):
        blobs.append(("sh", m.group(1)))
    # Heuristic: if the reference has a [tool:Bash] section, treat that
    # tail as a shell blob.
    for line in (reference or "").splitlines():
        if line.startswith("[tool:Bash]"):
            blobs.append(("sh", line[len("[tool:Bash]") :].strip()))
    return blobs


def _script_compiles(kind: str, body: str) -> bool:
    """py_compile or `bash -n` on the body in a temp file."""
    suffix = ".py" if kind == "py" else ".sh"
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8") as fh:
        fh.write(body)
        tmp_path = fh.name
    try:
        if kind == "py":
            try:
                py_compile.compile(tmp_path, doraise=True)
                return True
            except py_compile.PyCompileError:
                return False
        else:
            try:
                rc = subprocess.run(
                    ["bash", "-n", tmp_path],
                    capture_output=True,
                    timeout=10,
                ).returncode
                return rc == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def round_trip_gate(
    holdout_evals: list[dict],
    bundle_dir: Path,
    judge_model: str,
    api_key: str,
    seed: int,
    run_dir: Path,
) -> list[dict]:
    """Score up to 20 random holdouts' references as candidates; drop <0.90.

    Plus structural assertions per M2:
      - script_gen: extracted blobs must py_compile / bash -n cleanly.
      - skill_invoke: source_skill must exist under bundle_dir.parent/skills/.

    Logged to run_dir/eval_extraction_log.md. Returns the surviving evals
    (all rows that pass both judge-consistency AND structural checks).
    """
    log_path = run_dir / "eval_extraction_log.md"
    rng = random.Random(seed)

    sample = list(holdout_evals)
    rng.shuffle(sample)
    sample = sample[:20]
    keep_ids: set[str] = set()
    drop_ids: set[str] = set()

    # Path that holds peer skills for the skill_invoke check. Bundle dir
    # is the per-skill dir; its parent is .../skills/. We climb one more
    # level to find the project's skills/ directory.
    skills_root = bundle_dir.parent if bundle_dir.parent.name == "skills" else bundle_dir.parent

    for e in sample:
        eid = e.get("id", "")
        ref = e.get("reference") or ""
        rubric = e.get("rubric") or ""
        prompt = e.get("prompt") or ""

        # Judge self-consistency.
        s = score_single(
            prompt=prompt,
            rubric=rubric,
            completion=ref,
            judge_model=judge_model,
            api_key=api_key,
        )
        if s is None or s < 0.90:
            drop_ids.add(eid)
            _log_discard(
                log_path,
                "round_trip_judge_fail",
                f"id={eid} score={s}",
            )
            continue

        # Structural per-type asserts.
        etype = e.get("type")
        if etype == "script_gen":
            blobs = _extract_script_blobs(ref)
            if not blobs:
                # No extractable script means the reference was misclassified
                # as script_gen. Drop.
                drop_ids.add(eid)
                _log_discard(log_path, "round_trip_no_script_blob", f"id={eid}")
                continue
            if not all(_script_compiles(k, b) for k, b in blobs):
                drop_ids.add(eid)
                _log_discard(log_path, "round_trip_script_compile_fail", f"id={eid}")
                continue
        elif etype == "skill_invoke":
            slug = e.get("source_skill")
            if not slug or not (skills_root / str(slug)).exists():
                drop_ids.add(eid)
                _log_discard(
                    log_path,
                    "round_trip_missing_skill",
                    f"id={eid} slug={slug}",
                )
                continue

        keep_ids.add(eid)

    # Apply decisions only to the sampled subset; unsampled evals pass
    # through unchanged (the round-trip gate is a sanity probe, not an
    # exhaustive screen).
    survivors: list[dict] = []
    for e in holdout_evals:
        if e.get("id") in drop_ids:
            continue
        survivors.append(e)
    return survivors


# ─── Top-level orchestrator ────────────────────────────────────────────────


def run_eval_build(
    db_path: Path,
    source_repo: str,
    projects_json: Path,
    bundle_dir: Path,
    weak_model: str,
    judge_model: str,
    api_key: str,
    seed: int,
    days: int,
    run_dir: Path,
    max_candidates: int | None = None,
    max_workers: int = 4,
) -> tuple[list[dict], list[dict]]:
    """Compose Phases 1a–1f. Returns (train, holdout).

    The caller is responsible for building the AnonymizeContext (it has
    project_slugs / skill_slugs the eval_builder doesn't know about).
    """
    from .anonymize import build_context

    raw = pull_and_classify(
        db_path=db_path,
        source_repo=source_repo,
        projects_json=projects_json,
        bundle_dir=bundle_dir,
        days=days,
        weak_model=weak_model,
        judge_model=judge_model,
        api_key=api_key,
        seed=seed,
        run_dir=run_dir,
        max_candidates=max_candidates,
        max_workers=max_workers,
    )

    ctx = build_context(projects_json, bundle_dir, source_repo)
    train, holdout = build_eval_set(raw, seed, ctx, run_dir)

    # Round-trip gate only on holdout (per spec §1f: "for 20 random evals").
    holdout = round_trip_gate(
        holdout_evals=holdout,
        bundle_dir=bundle_dir,
        judge_model=judge_model,
        api_key=api_key,
        seed=seed,
        run_dir=run_dir,
    )

    # Re-write eval_set.jsonl with the post-gate holdout (train unchanged).
    out_path = run_dir / "eval_set.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for e in train + holdout:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    return train, holdout
