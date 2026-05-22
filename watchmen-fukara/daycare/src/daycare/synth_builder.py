"""Synthetic eval generation from watchmen skill bundles.

Replaces the conversation-extraction Phase 1 with LLM-generated Q&A pairs
grounded in the SKILL.md content. The skill IS the answer key — questions
test whether a 30B model knows what the skill teaches.

Why synthetic instead of extracting from corpus:
- Real conversations are either "you had to be there" operational tasks
  (score 0: needs live pod/run/context) or trivial general knowledge
  (score 1.0: model already knows). Nothing in the learnable middle band.
- Synthetic questions are designed to hit 0.2-0.7 without skill,
  0.7-0.9 with skill, by targeting load-bearing skill-specific details.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from .eval_builder import build_eval_set, calibrate_eval
from .runner import build_skill_system_prompt, run_rollout_subprocess
from .verifier import score_single


# ─── Constants ─────────────────────────────────────────────────────────────


_PROPOSER_SYSTEM_TMPL = (
    "You generate eval questions for a skill-testing benchmark. Given a SKILL.md file,\n"
    "generate {n} question-answer pairs that test whether a model knows the specific,\n"
    "non-obvious details the skill teaches.\n\n"
    "Rules:\n"
    "- Each question must be answerable from the SKILL.md content alone (no live infra needed)\n"
    "- Target load-bearing details: exact flag names, thresholds, file paths, provider names,\n"
    "  command patterns — things a general 30B model would guess wrong without the skill\n"
    "- Avoid \"you had to be there\" tokens: no instance IDs, pod IPs, run names — use placeholders\n"
    "- Each answer must be concise (≤200 words) and directly answerable\n"
    "- Include a rubric that awards 1.0 for the specific load-bearing detail,\n"
    "  0.5 for the right general approach, 0.0 for wrong/missing the key point\n"
    "- Mix question types: script_gen (write a command/script), procedural_qa (explain a procedure),\n"
    "  fact_recall (what is the exact value of X)\n\n"
    "Output JSON array: [{{\"question\": \"...\", \"answer\": \"...\", \"rubric\": \"...\", "
    "\"type\": \"script_gen|procedural_qa|fact_recall\", "
    "\"load_bearing_detail\": \"≤20 word description of what makes this question skill-specific\"}}]"
)


_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


# ─── Proposer call ─────────────────────────────────────────────────────────


def _proposer_call(
    system_prompt: str,
    user_payload: str,
    model: str,
    api_key: str,
    max_tokens: int = 8000,
    temperature: float = 0.7,
) -> str | None:
    """Single OR chat completion call for the proposer. Returns text or None."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(timeout=240.0) as client:
            r = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=body,
                headers=headers,
            )
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"]["content"] or ""
    except Exception as exc:  # noqa: BLE001
        print(f"[synth_builder] proposer_call_error: {exc}", file=sys.stderr)
        return None


def _extract_json_array(text: str) -> list | None:
    """Best-effort JSON-array extraction with one regex recovery pass."""
    if not text:
        return None
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    m = _JSON_ARRAY_RE.search(text)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return None


# ─── Step 1: question generation ───────────────────────────────────────────


def _read_skill_md(skill_dir: Path) -> str:
    """Read SKILL.md from a skill bundle directory; return '' on failure."""
    p = skill_dir / "SKILL.md"
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _extract_workflow_archetypes(running_md_text: str, max_chars: int = 500) -> str:
    """Pull a ≤max_chars excerpt of the 'Workflow archetypes' section."""
    if not running_md_text:
        return ""
    # Find the section heading.
    m = re.search(r"##\s*Workflow archetypes\s*\n", running_md_text, re.IGNORECASE)
    if not m:
        return running_md_text[:max_chars]
    start = m.end()
    # Next top-level "##" heading bounds the section.
    end_m = re.search(r"\n##\s+", running_md_text[start:])
    end = start + end_m.start() if end_m else len(running_md_text)
    excerpt = running_md_text[start:end].strip()
    return excerpt[:max_chars]


def generate_questions_for_skill(
    skill_dir: Path,
    running_md_text: str,
    claude_md_text: str,
    proposer_model: str,
    api_key: str,
    n: int = 20,
    seed: int = 42,
) -> list[dict]:
    """Call the proposer to generate N Q&A pairs grounded in SKILL.md.

    Returns a list of raw question dicts (before calibration). Each dict
    has keys: question, answer, rubric, type, load_bearing_detail.
    """
    skill_md = _read_skill_md(skill_dir)
    if not skill_md.strip():
        return []

    archetypes = _extract_workflow_archetypes(running_md_text, max_chars=500)
    claude_excerpt = (claude_md_text or "")[:500]

    system_prompt = _PROPOSER_SYSTEM_TMPL.format(n=n)

    user_payload = (
        f"SKILL.md content:\n```\n{skill_md}\n```\n\n"
        f"Workflow archetypes from project _running.md (excerpt):\n```\n{archetypes}\n```\n\n"
        f"Project CLAUDE.md (excerpt):\n```\n{claude_excerpt}\n```\n\n"
        f"Generate exactly {n} question-answer pairs as a JSON array, no prose around it."
    )

    raw = _proposer_call(system_prompt, user_payload, proposer_model, api_key)
    if raw is None:
        return []
    arr = _extract_json_array(raw)
    if arr is None:
        # One retry pass with stricter framing.
        retry_payload = user_payload + "\n\nReturn ONLY a JSON array. No prose. No markdown fences."
        raw = _proposer_call(system_prompt, retry_payload, proposer_model, api_key)
        arr = _extract_json_array(raw or "")
    if not arr:
        return []

    # Filter / normalise.
    out: list[dict] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        r = (item.get("rubric") or "").strip()
        t = (item.get("type") or "procedural_qa").strip()
        lbd = (item.get("load_bearing_detail") or "").strip()
        if not q or not a or not r:
            continue
        if t not in ("script_gen", "procedural_qa", "fact_recall"):
            t = "procedural_qa"
        out.append(
            {
                "question": q,
                "answer": a,
                "rubric": r,
                "type": t,
                "load_bearing_detail": lbd,
            }
        )
    return out


# ─── Step 2: calibration ───────────────────────────────────────────────────


def calibrate_synthetic(
    questions: list[dict],
    skill_dir: Path,
    weak_model: str,
    judge_model: str,
    api_key: str,
    seed: int,
    n_rollouts: int = 3,
    max_workers: int = 4,
    log_path: Path | None = None,
    skill_slug: str = "",
) -> list[dict]:
    """Score each question with weak-model + EMPTY skill; keep middle band.

    Wider keep-band than corpus extraction (0.05-0.80). Runs in parallel
    (max_workers threads) so N questions don't take N×3×rollout_time.
    """
    empty_bundle = skill_dir / "_synth_empty_bundle_placeholder"

    def _cal_one(args: tuple[int, dict]) -> dict | None:
        i, q = args
        prompt = q.get("question") or ""
        rubric = q.get("rubric") or ""
        if not prompt or not rubric:
            return None
        baseline = calibrate_eval(
            prompt=prompt,
            rubric=rubric,
            bundle_dir=empty_bundle,
            weak_model=weak_model,
            api_key=api_key,
            judge_model=judge_model,
            seed=seed + i,
            n_rollouts=n_rollouts,
        )
        label = "KEPT" if 0.05 <= baseline <= 0.80 else f"discard (score={baseline:.2f})"
        if log_path:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(f"- [{skill_slug}] q{i}: {label} | {prompt[:60]!r}\n")
        print(f"  [{skill_slug}] q{i}: score={baseline:.2f} → {label}", file=sys.stderr)
        if 0.05 <= baseline <= 0.80:
            enriched = dict(q)
            enriched["baseline_score"] = baseline
            return enriched
        return None

    survivors: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_cal_one, (i, q)): i for i, q in enumerate(questions)}
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                survivors.append(result)
    return survivors


# ─── Step 3: top-level orchestrator ────────────────────────────────────────


def _make_synth_id(question: str) -> str:
    """Stable 12-char id (sha256 prefix) keyed off the question text."""
    h = hashlib.sha256(question.encode("utf-8", errors="replace"))
    return h.hexdigest()[:12]


def _type_for_split(qtype: str) -> str:
    """Map fact_recall → procedural_qa for the downstream stratifier.

    The eval_builder split/dedup code only knows about
    script_gen / skill_invoke / procedural_qa — synthetic fact_recall
    rows ride along with procedural_qa for those mechanics.
    """
    if qtype == "fact_recall":
        return "procedural_qa"
    if qtype in ("script_gen", "procedural_qa", "skill_invoke"):
        return qtype
    return "procedural_qa"


def _write_extraction_log(
    log_path: Path,
    per_skill_counts: list[tuple[str, int, int]],
    total_before_dedup: int,
    total_train: int,
    total_holdout: int,
) -> None:
    """Append a synth_extraction_log.md describing generation + calibration."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# synth_extraction_log\n")
    lines.append("## Per-skill counts\n")
    lines.append("| slug | generated | calibrated |")
    lines.append("|------|-----------|------------|")
    for slug, gen, cal in per_skill_counts:
        lines.append(f"| {slug} | {gen} | {cal} |")
    lines.append("")
    lines.append("## Totals\n")
    lines.append(f"- pre-dedup combined calibrated: {total_before_dedup}")
    lines.append(f"- train: {total_train}")
    lines.append(f"- holdout: {total_holdout}")
    lines.append("")
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def run_synth_eval_build(
    bundle_dir: Path,
    watchmen_home: Path,
    project: str,
    weak_model: str,
    proposer_model: str,
    judge_model: str,
    api_key: str,
    seed: int,
    n_per_skill: int = 20,
    run_dir: Path | None = None,
    max_workers: int = 4,
) -> tuple[list[dict], list[dict]]:
    """Generate + calibrate + split synthetic evals across every skill in a bundle.

    bundle_dir is the project bundle root (e.g. ``~/.watchmen/bundles/ctf``);
    we iterate over ``bundle_dir/skills/<slug>/`` for each skill.
    """
    if run_dir is None:
        run_dir = bundle_dir / "_synth_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Read project-level context once.
    running_md_path = watchmen_home / "analyses" / project / "_running.md"
    claude_md_path = watchmen_home / "bundles" / project / "CLAUDE.md"
    try:
        running_md_text = running_md_path.read_text(encoding="utf-8", errors="replace") if running_md_path.exists() else ""
    except OSError:
        running_md_text = ""
    try:
        claude_md_text = claude_md_path.read_text(encoding="utf-8", errors="replace") if claude_md_path.exists() else ""
    except OSError:
        claude_md_text = ""

    skills_root = bundle_dir / "skills"
    per_skill_counts: list[tuple[str, int, int]] = []
    combined: list[dict] = []

    if not skills_root.exists():
        raise ValueError(f"no skills dir under {bundle_dir}")

    for skill_dir in sorted(skills_root.iterdir()):
        if not skill_dir.is_dir():
            continue
        if not (skill_dir / "SKILL.md").exists():
            continue
        slug = skill_dir.name

        questions = generate_questions_for_skill(
            skill_dir=skill_dir,
            running_md_text=running_md_text,
            claude_md_text=claude_md_text,
            proposer_model=proposer_model,
            api_key=api_key,
            n=n_per_skill,
            seed=seed,
        )

        log_path = run_dir / "synth_extraction_log.md"
        print(f"[synth] calibrating {len(questions)} questions for {slug}…", file=sys.stderr)
        calibrated = calibrate_synthetic(
            questions=questions,
            skill_dir=skill_dir,
            weak_model=weak_model,
            judge_model=judge_model,
            api_key=api_key,
            seed=seed,
            n_rollouts=3,
            max_workers=max_workers,
            log_path=log_path,
            skill_slug=slug,
        )
        print(f"[synth] {slug}: {len(calibrated)}/{len(questions)} survived calibration", file=sys.stderr)

        per_skill_counts.append((slug, len(questions), len(calibrated)))

        # Convert to eval-row format.
        for q in calibrated:
            question = q.get("question") or ""
            answer = q.get("answer") or ""
            rubric = q.get("rubric") or ""
            qtype_raw = q.get("type") or "procedural_qa"
            qtype = _type_for_split(qtype_raw)
            eid = _make_synth_id(question)
            combined.append(
                {
                    "id": eid,
                    # split assigned later by build_eval_set's _stratified_split
                    "type": qtype,
                    "prompt": question,
                    "anonymized_prompt": question,
                    "reference": answer,
                    "anonymized_reference": answer,
                    "rubric": rubric,
                    "anonymized_rubric": rubric,
                    "baseline_score": float(q.get("baseline_score") or 0.0),
                    "baseline_completion_len_tokens": 0,
                    "accepted": True,
                    "source_session": None,
                    "source_skill": slug,
                    # carry-through for downstream debugging (ignored by split/dedup)
                    "synth_type_raw": qtype_raw,
                    "load_bearing_detail": q.get("load_bearing_detail", ""),
                }
            )

    if not combined:
        raise ValueError("synth generation yielded zero calibrated questions")

    # Hand off to build_eval_set for dedup + stratified split + jsonl write.
    # build_eval_set re-anonymizes via the AnonymizeContext, but for synthetic
    # questions we already pre-set anonymized_* == raw. Build a minimal context.
    from .anonymize import build_context

    projects_json_path = watchmen_home / "projects.json"
    try:
        source_repo = ""
        if projects_json_path.exists():
            raw_proj = json.loads(projects_json_path.read_text(encoding="utf-8"))
            if isinstance(raw_proj, list):
                for entry in raw_proj:
                    if isinstance(entry, dict) and entry.get("project_key") == project:
                        source_repo = entry.get("source_repo") or ""
                        break
            elif isinstance(raw_proj, dict):
                entry = raw_proj.get(project) or {}
                source_repo = entry.get("source_repo") or "" if isinstance(entry, dict) else ""
    except Exception:  # noqa: BLE001
        source_repo = ""

    ctx = build_context(projects_json_path, bundle_dir, source_repo)

    # build_eval_set runs semantic_dedup, which requires ≥30 clusters; on
    # small synthetic sets we may not hit that. Bypass by writing directly
    # if combined < 60; otherwise reuse build_eval_set.
    try:
        train, holdout = build_eval_set(combined, seed, ctx, run_dir)
    except ValueError as exc:
        # Fall back: simple 50/50 stratified split without dedup.
        if "insufficient_distillable_surface" in str(exc) or "insufficient_stratification" in str(exc):
            train, holdout = _simple_split(combined, seed)
            out_path = run_dir / "eval_set.jsonl"
            with out_path.open("w", encoding="utf-8") as fh:
                for e in train + holdout:
                    fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        else:
            raise

    _write_extraction_log(
        run_dir / "synth_extraction_log.md",
        per_skill_counts,
        total_before_dedup=len(combined),
        total_train=len(train),
        total_holdout=len(holdout),
    )

    return train, holdout


def _simple_split(evals: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    """Fallback 50/50 split when dedup-gate fails (small synthetic sets).

    Avoids the spec's ≥30 cluster requirement for the synthetic path, which
    is intentionally allowed to run on tiny eval sets during bootstrap.
    """
    import random as _rand

    rng = _rand.Random(seed)
    by_stratum: dict[tuple, list[dict]] = {}
    for e in evals:
        key = (e.get("type"), bool(e.get("accepted")))
        by_stratum.setdefault(key, []).append(e)
    train: list[dict] = []
    holdout: list[dict] = []
    for items in by_stratum.values():
        shuffled = list(items)
        rng.shuffle(shuffled)
        cut = len(shuffled) // 2
        holdout.extend(shuffled[:cut])
        train.extend(shuffled[cut:])
    for e in train:
        e["split"] = "train"
    for e in holdout:
        e["split"] = "holdout"
    return train, holdout
