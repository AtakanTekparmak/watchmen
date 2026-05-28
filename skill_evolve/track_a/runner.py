"""Track A runner — the autoreason-style A/B/AB evolution loop.

Loop per pass:

  1. Critic surfaces concrete problems with the incumbent A (no fixes).
  2. Planner picks one structural op → apply to A to produce B.
  3. Synthesizer merges A and B into AB.
  4. Score A, B, AB via :func:`skill_evolve.evaluator.evaluate`.
  5. Greedy-accept the max composite; "do nothing" wins ties via
     ``(score, label == "A")`` lexicographic max.
  6. Streak >= 2 consecutive "A wins" → convergence, break.

Artifacts per run:

  runs/<id>/history.json                 — full pass log
  runs/<id>/pass_<N>/{A,B,AB}/           — the three folders evaluated
  runs/<id>/final/                       — last A (winning folder)

CLI:

  python -m skill_evolve.track_a.runner --seed seed_skills/ \
      --out runs/track_a_<ts>/ --max-passes 15 \
      [--no-verify] [--force-synthetic] \
      [--outer-model anthropic/claude-sonnet-4.6] \
      [--inner-model minimax/minimax-m2.7]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..evaluator import EvalResult, evaluate, have_live_keys
from .folder import SkillFolder
from .llm import LLMClient
from .ops import OpRecord, apply_op, pick_op
from .prompts import CRITIC_PROMPT, CRITIC_SYSTEM, DESCRIBE_GOAL
from .synth import synthesize
from .validate import validate, validates

logger = logging.getLogger(__name__)


DEFAULT_MAX_PASSES = 15
CONVERGENCE_STREAK = 2
MAX_OP_RETRIES = 3
NEG_INF = -math.inf

# Asymmetric model defaults: Sonnet 4.6 for the outer/meta LLM (critic /
# op-planner / body-writer / synthesizer) and M2.7 for the inner Hermes
# agent rollouts.  See V2_FIXES_PLAN.md (Fix 1).
DEFAULT_OUTER_MODEL = "moonshotai/kimi-k2.6"
DEFAULT_INNER_MODEL = "minimax/minimax-m2.7"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_failures_for_critic(eval_result: Optional[EvalResult]) -> str:
    if eval_result is None:
        return "(no prior evaluation — first pass)"
    lines: List[str] = []
    lines.append(
        f"success_rate={eval_result.success_rate:.3f} "
        f"composite={eval_result.composite:.4f} "
        f"n_tasks={eval_result.n_tasks}"
    )
    if eval_result.synthetic:
        lines.append("(synthetic evaluator result — treat as plumbing-only signal)")
    if eval_result.failures:
        lines.append("Failed tasks (task_id — last_msg):")
        for f in eval_result.failures[:8]:
            msg = (f.get("last_msg") or "").replace("\n", " ")[:200]
            lines.append(f"  - {f['task_id']}: {msg}")
    # Per-skill invocation counts attributable to failures.
    per_task = eval_result.per_task or []
    attrib: Dict[str, int] = {}
    for p in per_task:
        if p.get("success"):
            continue
        for sk in p.get("skills_invoked") or []:
            attrib[sk] = attrib.get(sk, 0) + 1
    if attrib:
        lines.append("Skill invocations on failing tasks:")
        for sk, c in sorted(attrib.items(), key=lambda kv: -kv[1]):
            lines.append(f"  - {sk}: {c}")
    return "\n".join(lines)


def _attributed_failures_for_skill(
    eval_result: Optional[EvalResult],
    skill_name: str,
) -> str:
    if eval_result is None:
        return "(no prior evaluation)"
    lines: List[str] = []
    for p in eval_result.per_task or []:
        if p.get("success"):
            continue
        invoked = p.get("skills_invoked") or []
        if skill_name in invoked:
            msg = (p.get("last_msg") or p.get("notes") or "").replace("\n", " ")[:240]
            lines.append(f"- {p['task_id']}: INVOKED but still failed — {msg}")
    if not lines:
        return "(no failures were attributed to this skill)"
    return "\n".join(lines)


def _score_of(result: Optional[EvalResult]) -> float:
    if result is None:
        return NEG_INF
    return result.composite


def _tiebreak_winner(
    score_a: float,
    score_b: float,
    score_ab: float,
) -> str:
    """Greedy max with "do nothing wins ties" (A is first-class).

    Priority on ties (highest -> lowest): **A > B > AB**.

    Implemented as lex-max of ``(score, priority)`` where A gets priority
    ``2``, B gets ``1``, AB gets ``0``. Because tuple comparison is
    lexicographic, two candidates with identical ``score`` are broken by
    ``priority``, so a true 3-way tie returns ``"A"`` and a 2-way tie
    between B and AB (both equal, both above A) returns ``"B"``.

    Precision caveat
    ----------------
    This function compares the *raw* floats it receives — no rounding or
    epsilon window. If two candidates' composite scores differ by even
    ``1e-12`` the larger one wins outright and the priority flag never
    applies. This can look like an inverted tie in logs that display
    rounded scores (e.g. history.json showing ``A=B=AB=0.45`` when the
    actual floats were ``0.325, 0.45, 0.45`` and B legitimately beat A).

    Note that in the pass loop the ``score_A`` logged in history.json is
    written **after** ``score_A`` may have been replaced by the winner's
    score, which is why rounded log values can appear to contradict the
    tiebreak outcome. See ``tests/test_tiebreak.py`` for the exact
    semantics under test.
    """
    candidates: List[Tuple[float, int, str]] = [
        (score_a, 2, "A"),  # A's tie-flag is 2 (highest)
        (score_b, 1, "B"),
        (score_ab, 0, "AB"),
    ]
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return candidates[0][2]


# ---------------------------------------------------------------------------
# Cost-warning helper
# ---------------------------------------------------------------------------


def _print_cost_warning(max_passes: int, verify: bool) -> None:
    # Rough order-of-magnitude guidance; the real cost depends on model
    # choice and task count. We just want the operator to know this isn't free.
    per_pass_lo = 0.50  # USD, synthetic shouldn't hit this path
    per_pass_hi = 5.00
    lo = max_passes * per_pass_lo
    hi = max_passes * per_pass_hi
    print("─" * 70, file=sys.stderr)
    print("Track A — REAL evaluator run", file=sys.stderr)
    print(f"  max_passes      : {max_passes}", file=sys.stderr)
    print(f"  verify (docker) : {verify}", file=sys.stderr)
    print(
        f"  est. cost range : ~${lo:.2f}–${hi:.2f} USD "
        "(per-pass: 3× evaluate() + ~3 LLM author/critic/plan calls)",
        file=sys.stderr,
    )
    print("  set --force-synthetic to dry-run with no LLM spend.", file=sys.stderr)
    print("─" * 70, file=sys.stderr)


# ---------------------------------------------------------------------------
# Pass result container
# ---------------------------------------------------------------------------


def _pass_record(
    pass_idx: int,
    winner: str,
    score_a: float,
    score_b: float,
    score_ab: float,
    op: Optional[OpRecord],
    streak: int,
    notes: str = "",
    b_valid: bool = True,
    ab_valid: bool = True,
) -> Dict[str, Any]:
    return {
        "pass": pass_idx,
        "winner": winner,
        "score_A": score_a,
        "score_B": score_b,
        "score_AB": (score_ab if not math.isinf(score_ab) else None),
        "op": (op.to_dict() if op else None),
        "streak": streak,
        "b_valid": b_valid,
        "ab_valid": ab_valid,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------


def _copy_seed_bytes(seed: Path, dest: Path) -> Path:
    """Byte-preserving copy of the seed folder into ``dest``.

    Track A keeps an in-memory :class:`SkillFolder` for mutations, but
    its ``write()`` path re-renders every SKILL.md through
    ``yaml.safe_dump``. That mangles frontmatter (description wrapping,
    inline-list expansion) and drops root-level files like INDEX.md —
    giving Track A a handicapped seed relative to Tracks B/C which
    ship byte-preserving ``FolderArtifact`` I/O.

    To keep the initial A comparable across tracks we ``copytree`` the
    seed here. Mutations (B, AB) continue to go through
    ``SkillFolder.write`` — those are produced in-memory by ops and must
    be serialized — so within-track deltas stay consistent.
    """
    dest = Path(dest).expanduser().resolve()
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(seed, dest)
    return dest


def run_loop(
    seed: Path,
    out: Path,
    *,
    max_passes: int = DEFAULT_MAX_PASSES,
    force_synthetic: bool = False,
    verify: bool = True,
    outer_model: str = DEFAULT_OUTER_MODEL,
    inner_model: str = DEFAULT_INNER_MODEL,
    max_workers: int = 1,
    repeats: int = 3,
    sources: Optional[List[str]] = None,
    rng_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute the A/B/AB loop; return the final history dict.

    Args:
        outer_model: model slug for the meta/outer LLM (critic, op-planner,
            body-writer, synthesizer) — threaded into :class:`LLMClient`.
        inner_model: model slug for the inner Hermes agent rollouts —
            threaded into :func:`skill_evolve.evaluator.evaluate` which
            passes it to ``run_agent.py --model=...``.
    """
    rng = random.Random(rng_seed) if rng_seed is not None else random.Random()

    out = Path(out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # -- load incumbent A -----------------------------------------------
    # We keep the SkillFolder in memory for mutation, but write the INITIAL
    # seed to disk byte-identically (see _copy_seed_bytes docstring for why).
    A = SkillFolder.load(seed)
    ok, errs = validate(A)
    if not ok:
        raise ValueError(f"seed folder does not validate: {errs}")

    client = LLMClient(model=outer_model, synthetic=force_synthetic)

    # Track whether A has been replaced by a mutation. If not, we can keep
    # the on-disk "final" folder byte-identical to the seed (same rationale
    # as Fix 1 for pass_0/A).
    a_is_seed = True

    # -- initial evaluation --------------------------------------------
    a_dir = out / "pass_0" / "A"
    _copy_seed_bytes(Path(seed), a_dir)
    eval_A = evaluate(
        a_dir,
        cascade=True,
        max_workers=max_workers,
        verify=verify,
        force_synthetic=force_synthetic,
        sources=sources,
        model=inner_model,
        repeats=repeats,
    )
    score_A = eval_A.composite
    last_failures = eval_A

    history: Dict[str, Any] = {
        "seed": str(Path(seed).resolve()),
        "out": str(out),
        "max_passes": max_passes,
        "force_synthetic": force_synthetic,
        "verify": verify,
        "outer_model": outer_model,
        "inner_model": inner_model,
        "passes": [
            {
                "pass": 0,
                "winner": "seed",
                "score_A": score_A,
                "score_B": None,
                "score_AB": None,
                "op": None,
                "streak": 0,
                "notes": "initial A evaluation",
                "eval_A_summary": _eval_summary(eval_A),
            }
        ],
        "converged": False,
        "final_score": score_A,
    }

    _dump_history(out, history)

    streak = 0

    # -- passes ---------------------------------------------------------
    for p in range(1, max_passes + 1):
        logger.info(
            "=== pass %d / %d (incumbent score=%.4f, streak=%d) ===",
            p,
            max_passes,
            score_A,
            streak,
        )

        pass_dir = out / f"pass_{p}"
        pass_dir.mkdir(parents=True, exist_ok=True)

        # 1. Critic
        critique = client.complete(
            CRITIC_SYSTEM,
            CRITIC_PROMPT.format(
                goal=DESCRIBE_GOAL,
                folder=A.render_summary(body_chars=1200),
                failures=_format_failures_for_critic(last_failures),
            ),
            tag="critic",
            max_tokens=1500,
        )
        (pass_dir / "critique.md").write_text(critique, encoding="utf-8")

        # 2. Propose B via op + retry
        op_record, B, b_notes = _propose_b(
            A,
            client,
            critique,
            last_failures,
            rng,
        )
        b_valid = B is not None
        if B is None:
            # Skip this pass gracefully — log and continue with streak logic
            # treating "no B" as "A wins this pass".
            logger.warning(
                "pass %d: could not produce valid B after retries — skipping", p
            )
            streak += 1
            history["passes"].append(
                _pass_record(
                    p,
                    "A",
                    score_A,
                    NEG_INF,
                    NEG_INF,
                    op_record,
                    streak,
                    notes=f"no valid B: {b_notes}",
                    b_valid=False,
                )
            )
            _dump_history(out, history)
            if streak >= CONVERGENCE_STREAK:
                history["converged"] = True
                break
            continue

        b_dir = pass_dir / "B"
        B.write(b_dir)

        # 3. Synthesize AB
        AB = synthesize(A, B, client, rng=rng)
        ab_dir: Optional[Path] = None
        if AB is not None:
            ab_dir = pass_dir / "AB"
            AB.write(ab_dir)

        # 4. Score B and AB (A already scored previously)
        eval_B = evaluate(
            b_dir,
            cascade=True,
            max_workers=max_workers,
            verify=verify,
            force_synthetic=force_synthetic,
            sources=sources,
            model=inner_model,
            repeats=repeats,
        )
        score_B = eval_B.composite

        if ab_dir is not None and AB is not None:
            eval_AB = evaluate(
                ab_dir,
                cascade=True,
                max_workers=max_workers,
                verify=verify,
                force_synthetic=force_synthetic,
                sources=sources,
                model=inner_model,
                repeats=repeats,
            )
            score_AB = eval_AB.composite
        else:
            eval_AB = None
            score_AB = NEG_INF

        # 5. Greedy accept, "do nothing wins ties"
        winner = _tiebreak_winner(score_A, score_B, score_AB)

        if winner == "A":
            streak += 1
            new_last_failures = last_failures  # unchanged
        else:
            streak = 0
            if winner == "B":
                A = B
                score_A = score_B
                new_last_failures = eval_B
            else:
                assert AB is not None
                A = AB
                score_A = score_AB
                new_last_failures = eval_AB
            a_is_seed = False
            # persist the chosen folder as the new incumbent path
            A.write(out / "current_A")
        last_failures = new_last_failures

        history["passes"].append(
            {
                "pass": p,
                "winner": winner,
                "score_A": score_A,
                "score_B": score_B,
                "score_AB": None if math.isinf(score_AB) else score_AB,
                "op": op_record.to_dict() if op_record else None,
                "streak": streak,
                "b_valid": b_valid,
                "ab_valid": AB is not None,
                "notes": "",
                "eval_B_summary": _eval_summary(eval_B),
                "eval_AB_summary": _eval_summary(eval_AB),
            }
        )
        history["final_score"] = score_A
        _dump_history(out, history)

        if streak >= CONVERGENCE_STREAK:
            history["converged"] = True
            logger.info(
                "convergence: streak=%d >= %d — stopping.", streak, CONVERGENCE_STREAK
            )
            break

    # -- persist final folder ------------------------------------------
    if a_is_seed:
        # No mutation ever won — keep final byte-identical to the seed.
        _copy_seed_bytes(Path(seed), out / "final")
    else:
        A.write(out / "final")
    history["final_score"] = score_A
    _dump_history(out, history)
    return history


# ---------------------------------------------------------------------------
# B-proposal helper (handles op choice + retry)
# ---------------------------------------------------------------------------


def _propose_b(
    A: SkillFolder,
    client: LLMClient,
    critique: str,
    last_eval: Optional[EvalResult],
    rng: random.Random,
) -> Tuple[Optional[OpRecord], Optional[SkillFolder], str]:
    last_failures = last_eval.failures if last_eval else []
    last_per_task = last_eval.per_task if last_eval else []
    # We pass skills_invoked-on-failure to pick_op's fallback heuristic.
    flat_failures: List[Dict[str, Any]] = []
    for p in last_per_task:
        if not p.get("success"):
            flat_failures.append(
                {
                    "task_id": p["task_id"],
                    "skills_invoked": p.get("skills_invoked") or [],
                }
            )

    last_err = ""
    op_record: Optional[OpRecord] = None
    for attempt in range(1, MAX_OP_RETRIES + 1):
        op_record = pick_op(A, critique, client=client, last_failures=flat_failures)
        # For RewriteSkillContent we enrich skill-specific failure context
        skill_failures = ""
        if op_record.op == "RewriteSkillContent":
            skill_failures = _attributed_failures_for_skill(
                last_eval, op_record.args.get("name", "")
            )
        try:
            B = apply_op(
                A,
                op_record,
                client,
                critique=critique,
                skill_failures=skill_failures,
            )
        except Exception as e:
            last_err = f"apply_op raised: {e!r}"
            logger.warning("propose_b attempt %d: %s", attempt, last_err)
            continue
        if validates(B):
            return op_record, B, ""
        last_err = f"B invalid after op {op_record.op}"
        logger.warning("propose_b attempt %d: %s", attempt, last_err)
    return op_record, None, last_err


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _eval_summary(res: Optional[EvalResult]) -> Optional[Dict[str, Any]]:
    if res is None:
        return None
    return {
        "success_rate": res.success_rate,
        "tool_calls_per_success": res.tool_calls_per_success,
        "composite": res.composite,
        "n_tasks": res.n_tasks,
        "verified_count": res.verified_count,
        "unverified_count": res.unverified_count,
        "synthetic": res.synthetic,
        "cascade_truncated": res.cascade_truncated,
        "failures": res.failures,
    }


def _dump_history(out: Path, history: Dict[str, Any]) -> None:
    path = out / "history.json"
    path.write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_patch_args(parser: argparse.ArgumentParser) -> None:
    """Group A — patch-format + smoke-test flags.

    Lives in its own helper so Group B can add ``_add_eval_source_args``
    to the same ``_cli()`` without textual conflict (plan section 7d).
    """
    parser.add_argument(
        "--patch-format",
        choices=["json-ops", "sentinel-blocks"],
        default="json-ops",
        help=(
            "outer-LLM patch format. ``json-ops`` (default) uses the "
            "existing JSON-op planner; ``sentinel-blocks`` uses the "
            "daycare-style ADD_FILE / EDIT_FILE / DELETE_FILE / "
            "REWRITE_FOLDER format with smoke-test gating."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        dest="smoke_test",
        action="store_true",
        default=True,
        help=(
            "run py_compile/bash -n on every candidate script before "
            "scoring; reject candidates with broken syntax (default on)"
        ),
    )
    parser.add_argument(
        "--no-smoke-test",
        dest="smoke_test",
        action="store_false",
        help="skip syntax validation; candidates are scored as-is",
    )


def _add_eval_source_args(parser: argparse.ArgumentParser) -> None:
    """Group B — --eval-source + behavioral adapter + validation-task-list.

    Symmetric with :func:`_add_patch_args`; lives in its own helper so
    Group A and Group B can both add argparse args to ``_cli()`` without
    textual conflict (plan section 7d). Track A is sentinel-mode by
    config, but exposing the same UX as Track B keeps the two CLIs
    consistent for users who switch between them.
    """
    parser.add_argument(
        "--eval-source",
        choices=("skillsbench", "behavioral"),
        default="skillsbench",
        help=(
            "Scoring backend. ``skillsbench`` (default) preserves the "
            "existing TBLite/SkillsBench agent-harness scoring. "
            "``behavioral`` scores candidates via a judge LLM against "
            "a daycare-format ``eval_set.jsonl`` (requires --eval-set)."
        ),
    )
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=None,
        help=(
            "Path to a daycare-style ``eval_set.jsonl`` (required when "
            "--eval-source=behavioral)."
        ),
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help=(
            "OpenRouter slug for the behavioral judge LLM. Required when "
            "--eval-source=behavioral and no stub is wired in."
        ),
    )
    parser.add_argument(
        "--validation-task-list",
        type=Path,
        default=None,
        help=(
            "Optional held-out task list scored after each accepted "
            "winner (records ``validation_score`` on the artifact)."
        ),
    )


def _cli(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m skill_evolve.track_a.runner",
        description="Autoreason-style A/B/AB evolution loop for skill folders.",
    )
    ap.add_argument(
        "--seed",
        type=Path,
        required=True,
        help="path to the seed skills folder (incumbent A_0)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="run directory for history + per-pass folders",
    )
    ap.add_argument("--max-passes", type=int, default=DEFAULT_MAX_PASSES)
    ap.add_argument(
        "--no-verify",
        action="store_true",
        help="skip real Docker/SWE-bench verification",
    )
    ap.add_argument(
        "--force-synthetic",
        action="store_true",
        help="canned LLM responses + synthetic evaluator (no spend)",
    )
    ap.add_argument(
        "--outer-model",
        default=DEFAULT_OUTER_MODEL,
        help="model slug for the outer/meta LLM (critic / "
        "op-planner / body-writer / synthesizer). "
        f"Default: {DEFAULT_OUTER_MODEL}",
    )
    ap.add_argument(
        "--inner-model",
        default=DEFAULT_INNER_MODEL,
        help="model slug for the inner Hermes agent rollouts "
        "(passed to run_agent.py --model=...). "
        f"Default: {DEFAULT_INNER_MODEL}",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="DEPRECATED alias — sets BOTH --outer-model and "
        "--inner-model. Prefer the split flags.",
    )
    ap.add_argument("--max-workers", type=int, default=1)
    ap.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="k repeat evaluations per candidate for "
        "majority-vote aggregation (default: 1 — no repeats)",
    )
    ap.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="restrict benchmark sources (e.g. tblite swebench)",
    )
    ap.add_argument("--rng-seed", type=int, default=None)
    _add_eval_source_args(ap)
    _add_patch_args(ap)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Handle deprecated --model alias: when set, it overrides BOTH split
    # flags. Emit a warning so pipeline owners transition to the new flags.
    if args.model is not None:
        print(
            "WARNING: --model is deprecated, use --outer-model and --inner-model",
            file=sys.stderr,
        )
        args.outer_model = args.model
        args.inner_model = args.model

    if not args.force_synthetic and have_live_keys():
        _print_cost_warning(args.max_passes, verify=not args.no_verify)
    elif not args.force_synthetic and not have_live_keys():
        print(
            "WARNING: no live LLM keys set. The evaluator will emit "
            "synthetic results and LLMClient will fail on instantiation. "
            "Pass --force-synthetic for a full dry run.",
            file=sys.stderr,
        )

    history = run_loop(
        seed=args.seed,
        out=args.out,
        max_passes=args.max_passes,
        force_synthetic=args.force_synthetic,
        verify=not args.no_verify,
        outer_model=args.outer_model,
        inner_model=args.inner_model,
        max_workers=args.max_workers,
        repeats=args.repeats,
        sources=args.sources,
        rng_seed=args.rng_seed,
    )

    print("─" * 70)
    print(f"final score       : {history['final_score']:.4f}")
    print(f"converged         : {history['converged']}")
    print(f"passes run        : {len(history['passes']) - 1}")
    print(f"history written to: {args.out / 'history.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
