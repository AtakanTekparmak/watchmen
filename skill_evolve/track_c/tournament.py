"""A/B/AB autoreason tournament as a MAP-Elites mutation operator.

For a given parent :class:`Program` (a Track-B archive entry):

* **A** = the parent itself (its folder + its known composite score).
* **B** = Track A's hybrid mutation — Critic → pick_op → apply_op.
* **AB** = Track A's synthesizer fed A and B in shuffled order.

We evaluate B and AB (A is already scored in the archive) and run the
tournament tiebreak ``(score, label == "A")`` — exactly the rule from
:mod:`skill_evolve.track_a.runner`. ``A`` wins any tie. The tournament's
*winner* is the label with the highest score (with the tiebreak), while
the *archive-insertions* include every valid non-A candidate regardless
of who won.

Design decision — INSERT-ALL-VALID, not winner-only
---------------------------------------------------

When the brief offered a choice between "insert only the tournament
winner" and "insert every valid candidate (B, AB) that survived
validation + evaluation", we went with **insert-all-valid**. Rationale:

* MAP-Elites' whole job is cell coverage across a behavioral grid.
  Dropping an evaluated candidate because it happened to lose *this
  parent's tournament* is wasteful — that candidate's
  (num_skills, total_tokens, avg_specificity) cell may still be empty,
  and its composite may still be the best-in-cell globally.
* The tournament's role in this hybrid is limited to deciding "was this
  parent's turn *progress*?" — which feeds the saturated-parent
  heuristic below. It is NOT a gate on what enters the archive.
* Track A's "do nothing wins ties" semantic is preserved structurally:
  when A wins, no new program is added for the parent's own cell
  (A is already there with score ``score_A``). We never insert a
  duplicate of A, so "A won" = "no mutation got accepted for that
  parent" = do-nothing.

Saturated-parent heuristic
--------------------------

Each time a parent is sampled and A wins the resulting tournament, we
increment ``metadata["a_wins_streak"]`` on the parent Program. When the
streak reaches ``SATURATION_STREAK`` (2 by default — a direct port of
autoreason's k=2 convergence), we set
``metadata["saturated"] = True``. :mod:`.iteration` and
:mod:`.controller` honor this flag by de-prioritizing saturated
programs in :meth:`~skill_evolve.track_b.openevolve_skills.database.ProgramDatabase.sample_parent`
(re-sampling up to a small number of times if the first draw is
saturated). This is the same "pick parents that haven't been beaten"
heuristic openevolve's parent sampler approximates, just expressed
explicitly via a flag so the cause is legible in the history log.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from skill_evolve.evaluator import EvalResult

from skill_evolve.track_a.folder import SkillDoc, SkillFolder, make_skill_doc
from skill_evolve.track_a.llm import LLMClient as TrackALLMClient
from skill_evolve.track_a.ops import OpRecord, apply_op, pick_op
from skill_evolve.track_a.synth import synthesize
from skill_evolve.track_a.validate import validates

from skill_evolve.track_b.openevolve_skills.database import (
    Program,
    new_program_id,
)
from skill_evolve.track_b.openevolve_skills.evaluator import (
    EvaluationResult,
    SkillFolderEvaluator,
)
from skill_evolve.track_b.openevolve_skills.folder_artifact import FolderArtifact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------

SATURATION_STREAK: int = 2
NEG_INF = -math.inf


# ---------------------------------------------------------------------------
# Tournament result container
# ---------------------------------------------------------------------------

@dataclass
class TournamentResult:
    """Return value of :func:`tournament_mutate`.

    ``winner_label`` is one of ``"A" | "B" | "AB"``. ``new_programs`` are
    the Program objects to insert into the MAP-Elites archive — A is
    never in this list (it's already in the archive).
    """

    parent_id: str
    winner_label: str
    winner_score: float
    score_A: float
    score_B: float
    score_AB: float
    op: Optional[OpRecord]
    b_valid: bool
    ab_valid: bool
    new_programs: List[Program] = field(default_factory=list)
    notes: str = ""

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "parent_id": self.parent_id,
            "winner_label": self.winner_label,
            "winner_score": self.winner_score,
            "score_A": self.score_A,
            "score_B": (None if math.isinf(self.score_B) else self.score_B),
            "score_AB": (None if math.isinf(self.score_AB) else self.score_AB),
            "op": (self.op.to_dict() if self.op else None),
            "b_valid": self.b_valid,
            "ab_valid": self.ab_valid,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# SkillFolder <-> FolderArtifact adapters
# ---------------------------------------------------------------------------
#
# Track A works with SkillFolder (list[SkillDoc] + source_path); Track B
# works with FolderArtifact (dict[path, content]). The two representations
# agree exactly on the on-disk layout (``<skill>/SKILL.md``), so we can
# convert losslessly in-memory without going through the filesystem.

def artifact_to_skill_folder(artifact: FolderArtifact) -> SkillFolder:
    """Build a :class:`SkillFolder` from a :class:`FolderArtifact`.

    Non-SKILL.md files are ignored; Track A's representation only knows
    about skill subfolders. This matches what
    :meth:`SkillFolder.load` would see on disk.
    """
    docs: List[SkillDoc] = []
    for name in artifact.skill_names():
        path = f"{name}/SKILL.md"
        content = artifact.files.get(path, "")
        docs.append(_parse_via_skill_folder(name, content))
    return SkillFolder(skills=docs)


def skill_folder_to_artifact(folder: SkillFolder) -> FolderArtifact:
    """Serialize a :class:`SkillFolder` into a :class:`FolderArtifact`.

    Emits one ``<folder_name>/SKILL.md`` per skill. No extra files are
    produced.
    """
    files: Dict[str, str] = {}
    for doc in folder.skills:
        files[f"{doc.folder_name}/SKILL.md"] = doc.render()
    return FolderArtifact(files=files)


def _parse_via_skill_folder(folder_name: str, content: str) -> SkillDoc:
    # Reuse the Track A parser without touching its private API by round-
    # tripping through a one-entry SkillFolder.
    from skill_evolve.track_a.folder import _parse_skill_md  # type: ignore
    return _parse_skill_md(folder_name, content)


# ---------------------------------------------------------------------------
# EvalResult reconstruction from a Program's metrics + eval_artifacts
# ---------------------------------------------------------------------------

def eval_result_from_program(program: Program) -> EvalResult:
    """Rebuild a Track-A-flavoured :class:`EvalResult` from a Program.

    Track A's critic / pick_op / rewrite helpers want an ``EvalResult``
    (``.failures``, ``.per_task``, ``.composite``, ``.n_tasks``, ...).
    Track B stores the same info split between ``program.metrics`` and
    ``program.eval_artifacts`` (JSON blobs). We stitch them back together
    so we can feed Track A's functions without extra evaluator calls.
    """
    m = program.metrics or {}
    a = program.eval_artifacts or {}

    failures_blob = a.get("failures", "[]")
    per_task_blob = a.get("per_task", "[]")
    try:
        failures = json.loads(failures_blob) if failures_blob else []
    except json.JSONDecodeError:
        failures = []
    try:
        per_task = json.loads(per_task_blob) if per_task_blob else []
    except json.JSONDecodeError:
        per_task = []

    return EvalResult(
        success_rate=float(m.get("success_rate", 0.0)),
        tool_calls_per_success=float(m.get("tool_calls_per_success", 0.0)),
        composite=float(m.get("composite", 0.0)),
        per_task=per_task,
        failures=failures,
        skills_folder="",
        n_tasks=int(m.get("n_tasks", len(per_task))),
        cascade_truncated=bool(int(a.get("cascade_truncated", "0") or 0)),
        synthetic=bool(int(a.get("synthetic", "0") or 0)),
        notes="reconstructed from Program artifacts",
        verified_count=int(m.get("verified_count", 0)),
        unverified_count=int(m.get("unverified_count", 0)),
    )


# ---------------------------------------------------------------------------
# Feedback-rendering helpers (parallels runner._format_failures_for_critic
# but inlined so track_a stays untouched).
# ---------------------------------------------------------------------------

def _format_failures_for_critic(ev: EvalResult) -> str:
    lines: List[str] = []
    lines.append(
        f"success_rate={ev.success_rate:.3f} "
        f"composite={ev.composite:.4f} "
        f"n_tasks={ev.n_tasks}"
    )
    if ev.synthetic:
        lines.append("(synthetic evaluator result — treat as plumbing-only signal)")
    if ev.failures:
        lines.append("Failed tasks (task_id — last_msg):")
        for f in ev.failures[:8]:
            msg = (f.get("last_msg") or "").replace("\n", " ")[:200]
            lines.append(f"  - {f.get('task_id')}: {msg}")
    attrib: Dict[str, int] = {}
    for p in ev.per_task or []:
        if p.get("success"):
            continue
        for sk in p.get("skills_invoked") or []:
            attrib[sk] = attrib.get(sk, 0) + 1
    if attrib:
        lines.append("Skill invocations on failing tasks:")
        for sk, c in sorted(attrib.items(), key=lambda kv: -kv[1]):
            lines.append(f"  - {sk}: {c}")
    return "\n".join(lines)


def _attributed_failures_for_skill(ev: EvalResult, skill_name: str) -> str:
    lines: List[str] = []
    for p in ev.per_task or []:
        if p.get("success"):
            continue
        if skill_name in (p.get("skills_invoked") or []):
            msg = (p.get("last_msg") or p.get("notes") or "").replace("\n", " ")[:240]
            lines.append(f"- {p.get('task_id')}: INVOKED but still failed — {msg}")
    if not lines:
        return "(no failures were attributed to this skill)"
    return "\n".join(lines)


def _flat_failures(ev: EvalResult) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in ev.per_task or []:
        if not p.get("success"):
            out.append({
                "task_id": p.get("task_id"),
                "skills_invoked": p.get("skills_invoked") or [],
            })
    return out


# ---------------------------------------------------------------------------
# Tournament
# ---------------------------------------------------------------------------

def _critic_call(
    A: SkillFolder,
    last_eval: EvalResult,
    client: TrackALLMClient,
) -> str:
    from skill_evolve.track_a.prompts import (
        CRITIC_PROMPT,
        CRITIC_SYSTEM,
        DESCRIBE_GOAL,
    )
    return client.complete(
        CRITIC_SYSTEM,
        CRITIC_PROMPT.format(
            goal=DESCRIBE_GOAL,
            folder=A.render_summary(body_chars=1200),
            failures=_format_failures_for_critic(last_eval),
        ),
        tag="critic",
        max_tokens=1500,
    )


def _propose_b(
    A: SkillFolder,
    critique: str,
    last_eval: EvalResult,
    client: TrackALLMClient,
    *,
    max_retries: int = 3,
) -> Tuple[Optional[OpRecord], Optional[SkillFolder], str]:
    """Mirror of :func:`skill_evolve.track_a.runner._propose_b`, inlined so
    Track C doesn't reach into runner's private helper."""
    flat = _flat_failures(last_eval)
    op_record: Optional[OpRecord] = None
    last_err = ""
    for attempt in range(1, max_retries + 1):
        op_record = pick_op(A, critique, client=client, last_failures=flat)
        skill_failures = ""
        if op_record.op == "RewriteSkillContent":
            skill_failures = _attributed_failures_for_skill(
                last_eval, op_record.args.get("name", "")
            )
        try:
            B = apply_op(A, op_record, client,
                         critique=critique, skill_failures=skill_failures)
        except Exception as e:
            last_err = f"apply_op raised: {e!r}"
            logger.warning("track_c propose_b attempt %d: %s", attempt, last_err)
            continue
        if validates(B):
            return op_record, B, ""
        last_err = f"B invalid after op {op_record.op}"
        logger.warning("track_c propose_b attempt %d: %s", attempt, last_err)
    return op_record, None, last_err


def _tiebreak_winner(score_a: float, score_b: float, score_ab: float) -> str:
    """Track A's tiebreak: ``A`` wins any tie; else ``B`` beats ``AB``."""
    candidates: List[Tuple[float, int, str]] = [
        (score_a, 2, "A"),
        (score_b, 1, "B"),
        (score_ab, 0, "AB"),
    ]
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return candidates[0][2]


def tournament_mutate(
    parent: Program,
    evaluator: SkillFolderEvaluator,
    client: TrackALLMClient,
    *,
    generation: int,
    rng: Optional[random.Random] = None,
) -> TournamentResult:
    """Generate B and AB from ``parent``, score them, declare a winner.

    Returns a :class:`TournamentResult` including the Program objects to
    insert into the archive (B and AB if valid — A is already there).
    """
    rng = rng or random.Random()

    # --- Rebuild Track-A-flavored eval context ----------------------------
    last_eval = eval_result_from_program(parent)
    score_A = float(parent.fitness())

    # --- A: wrap parent's artifact in a SkillFolder -----------------------
    A_folder = artifact_to_skill_folder(parent.artifact)
    if not validates(A_folder):
        # Parent is pathological — no tournament is meaningful. Treat as A-win.
        logger.warning("tournament: parent %s did not validate as SkillFolder",
                       parent.id[:8])
        return TournamentResult(
            parent_id=parent.id,
            winner_label="A", winner_score=score_A,
            score_A=score_A, score_B=NEG_INF, score_AB=NEG_INF,
            op=None, b_valid=False, ab_valid=False,
            notes="parent did not validate as SkillFolder",
        )

    # --- Critic -----------------------------------------------------------
    critique = _critic_call(A_folder, last_eval, client)

    # --- B via Track A's op pipeline --------------------------------------
    op_record, B_folder, b_notes = _propose_b(A_folder, critique, last_eval, client)
    b_valid = B_folder is not None

    # --- AB via Track A's synthesizer -------------------------------------
    AB_folder: Optional[SkillFolder] = None
    if B_folder is not None:
        AB_folder = synthesize(A_folder, B_folder, client, rng=rng)
    ab_valid = AB_folder is not None

    # --- Score B and AB (A already scored) --------------------------------
    score_B = NEG_INF
    score_AB = NEG_INF
    b_metrics: Dict[str, float] = {}
    b_artifacts: Dict[str, str] = {}
    ab_metrics: Dict[str, float] = {}
    ab_artifacts: Dict[str, str] = {}

    B_artifact: Optional[FolderArtifact] = None
    AB_artifact: Optional[FolderArtifact] = None

    if B_folder is not None:
        try:
            B_artifact = skill_folder_to_artifact(B_folder)
            ev_b: EvaluationResult = evaluator.evaluate_artifact(B_artifact)
            b_metrics = ev_b.metrics
            b_artifacts = ev_b.artifacts
            score_B = float(b_metrics.get("composite", 0.0))
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("tournament: eval(B) failed: %s", e)
            b_valid = False
            B_artifact = None

    if AB_folder is not None:
        try:
            AB_artifact = skill_folder_to_artifact(AB_folder)
            ev_ab: EvaluationResult = evaluator.evaluate_artifact(AB_artifact)
            ab_metrics = ev_ab.metrics
            ab_artifacts = ev_ab.artifacts
            score_AB = float(ab_metrics.get("composite", 0.0))
        except Exception as e:  # pragma: no cover
            logger.warning("tournament: eval(AB) failed: %s", e)
            ab_valid = False
            AB_artifact = None

    # --- Tournament (A wins ties; B beats AB on score tie) ---------------
    winner_label = _tiebreak_winner(score_A, score_B, score_AB)
    if winner_label == "A":
        winner_score = score_A
    elif winner_label == "B":
        winner_score = score_B
    else:
        winner_score = score_AB

    # --- Update saturated-parent bookkeeping on the PARENT Program -------
    # (The controller may also re-check this via sample_parent.)
    meta = parent.metadata
    if winner_label == "A":
        meta["a_wins_streak"] = int(meta.get("a_wins_streak", 0)) + 1
    else:
        meta["a_wins_streak"] = 0
    if meta["a_wins_streak"] >= SATURATION_STREAK:
        meta["saturated"] = True
    else:
        meta["saturated"] = meta.get("saturated", False)

    # --- Assemble new Programs to insert into the archive ----------------
    # INSERT-ALL-VALID: we insert B and AB regardless of who won. A is
    # already in the archive with score_A; not inserting a duplicate is
    # exactly the "do-nothing wins ties" semantic from Track A.
    new_programs: List[Program] = []

    if b_valid and B_artifact is not None:
        new_programs.append(Program(
            id=new_program_id(),
            artifact=B_artifact,
            parent_id=parent.id,
            generation=parent.generation + 1,
            iteration_found=generation,
            metrics=b_metrics,
            eval_artifacts=b_artifacts,
            metadata={
                "tournament_role": "B",
                "tournament_winner": winner_label,
                "op": (op_record.op if op_record else None),
                "a_wins_streak": 0,
                "saturated": False,
            },
        ))

    if ab_valid and AB_artifact is not None:
        new_programs.append(Program(
            id=new_program_id(),
            artifact=AB_artifact,
            parent_id=parent.id,
            generation=parent.generation + 1,
            iteration_found=generation,
            metrics=ab_metrics,
            eval_artifacts=ab_artifacts,
            metadata={
                "tournament_role": "AB",
                "tournament_winner": winner_label,
                "op": (op_record.op if op_record else None),
                "a_wins_streak": 0,
                "saturated": False,
            },
        ))

    return TournamentResult(
        parent_id=parent.id,
        winner_label=winner_label,
        winner_score=winner_score,
        score_A=score_A,
        score_B=score_B,
        score_AB=score_AB,
        op=op_record,
        b_valid=b_valid,
        ab_valid=ab_valid,
        new_programs=new_programs,
        notes=b_notes if not b_valid else "",
    )
