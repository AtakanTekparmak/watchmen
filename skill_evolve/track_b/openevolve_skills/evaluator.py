"""Wrapper that adapts ``skill_evolve.evaluator.evaluate`` to the
openevolve :class:`EvaluationResult` shape.

Responsibilities:

1. Take a :class:`Program`, write its ``FolderArtifact`` to a scratch
   directory, invoke :func:`skill_evolve.evaluator.evaluate`, collect
   the :class:`EvalResult`.
2. Surface the metrics the evolution loop needs to sort/bin programs
   (``composite``, ``success_rate``, ``tool_calls_per_success``).
3. Put failure messages + per-task ``skills_invoked`` into the
   ``artifacts`` channel so the prompt sampler can show them as
   feedback in the next mutation.

Design note: we keep this layer *thin*. All the real work (sandbox,
verifier, Docker, etc.) is inside :mod:`skill_evolve.evaluator`. Our job
here is purely translation.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from skill_evolve import evaluator as _skill_evaluator

from .database import Program
from .folder_artifact import FolderArtifact

logger = logging.getLogger(__name__)


# kai-skills patch (2026-04-27): task-name leakage prevention.
# The K2.6 outer LLM ingests evaluator artifacts (per_task / failures)
# that previously surfaced raw benchmark task IDs like
# "tblite/build-system-task-ordering". The outer model would then bake
# those names verbatim into SKILL.md descriptions; Hermes' router
# matches on description text, so this guarantees routing on those
# specific tasks but doesn't generalize. The anonymization scheme:
# sort the manifest's task IDs alphabetically and assign stable
# task_001, task_002, ... aliases. Both the bare segment
# ("build-system-task-ordering") and the fully-qualified prefix form
# ("tblite/build-system-task-ordering") are redacted.


def _short_name(task_id: str) -> str:
    """Return the bare segment of a task_id (drops "<source>/" prefix)."""
    return task_id.rsplit("/", 1)[-1]


def build_task_id_map(*, sources: Optional[List[str]] = None) -> Dict[str, str]:
    """Build a stable task_id -> ``task_NNN`` mapping for the active manifest.

    Pulls from :func:`skill_evolve.benchmark.load.load_subset` in offline
    mode (no HF lookups; we only need the IDs). The mapping covers both
    the fully-qualified ID ("tblite/foo") and its bare-segment form
    ("foo") so substring sanitization catches either spelling.
    """
    from skill_evolve.benchmark import load_subset as _load_subset

    tasks = _load_subset(offline_only=True, sources=sources)
    ordered = sorted({t["task_id"] for t in tasks})
    mapping: Dict[str, str] = {}
    for i, tid in enumerate(ordered, start=1):
        alias = f"task_{i:03d}"
        mapping[tid] = alias
        bare = _short_name(tid)
        # If two task IDs share the same bare segment (rare), prefer the
        # first encountered alphabetically so the alias is deterministic.
        mapping.setdefault(bare, alias)
    return mapping


def sanitize_text(text: str, mapping: Dict[str, str]) -> Tuple[str, List[str]]:
    """Replace every task name in ``text`` with its alias.

    Returns ``(sanitized_text, hits)`` where ``hits`` is the list of
    original names that matched (deduplicated, in match order). Longer
    keys are matched first so "tblite/foo" wins over "foo" when both
    appear in the mapping.
    """
    if not text:
        return text, []
    keys = sorted(mapping.keys(), key=len, reverse=True)
    pattern = re.compile("(" + "|".join(re.escape(k) for k in keys) + r")\b")
    hits: List[str] = []
    seen: set[str] = set()

    def _sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in seen:
            seen.add(name)
            hits.append(name)
        return mapping[name]

    return pattern.sub(_sub, text), hits


def sanitize_artifact(
    artifact: FolderArtifact, mapping: Dict[str, str]
) -> Tuple[int, List[Tuple[str, List[str]]]]:
    """In-place sanitize ``*.md`` files of ``artifact``.

    Returns ``(replaced_files_count, [(path, hits), ...])``. Non-md
    files are left untouched (scripts may legitimately reference task
    names in tests; we only redact the prose layer the router reads).
    """
    replaced = 0
    detail: List[Tuple[str, List[str]]] = []
    for path in list(artifact.files):
        if not path.endswith(".md"):
            continue
        original = artifact.files[path]
        cleaned, hits = sanitize_text(original, mapping)
        if hits:
            artifact.files[path] = cleaned
            replaced += 1
            detail.append((path, hits))
    return replaced, detail


def find_leaked_names(
    artifact: FolderArtifact, mapping: Dict[str, str]
) -> List[Tuple[str, str]]:
    """Return ``[(file, name), ...]`` for every verbatim task name found
    in any ``*.md`` file of ``artifact``. Empty list = clean."""
    leaks: List[Tuple[str, str]] = []
    keys = sorted(mapping.keys(), key=len, reverse=True)
    pattern = re.compile("(" + "|".join(re.escape(k) for k in keys) + r")\b")
    for path, content in artifact.files.items():
        if not path.endswith(".md"):
            continue
        for m in pattern.finditer(content or ""):
            leaks.append((path, m.group(1)))
    return leaks


# kai-skills patch end


# ---------------------------------------------------------------------------
# Local EvaluationResult mirror of openevolve's (metrics + artifacts channel).
# We redefine instead of importing to keep this sub-package standalone.
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    metrics: Dict[str, float]
    artifacts: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class SkillFolderEvaluator:
    """Evaluate a FolderArtifact by delegating to ``skill_evolve.evaluator``."""

    def __init__(
        self,
        *,
        force_synthetic: bool = False,
        verify: bool = True,
        cascade: bool = True,
        max_workers: int = 1,
        model: Optional[str] = None,
        repeats: int = 3,
        # kai-skills patch (2026-04-27): when True, task IDs are aliased
        # to ``task_NNN`` in everything that gets fed back to the outer
        # LLM (per_task, failures, unused_skills). Run logs and the
        # post-hoc best_meta.json are unaffected. Default OFF for
        # backwards compat with v6 reproducibility.
        anonymize_tasks: bool = False,
        # kai-skills patch (Phase E, 2026-04-29): SkillsBench task
        # source dispatch. Defaults preserve the existing tblite/hermes
        # flow byte-identically. See controller.py for the dispatch
        # wiring + anti-leak scanner.
        task_source: str = "tblite",
        agent_backend: str = "hermes",
        task_list: Optional["Path"] = None,
        leak_policy: str = "zero",
        # Phase E v7 patch (2026-05-05): inner agent harness selection
        # for the bench-cli backend (``claude-code`` | ``gemini``).
        # Plumbed through to ``evaluate()`` and onward to
        # ``BenchCliBackend.run_task(agent=...)``. Ignored by the hermes
        # backend.
        agent: str = "claude-code",
        # kai-skills patch (Group B, 2026-05-27): --eval-source + held-out
        # validation. ``eval_source="skillsbench"`` (default) preserves
        # the existing flow. ``eval_source="behavioral"`` dispatches via
        # ``skill_evolve.evaluator.evaluate(..., eval_source="behavioral",
        # eval_set_path=..., judge_model=...)``. ``validation_task_list``
        # is a path to a held-out task list scored separately on accepted
        # winners (recorded as ``validation_score``; no re-acceptance).
        eval_source: str = "skillsbench",
        eval_set_path: Optional["Path"] = None,
        judge_model: Optional[str] = None,
        validation_task_list: Optional["Path"] = None,
    ) -> None:
        self.force_synthetic = force_synthetic
        self.verify = verify
        self.cascade = cascade
        self.max_workers = max_workers
        self.model = model
        self.repeats = repeats
        # kai-skills patch (2026-04-27)
        self.anonymize_tasks = anonymize_tasks
        self._task_id_map: Optional[Dict[str, str]] = None
        # kai-skills patch (Phase E, 2026-04-29)
        self.task_source = task_source
        self.agent_backend = agent_backend
        self.task_list = task_list
        self.leak_policy = leak_policy
        # Phase E v7 patch (2026-05-05)
        self.agent = agent
        # kai-skills patch (Group B, 2026-05-27)
        self.eval_source = eval_source
        self.eval_set_path = eval_set_path
        self.judge_model = judge_model
        self.validation_task_list = validation_task_list
        # The Controller fills these once it has the resolved task
        # records. ``anonymizer`` is the dispatched module (in-file for
        # tblite, ``skillsbench_anonymize`` for skillsbench) and
        # exposes ``sanitize_text`` / ``sanitize_artifact`` /
        # ``find_leaked_names``-style functions. ``_task_records`` is
        # the hydrated task list (subset-filtered) for the SkillsBench
        # path's anti-leakage scanner.
        self.anonymizer: Any = None
        self._task_records: Optional[List[Any]] = None

    # kai-skills patch (2026-04-27)
    def task_id_map(self) -> Dict[str, str]:
        """Lazily-built stable ``task_id -> task_NNN`` map. Empty when
        anonymization is disabled."""
        if not self.anonymize_tasks:
            return {}
        if self._task_id_map is None:
            self._task_id_map = build_task_id_map()
        return self._task_id_map

    def set_task_id_map(self, mapping: Dict[str, str]) -> None:
        """Inject a pre-built id map (used by the Controller's
        SkillsBench dispatch path so the map domain matches the
        runtime's task records, not the global manifest sweep)."""
        self._task_id_map = dict(mapping)

    # kai-skills patch end

    def evaluate_program(self, program: Program) -> EvaluationResult:
        return self.evaluate_artifact(program.artifact, program_id=program.id)

    def evaluate_artifact(
        self, artifact: FolderArtifact, *, program_id: str = ""
    ) -> EvaluationResult:
        """Materialize, evaluate, translate.

        kai-skills patch (Phase E, 2026-04-29): on exit, also run the
        R-12 anti-leakage scanner. Hits attach a ``leak_warning`` blob
        to ``EvaluationResult.artifacts`` and (per ``leak_policy``)
        either zero the score or raise.
        """
        artifact.validate()
        # kai-skills patch (Phase E, 2026-04-29): plumb the
        # task_source / task_list / agent_backend the controller stored
        # on this evaluator through to the inner ``evaluate()`` call.
        # Without this, Phase E silently fell back to the default
        # tblite manifest (10 tasks) instead of the configured subset.
        sources_arg: Optional[List[str]] = None
        if self.task_source == "skillsbench":
            sources_arg = ["skillsbench"]
        elif self.task_source == "tblite":
            sources_arg = ["tblite"]
        # else: leave None — preserves the legacy "all sources" behavior
        # for any callers that pre-date the Phase E task_source field.

        task_ids_arg: Optional[List[str]] = None
        if self.task_list is not None:
            task_list_path = Path(self.task_list)
            if task_list_path.exists():
                try:
                    loaded_ids = json.loads(task_list_path.read_text(encoding="utf-8"))
                    if isinstance(loaded_ids, list):
                        task_ids_arg = [str(t) for t in loaded_ids]
                except Exception:
                    logger.warning(
                        "evaluate_artifact: failed to read task_list %s; "
                        "falling back to no task_id filter",
                        task_list_path,
                    )

        with tempfile.TemporaryDirectory(prefix="track_b_eval_") as tmp:
            skills_dir = Path(tmp) / "skills"
            # kai-skills patch (Group G, 2026-05-28; plan §7l): the
            # evaluator materializes the bundle for the inner agent —
            # ``meta_skill.md`` is training-only and MUST NOT leak into
            # the deployed bundle the agent sees.
            artifact.write_to(skills_dir, deployment=True)
            res = _skill_evaluator.evaluate(
                skills_dir,
                cascade=self.cascade,
                max_workers=self.max_workers,
                model=self.model,
                force_synthetic=self.force_synthetic,
                verify=self.verify,
                repeats=self.repeats,
                sources=sources_arg,
                task_ids=task_ids_arg,
                agent_backend=self.agent_backend,
                agent=self.agent,
                # kai-skills patch (Group B, 2026-05-27)
                eval_source=self.eval_source,
                eval_set_path=self.eval_set_path,
                judge_model=self.judge_model,
            )
        translated = self._translate(res, artifact, program_id=program_id)
        return self._apply_leak_policy(translated, artifact)

    # kai-skills patch (Group B, 2026-05-27): held-out validation eval.
    # Mirrors ``evaluate_artifact`` but overrides the task_list with the
    # ``--validation-task-list`` path. Returns just the composite/mean_score
    # pair so callers can stamp them onto an artifact / run record without
    # having to re-translate the full EvaluationResult shape.
    def evaluate_validation(
        self,
        artifact: FolderArtifact,
        *,
        program_id: str = "",
    ) -> Dict[str, float]:
        """Score ``artifact`` against the held-out validation task list.

        Returns ``{}`` when ``validation_task_list`` is unset. Otherwise
        returns ``{"validation_composite": float, "validation_mean_score":
        float | None, "validation_success_rate": float, "validation_n":
        int}`` so the controller can record them on the artifact without
        re-translating the EvaluationResult.
        """
        if self.validation_task_list is None:
            return {}
        artifact.validate()
        val_path = Path(self.validation_task_list)
        task_ids_arg: Optional[List[str]] = None
        if val_path.exists():
            try:
                loaded_ids = json.loads(val_path.read_text(encoding="utf-8"))
                if isinstance(loaded_ids, list):
                    task_ids_arg = [str(t) for t in loaded_ids]
            except Exception:
                logger.warning(
                    "evaluate_validation: failed to read %s; skipping",
                    val_path,
                )
                return {}
        sources_arg: Optional[List[str]] = None
        if self.task_source == "skillsbench":
            sources_arg = ["skillsbench"]
        elif self.task_source == "tblite":
            sources_arg = ["tblite"]
        with tempfile.TemporaryDirectory(prefix="track_b_val_") as tmp:
            skills_dir = Path(tmp) / "skills"
            # kai-skills patch (Group G, 2026-05-28; plan §7l): validation
            # eval uses the deployment-shape bundle — strip ``meta_skill.md``.
            artifact.write_to(skills_dir, deployment=True)
            res = _skill_evaluator.evaluate(
                skills_dir,
                cascade=self.cascade,
                max_workers=self.max_workers,
                model=self.model,
                force_synthetic=self.force_synthetic,
                verify=self.verify,
                repeats=self.repeats,
                sources=sources_arg,
                task_ids=task_ids_arg,
                agent_backend=self.agent_backend,
                agent=self.agent,
                eval_source=self.eval_source,
                eval_set_path=self.eval_set_path,
                judge_model=self.judge_model,
            )
        return {
            "validation_composite": float(res.composite),
            "validation_mean_score": (
                float(res.mean_score) if res.mean_score is not None else None
            ),
            "validation_success_rate": float(res.success_rate),
            "validation_n": int(res.n_tasks),
        }

    def _apply_leak_policy(
        self,
        result: "EvaluationResult",
        artifact: FolderArtifact,
    ) -> "EvaluationResult":
        """Run the R-12 scanner; honor ``leak_policy``.

        ``warn``  — log + attach ``leak_warning`` artifact, score kept.
        ``zero``  — additionally zero ``composite`` / ``success_rate``
                    / ``mean_score`` so the candidate cannot enter the
                    archive on a leaky exploit.
        ``raise`` — abort the run (RuntimeError).

        When anonymization is OFF (default for the existing tblite
        path), the scanner is a no-op — id_map is empty and no task
        records are loaded.
        """
        if not self.anonymize_tasks:
            return result
        from .leak_scanner import scan_artifact

        scan = scan_artifact(
            artifact,
            id_map=self.task_id_map(),
            task_records=self._task_records,
        )
        if scan.clean:
            return result
        warning_blob = scan.to_artifact_blob()
        logger.warning(
            "leak_scanner: %d hit(s) in candidate; policy=%s",
            len(scan.hits),
            self.leak_policy,
        )
        if self.leak_policy == "raise":
            raise RuntimeError(
                f"leak_scanner: {len(scan.hits)} hit(s); "
                f"policy=raise; abort. detail:\n{warning_blob}"
            )
        artifacts = dict(result.artifacts)
        artifacts["leak_warning"] = warning_blob
        new_metrics = dict(result.metrics)
        if self.leak_policy == "zero":
            for key in ("composite", "success_rate", "mean_score"):
                if key in new_metrics:
                    new_metrics[key] = 0.0
        return EvaluationResult(metrics=new_metrics, artifacts=artifacts)

    # -------------------- translation --------------------

    def _translate(
        self,
        res: "_skill_evaluator.EvalResult",
        artifact: FolderArtifact,
        *,
        program_id: str,
    ) -> EvaluationResult:
        metrics: Dict[str, float] = {
            # Primary fitness.
            "composite": float(res.composite),
            # Secondary signal (useful for logging + diagnostics).
            "success_rate": float(res.success_rate),
            "tool_calls_per_success": float(res.tool_calls_per_success),
            # Coverage signals.
            "n_tasks": float(res.n_tasks),
            "verified_count": float(res.verified_count),
            "unverified_count": float(res.unverified_count),
            # Broken-harness count — the controller reads this to abort a
            # run whose eval is all infra-errors rather than treating it as
            # "no improvement found" (2026-05-28 silent-fail incident).
            "errored_count": float(res.errored_count),
        }
        # Continuous-scoring signal (added 2026-04-20). ``mean_score`` is
        # ``None`` when no task exposed a structured breakdown — omit from
        # metrics in that case so downstream dashboards don't pick up a
        # spurious zero.
        if res.mean_score is not None:
            metrics["mean_score"] = float(res.mean_score)
            metrics["scored_task_count"] = float(res.scored_task_count)

        # kai-skills patch (2026-04-27): when --anonymize-tasks is set,
        # rewrite task IDs in everything that goes into the
        # ``artifacts`` channel (prompt-facing). The internal
        # logger.info call below still uses real names because run logs
        # are post-hoc telemetry, not LLM input.
        tid_map = self.task_id_map()

        def _alias(tid: Optional[str]) -> Optional[str]:
            if tid is None or not tid_map:
                return tid
            return tid_map.get(tid, tid_map.get(_short_name(tid), tid))

        # kai-skills patch (Phase E, 2026-04-29): dispatch through the
        # Controller-installed anonymizer module when available; fall
        # back to the in-file ``sanitize_text`` otherwise (tests +
        # backwards compat).
        _sanitize_text = (
            self.anonymizer.sanitize_text
            if getattr(self, "anonymizer", None) is not None
            else sanitize_text
        )

        def _alias_text(text: str) -> str:
            if not tid_map or not text:
                return text
            cleaned, _ = _sanitize_text(text, tid_map)
            return cleaned

        # kai-skills patch end

        # Feedback to fold into the next prompt.
        # v9d patch (2026-05-07): cap raised from 240 → 3000 chars per
        # entry. With 5 hot tasks and repeats=3, even 5×3000=15KB stays
        # well under DeepSeek-v4-pro's 128k context. The bench-cli backend
        # now fills last_msg with verifier status + agent final message +
        # last 6 execute titles for failures, instead of just the agent
        # error string. See _build_failure_last_msg in agents/bench_cli.py.
        failures_for_prompt = [
            {
                "task_id": _alias(f.get("task_id")),
                "last_msg": _alias_text((f.get("last_msg") or "")[:3000]),
            }
            for f in res.failures
        ]
        failures_blob = json.dumps(failures_for_prompt, indent=2)
        # Per-task detail. This is the structured record that downstream
        # consumers (Track B/C best_meta.json, Track C tournament rebuild)
        # read to answer "which task did evolution actually crack?". Keep
        # the schema stable — the three-state ``verified`` (True/False/None)
        # plus ``verifier_status`` is what distinguishes "we proved a pass"
        # from "we couldn't verify". ``last_msg`` is retained (trimmed) for
        # the Track A critic feedback path.
        per_task_list = [
            {
                "task_id": _alias(t.get("task_id")),
                "verified": t.get("verified"),
                "success": t.get("success"),
                "elapsed_s": t.get("elapsed_s"),
                "tool_calls": t.get("tool_calls"),
                "skills_invoked": list(t.get("skills_invoked") or []),
                "verifier_status": t.get("verifier_status"),
                "last_msg": _alias_text((t.get("last_msg") or "")[:240]),
            }
            for t in res.per_task
        ]

        # Aggregate invocation counts (what skills actually fired?).
        invocation_counts: Dict[str, int] = {}
        for t in per_task_list:
            for name in t["skills_invoked"]:
                invocation_counts[name] = invocation_counts.get(name, 0) + 1
        # Skills present but never invoked → dead weight signal.
        present = set(artifact.skill_names())
        invoked = set(invocation_counts.keys())
        unused = sorted(present - invoked)

        # kai-skills patch (2026-04-27): if a skill name happens to
        # contain a redacted task name, rewrite it for the prompt-facing
        # ``unused_skills`` blob too.
        unused_for_prompt = [_alias_text(s) for s in unused] if tid_map else unused
        # kai-skills patch end

        artifacts = {
            "failures": failures_blob,
            "per_task": json.dumps(per_task_list),
            "invocation_counts": json.dumps(invocation_counts, indent=2),
            "unused_skills": json.dumps(unused_for_prompt),
            "synthetic": "1" if res.synthetic else "0",
            "cascade_truncated": "1" if res.cascade_truncated else "0",
        }

        logger.info(
            "eval %s: composite=%.4f success=%.2f (%d tasks, %d verified)%s",
            (program_id or "")[:8],
            metrics["composite"],
            metrics["success_rate"],
            int(metrics["n_tasks"]),
            int(metrics["verified_count"]),
            " [synthetic]" if res.synthetic else "",
        )
        return EvaluationResult(metrics=metrics, artifacts=artifacts)
