"""Smoke test for the --anonymize-tasks task-name leakage guard.

Locks in three invariants of the kai-skills patch (2026-04-27):

* When ``anonymize_tasks=False`` (default), nothing changes — the
  evaluator's ``per_task`` artifacts contain real task IDs and the seed
  is left as-is.
* When ``anonymize_tasks=True``:
    - ``run_evolution()`` rewrites verbatim manifest task names in the
      seed's ``*.md`` files to ``task_NNN`` aliases at load time.
    - The evaluator's ``per_task`` / ``failures`` artifacts use the
      same aliases.
    - The validate-on-write guard rejects child patches that reintroduce
      a redacted name (covered by direct unit invocation; we don't run
      the full LLM loop because synthetic LLM doesn't emit task names).
"""

from __future__ import annotations

import json
from pathlib import Path


from skill_evolve.evaluator import EvalResult, TaskOutcome
from skill_evolve.track_b.openevolve_skills.controller import (
    RunConfig,
    run_evolution,
)
from skill_evolve.track_b.openevolve_skills.evaluator import (
    SkillFolderEvaluator,
    build_task_id_map,
    find_leaked_names,
    sanitize_text,
)
from skill_evolve.track_b.openevolve_skills.folder_artifact import (
    FolderArtifact,
)
from skill_evolve.track_b.openevolve_skills.llm_client import SyntheticLLM
from skill_evolve.track_b.openevolve_skills.prompt_sampler import PromptSampler


def _write_seed(seed_dir: Path, skill_md_body: str) -> None:
    """Build a 1-skill seed folder under ``seed_dir``."""
    seed_dir.mkdir(parents=True, exist_ok=True)
    skill = seed_dir / "router_match"
    skill.mkdir()
    (skill / "SKILL.md").write_text(skill_md_body, encoding="utf-8")


def test_build_task_id_map_is_stable():
    """Mapping is deterministic across calls and assigns task_NNN by sort order."""
    m1 = build_task_id_map()
    m2 = build_task_id_map()
    assert m1 == m2
    # Must contain at least the manifest IDs we know about.
    assert any(v.startswith("task_") for v in m1.values())
    # Every alias is task_NNN with NNN zero-padded.
    aliases = {v for v in m1.values()}
    for a in aliases:
        assert a.startswith("task_") and len(a) == len("task_001")


def test_sanitize_text_replaces_both_qualified_and_bare_forms():
    mapping = {
        "tblite/foo-bar": "task_001",
        "foo-bar": "task_001",
        "tblite/baz": "task_002",
        "baz": "task_002",
    }
    src = "We routed on tblite/foo-bar and on foo-bar; also baz."
    out, hits = sanitize_text(src, mapping)
    assert "foo-bar" not in out
    assert "tblite/foo-bar" not in out
    assert "task_001" in out
    assert "task_002" in out
    assert "tblite/foo-bar" in hits
    assert "foo-bar" in hits


def test_find_leaked_names_only_scans_md():
    mapping = {"foo-bar": "task_001"}
    art = FolderArtifact(
        files={
            "s/SKILL.md": "---\nname: s\ndescription: x\n---\n\nfoo-bar here",
            "s/scripts/run.sh": "#!/usr/bin/env bash\n# foo-bar in script - allowed\n",
        }
    )
    leaks = find_leaked_names(art, mapping)
    assert any(p.endswith("SKILL.md") for p, _ in leaks)
    assert not any("scripts" in p for p, _ in leaks)


def test_smoke_anonymize_off_leaves_seed_alone(tmp_path: Path):
    """Default OFF: seed prose with task names is preserved verbatim
    AND ``per_task`` artifacts use real IDs."""
    real_map = build_task_id_map()
    # Pick any real task name from the manifest.
    real_name = next(k for k in real_map if "/" not in k and k != real_map[k])
    body = (
        "---\n"
        "name: router_match\n"
        f"description: handles {real_name} tasks\n"
        "---\n\n"
        f"This skill is for {real_name}.\n"
    )
    seed_dir = tmp_path / "seed"
    _write_seed(seed_dir, body)

    evaluator = SkillFolderEvaluator(
        force_synthetic=True,
        verify=False,
        cascade=True,
        anonymize_tasks=False,
    )
    llm = SyntheticLLM(seed=0)
    result = run_evolution(
        seed_path=seed_dir,
        out_dir=tmp_path / "out",
        config=RunConfig(
            num_generations=0, num_islands=1, migration_interval=5, rng_seed=0
        ),
        evaluator=evaluator,
        llm=llm,
        prompt_sampler=PromptSampler(),
    )
    assert result.best is not None
    # Seed was loaded as-is; best/SKILL.md should still reference the real name.
    best_md = (tmp_path / "out" / "best" / "router_match" / "SKILL.md").read_text()
    assert real_name in best_md, (
        f"with anonymize_tasks=False, seed prose must be preserved; got: {best_md!r}"
    )


def test_smoke_anonymize_on_sanitizes_seed_and_artifacts(tmp_path: Path):
    """Flag ON: seed *.md files are rewritten and the evaluator's
    prompt-facing per_task uses task_NNN aliases."""
    real_map = build_task_id_map()
    real_name = next(k for k in real_map if "/" not in k and k != real_map[k])
    expected_alias = real_map[real_name]
    body = (
        "---\n"
        "name: router_match\n"
        f"description: handles {real_name} tasks\n"
        "---\n\n"
        f"This skill is for {real_name}.\n"
    )
    seed_dir = tmp_path / "seed"
    _write_seed(seed_dir, body)

    evaluator = SkillFolderEvaluator(
        force_synthetic=True,
        verify=False,
        cascade=True,
        anonymize_tasks=True,
    )
    llm = SyntheticLLM(seed=0)
    result = run_evolution(
        seed_path=seed_dir,
        out_dir=tmp_path / "out",
        config=RunConfig(
            num_generations=0, num_islands=1, migration_interval=5, rng_seed=0
        ),
        evaluator=evaluator,
        llm=llm,
        prompt_sampler=PromptSampler(),
    )
    assert result.best is not None
    # Seed was sanitized in-place: best/SKILL.md should NOT contain the real name.
    best_md = (tmp_path / "out" / "best" / "router_match" / "SKILL.md").read_text()
    assert real_name not in best_md, (
        f"with anonymize_tasks=True, seed prose must be sanitized; "
        f"still contains {real_name!r}: {best_md!r}"
    )
    assert expected_alias in best_md, (
        f"alias {expected_alias!r} not found in sanitized seed: {best_md!r}"
    )


def test_unit_translate_uses_aliases_when_anonymize_on():
    """Direct unit on _translate: synthesize an EvalResult with real task IDs
    and confirm anonymize_tasks=True replaces them in artifacts."""
    real_map = build_task_id_map()
    # Pick a fully-qualified ID we know is in the manifest.
    fq = next(k for k in real_map if "/" in k)
    alias = real_map[fq]

    art = FolderArtifact(
        files={
            "s/SKILL.md": "---\nname: s\ndescription: x\n---\n\nbody.",
        }
    )
    res = EvalResult(
        success_rate=0.5,
        tool_calls_per_success=2.0,
        composite=0.5,
        per_task=[
            TaskOutcome(
                task_id=fq,
                success=True,
                tool_calls=2,
                elapsed_s=1.0,
                skills_invoked=["s"],
                verified=True,
                verifier_status="ok",
            ).to_dict()
        ],
        failures=[{"task_id": fq, "last_msg": f"fail in {fq}"}],
        skills_folder="x",
        n_tasks=1,
        verified_count=1,
    )

    evaluator = SkillFolderEvaluator(force_synthetic=True, anonymize_tasks=True)
    out = evaluator._translate(res, art, program_id="t")
    per_task = json.loads(out.artifacts["per_task"])
    assert per_task[0]["task_id"] == alias
    assert fq not in out.artifacts["per_task"]
    failures = json.loads(out.artifacts["failures"])
    assert failures[0]["task_id"] == alias
    assert fq not in out.artifacts["failures"]


def test_unit_translate_preserves_real_ids_when_anonymize_off():
    real_map = build_task_id_map()
    fq = next(k for k in real_map if "/" in k)

    art = FolderArtifact(
        files={
            "s/SKILL.md": "---\nname: s\ndescription: x\n---\n\nbody.",
        }
    )
    res = EvalResult(
        success_rate=0.0,
        tool_calls_per_success=0.0,
        composite=0.0,
        per_task=[
            TaskOutcome(
                task_id=fq,
                success=False,
                tool_calls=0,
                elapsed_s=0.0,
            ).to_dict()
        ],
        failures=[{"task_id": fq, "last_msg": "fail"}],
        skills_folder="x",
        n_tasks=1,
    )
    evaluator = SkillFolderEvaluator(force_synthetic=True, anonymize_tasks=False)
    out = evaluator._translate(res, art, program_id="t")
    per_task = json.loads(out.artifacts["per_task"])
    assert per_task[0]["task_id"] == fq


def test_validate_on_write_guard_rejects_reintroduced_name():
    """Direct invocation: ``find_leaked_names`` flags any md file that
    contains a redacted task name; iteration uses this to reject patches."""
    mapping = build_task_id_map()
    # Pick a fully-qualified ID we know is in the manifest.
    real_name = next(k for k in mapping if "/" not in k and k != mapping[k])
    art = FolderArtifact(
        files={
            "s/SKILL.md": (
                f"---\nname: s\ndescription: handles {real_name}\n---\n\nbody."
            ),
        }
    )
    leaks = find_leaked_names(art, mapping)
    assert leaks, "expected at least one leak"
    assert any(name == real_name for _, name in leaks)
