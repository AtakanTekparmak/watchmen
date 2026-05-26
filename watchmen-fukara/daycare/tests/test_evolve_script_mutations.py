"""Unit tests for Group B — bundle-level evolution (script mutations,
bundle token cap, failure-mode categories, list_scripts tool).
"""

from __future__ import annotations

import json
from pathlib import Path

from daycare import evolve
from daycare.evolve import (
    MAX_BUNDLE_TOKENS,
    MAX_SKILL_TOKENS,
    _PROPOSER_SYSTEM_PROMPT,
    _bundle_tokens,
    _cluster_failures,
    _fallback_cluster_by_type,
    build_weakness_report,
    list_scripts,
    propose_candidates,
    type_to_category,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────


def _make_bundle(root: Path) -> Path:
    """Build a minimal bundle dir with SKILL.md + scripts/a.py + references/b.md."""
    bundle = root / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "SKILL.md").write_text(
        "# Skill\n\nThis is the SKILL.md body with several tokens.\n",
        encoding="utf-8",
    )
    scripts = bundle / "scripts"
    scripts.mkdir()
    (scripts / "a.py").write_text(
        "import argparse\n\ndef main():\n    pass\n\nif __name__ == '__main__':\n    main()\n",
        encoding="utf-8",
    )
    refs = bundle / "references"
    refs.mkdir()
    (refs / "b.md").write_text("# Reference\n\nSome reference text here.\n", encoding="utf-8")
    return bundle


# ─── Test 1: _bundle_tokens sums SKILL.md + scripts + references ──────────


def test_bundle_tokens_sums_skill_md_and_scripts(tmp_path):
    bundle = _make_bundle(tmp_path)
    total = _bundle_tokens(bundle)
    assert total > 0
    # Verify it strictly exceeds SKILL.md alone (it should also count scripts and references).
    skill_only_tokens = evolve._count_tokens((bundle / "SKILL.md").read_text())
    assert total > skill_only_tokens


def test_bundle_tokens_excludes_hidden_and_pycache(tmp_path):
    bundle = _make_bundle(tmp_path)
    baseline = _bundle_tokens(bundle)
    # Add hidden file and __pycache__ — should NOT be counted.
    (bundle / ".hidden.md").write_text("hidden content " * 100, encoding="utf-8")
    pyc = bundle / "scripts" / "__pycache__"
    pyc.mkdir()
    (pyc / "a.cpython-312.pyc").write_text("pyc content " * 100, encoding="utf-8")
    # Also a hidden file under scripts.
    (bundle / "scripts" / ".hidden.py").write_text("hidden " * 100, encoding="utf-8")
    after = _bundle_tokens(bundle)
    assert after == baseline


# ─── Test 2: bundle_too_large rejection in propose_candidates ─────────────


def test_bundle_too_large_rejected(tmp_path, monkeypatch):
    """Verify the propose_candidates guard rejects a candidate whose bundle
    exceeds MAX_BUNDLE_TOKENS, logging 'bundle_too_large' to history.jsonl.
    """
    # Set up a parent bundle.
    parent = _make_bundle(tmp_path / "parent")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    iter_n = 1

    # Force _bundle_tokens to report an over-limit value.
    monkeypatch.setattr(evolve, "_bundle_tokens", lambda p: MAX_BUNDLE_TOKENS + 1)

    # Stub proposer to emit a single trivial EDIT_FILE patch on SKILL.md.
    fake_patch = "<<<EDIT_FILE SKILL.md>>>\n# Skill\n\nUpdated body.\n<<<END_FILE>>>\n"

    def fake_run_one_proposer(**kwargs):
        slot = kwargs["slot"]
        cand_root = kwargs["candidate_dir_root"]
        (cand_root / f"c{slot}").mkdir(parents=True, exist_ok=True)
        return (
            slot,
            {
                "patch_text": fake_patch,
                "target_cluster": "cluster_1",
                "reasoning": "test",
            },
            [],
        )

    monkeypatch.setattr(evolve, "_run_one_proposer", fake_run_one_proposer)

    eval_set_path = tmp_path / "evals.jsonl"
    eval_set_path.write_text("", encoding="utf-8")

    surviving = propose_candidates(
        run_dir=run_dir,
        iter_n=iter_n,
        K=1,
        weakness_report_path=tmp_path / "missing_weakness.md",
        best_bundle_dir=parent,
        proposer_model="fake-model",
        api_key="fake-key",
        seed=0,
        max_workers=1,
        fingerprints=set(),
        eval_set_path=eval_set_path,
        leak_policy="warn",
    )

    assert surviving == []
    history_path = run_dir / "history.jsonl"
    assert history_path.exists()
    rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any("bundle_too_large" in (r.get("outcome") or "") for r in rows)


# ─── Test 3: proposer prompt mentions the 4 categories ────────────────────


def test_proposer_prompt_mentions_categories():
    for cat in (
        "missing_procedure_knowledge",
        "wrong_command_syntax",
        "missing_script_capability",
        "incomplete_workflow",
    ):
        assert cat in _PROPOSER_SYSTEM_PROMPT, f"category '{cat}' missing from proposer prompt"


# ─── Test 4: prompt token limits match constants (no drift) ───────────────


def test_proposer_prompt_token_limits_match_constants():
    assert str(MAX_SKILL_TOKENS) in _PROPOSER_SYSTEM_PROMPT
    assert str(MAX_BUNDLE_TOKENS) in _PROPOSER_SYSTEM_PROMPT
    # Regression guard for the historical "2500" literal drift.
    assert "2500" not in _PROPOSER_SYSTEM_PROMPT


# ─── Test 5: list_scripts returns sized paths ─────────────────────────────


def test_list_scripts_tool_returns_paths(tmp_path):
    bundle = _make_bundle(tmp_path)
    out = list_scripts(bundle)
    assert "scripts/a.py" in out
    assert "bytes" in out


def test_list_scripts_missing_dir(tmp_path):
    empty = tmp_path / "empty_bundle"
    empty.mkdir()
    assert list_scripts(empty) == "(no scripts/ in parent bundle)"


def test_list_scripts_empty_dir(tmp_path):
    bundle = tmp_path / "b"
    (bundle / "scripts").mkdir(parents=True)
    assert list_scripts(bundle) == "(scripts/ is empty)"


# ─── Test 6: cluster_failures sends category to judge ─────────────────────


def test_cluster_failures_includes_category_in_judge_payload(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "clusters": [
                                        {
                                            "name": "c1",
                                            "severity": 2,
                                            "category": "wrong_command_syntax",
                                            "mode": "reasoning",
                                            "examples": ["e1", "e2"],
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            }

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None, headers=None):
            captured["payload"] = json
            return FakeResponse()

    monkeypatch.setattr(evolve.httpx, "Client", FakeClient)

    failing = [
        {"id": "1", "type": "script_gen", "anonymized_prompt": "p1", "anonymized_rubric": "r1"},
        {"id": "2", "type": "procedural_qa", "anonymized_prompt": "p2", "anonymized_rubric": "r2"},
        {"id": "3", "type": "skill_invoke", "anonymized_prompt": "p3", "anonymized_rubric": "r3"},
        {"id": "4", "type": "behavioral_action", "anonymized_prompt": "p4", "anonymized_rubric": "r4"},
    ]

    _cluster_failures(failing, "judge-model", "fake-key")

    body = captured.get("payload")
    assert body is not None
    user_content = body["messages"][1]["content"]
    inner = json.loads(user_content)
    rows = inner["failing_evals"]
    expected = {
        "script_gen": "wrong_command_syntax",
        "procedural_qa": "missing_procedure_knowledge",
        "skill_invoke": "missing_script_capability",
        "behavioral_action": "incomplete_workflow",
    }
    for row in rows:
        assert "category" in row
        assert row["category"] == expected[row["type"]]


# ─── Test 7: cluster_failures output preserves category ───────────────────


def test_cluster_failures_includes_category_in_output(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "clusters": [
                                        {
                                            "name": "needs_flag",
                                            "severity": 3,
                                            "category": "wrong_command_syntax",
                                            "mode": "reasoning",
                                            "examples": ["ex1"],
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            }

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **kw):
            return FakeResponse()

    monkeypatch.setattr(evolve.httpx, "Client", FakeClient)

    failing = [{"id": "1", "type": "script_gen", "anonymized_prompt": "p", "anonymized_rubric": "r"}]
    clusters = _cluster_failures(failing, "judge", "key")
    assert clusters
    assert clusters[0]["category"] == "wrong_command_syntax"


def test_cluster_failures_category_defaults_when_missing(monkeypatch):
    """Judge omits category → default to missing_procedure_knowledge."""

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "clusters": [
                                        {
                                            "name": "x",
                                            "severity": 1,
                                            "mode": "reasoning",
                                            "examples": [],
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            }

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **kw):
            return FakeResponse()

    monkeypatch.setattr(evolve.httpx, "Client", FakeClient)
    clusters = _cluster_failures(
        [{"id": "1", "type": "script_gen", "anonymized_prompt": "p", "anonymized_rubric": "r"}],
        "judge",
        "key",
    )
    assert clusters
    assert clusters[0]["category"] == "missing_procedure_knowledge"


# ─── Test 8: fallback cluster assigns category by type ────────────────────


def test_fallback_cluster_assigns_category_by_type():
    rows = [
        {"id": "a", "type": "behavioral_action", "anonymized_prompt": "p1"},
        {"id": "b", "type": "script_gen", "anonymized_prompt": "p2"},
        {"id": "c", "type": "procedural_qa", "anonymized_prompt": "p3"},
        {"id": "d", "type": "skill_invoke", "anonymized_prompt": "p4"},
    ]
    clusters = _fallback_cluster_by_type(rows)
    by_name = {c["name"]: c for c in clusters}
    assert by_name["behavioral_action_failures"]["category"] == "incomplete_workflow"
    assert by_name["script_gen_failures"]["category"] == "wrong_command_syntax"
    assert by_name["procedural_qa_failures"]["category"] == "missing_procedure_knowledge"
    assert by_name["skill_invoke_failures"]["category"] == "missing_script_capability"


# ─── Test 9: weakness report renders category ─────────────────────────────


def test_weakness_report_renders_category(tmp_path, monkeypatch):
    """build_weakness_report should include category=... in cluster headers.

    Stub out the expensive parts (train scoring + per-eval scoring + invocations
    + judge clustering) so the test focuses on rendering.
    """
    bundle = _make_bundle(tmp_path / "parent")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    fake_clusters = [
        {
            "name": "bad_flag",
            "severity": 4,
            "category": "wrong_command_syntax",
            "mode": "reasoning",
            "examples": ["e1", "e2"],
        }
    ]
    monkeypatch.setattr(evolve, "_cluster_failures", lambda *a, **k: fake_clusters)

    # Stub score_bundle, score_single, run_rollout_subprocess, _count_invocations.
    class FakeEvalSummary:
        holdout_score = 0.5
        fitness = 0.5
        tokens_skill_md = 100
        penalty = 0.0
        lambda_n = 0.0
        n_holdout = 1
        by_length_quartile = {"q1": 0.5, "q2": 0.5, "q3": 0.5, "q4": 0.5}

    monkeypatch.setattr(evolve, "score_bundle", lambda **kw: FakeEvalSummary())

    class FakeRollout:
        error = None
        completion = "no skill use here"

    monkeypatch.setattr(evolve, "run_rollout_subprocess", lambda **kw: FakeRollout())
    monkeypatch.setattr(evolve, "score_single", lambda **kw: 0.0)
    monkeypatch.setattr(
        evolve,
        "_count_invocations",
        lambda *a, **kw: (__import__("collections").Counter(), 1),
    )

    train_evals = [
        {
            "id": "t1",
            "split": "train",
            "type": "script_gen",
            "prompt": "p",
            "rubric": "r",
            "baseline_completion_len_tokens": 10,
        }
    ]

    out_path = build_weakness_report(
        run_dir=run_dir,
        iter_n=1,
        train_evals=train_evals,
        best_bundle_dir=bundle,
        model="m",
        judge_model="j",
        api_key="k",
        rollouts=1,
        baseline_len_quartiles=[5.0, 10.0, 20.0],
        max_workers=1,
    )

    rendered = out_path.read_text(encoding="utf-8")
    assert "category=" in rendered
    assert "category=wrong_command_syntax" in rendered


# ─── Sanity: type_to_category is module-level and complete ────────────────


def test_type_to_category_mapping_keys():
    assert type_to_category["script_gen"] == "wrong_command_syntax"
    assert type_to_category["procedural_qa"] == "missing_procedure_knowledge"
    assert type_to_category["skill_invoke"] == "missing_script_capability"
    assert type_to_category["behavioral_action"] == "incomplete_workflow"
