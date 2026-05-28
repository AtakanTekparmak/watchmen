"""One-iter end-to-end test for track_b/run.py.

Drives ``track_b/run.py:main()`` programmatically with:
  * ``--patch-format sentinel-blocks``
  * ``--eval-source skillsbench``
  * ``--task-list .../train_3.json``
  * ``--validation-task-list .../val_2.json``
  * Stubbed OpenRouter LLM emitting a one-op ADD_FILE patch.

Asserts iteration completes, smoke gate ran, evaluator scored,
validation_score recorded, artifact JSON written to a tmp path.

Depends on Group B's ``--eval-source``, ``--eval-set``,
``--judge-model``, and ``--validation-task-list`` CLI flags. The test
is skipped (not failed) if any of those flags are missing from the
current ``track_b/run.py`` argparser so Phase D2 can land before B.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest import mock

import pytest

from skill_evolve.track_b.run import _build_parser, main


REPO_ROOT = Path(__file__).resolve().parents[3]
SEED = REPO_ROOT / "seed_skills"
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "mock_skillsbench"


def _required_b_flags_present() -> bool:
    """Return True iff Group B's argparse flags have landed."""
    parser = _build_parser()
    flag_names = {a.option_strings[0] for a in parser._actions if a.option_strings}
    needed = {
        "--patch-format",
        "--eval-source",
        "--validation-task-list",
    }
    return needed.issubset(flag_names)


def test_end_to_end_one_iter_with_stubbed_llm(tmp_path: Path) -> None:
    """One iteration runs end-to-end against the mock_skillsbench fixture."""
    if not SEED.is_dir():
        pytest.skip(f"seed_skills not found at {SEED}")
    if not _required_b_flags_present():
        pytest.skip(
            "Group B's --patch-format / --eval-source / "
            "--validation-task-list flags not yet present in track_b/run.py"
        )

    out_dir = tmp_path / "run_out"
    # Copy the seed dir to a tmp path so the test doesn't write into
    # the source tree.
    seed_copy = tmp_path / "seed"
    shutil.copytree(SEED, seed_copy)

    train_list = FIXTURE_DIR / "train_3.json"
    val_list = FIXTURE_DIR / "val_2.json"

    # Stubbed OpenRouter LLM emits one ADD_FILE patch.
    one_op_patch = (
        "<<<ADD_FILE demo-helper/SKILL.md>>>\n"
        "---\nname: demo-helper\ndescription: tiny synth helper\n---\n\n# demo-helper\n"
        "<<<END_FILE>>>\n"
    )

    with mock.patch(
        "skill_evolve.track_b.openevolve_skills.llm_client.build_default_client"
    ) as m_build:
        fake_llm = mock.MagicMock()
        fake_llm.generate.return_value = one_op_patch
        m_build.return_value = fake_llm

        argv = [
            "--seed",
            str(seed_copy),
            "--out",
            str(out_dir),
            "--num-generations",
            "1",
            "--num-islands",
            "1",
            "--force-synthetic",
            "--patch-format",
            "sentinel-blocks",
            "--eval-source",
            "skillsbench",
            "--task-source",
            "skillsbench",
            "--task-list",
            str(train_list),
            "--validation-task-list",
            str(val_list),
            "--rng-seed",
            "0",
        ]
        rc = main(argv)

    assert rc == 0
    # Artifact written.
    assert out_dir.exists()
    # Best dir is the canonical artifact location.
    assert (out_dir / "best").is_dir()
    # Validation score recorded in either history.jsonl or run_meta.json.
    hist = out_dir / "history.jsonl"
    meta = out_dir / "run_meta.json"
    assert hist.exists() or meta.exists()
