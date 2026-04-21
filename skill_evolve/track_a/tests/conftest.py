"""Test fixtures for Track A.

Loads the seed skills folder and gives each test a temporary copy to
mutate.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from skill_evolve.track_a.folder import SkillFolder
from skill_evolve.track_a.llm import LLMClient


REPO_ROOT = Path(__file__).resolve().parents[3]
SEED_SKILLS = REPO_ROOT / "seed_skills"


@pytest.fixture
def seed_path(tmp_path: Path) -> Path:
    """Return a fresh temp copy of the seed skills folder."""
    dest = tmp_path / "seed_skills"
    shutil.copytree(SEED_SKILLS, dest)
    # Drop non-skill files like INDEX.md from the copy so SkillFolder.load
    # doesn't confuse them — SkillFolder already ignores non-dir children,
    # so INDEX.md is harmless, but we leave it in to mirror real usage.
    return dest


@pytest.fixture
def seed_folder(seed_path: Path) -> SkillFolder:
    return SkillFolder.load(seed_path)


@pytest.fixture
def synthetic_client() -> LLMClient:
    return LLMClient(synthetic=True)
