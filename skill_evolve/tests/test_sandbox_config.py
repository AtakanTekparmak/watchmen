"""Sandbox config tests.

Confirms that per-run ``config.yaml`` carries over the full ``model:``
block from the user's global ``~/.hermes/config.yaml`` — including
``model.temperature``. Prior audit found sampling-noise variance was
driven by non-zero temperature, and the fix sets ``model.temperature=0.0``
globally; we need the sandbox to forward that through so the inner
Hermes subprocess actually respects it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest
import yaml

from skill_evolve import sandbox as _sandbox_mod
from skill_evolve.sandbox import make_sandbox


def _write_user_cfg(tmp_home: Path, model_block: Dict) -> Path:
    cfg = tmp_home / ".hermes" / "config.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(yaml.safe_dump({"model": model_block}), encoding="utf-8")
    return cfg


@pytest.fixture
def fake_user_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the module's _USER_GLOBAL_CONFIG to a temp file and write one."""
    fake_home = tmp_path / "home"
    cfg = _write_user_cfg(
        fake_home,
        {"provider": "openrouter", "default": "minimax/minimax-m2.7",
         "temperature": 0.0},
    )
    monkeypatch.setattr(_sandbox_mod, "_USER_GLOBAL_CONFIG", cfg)
    return cfg


def test_sandbox_carries_full_model_block_including_temperature(
    tmp_path: Path,
    fake_user_config: Path,
) -> None:
    # Build a dummy skills dir that validates (just needs to be a directory)
    skills_dir = tmp_path / "skills"
    (skills_dir / "demo").mkdir(parents=True)
    (skills_dir / "demo" / "SKILL.md").write_text(
        "---\nname: demo\ndescription: demo\n---\nbody\n",
        encoding="utf-8",
    )

    handle = make_sandbox(skills_dir, run_id="cfgtest", tmp_root=tmp_path)
    try:
        cfg = handle.home / "config.yaml"
        loaded = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        model = loaded.get("model") or {}
        assert model.get("provider") == "openrouter"
        assert model.get("default") == "minimax/minimax-m2.7"
        # The critical assertion: temperature forwarded through.
        assert "temperature" in model, f"model block missing temperature: {model}"
        assert float(model["temperature"]) == 0.0
    finally:
        import shutil
        shutil.rmtree(handle.home, ignore_errors=True)


def test_sandbox_model_block_preserves_arbitrary_extra_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure _load_user_model_config returns the full block, not a subset."""
    fake_home = tmp_path / "home"
    cfg = _write_user_cfg(
        fake_home,
        {
            "provider": "openrouter",
            "default": "anthropic/claude-opus-4.6",
            "temperature": 0.0,
            "max_tokens": 4096,
            "top_p": 0.95,
        },
    )
    monkeypatch.setattr(_sandbox_mod, "_USER_GLOBAL_CONFIG", cfg)

    skills_dir = tmp_path / "skills"
    (skills_dir / "demo").mkdir(parents=True)
    (skills_dir / "demo" / "SKILL.md").write_text(
        "---\nname: demo\ndescription: demo\n---\nbody\n",
        encoding="utf-8",
    )

    handle = make_sandbox(skills_dir, run_id="cfgtest2", tmp_root=tmp_path)
    try:
        loaded = yaml.safe_load(
            (handle.home / "config.yaml").read_text(encoding="utf-8")
        )
        model = loaded.get("model") or {}
        for k in ("provider", "default", "temperature", "max_tokens", "top_p"):
            assert k in model, f"model block missing {k}: {model}"
        assert float(model["temperature"]) == 0.0
    finally:
        import shutil
        shutil.rmtree(handle.home, ignore_errors=True)
