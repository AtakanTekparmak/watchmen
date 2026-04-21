"""Per-evaluation HERMES_HOME sandbox.

Hermes reads its config from ``$HERMES_HOME/config.yaml`` (see
``hermes-agent/hermes_constants.py::get_hermes_home``). Custom skill folders
are wired in via ``skills.external_dirs`` in that config (see
``hermes-agent/agent/skill_utils.py::get_external_skills_dirs``).

Parallel evaluators must NOT share a HERMES_HOME — concurrent skill mutations,
log writes, and trajectory append races would scramble results. This module
gives each run its own throw-away HERMES_HOME pointing at the candidate skills
folder, plus a context manager for cleanup.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional

import yaml


_DEFAULT_TMP_PREFIX = "hermes-eval-"


def _resolve_tmp_root(explicit: Optional[Path]) -> Path:
    """Pick the host directory under which per-run HERMES_HOME dirs live.

    Resolution order:
    1. An explicit ``tmp_root`` arg always wins (caller knows best).
    2. ``SKILL_EVOLVE_TMP_ROOT`` env var (user override for sandbox location).
    3. On macOS (``sys.platform == "darwin"``), default to
       ``~/.cache/skill_evolve`` — Docker Desktop's default VirtioFS shared
       path set does NOT include ``/tmp`` (nor the macOS ``/var/folders/...``
       that ``tempfile.gettempdir()`` resolves to), so bind mounts of those
       paths silently yield empty directories inside containers.
    4. Otherwise fall back to ``tempfile.gettempdir()`` (Linux preserves
       historical behavior — ``/tmp`` is shareable under typical Docker setups).

    Paths from branches 2 and 3 are created if missing so callers can rely on
    the returned directory existing.
    """
    if explicit is not None:
        return Path(explicit)
    env_override = os.environ.get("SKILL_EVOLVE_TMP_ROOT")
    if env_override:
        root = Path(env_override).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        return root
    if sys.platform == "darwin":
        root = Path.home() / ".cache" / "skill_evolve"
        root.mkdir(parents=True, exist_ok=True)
        return root
    return Path(tempfile.gettempdir())


@dataclass
class SandboxHandle:
    """Handle for an active sandbox.

    Attributes:
        run_id: Stable identifier for the run (used in log/file names).
        home: Path of the temporary HERMES_HOME directory.
        skills_dir: Path of the candidate skills folder being evaluated.
        env: Environment dict suitable for ``subprocess.run(..., env=env)``.
              Inherits the parent process env and overrides HERMES_HOME +
              a few hardening knobs.
        run_dir: A workdir inside ``home`` where the evaluator can write
                 trajectory_samples.jsonl / failed_trajectories.jsonl. Hermes
                 writes those to the *current working directory* by default,
                 so callers should ``cwd=run_dir`` when invoking subprocesses.
    """

    run_id: str
    home: Path
    skills_dir: Path
    env: Dict[str, str]
    run_dir: Path


_USER_GLOBAL_CONFIG = Path.home() / ".hermes" / "config.yaml"


def _load_user_model_config() -> Dict:
    """Read ~/.hermes/config.yaml and return its ``model:`` block (or {})."""
    if not _USER_GLOBAL_CONFIG.exists():
        return {}
    try:
        loaded = yaml.safe_load(_USER_GLOBAL_CONFIG.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}
    return loaded.get("model", {}) if isinstance(loaded, dict) else {}


def _build_minimal_config(external_skills_dir: Path) -> Dict:
    """Return a minimal config.yaml dict that points at one external skills dir.

    We deliberately keep this small so Hermes' lazy YAML loader doesn't trip
    on unfamiliar fields. The ``model:`` block is inherited from the user's
    global ~/.hermes/config.yaml so the inner agent knows which provider and
    model to use — without it Hermes has no LLM configured and silently no-ops
    every task rollout.
    """
    cfg: Dict = {
        "skills": {
            # Hermes expands ~ and $VAR for entries here, but we hand it an
            # already-resolved absolute path.
            "external_dirs": [str(external_skills_dir.resolve())],
            # Don't disable anything; the candidate folder *is* the skill set.
            "disabled": [],
        },
        # Don't try to talk to Honcho/Skills-Hub from inside an eval.
        "memory": {"provider": "none"},
    }
    user_model = _load_user_model_config()
    if user_model:
        cfg["model"] = user_model
    return cfg


def _materialize_home(home: Path, skills_dir: Path) -> None:
    """Write config.yaml and create the sub-directories Hermes expects."""
    home.mkdir(parents=True, exist_ok=True)
    # Hermes creates ~/.hermes/skills/, logs/, etc. on demand, but we
    # pre-create them so subprocess writes never race against mkdir.
    (home / "skills").mkdir(exist_ok=True)
    (home / "logs").mkdir(exist_ok=True)
    (home / "cache").mkdir(exist_ok=True)
    config_path = home / "config.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            _build_minimal_config(skills_dir),
            f,
            default_flow_style=False,
            sort_keys=False,
        )


def make_sandbox(
    skills_folder_path: Path,
    run_id: Optional[str] = None,
    *,
    base_env: Optional[Dict[str, str]] = None,
    tmp_root: Optional[Path] = None,
) -> SandboxHandle:
    """Provision a HERMES_HOME sandbox without entering it.

    Use the :func:`sandbox` context manager for the common case (auto-cleanup);
    call this directly if you need to manage lifetime manually (e.g. when
    handing off to another process tree).
    """
    skills_folder_path = Path(skills_folder_path).expanduser().resolve()
    if not skills_folder_path.is_dir():
        raise FileNotFoundError(f"skills folder does not exist: {skills_folder_path}")

    rid = run_id or uuid.uuid4().hex[:12]
    tmp_root = _resolve_tmp_root(tmp_root)
    home = tmp_root / f"{_DEFAULT_TMP_PREFIX}{rid}"
    if home.exists():
        # Reusing a stale dir would defeat isolation — refuse.
        raise FileExistsError(f"sandbox HERMES_HOME already exists: {home}")
    _materialize_home(home, skills_folder_path)

    run_dir = home / "run"
    run_dir.mkdir(exist_ok=True)

    env = dict(base_env if base_env is not None else os.environ)
    env["HERMES_HOME"] = str(home)
    # Suppress noisy "first-run" telemetry / interactive prompts.
    env.setdefault("HERMES_DISABLE_TELEMETRY", "1")
    env.setdefault("HERMES_NONINTERACTIVE", "1")
    # Some Hermes code paths inspect HOME for git/ssh config — point those at
    # an isolated dir so we don't pollute the user's real config.
    env.setdefault("HERMES_PLATFORM", "eval")

    return SandboxHandle(
        run_id=rid,
        home=home,
        skills_dir=skills_folder_path,
        env=env,
        run_dir=run_dir,
    )


@contextmanager
def sandbox(
    skills_folder_path: Path,
    run_id: Optional[str] = None,
    *,
    base_env: Optional[Dict[str, str]] = None,
    tmp_root: Optional[Path] = None,
    keep_on_exit: bool = False,
) -> Iterator[SandboxHandle]:
    """Context manager: provision a sandbox, yield the handle, then clean up.

    Set ``keep_on_exit=True`` to leave the temp dir behind for inspection
    (useful when a run fails and you want to look at the trajectory files).
    """
    handle = make_sandbox(
        skills_folder_path,
        run_id=run_id,
        base_env=base_env,
        tmp_root=tmp_root,
    )
    try:
        yield handle
    finally:
        if not keep_on_exit and handle.home.exists():
            shutil.rmtree(handle.home, ignore_errors=True)


def smoke_test(skills_folder_path: Path) -> None:
    """Stand up a sandbox and shell out to ``batch_runner.py --help``.

    Verifies the env wiring without spending any LLM credits. Raises on
    non-zero exit.
    """
    import subprocess

    repo_root = Path(__file__).resolve().parent.parent
    batch_runner = repo_root / "hermes-agent" / "batch_runner.py"
    if not batch_runner.exists():
        raise FileNotFoundError(f"batch_runner.py not found at {batch_runner}")

    with sandbox(skills_folder_path, run_id="smoke") as h:
        proc = subprocess.run(
            ["python", str(batch_runner), "--", "--help"],
            env=h.env,
            cwd=h.run_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"batch_runner --help failed under sandbox\n"
                f"stdout: {proc.stdout[-400:]}\n"
                f"stderr: {proc.stderr[-400:]}"
            )
        # Verify config.yaml landed where we expect.
        cfg = h.home / "config.yaml"
        assert cfg.exists(), f"missing {cfg}"
        loaded = yaml.safe_load(cfg.read_text())
        ext = loaded["skills"]["external_dirs"]
        assert str(h.skills_dir) in ext, f"external_dirs misconfigured: {ext}"
        print(f"sandbox smoke OK: HERMES_HOME={h.home} skills_dir={h.skills_dir}")


if __name__ == "__main__":
    import sys

    target = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parent.parent / "seed_skills"
    )
    smoke_test(target)
