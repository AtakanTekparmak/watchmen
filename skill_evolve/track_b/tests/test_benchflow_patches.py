"""Tests for the SG-2 monkey-patch on benchflow.deploy_skills.

The bug: ``benchflow._agent_setup.deploy_skills`` runs
``mkdir -p /home/{u}/.claude/skills && cp -r /skills/. ...`` as root
*after* ``setup_sandbox_user`` chowned ``/home/{u}`` to ``{u}:{u}``.
The freshly-created ``/home/{u}/.claude/`` ends up root-owned and the
agent later hits ``EACCES`` when it tries to ``mkdir
/home/{u}/.claude/session-env``.

These tests verify:

  1. The patch wraps ``benchflow._agent_setup.deploy_skills``.
  2. The wrapper calls the original first, then issues
     ``chown -R {user}:{user} /home/{user}`` on the same ``env``.
  3. The wrapper does *not* chown when ``sandbox_user`` is ``None``.
  4. The wrapper does *not* chown when no skills were distributed.
  5. The patch is idempotent — applying twice leaves one wrapper.
  6. The bench-cli argv shim imports the patches module before
     entering ``benchflow.cli.main``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock


import benchflow._agent_setup as _agent_setup
from skill_evolve.agents import _benchflow_patches as patches


def _fresh_env_mock() -> SimpleNamespace:
    """Build a minimal env stub mirroring the surface deploy_skills uses."""
    env = SimpleNamespace()
    env.exec = AsyncMock()
    env.upload_dir = AsyncMock()
    return env


def _fake_task(task_skills_dir: str | None = None) -> SimpleNamespace:
    """Minimal Task-shaped stub: only the attrs deploy_skills reads."""
    return SimpleNamespace(
        config=SimpleNamespace(
            environment=SimpleNamespace(skills_dir=task_skills_dir),
        ),
    )


def _fake_agent_cfg(
    skill_paths: List[str] | None = None,
    name: str = "claude-agent-acp",
) -> SimpleNamespace:
    return SimpleNamespace(name=name, skill_paths=skill_paths or [])


def test_apply_wraps_deploy_skills() -> None:
    """After apply(), deploy_skills carries the patch sentinel."""
    # apply() ran on import. Verify it's installed.
    assert patches.is_applied(), "patches module imports should auto-apply the wrapper"
    fn = _agent_setup.deploy_skills
    assert getattr(fn, patches._PATCH_SENTINEL, False) is True


def test_apply_is_idempotent() -> None:
    """Calling apply() twice leaves exactly one wrapper installed."""
    patches.apply()
    first = _agent_setup.deploy_skills
    patches.apply()
    second = _agent_setup.deploy_skills
    assert first is second
    # Ensure we didn't double-wrap by checking sentinel is still
    # exactly the wrapper, not a wrapper-of-wrapper. Double-wrapping
    # would also set the sentinel, but we'd see two layers; identity
    # preservation is the strongest test.
    assert getattr(second, patches._PATCH_SENTINEL, False) is True


def test_wrapper_chowns_after_original() -> None:
    """When sandbox_user is set and skills distribute, the chown follows."""
    env = _fresh_env_mock()
    task = _fake_task()
    agent_cfg = _fake_agent_cfg(skill_paths=["$HOME/.claude/skills"])

    async def run() -> None:
        await _agent_setup.deploy_skills(
            env=env,
            task_path=SimpleNamespace(),  # accessed only via attribute on real call
            skills_dir="/local/skills/dir",  # truthy → original would distribute
            agent_cfg=agent_cfg,
            sandbox_user="agent",
            agent_cwd="/app",
            task=task,
        )

    # We can't run the real deploy_skills (it would call
    # env.upload_dir on a non-existent path). Replace the underlying
    # ``original`` inside the wrapper by monkeypatching the
    # _agent_setup module to a stub original, re-applying, then
    # invoking. We re-apply at the module level because the wrapper
    # closure captured the prior ``original``.
    real = _agent_setup.deploy_skills

    async def stub_original(
        env, task_path, skills_dir, agent_cfg, sandbox_user, agent_cwd, task
    ) -> None:
        # Mirror the original distribution exec so the wrapper has
        # something on env.exec history to detect — i.e. one cp call.
        await env.exec("mkdir -p /home/agent/.claude/skills && cp ...", timeout_sec=15)

    # Reset the patch chain: install the stub as the unpatched
    # baseline, then re-apply so the wrapper wraps the stub.
    _agent_setup.deploy_skills = stub_original
    setattr(stub_original, patches._PATCH_SENTINEL, False)
    patches.apply()
    try:
        asyncio.run(run())
    finally:
        # Restore the production wrapper for downstream tests.
        _agent_setup.deploy_skills = real

    # env.exec was called twice: once by the stub original, once by
    # the wrapper's chown follow-up.
    calls = env.exec.await_args_list
    assert len(calls) == 2, (
        f"expected stub-original + chown = 2 exec calls, got {len(calls)}"
    )
    chown_cmd = calls[-1].args[0] if calls[-1].args else calls[-1].kwargs.get("cmd")
    assert "chown -R" in chown_cmd
    assert "agent:agent" in chown_cmd
    assert "/home/agent" in chown_cmd


def test_wrapper_skips_chown_when_no_sandbox_user() -> None:
    """sandbox_user=None → no chown (original writes to /root)."""
    env = _fresh_env_mock()
    task = _fake_task()
    agent_cfg = _fake_agent_cfg(skill_paths=["$HOME/.claude/skills"])

    real = _agent_setup.deploy_skills

    async def stub_original(
        env, task_path, skills_dir, agent_cfg, sandbox_user, agent_cwd, task
    ) -> None:
        await env.exec("noop", timeout_sec=15)

    _agent_setup.deploy_skills = stub_original
    setattr(stub_original, patches._PATCH_SENTINEL, False)
    patches.apply()

    async def run() -> None:
        await _agent_setup.deploy_skills(
            env=env,
            task_path=SimpleNamespace(),
            skills_dir="/local/skills/dir",
            agent_cfg=agent_cfg,
            sandbox_user=None,  # ← key
            agent_cwd="/app",
            task=task,
        )

    try:
        asyncio.run(run())
    finally:
        _agent_setup.deploy_skills = real

    # Only the stub-original call; no chown follow-up.
    assert len(env.exec.await_args_list) == 1


def test_wrapper_skips_chown_when_no_distribution() -> None:
    """Agent has no skill_paths and no task skills_dir → no chown."""
    env = _fresh_env_mock()
    task = _fake_task(task_skills_dir=None)
    agent_cfg = _fake_agent_cfg(skill_paths=[])  # empty → no distribution

    real = _agent_setup.deploy_skills

    async def stub_original(
        env, task_path, skills_dir, agent_cfg, sandbox_user, agent_cwd, task
    ) -> None:
        return None  # nothing distributed → no env.exec at all

    _agent_setup.deploy_skills = stub_original
    setattr(stub_original, patches._PATCH_SENTINEL, False)
    patches.apply()

    async def run() -> None:
        await _agent_setup.deploy_skills(
            env=env,
            task_path=SimpleNamespace(),
            skills_dir=None,  # ← no runtime skills
            agent_cfg=agent_cfg,
            sandbox_user="agent",
            agent_cwd="/app",
            task=task,
        )

    try:
        asyncio.run(run())
    finally:
        _agent_setup.deploy_skills = real

    assert env.exec.await_args_list == []


def test_wrapper_uses_dynamic_sandbox_user() -> None:
    """A non-default sandbox_user (e.g. 'evaluator') flows through."""
    env = _fresh_env_mock()
    task = _fake_task()
    agent_cfg = _fake_agent_cfg(skill_paths=["$HOME/.claude/skills"])

    real = _agent_setup.deploy_skills

    async def stub_original(
        env, task_path, skills_dir, agent_cfg, sandbox_user, agent_cwd, task
    ) -> None:
        return None

    _agent_setup.deploy_skills = stub_original
    setattr(stub_original, patches._PATCH_SENTINEL, False)
    patches.apply()

    async def run() -> None:
        await _agent_setup.deploy_skills(
            env=env,
            task_path=SimpleNamespace(),
            skills_dir="/local/skills",
            agent_cfg=agent_cfg,
            sandbox_user="evaluator",
            agent_cwd="/app",
            task=task,
        )

    try:
        asyncio.run(run())
    finally:
        _agent_setup.deploy_skills = real

    chown_call = env.exec.await_args_list[-1]
    chown_cmd = chown_call.args[0] if chown_call.args else chown_call.kwargs["cmd"]
    assert "evaluator:evaluator" in chown_cmd
    assert "/home/evaluator" in chown_cmd


def test_bench_cli_argv_uses_patch_shim() -> None:
    """BenchCliBackend's argv enters bench through the patches shim."""
    from skill_evolve.agents.bench_cli import (
        _BENCH_SHIM_CODE,
        _bench_cli_argv,
    )

    argv = _bench_cli_argv("/tmp/scene.yaml", "/tasks/foo", "claude-opus-4-7")
    # Must invoke a real Python interpreter (not the bench script) so
    # the shim's import-then-patch sequence runs in the same process.
    assert argv[0].endswith("python") or "python" in argv[0]
    assert argv[1] == "-c"
    assert argv[2] == _BENCH_SHIM_CODE
    assert "skill_evolve.agents._benchflow_patches" in _BENCH_SHIM_CODE
    # The patch import must come before the bench import inside the shim.
    assert _BENCH_SHIM_CODE.index("_benchflow_patches") < _BENCH_SHIM_CODE.index(
        "benchflow.cli.main"
    )
    # Standard bench eval create flags survive. benchflow 0.3.4 accepts only
    # the long forms (--config/--tasks-dir/--agent/--model); the short
    # -f/-t/-a/-m aliases were dropped and silently zeroed every candidate at
    # arg-parse (2026-05-28 silent-fail incident).
    assert argv[3:5] == ["eval", "create"]
    assert "--config" in argv and "/tmp/scene.yaml" in argv
    assert "--tasks-dir" in argv and "/tasks/foo" in argv
    # ``--agent`` resolves to the SG-1-registered ``claude-code`` agent in the
    # subprocess registry; the shim above performs that registration before
    # ``benchflow.cli.main`` resolves the ``--agent`` flag.
    assert "--agent" in argv and "claude-code" in argv
    assert "--model" in argv and "claude-opus-4-7" in argv
