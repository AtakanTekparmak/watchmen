"""Runtime monkey-patches for ``benchflow`` (SG-2 fix).

benchflow's ``deploy_skills`` (``benchflow/_agent_setup.py:65``) runs
``mkdir -p /home/{user}/.claude/skills && cp -r /skills/. ...`` as root
*after* ``setup_sandbox_user`` has already chowned ``/home/{user}`` to
``{user}:{user}``. The freshly-created ``/home/{user}/.claude/`` and
``/home/{user}/.claude/skills`` end up root-owned, so when the agent
(running as ``{user}``) later tries to ``mkdir
/home/{user}/.claude/session-env`` it hits ``EACCES``. Evidence:
61/353 with-skills trials in ``acp_trajectory.jsonl`` recorded
``EACCES: permission denied, mkdir '/home/agent/.claude/session-env'``;
zero on the no-skills arm (which never deploys skills).

This module wraps ``deploy_skills`` so that immediately after the
original distribution finishes it issues a ``chown -R {user}:{user}
/home/{user}`` to restore agent ownership of every directory the cp
created. The original sandbox lockdown in ``setup_sandbox_user``
already chowned the home dir tree, so this is a re-application after
the root-side cp -r — not a new privilege grant.

Why a wrapper rather than editing the cp shell command in place: the
distribution loop builds one chained shell string per agent skill_path,
and the chown needs to follow *all* those mkdir+cp segments (not just
the last one) — and only when ``sandbox_user`` is set. A one-shot
follow-up exec keeps the patch surgical and makes the chown
idempotent regardless of how many skill_paths the agent declared.

Idempotency:
    Importing this module twice must not double-wrap. We sentinel via
    a ``__benchflow_sg2_patched`` attribute on the wrapped function.

Subprocess reach:
    The ``BenchCliBackend`` invokes ``bench eval create`` as a
    subprocess (see ``skill_evolve/agents/bench_cli.py``). Patching
    this parent process accomplishes nothing for that path. The
    backend's ``argv`` is shaped to enter the bench CLI through a
    ``-c`` shim that imports this module first; see
    ``_bench_cli_argv`` below.
"""

from __future__ import annotations

import logging
import shlex
import sys
from typing import Any

import benchflow._agent_setup as _agent_setup

logger = logging.getLogger(__name__)


_PATCH_SENTINEL = "__benchflow_sg2_patched"


def _build_chown_cmd(sandbox_user: str) -> str:
    """Build the chown shell command run after deploy_skills finishes."""
    user_q = shlex.quote(sandbox_user)
    home_q = shlex.quote(f"/home/{sandbox_user}")
    return f"chown -R {user_q}:{user_q} {home_q}"


def _wrap_deploy_skills(original):
    """Wrap ``deploy_skills`` so it chowns the sandbox user's home dir.

    The wrapper preserves the original signature exactly, awaits the
    original coroutine, and — when ``sandbox_user`` is set and skills
    were actually distributed (i.e. the cp -r ran) — issues a
    follow-up chown. If ``sandbox_user`` is ``None`` the original
    function already targets ``/root``, so no chown is needed.
    """

    async def wrapped_deploy_skills(
        env,
        task_path,
        skills_dir,
        agent_cfg,
        sandbox_user,
        agent_cwd,
        task,
    ) -> None:
        await original(
            env,
            task_path,
            skills_dir,
            agent_cfg,
            sandbox_user,
            agent_cwd,
            task,
        )
        # Only chown when there's a non-root sandbox user to restore
        # ownership for. The original function only writes into
        # /home/{user} when sandbox_user is truthy; otherwise it
        # writes into /root which is already root-owned and correct.
        if not sandbox_user:
            return
        # Skip when no distribution happened (agent_cfg or
        # skill_paths absent, or no effective skills dir). Mirrors the
        # original function's distribution guard.
        task_skills_dir = task.config.environment.skills_dir
        effective_skills = "/skills" if skills_dir else task_skills_dir
        if not (effective_skills and agent_cfg and agent_cfg.skill_paths):
            return
        chown_cmd = _build_chown_cmd(sandbox_user)
        try:
            await env.exec(chown_cmd, timeout_sec=15)
            logger.info(
                "SG-2 patch: chowned /home/%s back to %s after deploy_skills",
                sandbox_user,
                sandbox_user,
            )
        except Exception:  # pragma: no cover — defensive logging
            logger.exception(
                "SG-2 patch: chown of /home/%s failed after deploy_skills",
                sandbox_user,
            )
            raise

    setattr(wrapped_deploy_skills, _PATCH_SENTINEL, True)
    # Preserve identity-friendly metadata for any code that
    # introspects deploy_skills.__name__ etc.
    wrapped_deploy_skills.__name__ = getattr(original, "__name__", "deploy_skills")
    wrapped_deploy_skills.__qualname__ = getattr(
        original, "__qualname__", "deploy_skills"
    )
    wrapped_deploy_skills.__doc__ = getattr(original, "__doc__", None)
    return wrapped_deploy_skills


def apply() -> None:
    """Install the SG-2 patch on ``benchflow._agent_setup.deploy_skills``.

    Idempotent: calling this twice (or importing this module from two
    paths) leaves exactly one wrapper installed.
    """
    current = getattr(_agent_setup, "deploy_skills", None)
    if current is None:  # pragma: no cover — defensive
        logger.warning(
            "SG-2 patch: benchflow._agent_setup.deploy_skills missing; skipping"
        )
        return
    if getattr(current, _PATCH_SENTINEL, False):
        return  # already wrapped
    wrapped = _wrap_deploy_skills(current)
    _agent_setup.deploy_skills = wrapped

    # Also rebind in any module that has already done
    # ``from benchflow._agent_setup import deploy_skills`` (notably
    # ``benchflow.trial`` and ``benchflow.sdk``). Without this, those
    # modules keep a reference to the original unwrapped function.
    for mod_name in ("benchflow.trial", "benchflow.sdk"):
        mod = sys.modules.get(mod_name)
        if mod is not None and getattr(mod, "deploy_skills", None) is current:
            mod.deploy_skills = wrapped


def is_applied() -> bool:
    """Return True iff the wrapper is currently installed."""
    fn: Any = getattr(_agent_setup, "deploy_skills", None)
    return bool(fn and getattr(fn, _PATCH_SENTINEL, False))


# Apply on import so a single ``import skill_evolve.agents._benchflow_patches``
# is enough to activate the fix.
apply()
