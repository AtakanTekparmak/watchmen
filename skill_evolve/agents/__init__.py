"""Agent backend registry.

Two backends ship today:

* ``"hermes"`` — wraps the existing ``_run_one_task`` Hermes/Docker
  subprocess flow (behavior-preserving relocation of
  ``skill_evolve/evaluator.py:314``).
* ``"bench-cli"`` — shells out to the official SkillsBench
  ``bench eval create`` CLI driving the ``claude-code`` agent (Anthropic's
  first-party CLI, pinned to the paper's version 2.1.19). The
  ``claude-code`` entry isn't in benchflow's built-in registry; the SG-1
  fix in :mod:`skill_evolve.agents.register_claude_code` patches it in at
  runtime to match
  ``skill_evolve/benchmark/vendor/skillsbench/experiments/configs/main-with-skills.yaml``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# SG-1: register the ``claude-code`` agent in benchflow's runtime registry.
# Side-effect import — the module's top-level ``register()`` call adds the
# entry. Idempotent. The bench-cli backend additionally injects an
# ``import skill_evolve.agents.register_claude_code`` shim into the subprocess
# argv so the same registration reaches the bench CLI's interpreter (which
# loads its own benchflow module copy).
from skill_evolve.agents import register_claude_code as register_claude_code  # noqa: F401

# SG-2: monkey-patch ``benchflow._agent_setup.deploy_skills`` so it chowns
# the sandbox user's home dir back after the root-side ``cp -r`` of skills.
# Side-effect-only; the module's top-level ``apply()`` call installs the
# wrapper. Idempotent — see ``_benchflow_patches.py`` for details. Note:
# this only patches the *parent* process; the bench-cli backend additionally
# injects an ``import skill_evolve.agents._benchflow_patches`` shim into the
# subprocess argv so the same patch reaches the bench CLI's interpreter.
from skill_evolve.agents import _benchflow_patches as _benchflow_patches  # noqa: F401

if TYPE_CHECKING:  # pragma: no cover
    from skill_evolve.agents.base import AgentBackend


__all__ = ["get_backend"]


def get_backend(name: str) -> "AgentBackend":
    """Resolve a backend name to an instantiated backend.

    Imports are deferred until lookup so importing this package does
    not pull in ``subprocess`` shims, evaluator state, or any heavy
    Hermes/Docker module at import time — which also avoids a circular
    import between ``evaluator`` and the Hermes backend.
    """
    if name == "hermes":
        from skill_evolve.agents.hermes import HermesBackend

        return HermesBackend()
    if name == "bench-cli":
        from skill_evolve.agents.bench_cli import BenchCliBackend

        return BenchCliBackend()
    raise ValueError(f"unknown agent backend: {name!r} (known: 'hermes', 'bench-cli')")
