"""Register a `claude-code` agent in benchflow's runtime registry.

The SkillsBench paper's experiment configs (vendored at
``skill_evolve/benchmark/vendor/skillsbench/experiments/configs/main-with-skills.yaml:102-109``)
identify the agent as ``name: claude-code, version: 2.1.19`` — Anthropic's
first-party CLI (``@anthropic-ai/claude-code``). Benchflow's built-in registry
ships ``claude-agent-acp`` (Zed's third-party ACP shim) but no ``claude-code``
entry, so calling ``bench eval create -a claude-code`` fails out of the box.

This module patches that gap at import time by calling
``benchflow.agents.registry.register_agent`` with:

  * ``install_cmd`` that pins the upstream Anthropic CLI to version 2.1.19
    (``npm install -g @anthropic-ai/claude-code@2.1.19``) AND installs Zed's
    ``@zed-industries/claude-agent-acp`` ACP shim. The pinned ``claude``
    binary lands on PATH so any task Dockerfile that calls ``claude`` directly
    (e.g. ``tasks/suricata-custom-exfil/environment/Dockerfile``) gets the
    paper-canonical version. The ACP shim is a separate package that uses
    ``@anthropic-ai/claude-agent-sdk`` (the same Anthropic SDK), so its
    behaviour is bit-compatible with what the paper measured even though the
    binary name differs.
  * ``launch_cmd: claude-agent-acp`` — benchflow's runtime is ACP-only
    (``benchflow/_acp_run.py``), so the launched process must speak ACP on
    stdio. Anthropic's bare ``claude`` CLI is a REPL/stream-json interface,
    not an ACP server, and benchflow has no driver for it (the
    ``benchflow.trajectories.claude_code`` module is a postprocessing ATIF
    parser only). Routing through ``claude-agent-acp`` is the only path that
    works without writing a new shim.
  * Same ``requires_env``, ``api_protocol``, ``env_mapping``, ``skill_paths``
    and ``subscription_auth`` as the built-in ``claude-agent-acp`` entry, so
    auth / model selection / skill mounting behave identically.

Side-effect import: simply doing ``import skill_evolve.agents.register_claude_code``
adds the agent to ``benchflow.agents.registry.AGENTS`` for the current Python
process. ``skill_evolve.agents.bench_cli`` invokes the bench CLI as a
subprocess via ``python -c "import skill_evolve.agents.register_claude_code;
from benchflow.cli.main import app; app()"`` so the registration runs inside
the subprocess too — without that pre-import, the subprocess sees only
benchflow's built-in registry and rejects ``-a claude-code``.

Re-imports are idempotent: ``register_agent`` overwrites by name.
"""

from __future__ import annotations

from benchflow.agents.registry import (
    HostAuthFile,
    SubscriptionAuth,
    register_agent,
)

# Pin matches main-with-skills.yaml:102-109 exactly. Bump only when the paper
# config bumps.
_CLAUDE_CODE_VERSION = "2.1.19"

# Node 22 bootstrap mirrors the snippet in benchflow's registry. Duplicated
# here (rather than imported) because benchflow's `_NODE_INSTALL` is a
# module-private constant; importing it would couple us to a private API and
# break on a benchflow patch release.
_NODE_INSTALL = (
    "set -o pipefail; "
    "export DEBIAN_FRONTEND=noninteractive; "
    "NODE_OK=0; "
    "if command -v node >/dev/null 2>&1; then "
    "  NODE_VER=$(node -e 'console.log(process.versions.node.split(\".\")[0])' 2>/dev/null || echo 0); "
    '  [ "$NODE_VER" -ge 22 ] 2>/dev/null && NODE_OK=1; '
    "fi; "
    'if [ "$NODE_OK" = 0 ]; then '
    "  apt-get update -qq && "
    "  apt-get install -y -qq curl ca-certificates && "
    "  curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && "
    "  apt-get install -y -qq nodejs; "
    "fi >/dev/null 2>&1"
)

# Install order:
#   1. Node 22 (idempotent).
#   2. Pinned upstream Anthropic CLI — must be installed BEFORE the ACP shim
#      so that any task / Dockerfile that runs `claude` directly hits 2.1.19.
#   3. Zed's ACP shim — the binary benchflow's launcher actually drives.
_INSTALL_CMD = (
    f"{_NODE_INSTALL} && "
    "( command -v claude >/dev/null 2>&1 && "
    f'  [ "$(claude --version 2>/dev/null | awk \'{{print $1}}\')" = "{_CLAUDE_CODE_VERSION}" ] || '
    f"  npm install -g @anthropic-ai/claude-code@{_CLAUDE_CODE_VERSION} >/dev/null 2>&1 ) && "
    "( command -v claude-agent-acp >/dev/null 2>&1 || "
    "  npm install -g @zed-industries/claude-agent-acp@latest >/dev/null 2>&1 ) && "
    "command -v claude-agent-acp >/dev/null 2>&1 && "
    "command -v claude >/dev/null 2>&1"
)


def register() -> None:
    """Add ``claude-code`` to benchflow's agent registry. Idempotent."""
    cfg = register_agent(
        name="claude-code",
        description=(
            f"Anthropic Claude Code CLI v{_CLAUDE_CODE_VERSION} "
            "(driven via @zed-industries/claude-agent-acp ACP shim)"
        ),
        skill_paths=["$HOME/.claude/skills"],
        install_cmd=_INSTALL_CMD,
        launch_cmd="claude-agent-acp",
        protocol="acp",
        requires_env=["ANTHROPIC_API_KEY"],
        env_mapping={
            "BENCHFLOW_PROVIDER_BASE_URL": "ANTHROPIC_BASE_URL",
            "BENCHFLOW_PROVIDER_API_KEY": "ANTHROPIC_AUTH_TOKEN",
            "BENCHFLOW_PROVIDER_MODEL": "ANTHROPIC_MODEL",
        },
        subscription_auth=SubscriptionAuth(
            replaces_env="ANTHROPIC_API_KEY",
            detect_file="~/.claude/.credentials.json",
            files=[
                HostAuthFile(
                    "~/.claude/.credentials.json",
                    "{home}/.claude/.credentials.json",
                ),
            ],
        ),
    )
    # ``register_agent`` doesn't expose ``api_protocol`` as a kwarg, but the
    # built-in ``claude-agent-acp`` entry sets it to ``anthropic-messages``
    # so the SDK picks the right endpoint when a provider exposes multiple
    # (see ``benchflow/_agent_env.py``). Assign it post-hoc to stay
    # bit-compatible with the original entry.
    cfg.api_protocol = "anthropic-messages"


# Run at import time — `bench_cli.py` relies on `import register_claude_code`
# being a no-arg side-effect that registers the agent.
register()


__all__ = ["register"]
