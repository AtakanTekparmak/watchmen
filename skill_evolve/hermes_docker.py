"""Per-task Docker backend configuration for the Hermes agent subprocess.

This module produces the env-var patch that makes ``run_agent.py`` execute
commands *inside* the task's Docker container instead of on the host. It is
called by the evaluator when a tblite task is being rolled out with
real-verifier mode enabled.

Why env vars (and not ``config.yaml``)
--------------------------------------
``run_agent.py`` does NOT read ``$HERMES_HOME/config.yaml``'s ``terminal:``
block — only the interactive ``hermes`` CLI does. The env-var bridge at
``hermes-agent/tools/terminal_tool.py::_get_env_config`` is the only way to
configure the terminal backend for a subprocess-spawned agent. So we emit
``TERMINAL_*`` env vars and rely on that bridge; adding a ``terminal:`` block
to the sandbox config would be silently ignored and would mislead future
readers.

The double-mount trick
----------------------
``DockerEnvironment`` hardcodes ``/workspace`` as the container's bind-mount
target (see ``hermes-agent/tools/environments/docker.py:340``). The TBLite
task scripts, however, expect files under ``/app``. We can't change the
hardcoded target without patching Hermes' source — but we CAN add extra
``-v`` mounts via ``TERMINAL_DOCKER_VOLUMES``. We therefore mount the host
workspace dir twice: once at ``/workspace`` (Hermes' default cwd) and once
at ``/app`` (where test.sh and the task prompt look). Both mount targets
reference the same host files, so agent edits at ``/app/foo`` appear at
``/workspace/foo`` and vice-versa.

Reference: the exact precedent for the per-task env-var layout is at
``hermes-agent/environments/benchmarks/terminalbench_2/terminalbench2_env.py:505-512``.
The image pre-pull pattern lifted here is at
``hermes-agent/batch_runner.py:256-290``.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Env patch builder
# ---------------------------------------------------------------------------


def build_env_patch(
    task: Dict[str, Any],
    workspace_host_dir: Path,
) -> Dict[str, str]:
    """Build the ``TERMINAL_*`` env patch for a Hermes agent subprocess.

    The patch configures the terminal tool to route all shell commands
    through an ephemeral Docker container spawned from the task's own
    image, with the host workspace double-mounted at ``/app`` and
    ``/workspace`` (see module docstring for rationale).

    Args:
        task: Benchmark task dict. Must have a
            ``success_check_payload.docker_image`` entry.
        workspace_host_dir: Host directory that should be made visible
            inside the container. Resolved to an absolute path before
            being embedded in the volume spec — relative paths are
            silently rejected by Docker's ``-v`` flag.

    Returns:
        Mapping suitable for merging into a subprocess env dict
        (``env.update(patch)``). Keys:

          * ``TERMINAL_ENV = "docker"``
          * ``TERMINAL_DOCKER_IMAGE``
          * ``TERMINAL_CWD = "/app"``
          * ``TERMINAL_CONTAINER_PERSISTENT = "false"``
          * ``TERMINAL_DOCKER_VOLUMES`` — JSON list of two ``host:container``
            bind-mount specs
          * ``TERMINAL_DOCKER_FORWARD_ENV`` — JSON list of host env names to
            propagate (currently just ``OPENROUTER_API_KEY``)

    Raises:
        ValueError: if the task has no ``docker_image`` in its payload.
    """
    payload = task.get("success_check_payload") or {}
    image = payload.get("docker_image")
    if not image:
        raise ValueError(
            "task has no success_check_payload.docker_image; "
            "cannot configure per-task docker backend "
            f"(task_id={task.get('task_id')!r})"
        )

    abs_workspace = str(Path(workspace_host_dir).resolve())

    # Double-mount: /app is where test.sh + the task prompt look; /workspace
    # is Hermes' hardcoded default cwd. Both reference the same host dir so
    # files stay in sync.
    volumes = [
        f"{abs_workspace}:/app",
        f"{abs_workspace}:/workspace",
    ]
    # OPENROUTER_API_KEY is forwarded so the agent's LLM client can reach
    # the provider from inside the container. Expand this list if future
    # tasks need more env vars (e.g. ANTHROPIC_API_KEY).
    forward_env = ["OPENROUTER_API_KEY"]

    return {
        "TERMINAL_ENV": "docker",
        "TERMINAL_DOCKER_IMAGE": str(image),
        "TERMINAL_CWD": "/app",
        "TERMINAL_CONTAINER_PERSISTENT": "false",
        "TERMINAL_DOCKER_VOLUMES": json.dumps(volumes),
        "TERMINAL_DOCKER_FORWARD_ENV": json.dumps(forward_env),
    }


# ---------------------------------------------------------------------------
# Image pre-pull
# ---------------------------------------------------------------------------


def ensure_image_pulled(image: str, timeout_s: int = 300) -> None:
    """Guarantee the given Docker image is present locally, pulling if not.

    Fast path: ``docker image inspect`` — if returncode is 0 the image is
    already cached and we return immediately (typical case after a warm
    benchmark run). Only falls through to ``docker pull`` on the first
    invocation for a given image, matching the pattern used by
    ``hermes-agent/batch_runner.py:256-290``.

    Args:
        image: Fully-qualified image reference (e.g.
            ``nousresearch/tblite-log-summary:latest``).
        timeout_s: Wall-clock cap for the ``docker pull`` call. Large TBLite
            images on slow connections can take several minutes on first
            fetch, so the default (300s) is deliberately generous.

    Raises:
        RuntimeError: if the pull fails. Message includes a tail of stderr
            so the caller can surface it to the user without digging through
            log files.
    """
    # Fast path: already cached?
    probe = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        check=False,
    )
    if probe.returncode == 0:
        logger.debug("docker image %s already present; skipping pull", image)
        return

    logger.info("pulling docker image %s (timeout=%ss)", image, timeout_s)
    try:
        pull = subprocess.run(
            ["docker", "pull", image],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"docker pull {image} exceeded {timeout_s}s timeout"
        ) from exc

    if pull.returncode != 0:
        stderr_tail = (pull.stderr or "")[-500:]
        raise RuntimeError(
            f"docker pull {image} failed (rc={pull.returncode}): {stderr_tail}"
        )


# ---------------------------------------------------------------------------
# Daemon availability probe
# ---------------------------------------------------------------------------

_DOCKER_AVAILABLE_CACHE: Optional[bool] = None


def docker_backend_available() -> bool:
    """Return True iff the Docker daemon responds to ``docker info``.

    Cached at module scope (same convention as
    ``skill_evolve.verifiers._DOCKER_AVAILABLE_CACHE``) so callers can probe
    cheaply in hot loops. The cache is only invalidated by
    :func:`_reset_docker_cache` — intended for tests.
    """
    global _DOCKER_AVAILABLE_CACHE
    if _DOCKER_AVAILABLE_CACHE is not None:
        return _DOCKER_AVAILABLE_CACHE
    docker = shutil.which("docker")
    if docker is None:
        _DOCKER_AVAILABLE_CACHE = False
        return False
    try:
        proc = subprocess.run(
            [docker, "info"],
            capture_output=True,
            timeout=5,
        )
        _DOCKER_AVAILABLE_CACHE = proc.returncode == 0
    except Exception as exc:
        logger.warning("docker info probe failed: %s", exc)
        _DOCKER_AVAILABLE_CACHE = False
    return _DOCKER_AVAILABLE_CACHE


def _reset_docker_cache() -> None:
    """Test hook — re-probe Docker on next :func:`docker_backend_available` call."""
    global _DOCKER_AVAILABLE_CACHE
    _DOCKER_AVAILABLE_CACHE = None
