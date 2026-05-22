"""Thin wrapper around watchmen's provider layer.

Daycare deliberately does NOT re-implement provider routing — we lean on
watchmen.providers / watchmen.agent and only add:
  - raw_log_call(): K5 (write every OR response to disk when
    OPENROUTER_RAW_LOG_DIR is set).
  - ping_model(): convenience wrapper around Provider.probe() for the
    Phase 0 pre-launch ping.

Public re-exports live at the top so callers can write:

    from daycare.providers import Agent, chat_call, Provider, raw_log_call
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Re-exports — these are the only places daycare touches watchmen for LLM I/O.
from watchmen.agent import Agent, chat_call  # noqa: F401
from watchmen.providers import Provider  # noqa: F401
from watchmen import providers as _providers  # noqa: F401


def raw_log_call(response_json: dict, call_id: str) -> None:
    """Write a raw OpenRouter response to disk for offline debugging.

    Gated on the OPENROUTER_RAW_LOG_DIR env var (K5). Silently no-ops if
    the env var is unset — that's the documented "off" state.

    Files are named ``<UTC-iso-timestamp>_<call_id>.json`` so a directory
    listing sorts chronologically.
    """
    log_dir = os.environ.get("OPENROUTER_RAW_LOG_DIR")
    if not log_dir:
        return

    out_dir = Path(log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use microsecond precision to avoid collisions across rapid-fire calls
    # from a single rollout batch. Colons are illegal on Windows filesystems
    # so we use a filename-safe iso variant.
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    out_path = out_dir / f"{ts}_{call_id}.json"
    out_path.write_text(json.dumps(response_json, indent=2, default=str))


def ping_model(model_slug: str, provider_name: str = "openrouter") -> dict:
    """Run a cheap liveness probe against the named provider+model.

    Resolves the provider from watchmen's registry, calls ``.probe()``, and
    returns the result as a plain dict so callers (Phase 0 doctor) can
    serialise it into ``doctor.json``.

    The spec's signature passes ``model_slug`` as the first arg; watchmen's
    Provider.probe() signature takes an api_key. We resolve the api_key
    from watchmen's standard load path before delegating, and surface the
    model_slug in the returned payload for downstream version-pinning.
    """
    provider = _providers.get_provider(provider_name)

    # Pull the configured api_key the same way watchmen's agent.load_api_key
    # does — provider.resolve_api_key handles keychain / env / file fallbacks.
    from watchmen.agent import load_api_key

    api_key = load_api_key(provider_name)
    probe = provider.probe(api_key)

    # Surface as a plain dict so the doctor.json serialiser doesn't need
    # to know about ProbeResult.
    return {
        "ok": bool(getattr(probe, "ok", False)),
        "detail": str(getattr(probe, "detail", "")),
        "provider": provider_name,
        "model_slug": model_slug,
    }
