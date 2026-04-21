"""Minimal LLM client abstraction for Track B.

Two implementations:

* :class:`SyntheticLLM` — deterministic, offline. Produces a small,
  valid patch on every call. Used for dev runs, tests, and whenever
  no API keys are present. Default for ``--force-synthetic``.
* :class:`OpenRouterLLM` — thin wrapper around the ``openai`` SDK
  pointed at OpenRouter's OpenAI-compatible endpoint. OpenRouter routes
  to the correct upstream provider based on the model slug (e.g.
  ``anthropic/claude-opus-4.6``, ``minimax/minimax-m2.7``).

We deliberately do *not* fork openevolve's ``LLMEnsemble`` machinery:
it is built around process-pool workers + retry policies we don't need
for single-threaded Track-B runs.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Optional, Protocol

from .folder_artifact import FolderArtifact

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    def generate(self, *, system: str, user: str) -> str: ...


# ---------------------------------------------------------------------------
# Synthetic client — deterministic, no network, always valid patch.
# ---------------------------------------------------------------------------

class SyntheticLLM:
    """Produces a cheap but syntactically-valid patch on every call.

    Strategy: pick one skill at random from the parent (we get this via
    the user prompt; parsing is fragile, so we instead receive the
    parent artifact directly via :meth:`set_parent`). Emit an EDIT_FILE
    that appends a short "evolved" footer to its body. This exercises
    the whole pipeline (parse → apply → validate → evaluate → archive)
    without needing a live model.

    When called with no parent context, falls back to a no-op empty
    patch (parser will reject it; caller should skip the iteration).
    """

    def __init__(self, *, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._parent: Optional[FolderArtifact] = None
        self._counter = 0

    def set_parent(self, artifact: FolderArtifact) -> None:
        self._parent = artifact

    def generate(self, *, system: str, user: str) -> str:
        # We ignore the rendered prompt and synthesize deterministically.
        self._counter += 1
        if self._parent is None or self._parent.num_skills() == 0:
            # Add a new minimal skill so the caller has something to evaluate.
            name = f"synthetic-helper-{self._counter:02d}"
            body = (
                f"---\n"
                f"name: {name}\n"
                f"description: Synthetic placeholder skill emitted by the dev LLM. "
                f"Exists only to exercise the mutation pipeline.\n"
                f"version: 0.0.{self._counter}\n"
                f"---\n\n# {name}\n\nSynthetic content.\n"
            )
            return (
                f"<<<ADD_FILE {name}/SKILL.md>>>\n"
                f"{body}"
                f"<<<END_FILE>>>\n"
            )

        # Randomly pick an existing skill, append a generation footer.
        names = sorted(self._parent.skill_names())
        target = self._rng.choice(names)
        src_path = f"{target}/SKILL.md"
        src = self._parent.files[src_path]
        footer = (
            f"\n\n<!-- synthetic-evolve pass #{self._counter} -->\n"
            f"## Evolution note (auto)\n\nRefined by the dev LLM on pass "
            f"{self._counter}.\n"
        )
        new_content = src.rstrip("\n") + footer
        return (
            f"<<<EDIT_FILE {src_path}>>>\n"
            f"{new_content}\n"
            f"<<<END_FILE>>>\n"
        )


# ---------------------------------------------------------------------------
# OpenRouter client — used when an API key is present.
# ---------------------------------------------------------------------------

class OpenRouterLLM:
    """OpenAI-compatible client pointed at OpenRouter. Lazily imports the SDK.

    Model slugs follow OpenRouter naming (e.g. ``anthropic/claude-opus-4.6``,
    ``minimax/minimax-m2.7``); OpenRouter handles routing to the correct
    upstream provider.
    """

    def __init__(self, *, model: str = "anthropic/claude-opus-4.6",
                 max_tokens: int = 4000,
                 api_key: Optional[str] = None) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover — optional dep
            raise RuntimeError(
                "openai SDK not installed; pip install openai") from exc
        self._model = model
        self._max_tokens = max_tokens
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self._api_key:  # pragma: no cover
            raise RuntimeError("OPENROUTER_API_KEY not set")
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self._api_key,
        )

    def generate(self, *, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


def build_default_client(*, force_synthetic: bool, model: str,
                         seed: Optional[int] = None) -> LLMClient:
    """Factory — returns a synthetic client if no keys or forced, else
    an :class:`OpenRouterLLM` instance.
    """
    if force_synthetic or not os.environ.get("OPENROUTER_API_KEY"):
        return SyntheticLLM(seed=seed)
    return OpenRouterLLM(model=model)
