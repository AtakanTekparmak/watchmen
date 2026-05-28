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
            return f"<<<ADD_FILE {name}/SKILL.md>>>\n{body}<<<END_FILE>>>\n"

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
        return f"<<<EDIT_FILE {src_path}>>>\n{new_content}\n<<<END_FILE>>>\n"


# ---------------------------------------------------------------------------
# OpenRouter client — used when an API key is present.
# ---------------------------------------------------------------------------


class OpenRouterLLM:
    """OpenAI-compatible client pointed at OpenRouter. Lazily imports the SDK.

    Model slugs follow OpenRouter naming (e.g. ``anthropic/claude-opus-4.6``,
    ``minimax/minimax-m2.7``); OpenRouter handles routing to the correct
    upstream provider.
    """

    # kai-skills patch (2026-04-25 — fix track-b K2.6 parse_error)
    # Reasoning models (kimi-k2.6, deepseek-r-style) silently spend the
    # full max_tokens budget on internal reasoning and return content="".
    # Empirically kimi-k2.6 burns ~13K reasoning tokens on a track-b
    # mutation prompt, so we budget 20K total. The reasoning_max_tokens
    # cap is best-effort (Moonshot upstream ignores it); the real safety
    # net is the larger overall ceiling.
    def __init__(
        self,
        *,
        model: str = "anthropic/claude-opus-4.6",
        max_tokens: int = 20000,
        reasoning_max_tokens: Optional[int] = 2000,
        api_key: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover — optional dep
            raise RuntimeError("openai SDK not installed; pip install openai") from exc
        self._model = model
        self._max_tokens = max_tokens
        self._reasoning_max_tokens = reasoning_max_tokens
        # kai-skills patch (Group E, 2026-05-28): optional request-level
        # ``seed`` param for OpenRouter providers that honor determinism
        # (OpenAI, Together, etc.). Threaded into ``chat.completions.create``
        # below when not None so the canonical replicability protocol
        # (n>=3 seeded runs) gets real per-call seeding.
        self._seed = seed
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self._api_key:  # pragma: no cover
            raise RuntimeError("OPENROUTER_API_KEY not set")
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self._api_key,
        )
        # kai-skills patch (2026-04-25 — fix track-b K2.6 parse_error)
        # Optional raw-response sink: if OPENROUTER_RAW_LOG_DIR is set, we
        # dump every (system, user, content, reasoning, finish_reason)
        # tuple as a numbered .json there. Cheap to leave in; only writes
        # when env var is set.
        self._raw_log_dir = os.environ.get("OPENROUTER_RAW_LOG_DIR")
        self._raw_log_counter = 0

    def generate(self, *, system: str, user: str) -> str:
        # kai-skills patch (2026-04-25 — fix track-b K2.6 parse_error)
        kwargs: dict = dict(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if self._reasoning_max_tokens is not None:
            kwargs["extra_body"] = {
                "reasoning": {"max_tokens": self._reasoning_max_tokens}
            }
        # kai-skills patch (Group E, 2026-05-28): forward optional seed.
        if self._seed is not None:
            kwargs["seed"] = self._seed
        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        content = msg.content or ""
        finish_reason = resp.choices[0].finish_reason
        reasoning = getattr(msg, "reasoning", None) or ""

        # kai-skills patch (2026-04-25 — fix track-b K2.6 parse_error)
        if self._raw_log_dir:
            try:
                import json as _json
                from pathlib import Path as _Path

                self._raw_log_counter += 1
                d = _Path(self._raw_log_dir)
                d.mkdir(parents=True, exist_ok=True)
                (d / f"call_{self._raw_log_counter:04d}.json").write_text(
                    _json.dumps(
                        {
                            "model": self._model,
                            "finish_reason": finish_reason,
                            "content_len": len(content),
                            "reasoning_len": len(reasoning),
                            "content": content,
                            "reasoning": reasoning[:4000],
                            "system_first_400": system[:400],
                            "user_first_400": user[:400],
                        },
                        indent=2,
                    )
                )
            except Exception:  # pragma: no cover — debug sink, never fatal
                pass

        # kai-skills patch (2026-04-25 — fix track-b K2.6 parse_error)
        # Loud diagnostic when the model returned no content. This is the
        # exact failure mode that tanked v1/v2 (kimi-k2.6 burned the full
        # max_tokens budget on reasoning). Logging it here makes future
        # regressions instantly visible in track_b.log.
        if not content.strip():
            logger.warning(
                "LLM returned empty content (model=%s, finish_reason=%s, "
                "reasoning_chars=%d). If this repeats, raise max_tokens or "
                "lower reasoning.max_tokens.",
                self._model,
                finish_reason,
                len(reasoning),
            )
        return content


def build_default_client(
    *, force_synthetic: bool, model: str, seed: Optional[int] = None
) -> LLMClient:
    """Factory — returns a synthetic client if no keys or forced, else
    an :class:`OpenRouterLLM` instance.

    Bug 15: a missing ``OPENROUTER_API_KEY`` was silently downgraded to
    SyntheticLLM (random mutations). Live runs would burn the inner
    trial budget against a synthetic outer LLM and never converge.
    Now logged at WARNING so the regression is visible in run logs even
    when the upstream pre-flight check is skipped.
    """
    if force_synthetic:
        return SyntheticLLM(seed=seed)
    if not os.environ.get("OPENROUTER_API_KEY"):
        logger.warning(
            "OPENROUTER_API_KEY missing; falling back to SyntheticLLM "
            "(random mutations). Outer LLM model=%r will NOT be called. "
            "Pass --force-synthetic to silence this warning, or set "
            "OPENROUTER_API_KEY for a live run.",
            model,
        )
        return SyntheticLLM(seed=seed)
    return OpenRouterLLM(model=model, seed=seed)
