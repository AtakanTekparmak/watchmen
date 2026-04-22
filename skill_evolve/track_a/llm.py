"""Thin LLM wrapper around the OpenAI SDK (pointed at OpenRouter) with a
``synthetic`` mode.

Two knobs matter:

* ``LLMClient.synthetic`` — when True, :meth:`complete` returns a canned
  response from :data:`SYNTHETIC_CANNED` keyed by the *role tag* you pass
  in. Lets the whole A/B/AB loop run offline for dev + tests.

* ``LLMClient.model`` — OpenRouter model slug (e.g.
  ``anthropic/claude-opus-4.6``, ``minimax/minimax-m2.7``). OpenRouter
  routes to the correct upstream provider based on this slug.

We pass the *system* prompt as the first message with ``role="system"``
and the user prompt as a second message with ``role="user"``. None of
our prompts are multi-turn.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# OpenRouter slug for the default model. Verified via
# https://openrouter.ai/anthropic/claude-opus-4.6 — OpenRouter uses a dot
# in the minor-version (``4.6``), not a hyphen (``4-6``).
DEFAULT_MODEL = "anthropic/claude-opus-4.6"


# ---------------------------------------------------------------------------
# Canned synthetic responses
# ---------------------------------------------------------------------------
#
# Keyed by a short tag the caller passes in ``tag=``. These responses are
# engineered to pass the loop's JSON/format parsers so the synthetic mode
# exercises real code paths, just without LLM spend.

SYNTHETIC_CANNED = {
    "critic": (
        "1. ask-the-environment: description is too generic; cites no "
        "concrete tool names from terminal-bench. Offending passage: "
        "'Prefer a cheap read-only command over guessing.' — this is "
        "platitude, not guidance.\n"
        "2. read-before-write: overlaps substantially with patch-then-verify's "
        "'Re-read the patched region' step. Risk: agent loads both and gets "
        "conflicting advice on re-read cadence.\n"
        "3. MISSING: no skill covers interpretation of SWE-bench harness "
        "error formats. The agent has no guidance on decoding harness "
        "output when a patch fails to apply.\n"
    ),
    "op_planner": (
        '{"op": "RewriteSkillContent", "name": "ask-the-environment", '
        '"critique_excerpt": "description is too generic; cites no concrete '
        'tool names from terminal-bench."}'
    ),
    "rewrite_body": (
        "# Ask the Environment\n\n"
        "## Overview\n\n"
        "Read-only inspection grounds every action. When you need to know a "
        "path, version, branch, or file state, run the one command that "
        "answers the question.\n\n"
        "## Iron Law\n\n"
        "```\nNO ACTION ON AN UNVERIFIED ASSUMPTION\n```\n\n"
        "## When to Use\n\n"
        "Before running builds, tests, installs, edits, or commits.\n\n"
        "## Steps\n\n"
        "1. Form the specific question.\n"
        "2. Pick the minimal command.\n"
        "3. Run it, read output.\n"
        "4. Act on observed reality.\n\n"
        "## Red Flags\n\n"
        "- Calling `python` without `which python`.\n"
        "- Assuming $FOO is set without `printenv FOO`.\n"
    ),
    "new_body": (
        "# New Skill\n\n"
        "## Overview\n\n"
        "Placeholder body emitted by synthetic LLM mode.\n\n"
        "## Iron Law\n\n"
        "```\nPLACEHOLDER IRON LAW\n```\n\n"
        "## When to Use\n\n"
        "- Never in production; this is a synthetic-mode artifact.\n\n"
        "## Steps\n\n"
        "1. Step one.\n2. Step two.\n"
    ),
    "split": (
        "<<<SKILL_A>>>\n"
        "# Skill A\n\nSynthetic split half A.\n\n## Iron Law\nA\n"
        "<<<END_SKILL_A>>>\n"
        "<<<SKILL_B>>>\n"
        "# Skill B\n\nSynthetic split half B.\n\n## Iron Law\nB\n"
        "<<<END_SKILL_B>>>\n"
    ),
    "merge": (
        "# Merged Skill\n\n"
        "## Overview\n\n"
        "Synthetic merge of two overlapping skills.\n\n"
        "## Iron Law\n\nMERGED\n\n"
        "## When to Use\n\n- Synthetic mode only.\n"
    ),
    "synth": (
        '{"skills": []}'
    ),
    "new_script": (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "\n"
        "# Synthetic-mode placeholder script. Writes a marker to the first\n"
        "# positional argument (default /app/probe.txt) and exits 0.\n"
        "OUT=\"${1:-/app/probe.txt}\"\n"
        "echo \"ok $(date -Iseconds)\" > \"$OUT\"\n"
    ),
    "rewrite_script": (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "\n"
        "# Synthetic rewrite_script placeholder. Preserves the path, adds\n"
        "# a minimal shebang + exit-on-error guard.\n"
        "exit 0\n"
    ),
}


@dataclass
class LLMClient:
    model: str = DEFAULT_MODEL
    synthetic: bool = False
    max_tokens: int = 4096
    temperature: float = 0.7
    _client: Optional[object] = None

    def __post_init__(self) -> None:
        if self.synthetic:
            return
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "openai SDK not installed; pip install openai, or "
                "pass --force-synthetic"
            ) from e
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set; export it or pass "
                "--force-synthetic to use canned responses."
            )
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    # ------------------------------------------------------------------

    def complete(
        self,
        system: str,
        user: str,
        *,
        tag: str = "generic",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Send one (system, user) pair; return the assistant text."""
        if self.synthetic:
            return SYNTHETIC_CANNED.get(tag, "(synthetic placeholder response)")

        assert self._client is not None
        resp = self._client.chat.completions.create(  # type: ignore[attr-defined]
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            temperature=self.temperature if temperature is None else temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""
