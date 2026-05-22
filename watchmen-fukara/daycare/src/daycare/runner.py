"""Subprocess-isolated weak-model rollout runner (K10).

Each weak-model invocation runs in its own ``python -c`` subprocess to
prevent global state mutation between rollouts (per K10 of the spec).
The subprocess makes a single OpenRouter chat completion call via httpx,
prints JSON to stdout, and exits. The parent deserialises and packages
the result into a ``RunResult``.

Also exposes ``build_skill_system_prompt(bundle_dir)`` — the spec's
injection format (Phase "Skill injection format"): strip SKILL.md
frontmatter, inline each script with a header + 4000-char truncation,
prepend a base-directory marker line.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


# ─── Public dataclass ──────────────────────────────────────────────────────


@dataclass
class RunResult:
    """One rollout's outcome.

    score is filled in by the verifier later (the runner itself doesn't
    score — it just produces the completion). We pre-populate it here so
    downstream W9 aggregation can treat the type uniformly.
    """

    score: float
    completion: str
    completion_len_tokens: int
    error: str | None


# ─── Skill prompt construction (Phase "Skill injection format") ───────────


_SKILL_PLACEHOLDER = "<SKILL_SLUG_PLACEHOLDER>"
_SCRIPT_TRUNC_CHARS = 4000


def _strip_frontmatter(text: str) -> str:
    """Strip a leading YAML frontmatter block delimited by ``---`` lines.

    Mirrors Claude Code's behaviour: if the first non-empty line is
    ``---``, everything up to the next ``---`` line is dropped. If no
    closing fence exists, the text is returned unchanged (defensive — a
    malformed SKILL.md shouldn't silently lose all content).
    """
    if not text:
        return text
    lines = text.splitlines()
    # Find first non-empty line.
    i = 0
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    if i >= len(lines) or lines[i].strip() != "---":
        return text
    # Look for closing fence after the opener.
    j = i + 1
    while j < len(lines) and lines[j].strip() != "---":
        j += 1
    if j >= len(lines):
        # No closing fence; leave the file alone.
        return text
    # Drop lines [i .. j] inclusive (the fence + body).
    remaining = lines[j + 1 :]
    # Strip a leading blank line if present so the output doesn't start
    # with awkward whitespace.
    while remaining and remaining[0].strip() == "":
        remaining.pop(0)
    return "\n".join(remaining)


def build_skill_system_prompt(bundle_dir: Path) -> str:
    """Build the system-prompt-prefix string injected into every rollout.

    Layout (per spec §"Skill injection format"):

        Base directory for this skill: <SKILL_SLUG_PLACEHOLDER>

        <SKILL.md body — frontmatter STRIPPED>

        --- File: scripts/<name>
        <file contents, truncated to 4000 chars with "...[truncated]">

    Returns an empty string if ``bundle_dir`` has no SKILL.md (this is
    the empty-bundle case used for Baseline A floor scoring — caller
    detects it and feeds an empty system prompt).
    """
    skill_md_path = bundle_dir / "SKILL.md"
    if not skill_md_path.exists():
        return ""

    try:
        raw = skill_md_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""

    body = _strip_frontmatter(raw)

    parts: list[str] = []
    parts.append(f"Base directory for this skill: {_SKILL_PLACEHOLDER}\n")
    parts.append(body)

    scripts_dir = bundle_dir / "scripts"
    if scripts_dir.exists() and scripts_dir.is_dir():
        # Stable file ordering — sorted by name so the same bundle always
        # produces byte-identical prompts (load-bearing for the hash pin).
        for entry in sorted(scripts_dir.iterdir()):
            if not entry.is_file():
                continue
            try:
                content = entry.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if len(content) > _SCRIPT_TRUNC_CHARS:
                content = content[:_SCRIPT_TRUNC_CHARS] + "...[truncated]"
            parts.append(f"\n--- File: scripts/{entry.name}\n{content}")

    return "\n".join(parts)


# ─── Subprocess-isolated rollout ───────────────────────────────────────────


# Inline script run inside the child. Reads inputs from argv (a single
# JSON blob on stdin via subprocess.run's input=) and writes one JSON
# line to stdout. Keeps imports minimal — only httpx + stdlib.
_CHILD_SCRIPT = r"""
import json, sys, os

def main():
    payload = json.loads(sys.stdin.read())
    api_key = payload["api_key"]
    model = payload["model"]
    system_prompt = payload["system_prompt"]
    user_prompt = payload["user_prompt"]
    seed = payload["seed"]
    temperature = payload["temperature"]
    max_tokens = payload["max_tokens"]

    import httpx

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=90.0) as client:
            r = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=body,
                headers=headers,
            )
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        sys.stdout.write(json.dumps({"ok": False, "error": f"http_error: {exc}"}))
        return

    try:
        completion = data["choices"][0]["message"]["content"] or ""
    except Exception as exc:
        sys.stdout.write(json.dumps({"ok": False, "error": f"parse_error: {exc}", "raw": data}))
        return

    usage = data.get("usage") or {}
    completion_len_tokens = int(usage.get("completion_tokens") or 0)

    sys.stdout.write(json.dumps({
        "ok": True,
        "completion": completion,
        "completion_len_tokens": completion_len_tokens,
    }))

main()
"""


def run_rollout_subprocess(
    prompt: str,
    skill_system_prompt: str,
    model: str,
    api_key: str,
    seed: int,
    temperature: float,
    max_tokens: int = 2048,
) -> RunResult:
    """Spawn a fresh subprocess, make one OpenRouter chat completion call,
    return a ``RunResult`` (K10 — subprocess isolation per rollout).

    On any failure (subprocess crash, JSON parse error, timeout, HTTP
    error), returns RunResult(score=0.0, completion="",
    completion_len_tokens=0, error=str(...)). Never raises — callers
    expect to aggregate over many rollouts and treat errors as one
    bucket per W9.
    """
    payload = {
        "api_key": api_key,
        "model": model,
        "system_prompt": skill_system_prompt or "",
        "user_prompt": prompt,
        "seed": seed,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD_SCRIPT],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        return RunResult(score=0.0, completion="", completion_len_tokens=0, error=f"timeout: {exc}")
    except Exception as exc:  # noqa: BLE001 — runner contract: never raise
        return RunResult(score=0.0, completion="", completion_len_tokens=0, error=f"subprocess: {exc}")

    if proc.returncode != 0:
        return RunResult(
            score=0.0,
            completion="",
            completion_len_tokens=0,
            error=f"nonzero_rc={proc.returncode}: {proc.stderr[:500]}",
        )

    try:
        data = json.loads(proc.stdout)
    except Exception as exc:  # noqa: BLE001
        return RunResult(
            score=0.0,
            completion="",
            completion_len_tokens=0,
            error=f"json_parse: {exc}: stdout={proc.stdout[:500]}",
        )

    if not data.get("ok"):
        return RunResult(
            score=0.0,
            completion="",
            completion_len_tokens=0,
            error=str(data.get("error") or "unknown_error"),
        )

    return RunResult(
        score=0.0,
        completion=str(data.get("completion") or ""),
        completion_len_tokens=int(data.get("completion_len_tokens") or 0),
        error=None,
    )
