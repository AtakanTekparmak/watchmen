"""Prompt rendering for Track B.

Fork of openevolve/prompt/sampler.py — we keep the same mental model
(system + user message, parent + inspiration slots, feedback-from-last-
evaluation), but render a *folder of files* instead of a single source
file, and ask the LLM to reply with patch-style ops (see
:mod:`.patch_parser` for the format).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .database import Program, cell_key, compute_features, FEATURE_DIMENSIONS
from .folder_artifact import FolderArtifact


SYSTEM_TEMPLATE = """\
You are evolving a *folder of Hermes-agent skills*. Each skill is a
subfolder containing a single `SKILL.md` file with YAML frontmatter
(name, description, version, tags) and a markdown body.

Your goal: produce a *mutation* of the parent folder that scores higher
on the composite fitness metric (success_rate minus a small tool-call
overhead penalty) on the benchmark. Focus on things that actually
change agent behavior — clearer triggers, better boundaries, cutting
dead weight, fixing a failure mode surfaced in the feedback.

You MUST reply using the patch format described in the user message.
Reply with the patch only — no explanation, no surrounding prose,
no markdown code fences.
"""


USER_TEMPLATE = """\
# Current parent folder
Cell (MAP-Elites): {cell_key}
Fitness (composite): {fitness:.4f}   success_rate: {success_rate:.3f}   tool_calls/success: {tool_calls_per_success:.2f}
Features: num_skills={num_skills}  total_tokens≈{total_tokens:.0f}  avg_specificity={avg_specificity:.1f}

## Skills folder ({n_skills} skills, ~{total_tokens_int} tokens)

{folder_render}

## Recent evaluation feedback

Failures (per task):
{failures_render}

Per-skill invocation counts (last run):
{invocations_render}

Skills that were present but NEVER invoked by the agent:
{unused_render}

## Inspiration program (a peer from the same island)

{inspiration_render}

## How to reply

Emit zero or more of these commands. Each command operates on one file.

    <<<ADD_FILE path/to/new_skill/SKILL.md>>>
    ---
    name: ...
    description: ...
    ---

    # markdown body
    <<<END_FILE>>>

    <<<EDIT_FILE path/to/existing/SKILL.md>>>
    <entire new contents>
    <<<END_FILE>>>

    <<<DELETE_FILE path/to/stale/SKILL.md>>>

Constraints:
* ≤ {max_files} files total in the resulting folder
* ≤ {max_kb} KiB total across all files
* Every skill must live in its own subfolder containing `SKILL.md`
* At least one skill must remain after the patch
* Do NOT use `..` in paths

Focus on the single most promising change, not a rewrite.
"""


@dataclass
class RenderedPrompt:
    system: str
    user: str


class PromptSampler:
    """Assemble a prompt for a folder-mutation LLM call."""

    def __init__(self, *, max_files: int = 20, max_total_bytes: int = 100 * 1024) -> None:
        self.max_files = max_files
        self.max_total_bytes = max_total_bytes

    def build(
        self,
        parent: Program,
        *,
        inspiration: Optional[Program] = None,
    ) -> RenderedPrompt:
        feats = compute_features(parent.artifact)
        key = cell_key(parent.artifact)

        # Folder body — multi-file context block.
        folder_render = render_folder(parent.artifact, include_body=True)

        # Feedback from the last eval.
        failures_render = _render_json_block(
            parent.eval_artifacts.get("failures", "[]"), default="(none)")
        invocations_render = _render_json_block(
            parent.eval_artifacts.get("invocation_counts", "{}"), default="(none)")
        unused_render = _render_json_block(
            parent.eval_artifacts.get("unused_skills", "[]"), default="(none)")

        # Inspiration — a lighter-weight render (names + descriptions only,
        # no full bodies, to keep token budget under control).
        inspiration_render = (
            render_folder(inspiration.artifact, include_body=False)
            if inspiration is not None else "(no inspiration available)"
        )

        user = USER_TEMPLATE.format(
            cell_key=str(list(key)),
            fitness=parent.fitness(),
            success_rate=float(parent.metrics.get("success_rate", 0.0)),
            tool_calls_per_success=float(parent.metrics.get("tool_calls_per_success", 0.0)),
            num_skills=int(feats["num_skills"]),
            total_tokens=feats["total_tokens"],
            total_tokens_int=int(feats["total_tokens"]),
            avg_specificity=feats["avg_specificity"],
            n_skills=parent.artifact.num_skills(),
            folder_render=folder_render,
            failures_render=failures_render,
            invocations_render=invocations_render,
            unused_render=unused_render,
            inspiration_render=inspiration_render,
            max_files=self.max_files,
            max_kb=self.max_total_bytes // 1024,
        )
        return RenderedPrompt(system=SYSTEM_TEMPLATE, user=user)


def render_folder(artifact: FolderArtifact, *, include_body: bool) -> str:
    """Multi-file context block. ``include_body=False`` for inspiration."""
    chunks: List[str] = []
    for name in artifact.skill_names():
        skill_md_path = f"{name}/SKILL.md"
        content = artifact.files.get(skill_md_path, "")
        if include_body:
            chunks.append(f"### skill: {name}\n```md\n{content.rstrip()}\n```")
        else:
            # Just the frontmatter for inspiration.
            head = _frontmatter_only(content)
            chunks.append(f"### skill: {name}\n```yaml\n{head.rstrip()}\n```")

    # Any non-SKILL.md files (e.g. INDEX.md) — surface them too.
    extras = [p for p in sorted(artifact.files) if not p.endswith("SKILL.md")]
    for p in extras:
        content = artifact.files[p]
        if include_body:
            chunks.append(f"### file: {p}\n```\n{content.rstrip()}\n```")

    return "\n\n".join(chunks) if chunks else "(empty folder)"


def _frontmatter_only(content: str) -> str:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return content[:400]
    out: List[str] = ["---"]
    for line in lines[1:]:
        out.append(line)
        if line.strip() == "---":
            break
    return "\n".join(out)


def _render_json_block(blob: str, *, default: str) -> str:
    """Pretty-print JSON if it's JSON, else return the raw blob / default."""
    if not blob or blob.strip() in ("[]", "{}", ""):
        return default
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        return blob.strip() or default
    if not parsed:
        return default
    return json.dumps(parsed, indent=2)
