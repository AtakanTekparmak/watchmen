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
from typing import List, Optional

from .database import Program, cell_key, compute_features
from .folder_artifact import FolderArtifact, MAX_FILES, MAX_TOTAL_BYTES


SYSTEM_TEMPLATE = """\
You are evolving a *folder of Hermes-agent skills*. Each skill lives
in its own subdirectory. A skill's directory contains:

  * `SKILL.md` (required) — YAML frontmatter (name, description,
    version, tags) + a prose body the agent reads before acting.
  * `scripts/` (optional) — executable helpers (`.sh` with shebang,
    `.py` with shebang) the agent invokes from inside its task
    container to do heavy lifting. Bash / python only; no binaries.
  * `references/` (optional) — supporting docs, specs, examples.
  * `templates/` (optional) — output-format boilerplate.
  * `assets/` (optional) — supplementary static files.

Scripts under `scripts/` are written out with the executable bit set,
so SKILL.md prose can tell the agent to `bash scripts/analyze.sh
<args>` from the task workspace (mounted at `/app` in the task
container).

Your goal: produce a *mutation* of the parent folder that scores higher
on the composite fitness metric (success_rate minus a small tool-call
overhead penalty) on the benchmark. Focus on things that actually
change agent behavior — clearer triggers, better boundaries, cutting
dead weight, fixing a failure mode surfaced in the feedback, or
adding/fixing a script that short-circuits repeated agent work.

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
Paths are interpreted under the skills-folder root, so a script at
`<skill>/scripts/foo.sh` is written as `my_skill/scripts/foo.sh`.

    <<<ADD_FILE my_skill/SKILL.md>>>
    ---
    name: my_skill
    description: One-line trigger for when the agent should use this skill.
    ---

    # My Skill

    Prose the agent reads before acting. Reference helpers as
    `bash scripts/analyze.sh <path>` so the agent knows to invoke them.
    <<<END_FILE>>>

    <<<ADD_FILE my_skill/scripts/analyze.sh>>>
    #!/usr/bin/env bash
    set -euo pipefail
    target="${{1:-/app}}"
    # ... do the thing ...
    <<<END_FILE>>>

    <<<EDIT_FILE my_skill/scripts/analyze.sh>>>
    #!/usr/bin/env bash
    set -euo pipefail
    # entire rewritten body, shebang included
    <<<END_FILE>>>

    <<<DELETE_FILE my_skill/scripts/stale_helper.sh>>>

Constraints:
* <= {max_files} files total in the resulting folder
* <= {max_kb} KiB total across all files
* Every skill must live in its own subfolder containing `SKILL.md`
* At least one skill must remain after the patch
* Do NOT use `..` in paths
* New scripts MUST start with a shebang: `#!/usr/bin/env bash` followed
  by `set -euo pipefail` for bash, or `#!/usr/bin/env python3` for
  python. Executable bit is set automatically for `.sh` / `.py` files
  under any skill's `scripts/` subdir.
* SKILL.md YAML frontmatter must parse (the validator rejects patches
  that produce broken frontmatter — quote descriptions containing `:`).

Focus on the single most promising change, not a rewrite. Prefer
adding/editing a script when the failure is the agent re-implementing
the same 10-20 lines of shell / python across tasks. Prefer editing
SKILL.md prose when the failure is the agent not invoking the right
skill, or invoking it at the wrong time.
"""


@dataclass
class RenderedPrompt:
    system: str
    user: str


class PromptSampler:
    """Assemble a prompt for a folder-mutation LLM call."""

    def __init__(
        self,
        *,
        max_files: int = MAX_FILES,
        max_total_bytes: int = MAX_TOTAL_BYTES,
    ) -> None:
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
            parent.eval_artifacts.get("failures", "[]"), default="(none)"
        )
        invocations_render = _render_json_block(
            parent.eval_artifacts.get("invocation_counts", "{}"), default="(none)"
        )
        unused_render = _render_json_block(
            parent.eval_artifacts.get("unused_skills", "[]"), default="(none)"
        )

        # Inspiration — a lighter-weight render (names + descriptions only,
        # no full bodies, to keep token budget under control).
        inspiration_render = (
            render_folder(inspiration.artifact, include_body=False)
            if inspiration is not None
            else "(no inspiration available)"
        )

        user = USER_TEMPLATE.format(
            cell_key=str(list(key)),
            fitness=parent.fitness(),
            success_rate=float(parent.metrics.get("success_rate", 0.0)),
            tool_calls_per_success=float(
                parent.metrics.get("tool_calls_per_success", 0.0)
            ),
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
