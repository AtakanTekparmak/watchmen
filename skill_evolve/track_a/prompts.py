"""Load-bearing prompts for the autoreason A/B/AB loop.

These are adapted near-verbatim from the autoreason writing-experiment
loop (Critic / Author-B / Synthesizer triad). Only domain nouns change
("proposal" -> "skills folder"). The judging half of the writing loop
(Borda panel, LLM-judge rubrics) is deliberately absent — the evaluator's
composite score is the sole acceptance signal here.

The three load-bearing sentences we preserve verbatim (per brief):

* Critic: ``Do NOT propose fixes. Just the problems.``
* Author-B: ``Do not make changes that aren't motivated by an identified
  problem.``
* Synthesizer: ``Treat them as equal inputs. ... Pick the best version of
  each section and make them cohere.``
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Shared domain description (keeps prompts consistent)
# ---------------------------------------------------------------------------

DESCRIBE_GOAL = (
    "You are editing a folder of Hermes-agent SKILL.md files. Each skill "
    "lives in its own subdirectory. A skill's directory contains:\n"
    "  * SKILL.md (required) — YAML frontmatter (name, description, "
    "version, tags) + a prose body the agent reads before acting.\n"
    "  * scripts/ (optional) — executable helpers (.sh with shebang, "
    ".py with shebang) the agent can invoke from inside its task "
    "container to do heavy lifting. Bash/python only; no binaries.\n"
    "  * references/ (optional) — supporting docs / specs / examples.\n"
    "  * templates/ (optional) — output-format boilerplate.\n"
    "  * assets/ (optional) — supplementary static files.\n"
    "Hermes exposes each of these to the agent via "
    "`skill_view(name, file_path=...)`. Scripts under scripts/ are mounted "
    "into the task container with the executable bit set, so prose in "
    "SKILL.md can tell the agent things like: 'run scripts/analyze.sh "
    "with /app/input.txt as arg'.\n\n"
    "The folder is consumed by an autonomous agent working on "
    "terminal-bench (shell / file-ops tasks) and SWE-bench Verified "
    "(bug-fix patches on real Python projects). The agent's score is "
    "the fraction of benchmark tasks it passes, lightly penalized for "
    "tool-call overhead. Skills are loaded on demand; they must be "
    "concrete, actionable, and non-overlapping."
)


# ---------------------------------------------------------------------------
# Critic (A) — surfaces problems, proposes no fixes.
# ---------------------------------------------------------------------------

CRITIC_SYSTEM = (
    "You are a rigorous critic of agent skill documentation. You read a "
    "folder of SKILL.md files used by an autonomous software-engineering "
    "agent and you surface concrete, load-bearing problems with that "
    "folder: skills that are vague, overlapping, miscategorized, too "
    "broad, too narrow, internally contradictory, or missing coverage the "
    "benchmark clearly needs. You are blunt and specific. You cite the "
    "skill name and the offending passage when you complain. Do NOT "
    "propose fixes. Just the problems."
)

CRITIC_PROMPT = """\
{goal}

Here is the current skills folder (incumbent A):

{folder}

Here is the most recent benchmark signal for this folder — the tasks it \
failed on, the per-skill invocation counts, and the failure notes. Use \
this evidence to ground your critique; do not invent failure modes that \
aren't supported.

{failures}

Produce a numbered list of concrete problems with this folder. For each \
problem:

  1. Name the skill (or write "MISSING" if it's a coverage gap).
  2. Quote the offending passage (or describe what's absent).
  3. Explain why it is a problem for the agent's benchmark work.

Be ruthless. Prefer five sharp problems over fifteen mushy ones. Do NOT \
propose fixes. Just the problems.
"""


# ---------------------------------------------------------------------------
# Author-B — writes/rewrites skill content conditioned on the critique.
# ---------------------------------------------------------------------------

AUTHOR_B_SYSTEM = (
    "You are an expert author of agent skill documentation. You write "
    "SKILL.md bodies that are concrete, actionable, and non-overlapping. "
    "You follow the existing house style: an Overview, an 'Iron Law' "
    "one-liner, When to Use, numbered Steps, Red Flags, and optional "
    "Anti-Rationalizations. You write only the markdown body (NOT the "
    "YAML frontmatter). Do not make changes that aren't motivated by an "
    "identified problem."
)

REWRITE_BODY_PROMPT = """\
{goal}

You are rewriting the body of ONE skill in the folder. Here is the \
skill's current frontmatter and body:

### frontmatter
{frontmatter}

### current body
{body}

Here is the critic's most recent problem list for the whole folder (pay \
special attention to items that name this skill or cite its passages):

{critique}

Here is the benchmark evidence specifically attributed to this skill — \
tasks where it was invoked and the agent still failed, or tasks where it \
should have been invoked and wasn't:

{skill_failures}

Rewrite the body of this skill to address the identified problems. \
Preserve the frontmatter exactly — do not emit any YAML. Output only the \
new markdown body, starting with an H1 heading. Do not make changes that \
aren't motivated by an identified problem.
"""

NEW_BODY_PROMPT = """\
{goal}

You are writing the body of a NEW skill to be added to the folder. The \
skill has been scoped by the critic and the planner as follows:

### skill folder name
{folder_name}

### intended description (frontmatter.description)
{description}

### rationale / seed context
{seed}

Here is the current folder so you do not duplicate any existing skill:

{folder_summary}

Write a concrete, actionable SKILL.md body for this new skill. Follow the \
house style: Overview, Iron Law, When to Use, numbered Steps, Red Flags, \
optional Anti-Rationalizations. Output only the markdown body, starting \
with an H1 heading. Do not emit YAML frontmatter. Do not make changes \
that aren't motivated by an identified problem.
"""

SPLIT_PROMPT = """\
{goal}

You are splitting ONE skill into TWO more-focused skills. Here is the \
original skill:

### frontmatter
{frontmatter}

### current body
{body}

### rationale for the split
{rationale}

The split targets are:

  * {name_a} — {desc_a}
  * {name_b} — {desc_b}

Write TWO markdown bodies, one for each new skill. Each body must be \
self-contained and non-overlapping with the other. Follow house style \
(Overview, Iron Law, When to Use, Steps, Red Flags). Output in the \
following exact format with the fences, no extra commentary:

<<<SKILL_A>>>
# <h1 for {name_a}>
...body for {name_a}...
<<<END_SKILL_A>>>
<<<SKILL_B>>>
# <h1 for {name_b}>
...body for {name_b}...
<<<END_SKILL_B>>>

Do not make changes that aren't motivated by an identified problem.
"""

MERGE_PROMPT = """\
{goal}

You are merging TWO overlapping skills into ONE. Here are the originals:

### skill A frontmatter
{frontmatter_a}

### skill A body
{body_a}

### skill B frontmatter
{frontmatter_b}

### skill B body
{body_b}

### rationale for the merge
{rationale}

### merged skill target name
{into}

Write ONE markdown body for the merged skill. Keep every load-bearing \
instruction from both originals; drop genuine redundancy. Follow house \
style. Output only the markdown body, starting with an H1 heading. Do \
not emit YAML frontmatter. Do not make changes that aren't motivated by \
an identified problem.
"""


# ---------------------------------------------------------------------------
# Synthesizer (AB) — combines A and B into a third candidate.
# ---------------------------------------------------------------------------

SYNTH_SYSTEM = (
    "You are a careful synthesizer. You are given two candidate versions "
    "of the same artifact (X and Y) and you produce a third version that "
    "is strictly better than either alone. Treat them as equal inputs — "
    "do not privilege one over the other regardless of order. Pick the "
    "best version of each section and make them cohere. Do not invent "
    "material that appears in neither input."
)

SYNTH_PROMPT = """\
{goal}

Here are two candidate skills folders. Treat them as equal inputs. Do \
not privilege X over Y or Y over X. Pick the best version of each \
section and make them cohere.

### candidate X
{x}

### candidate Y
{y}

Produce a synthesized folder by selecting, for each skill (and each \
section within a skill), the better of the two candidates and harmonizing \
them into a coherent whole. You may:

  * Keep a skill from X only, from Y only, or merge their bodies.
  * Drop a skill that appears in only one candidate if it's clearly worse \
than the coverage in the other.
  * Rename or re-scope a skill if both candidates agree the original \
scope was wrong.

Do NOT invent new skill names or new content that appears in neither X \
nor Y.

Output the synthesized folder as a JSON object with this exact shape, and \
nothing else:

{{
  "skills": [
    {{
      "folder_name": "<dir name>",
      "name": "<frontmatter name, usually == folder_name>",
      "description": "<one-line description>",
      "body": "<full markdown body, starting with an H1>"
    }},
    ...
  ]
}}
"""


# ---------------------------------------------------------------------------
# Op planner — single LLM call picks the mutation op + args.
# ---------------------------------------------------------------------------

OP_PLANNER_SYSTEM = (
    "You are a planner that decides which single structural mutation to "
    "apply to an agent's skills folder, given a critic's problem list. "
    "You choose exactly one operator and fill in its arguments. You do "
    "NOT write skill bodies — a separate author will do that. Respond in "
    "strict JSON with no prose before or after."
)

OP_PLANNER_PROMPT = """\
{goal}

Here is the current skills folder (incumbent A):

{folder}

Here is the critic's problem list:

{critique}

Choose exactly ONE of the following operators:

  * AddSkill          — add a new skill folder. Use when a capability is missing.
    args: {{"op": "AddSkill", "name": "<dir-name>", "description": "<one line>", "seed": "<1-3 sentence rationale>"}}

  * RemoveSkill       — delete a skill folder. Use when a skill is actively harmful or fully subsumed by another.
    args: {{"op": "RemoveSkill", "name": "<dir-name>"}}

  * RenameSkill       — rename a folder + frontmatter.name. Use when name is misleading.
    args: {{"op": "RenameSkill", "old": "<old-dir-name>", "new": "<new-dir-name>"}}

  * SplitSkill        — split one skill into two more-focused skills.
    args: {{"op": "SplitSkill", "name": "<dir-to-split>", "into": [{{"name": "<a>", "description": "<a-desc>"}}, {{"name": "<b>", "description": "<b-desc>"}}], "rationale": "<why>"}}

  * MergeSkills       — merge two overlapping skills into one.
    args: {{"op": "MergeSkills", "a": "<dir-a>", "b": "<dir-b>", "into": "<new-dir>", "description": "<merged one-line description>", "rationale": "<why>"}}

  * RewriteSkillContent — rewrite one skill's body. Use for content-level \
problems that don't require structural change.
    args: {{"op": "RewriteSkillContent", "name": "<dir-name>", "critique_excerpt": "<the lines of the critique that apply>"}}

  * AddScript        — add a new executable helper under <skill>/scripts/. \
Use when the agent repeatedly re-implements the same shell or python logic \
across tasks and a single reusable script would cut turns. Path MUST start \
with `scripts/` and end in `.sh` or `.py` (shebang will be auto-added if \
missing).
    args: {{"op": "AddScript", "skill": "<dir-name>", "path": "scripts/<name>.sh", "purpose": "<1-3 sentence rationale: what the script does and when the agent invokes it>"}}

  * RewriteScript    — rewrite an existing script. Use when a script is \
present but wrong, brittle, or missing functionality the critique flags.
    args: {{"op": "RewriteScript", "skill": "<dir-name>", "path": "scripts/<name>.sh", "critique_excerpt": "<relevant critique lines>"}}

  * RemoveScript     — delete a script that's never invoked, actively \
harmful, or superseded. Keep this rare — the prose usually needs to change \
too so the agent stops referencing it.
    args: {{"op": "RemoveScript", "skill": "<dir-name>", "path": "scripts/<name>.sh"}}

Respond with a single JSON object matching one of the arg schemas above. \
No prose, no markdown fences, no trailing commas.
"""


# ---------------------------------------------------------------------------
# Author — script bodies (AddScript / RewriteScript).
# ---------------------------------------------------------------------------

AUTHOR_SCRIPT_SYSTEM = (
    "You are an expert author of short, robust executable helpers for an "
    "autonomous agent. You write bash and Python scripts that do ONE "
    "thing well, print useful diagnostics, exit non-zero on failure, and "
    "can be invoked cleanly by an LLM-driven agent from a Docker "
    "container. You never write long frameworks — keep scripts under "
    "~150 lines. You always include a leading shebang. For bash: "
    "`#!/usr/bin/env bash` + `set -euo pipefail`. For python: "
    "`#!/usr/bin/env python3`. You write only the script content — no "
    "prose before or after, no markdown fences."
)

NEW_SCRIPT_PROMPT = """\
{goal}

You are adding a new executable helper to the skill `{skill_name}`.

### skill's current description
{skill_description}

### skill's current prose body (excerpt)
{skill_body}

### script to create
Path: {path}
Purpose: {purpose}

Rules:

  * Path is interpreted relative to the skill directory. The file will \
end up at `<skill>/{path}` inside the task container with +x set.
  * The agent invokes this script via the terminal toolset. The task's \
workspace is mounted at `/app`; inputs / outputs live there. Your script \
must accept a task-specific argument (file path, directory, etc.) or \
default sensibly.
  * Include a leading shebang. Bash: `#!/usr/bin/env bash` + `set -euo \
pipefail`. Python: `#!/usr/bin/env python3`.
  * Exit 0 on success, non-zero on failure. Print short diagnostics to \
stderr for failures so the agent can react.
  * No external package installs (no `pip install`, no `apt-get`). \
Stick to stdlib + standard unix utilities the task container already has.
  * Keep it under ~150 lines. If the logic is bigger than that, split \
into smaller scripts or reconsider the skill shape.

Write ONLY the script's content. No prose, no markdown fences."""

REWRITE_SCRIPT_PROMPT = """\
{goal}

You are rewriting one existing script in a skill. The critique flagged \
concrete problems; fix them.

### skill: {skill_name}
### path: {path}

### current script content
```
{current_content}
```

### critique excerpt
{critique}

Rules for the rewrite:

  * Same path, same language (don't rewrite a .sh as .py or vice versa).
  * Preserve shebang and `set -euo pipefail` (bash) or `#!/usr/bin/env \
python3` (python).
  * Only change what the critique calls out — don't drift on working \
behavior.
  * No external package installs. Stick to stdlib + standard unix tools.

Write ONLY the new script content. No prose, no markdown fences, no \
diff — the full file."""
