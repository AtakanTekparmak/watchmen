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


# ---------------------------------------------------------------------------
# Sentinel-block proposer prompt (ported from daycare evolve.py:620-685)
# ---------------------------------------------------------------------------
#
# The HARD LIMITS block embeds the MAX_SKILL_TOKENS / MAX_BUNDLE_TOKENS
# constants verbatim so a runtime check (below) can spot drift if the
# constants ever change but the prompt isn't re-rendered. The CRITICAL
# warning on <<<END_FILE>>> is preserved verbatim — DeepSeek omits the
# terminator ~50% of the time without it (feedback_proposer_sentinel_quality
# memory note).

from skill_evolve.shared.bundle_ops import (  # noqa: E402
    MAX_BUNDLE_TOKENS,
    MAX_SKILL_TOKENS,
)


_SENTINEL_PROPOSER_SYSTEM_PROMPT_TEMPLATE = """You are a skill-bundle mutator. Emit ONE mutation
to the current best bundle, targeting the assigned failure cluster.

The bundle is the FULL skill package — SKILL.md + scripts/ + references/.
Mutate whichever file(s) best fix the failure. SKILL.md changes alter what the
weak model knows; script changes alter what the weak model can do.

Choose your target by reading the weakness report's cluster.category field:
  - missing_procedure_knowledge → edit SKILL.md (add the missing step/value)
  - wrong_command_syntax         → edit SKILL.md (correct flag/argument)
  - missing_script_capability    → ADD_FILE or EDIT_FILE a script under scripts/
  - incomplete_workflow          → may require both (SKILL.md + script)

WORKFLOW — follow exactly:
1. Call read_weakness_report() — read the cluster.category for your target.
2. Call list_parent_bundle_files() — see what's there.
3. Call read_parent_bundle_file("SKILL.md") and any relevant scripts/ files.
4. If you plan a script change, call list_scripts() to confirm naming/layout.
5. Draft your mutation. Call validate_sentinel_patch() to dry-run.
6. For EVERY script you add or modify, call lint_script(path, content) BEFORE
   finish_candidate. Fix any syntax errors.
7. For SKILL.md changes, call count_skill_tokens(content) — must be ≤ {max_skill}.
8. Call finish_candidate(patch_text, target_cluster, reasoning).

HARD LIMITS:
- ≤ 16 tool calls total.
- An empty patch_text is REJECTED.
- SKILL.md ≤ {max_skill} cl100k_base tokens.
- Whole bundle ≤ {max_bundle} cl100k_base tokens.
- Each script ≤ 150 lines.
- No `pip install` calls; stdlib + requirements.txt only.
- All script CLIs via argparse. No hardcoded paths.
- py_compile / `bash -n` must pass.
- Generic guidance only — no session IDs, user names, project identifiers.

Sentinel-block format (the ONLY accepted format):

    <<<ADD_FILE path/relative/to/bundle>>>
    ... full file content ...
    <<<END_FILE>>>

    <<<EDIT_FILE path/relative/to/bundle>>>
    ... FULL rewritten body (NOT a diff) ...
    <<<END_FILE>>>

    <<<DELETE_FILE path/relative/to/bundle>>>

    <<<REWRITE_FOLDER scripts>>>
    --- file: scripts/foo.py
    ... content ...
    --- file: scripts/bar.py
    ... content ...
    <<<END_REWRITE>>>

## EDIT BUDGET

You may propose AT MOST {edit_budget_line} file operations this iteration. If
you propose more, only the top {edit_budget_line} will be applied (in the
order you list them, with priority ADD > EDIT > DELETE > REWRITE on ties).

## META-SKILL

The following audit log records prior-iteration patch summaries, lessons,
and persistent failures. Treat it as cross-iteration memory — what mutations
have already been tried, what tended to help, what failure modes keep
recurring. Do NOT re-propose a mutation shape that the log shows already
failed; pick a different angle of attack.

If the SKILL.md you see contains a region between
<!-- SLOW_UPDATE_START --> and <!-- SLOW_UPDATE_END --> markers, that region
is PROTECTED — a separate slow-update consolidator owns it. Do NOT propose
any EDIT_FILE / DELETE_FILE / REWRITE_FOLDER op that modifies, deletes, or
mutates bytes inside that region. Out-of-fence edits to SKILL.md are fine.

{meta_skill_body}

## RECENT REJECTIONS — DO NOT REPEAT

The following recent candidate patches were rejected by the validation
gate (strict-mode acceptance requires train_score >= parent AND val_score
strictly above the best-seen). Do NOT re-propose the same shape — pick a
different mutation target or strategy:

{recent_rejections}

CRITICAL: every ADD_FILE and EDIT_FILE block MUST end with <<<END_FILE>>> on
its own line. A patch missing any <<<END_FILE>>> terminator is REJECTED outright
— use validate_sentinel_patch() to verify before finishing.

You may NOT read held-out slice data. Use only the proposer tools.
Reasoning ≤ 200 words.
"""

# The ``{recent_rejections}`` slot is filled at proposer-call time by
# ``RejectedBuffer.render_for_prompt()``. We render the literal sentinel
# ``(none yet)`` here so the module-level constant is a valid prompt even
# before any rejection has occurred — runtime call sites re-render with
# the live buffer via ``str.format(recent_rejections=...)``.
#
# kai-skills patch (Group F, 2026-05-28): the ``{edit_budget_line}`` slot
# is filled per-iteration with the live L_t value (see
# ``shared.edit_budget.compute_lt``). The module-level constant renders
# the canonical default ``8`` so the prompt is a valid string even before
# a runtime call site re-renders.
# kai-skills patch (Group G, 2026-05-28; plan §7l): the
# ``{meta_skill_body}`` slot carries the per-iteration audit log
# (markdown sections rendered by ``MetaSkill.render_for_prompt``). The
# module-level constant renders the locked literal
# ``(empty — no consolidated patterns yet)`` so the prompt is valid
# even on the first iter; runtime call sites re-render with the live
# meta-skill via ``render_sentinel_proposer_prompt(meta_skill=...)``.
SENTINEL_PROPOSER_SYSTEM_PROMPT = _SENTINEL_PROPOSER_SYSTEM_PROMPT_TEMPLATE.format(
    max_skill=MAX_SKILL_TOKENS,
    max_bundle=MAX_BUNDLE_TOKENS,
    edit_budget_line=8,
    meta_skill_body="(empty — no consolidated patterns yet)",
    recent_rejections="(none yet)",
)


def _assert_prompt_well_formed() -> None:
    """Runtime check — daycare used bare ``assert`` which strips under
    ``python -O``. Promote to explicit ``RuntimeError`` so the check
    lives in all runtime modes.
    """
    if str(MAX_SKILL_TOKENS) not in SENTINEL_PROPOSER_SYSTEM_PROMPT:
        raise RuntimeError(
            "MAX_SKILL_TOKENS must appear verbatim in the proposer prompt"
        )
    if str(MAX_BUNDLE_TOKENS) not in SENTINEL_PROPOSER_SYSTEM_PROMPT:
        raise RuntimeError(
            "MAX_BUNDLE_TOKENS must appear verbatim in the proposer prompt"
        )
    # kai-skills patch (Group E, 2026-05-28): RECENT REJECTIONS section
    # must be present in the rendered prompt so the proposer always sees
    # the strict-gate feedback channel (rendered as ``(none yet)`` when
    # the buffer is empty).
    if "## RECENT REJECTIONS" not in SENTINEL_PROPOSER_SYSTEM_PROMPT:
        raise RuntimeError(
            "## RECENT REJECTIONS section must appear in the proposer prompt"
        )
    # kai-skills patch (Group F, 2026-05-28): EDIT BUDGET section must
    # be present and the ``{edit_budget_line}`` slot must have been
    # rendered (the caller fills it with the live L_t value). We check
    # the section header and that no unrendered slot remains.
    if "## EDIT BUDGET" not in SENTINEL_PROPOSER_SYSTEM_PROMPT:
        raise RuntimeError("## EDIT BUDGET section must appear in the proposer prompt")
    if "{edit_budget_line}" in SENTINEL_PROPOSER_SYSTEM_PROMPT:
        raise RuntimeError(
            "{edit_budget_line} slot in proposer prompt was not rendered"
        )
    # kai-skills patch (Group G, 2026-05-28; plan §7l): META-SKILL
    # section must be present in the rendered prompt so the proposer
    # always sees the cross-iter audit log slot (rendered as
    # ``(empty — no consolidated patterns yet)`` when the meta-skill is
    # empty). The CONSOLIDATOR prompt must cite both fence markers
    # verbatim.
    if "## META-SKILL" not in SENTINEL_PROPOSER_SYSTEM_PROMPT:
        raise RuntimeError("## META-SKILL section must appear in the proposer prompt")
    if "{meta_skill_body}" in SENTINEL_PROPOSER_SYSTEM_PROMPT:
        raise RuntimeError("{meta_skill_body} slot in proposer prompt was not rendered")
    if (
        "<!-- SLOW_UPDATE_START -->" not in CONSOLIDATOR_PROPOSER_SYSTEM_PROMPT
        or "<!-- SLOW_UPDATE_END -->" not in CONSOLIDATOR_PROPOSER_SYSTEM_PROMPT
    ):
        raise RuntimeError(
            "CONSOLIDATOR_PROPOSER_SYSTEM_PROMPT must cite both "
            "<!-- SLOW_UPDATE_START --> and <!-- SLOW_UPDATE_END --> verbatim"
        )


# kai-skills patch (Group E, 2026-05-28): live-render helper for the
# RECENT REJECTIONS slot. The module-level ``SENTINEL_PROPOSER_SYSTEM_PROMPT``
# is rendered with the ``(none yet)`` literal; runtime proposer call sites
# re-format with the live buffer via this helper so the prompt sees an
# up-to-date list of recently-rejected patches.
def _render_recent_rejections(buffer) -> str:  # type: ignore[no-untyped-def]
    """Render the ``{recent_rejections}`` slot from a ``RejectedBuffer``.

    Returns the locked literal ``(none yet)`` when the buffer is empty
    (per plan section 7j) so the section never collapses to a bare
    header. ``buffer`` may be ``None`` (callers that haven't wired the
    buffer through yet) — same fallback applies.
    """
    if buffer is None:
        return "(none yet)"
    body = buffer.render_for_prompt()
    if not body:
        return "(none yet)"
    return body


def render_sentinel_proposer_prompt(  # type: ignore[no-untyped-def]
    buffer=None,
    *,
    edit_budget_line: int = 8,
    meta_skill_body: str = "(empty — no consolidated patterns yet)",
) -> str:
    """Render ``SENTINEL_PROPOSER_SYSTEM_PROMPT`` with live slot values.

    Convenience wrapper for the runtime proposer call site: re-renders
    the template with the live rejected buffer + edit-budget L_t + meta-
    skill body in place of the module-default literals baked in at
    module import.
    """
    return _SENTINEL_PROPOSER_SYSTEM_PROMPT_TEMPLATE.format(
        max_skill=MAX_SKILL_TOKENS,
        max_bundle=MAX_BUNDLE_TOKENS,
        edit_budget_line=edit_budget_line,
        meta_skill_body=meta_skill_body or "(empty — no consolidated patterns yet)",
        recent_rejections=_render_recent_rejections(buffer),
    )


# ---------------------------------------------------------------------------
# Consolidator (slow-update) proposer prompt — Group G (plan §7l)
# ---------------------------------------------------------------------------
#
# Fires every K iters from track_b/openevolve_skills/iteration.py. Output
# is a SINGLE EDIT_FILE sentinel block on SKILL.md whose body REPLACES
# the slow-update fenced region only — the iteration loop double-guards
# via shared.slow_update.extract/replace so even if the consolidator's
# body lies about out-of-fence content, only in-fence bytes are taken
# (per plan §7l "consolidator I/O" contract).
#
# Paper-divergence (plan §7l): input is simplified relative to
# skillopt/prompts/slow_update.md — no A-vs-B skill comparison, no
# four-way regression / persistent-failure / improvement / stable-success
# categorization, no prior-guidance self-reflection slot. skill_evolve
# materializes accumulated cross-iter evidence (accepted-edit log + the
# rejected buffer + persistent-failure list) into a single best_skill
# input.

CONSOLIDATOR_PROPOSER_SYSTEM_PROMPT = """\
You are the SLOW-UPDATE consolidator for a skill-bundle evolution loop.

You run every few iterations and rewrite a small protected region inside
SKILL.md that the fast (per-iter) proposer cannot touch. Your channel is
called the "slow-update field" and is delimited verbatim by the markers:

    <!-- SLOW_UPDATE_START -->
    ... your content ...
    <!-- SLOW_UPDATE_END -->

Your job is to consolidate accumulated cross-iteration evidence —
which mutation shapes have helped, which keep failing, which task
clusters keep regressing — into concise, generalizable guidance that
the fast proposer will read on every subsequent iter.

## CURRENT BEST SKILL

Here is the current best SKILL.md including its slow-update fence:

{best_skill}

## RECENT EDIT HISTORY

Recent accepted edits + the live RejectedBuffer (most-recent first):

{edit_history}

## PERSISTENT FAILURES

Task IDs that have failed in the last K consecutive iters (these are
the candidates for slow-update guidance):

{persistent_failures}

## OUTPUT CONTRACT

Emit EXACTLY ONE sentinel block, no prose before or after:

    <<<EDIT_FILE SKILL.md>>>
    ... full rewritten SKILL.md ...
    <<<END_FILE>>>

Within that block:

* The <!-- SLOW_UPDATE_START --> and <!-- SLOW_UPDATE_END --> markers
  MUST appear verbatim, in that order, exactly once each, at the END
  of the file (paper-faithful position).
* Bytes OUTSIDE the fence MUST be byte-identical to the current best
  SKILL.md (the iteration loop double-guards via extract/replace so
  drift is caught; emit the file faithfully anyway).
* Bytes INSIDE the fence are YOUR consolidated guidance. Concise,
  generalizable, no task-specific identifiers, no session IDs, no
  user names. Treat the region as a rolling "lessons learned" memo
  for the fast proposer — append-or-revise, do not duplicate prior
  iters' content verbatim.

CRITICAL: the <<<EDIT_FILE>>> block MUST end with <<<END_FILE>>> on its
own line. A patch missing the terminator is REJECTED outright.
"""


def render_consolidator_prompt(
    *,
    best_skill: str,
    edit_history: str,
    persistent_failures: str,
) -> str:
    """Render the consolidator system prompt with live slot values."""
    return CONSOLIDATOR_PROPOSER_SYSTEM_PROMPT.format(
        best_skill=best_skill or "(empty)",
        edit_history=edit_history or "(no recent edits)",
        persistent_failures=persistent_failures or "(none)",
    )


# Module-level assertion call — runs after CONSOLIDATOR_PROPOSER_SYSTEM_PROMPT
# is defined so the META-SKILL / fence-marker checks succeed at import.
_assert_prompt_well_formed()


# ---------------------------------------------------------------------------
# Failure / success reflection proposer prompts (Group H, SkillOpt port)
# ---------------------------------------------------------------------------
#
# These two prompts replace the SINGLE proposer call under
# ``--reflection-mode partition``. The iteration loop runs the two prompts
# in parallel via ``concurrent.futures.ThreadPoolExecutor(max_workers=2)``;
# the resulting per-side op lists are merged via
# ``skill_evolve.shared.reflection.merge_patches`` (failure-priority
# keyed-dict resolver — see §7l divergence note).
#
# Paper anchors (per §7m, preserved as instructional bones — common-pattern
# focus, no edge-case hardcoding, generalizability mandate, slow-update
# protection notice):
#   * ``skillopt/prompts/analyst_error.md``
#   * ``skillopt/prompts/analyst_success.md``
#
# The slot shape mirrors SENTINEL_PROPOSER_SYSTEM_PROMPT:
#   * {max_skill}, {max_bundle}      — HARD LIMITS verbatim numbers.
#   * {edit_budget_line}             — F's L_t per-iter cap; the caller
#                                       fills this slot at proposer-call
#                                       time via ``compute_lt(...)``.
#   * {recent_rejections}            — E's RejectedBuffer render.
#   * {meta_skill_body}              — G's MetaSkill render. Optional;
#                                       falls back to an empty marker when
#                                       G hasn't merged.
#   * {trajectories}                 — H-specific: the failure or success
#                                       minibatch trajectories rendered
#                                       per-task by the caller.

_FAILURE_REFLECTION_PROPOSER_SYSTEM_PROMPT_TEMPLATE = """You are an expert failure-analysis agent for AI agent tasks.

You will be given MULTIPLE failed agent trajectories from a single minibatch
and the current skill document. Your job is to identify the most important
COMMON failure patterns across the batch and propose a concise set of skill
edits as a sentinel-block patch.

## Analysis Process

1. Read ALL trajectories in the minibatch.
2. Identify the most prevalent, systematic failure patterns across them.
3. For each pattern, classify its failure type (missing_procedure_knowledge,
   wrong_command_syntax, missing_script_capability, incomplete_workflow).
4. Propose skill edits that address the COMMON patterns — not individual
   edge cases.
5. Edits MUST be generalizable; do NOT hardcode task-specific values
   (task IDs, file names from a single trajectory, user names, etc.).
6. Only patch gaps in the skill — do NOT duplicate existing content.

You will be told the maximum number of edits (the budget L_t). Produce AT
MOST L_t edits, focusing on the highest-impact failure patterns. You may
produce fewer if warranted.

IMPORTANT: The skill document may contain a section between
<!-- SLOW_UPDATE_START --> and <!-- SLOW_UPDATE_END --> markers.
This is a PROTECTED section managed by a separate slow-update process.
Do NOT propose any edits that target, modify, or delete content within
these markers.

HARD LIMITS:
- An empty patch is REJECTED.
- SKILL.md ≤ {max_skill} cl100k_base tokens.
- Whole bundle ≤ {max_bundle} cl100k_base tokens.
- Each script ≤ 150 lines.
- No `pip install` calls; stdlib + requirements.txt only.
- All script CLIs via argparse. No hardcoded paths.
- py_compile / `bash -n` must pass.
- Generic guidance only — no session IDs, user names, project identifiers.

Sentinel-block format (the ONLY accepted format):

    <<<ADD_FILE path/relative/to/bundle>>>
    ... full file content ...
    <<<END_FILE>>>

    <<<EDIT_FILE path/relative/to/bundle>>>
    ... FULL rewritten body (NOT a diff) ...
    <<<END_FILE>>>

    <<<DELETE_FILE path/relative/to/bundle>>>

    <<<REWRITE_FOLDER scripts>>>
    --- file: scripts/foo.py
    ... content ...
    <<<END_REWRITE>>>

## EDIT BUDGET

You may propose AT MOST {edit_budget_line} file operations this iteration.
If you propose more, only the top {edit_budget_line} will be applied (in
the order you list them). Focus on the highest-impact failure-pattern fixes.

## META-SKILL

{meta_skill_body}

## RECENT REJECTIONS — DO NOT REPEAT

The following recent candidate patches were rejected by the validation
gate. Do NOT re-propose the same shape — pick a different mutation target
or strategy:

{recent_rejections}

## FAILURE TRAJECTORIES

{trajectories}

CRITICAL: every ADD_FILE and EDIT_FILE block MUST end with <<<END_FILE>>>
on its own line. A patch missing any <<<END_FILE>>> terminator is REJECTED
outright.

Reasoning ≤ 200 words.
"""


_SUCCESS_REFLECTION_PROPOSER_SYSTEM_PROMPT_TEMPLATE = """You are an expert success-pattern analyst for AI agents.

You will be given MULTIPLE successful agent trajectories from a single
minibatch and the current skill document. Your job is to identify
generalizable behavior patterns that are COMMON across the batch and
worth encoding in the skill, then emit a sentinel-block patch that
reinforces / generalizes those patterns.

## Rules

- Only propose patches for patterns NOT already covered in the skill.
- Focus on patterns that appear across MULTIPLE trajectories in the batch.
- Be concise. Patterns MUST generalize beyond specific tasks.
- Prefer reinforcing existing sections over adding new top-level sections.
- Do NOT hardcode task-specific values (task IDs, file names from a single
  trajectory, user names, etc.).

You will be told the maximum number of edits (the budget L_t). Produce AT
MOST L_t edits, focusing on the most broadly applicable success patterns.
You may produce fewer if warranted.

IMPORTANT: The skill document may contain a section between
<!-- SLOW_UPDATE_START --> and <!-- SLOW_UPDATE_END --> markers.
This is a PROTECTED section managed by a separate slow-update process.
Do NOT propose any edits that target, modify, or delete content within
these markers.

HARD LIMITS:
- An empty patch is REJECTED.
- SKILL.md ≤ {max_skill} cl100k_base tokens.
- Whole bundle ≤ {max_bundle} cl100k_base tokens.
- Each script ≤ 150 lines.
- No `pip install` calls; stdlib + requirements.txt only.
- All script CLIs via argparse. No hardcoded paths.
- py_compile / `bash -n` must pass.
- Generic guidance only — no session IDs, user names, project identifiers.

Sentinel-block format (the ONLY accepted format):

    <<<ADD_FILE path/relative/to/bundle>>>
    ... full file content ...
    <<<END_FILE>>>

    <<<EDIT_FILE path/relative/to/bundle>>>
    ... FULL rewritten body (NOT a diff) ...
    <<<END_FILE>>>

    <<<DELETE_FILE path/relative/to/bundle>>>

    <<<REWRITE_FOLDER scripts>>>
    --- file: scripts/foo.py
    ... content ...
    <<<END_REWRITE>>>

## EDIT BUDGET

You may propose AT MOST {edit_budget_line} file operations this iteration.
If you propose more, only the top {edit_budget_line} will be applied (in
the order you list them). Focus on the broadest-applicability patterns.

## META-SKILL

{meta_skill_body}

## RECENT REJECTIONS — DO NOT REPEAT

The following recent candidate patches were rejected by the validation
gate. Do NOT re-propose the same shape — pick a different mutation target
or strategy:

{recent_rejections}

## SUCCESS TRAJECTORIES

{trajectories}

CRITICAL: every ADD_FILE and EDIT_FILE block MUST end with <<<END_FILE>>>
on its own line. A patch missing any <<<END_FILE>>> terminator is REJECTED
outright.

Reasoning ≤ 200 words.
"""


# Module-level constants — rendered with canonical defaults at import so
# the strings are valid prompts even before a runtime call site re-renders.
# The {edit_budget_line} slot defaults to ``8`` (paper L_0); runtime
# callers re-render with the live L_t value via
# ``str.format(edit_budget_line=...)``. Same pattern for the other slots.
FAILURE_REFLECTION_PROPOSER_SYSTEM_PROMPT = (
    _FAILURE_REFLECTION_PROPOSER_SYSTEM_PROMPT_TEMPLATE.format(
        max_skill=MAX_SKILL_TOKENS,
        max_bundle=MAX_BUNDLE_TOKENS,
        edit_budget_line=8,
        meta_skill_body="(empty — no consolidated patterns yet)",
        recent_rejections="(none yet)",
        trajectories="(none yet)",
    )
)

SUCCESS_REFLECTION_PROPOSER_SYSTEM_PROMPT = (
    _SUCCESS_REFLECTION_PROPOSER_SYSTEM_PROMPT_TEMPLATE.format(
        max_skill=MAX_SKILL_TOKENS,
        max_bundle=MAX_BUNDLE_TOKENS,
        edit_budget_line=8,
        meta_skill_body="(empty — no consolidated patterns yet)",
        recent_rejections="(none yet)",
        trajectories="(none yet)",
    )
)


def _assert_reflection_prompts_well_formed() -> None:
    """Runtime check for the H reflection prompts.

    Mirrors ``_assert_prompt_well_formed`` above (explicit RuntimeError
    not bare assert, so the check lives under ``python -O``).
    """
    for name, prompt in (
        (
            "FAILURE_REFLECTION_PROPOSER_SYSTEM_PROMPT",
            FAILURE_REFLECTION_PROPOSER_SYSTEM_PROMPT,
        ),
        (
            "SUCCESS_REFLECTION_PROPOSER_SYSTEM_PROMPT",
            SUCCESS_REFLECTION_PROPOSER_SYSTEM_PROMPT,
        ),
    ):
        if "<<<END_FILE>>>" not in prompt:
            raise RuntimeError(
                f"<<<END_FILE>>> sentinel must appear verbatim in {name}"
            )
        if "## EDIT BUDGET" not in prompt:
            raise RuntimeError(f"## EDIT BUDGET section must appear in {name}")
        if str(MAX_SKILL_TOKENS) not in prompt:
            raise RuntimeError(f"MAX_SKILL_TOKENS must appear verbatim in {name}")
        if str(MAX_BUNDLE_TOKENS) not in prompt:
            raise RuntimeError(f"MAX_BUNDLE_TOKENS must appear verbatim in {name}")
        if "{edit_budget_line}" in prompt:
            raise RuntimeError(f"{{edit_budget_line}} slot in {name} was not rendered")


_assert_reflection_prompts_well_formed()


def render_failure_reflection_prompt(
    *,
    edit_budget_line: int | str = 8,
    recent_rejections: str = "(none yet)",
    meta_skill_body: str = "(empty — no consolidated patterns yet)",
    trajectories: str = "(none yet)",
) -> str:
    """Render ``FAILURE_REFLECTION_PROPOSER_SYSTEM_PROMPT`` with live slots.

    Convenience wrapper for the runtime proposer call site under
    ``--reflection-mode partition``. Re-renders the template with the
    live L_t / rejected-buffer / meta-skill / trajectory slots in place
    of the module-import defaults.
    """
    return _FAILURE_REFLECTION_PROPOSER_SYSTEM_PROMPT_TEMPLATE.format(
        max_skill=MAX_SKILL_TOKENS,
        max_bundle=MAX_BUNDLE_TOKENS,
        edit_budget_line=edit_budget_line,
        meta_skill_body=meta_skill_body,
        recent_rejections=recent_rejections,
        trajectories=trajectories,
    )


def render_success_reflection_prompt(
    *,
    edit_budget_line: int | str = 8,
    recent_rejections: str = "(none yet)",
    meta_skill_body: str = "(empty — no consolidated patterns yet)",
    trajectories: str = "(none yet)",
) -> str:
    """Render ``SUCCESS_REFLECTION_PROPOSER_SYSTEM_PROMPT`` with live slots."""
    return _SUCCESS_REFLECTION_PROPOSER_SYSTEM_PROMPT_TEMPLATE.format(
        max_skill=MAX_SKILL_TOKENS,
        max_bundle=MAX_BUNDLE_TOKENS,
        edit_budget_line=edit_budget_line,
        meta_skill_body=meta_skill_body,
        recent_rejections=recent_rejections,
        trajectories=trajectories,
    )
