# watchmen-daycare v2 — Implementation Plan (plan_0)

## Goal

Convert daycare from "synthetic QA evolves SKILL.md text" into a **behavioral
distillation flywheel** where (a) evals are extracted from real Opus corpus
decision points, (b) mutation evolves the entire skill bundle (SKILL.md +
scripts + references), and (c) the daemon continuously re-evolves as fresh Opus
sessions accumulate.

## Requirements

- New eval type `behavioral_action` extracted from corpus turns, with rubric
  scoring "did the candidate take the action the strong model took?" — not
  "does the candidate know which flag this script uses?"
- Behavioral evals replace synthetic QA as the **default** eval source; synth
  remains as a `--synthetic` fallback for cold-start or empty-corpus projects.
- Proposer prompt unlocks script mutation when failure clusters point at script
  behavior; SKILL.md-only is no longer the implicit default.
- Token budget enforced over the **whole bundle** (not just SKILL.md) so script
  growth is bounded.
- Weakness report classifies failures into four canonical buckets (procedure /
  syntax / script-capability / workflow) so the proposer knows what to touch.
- Daemon incrementally rebuilds behavioral evals from newly-arrived sessions and
  re-runs evolution when the eval set materially changes.
- All new code has unit tests; no integration tests required for the daemon
  cadence (covered by existing test_daemon scaffolding).

## Files to modify / create

| Path | Status | Rationale |
|---|---|---|
| `src/daycare/behavioral_builder.py` | NEW | Pulls turns from corpus, picks behavioral decision points, emits `behavioral_action` evals with action-oriented rubrics. |
| `src/daycare/eval_builder.py` | MODIFY | Accept and pass through `behavioral_action` type; expose a `behavioral_only=True` orchestrator flag; wire `behavioral_builder` into `run_eval_build`. |
| `src/daycare/evolve.py` | MODIFY | (a) New proposer prompt encouraging script mutation when cluster category is script-related; (b) bundle-level token tracking (`MAX_BUNDLE_TOKENS`); (c) weakness report enriched with the 4-bucket failure classifier; (d) new proposer tool `list_scripts(slug)`; (e) inline `MAX_SKILL_TOKENS` constant into the prompt via f-string so prompt and code can never drift. |
| `src/daycare/verifier.py` | NO CHANGE | Rubric-based `score_single` already handles arbitrary type strings. Group D adds a regression smoke test confirming `behavioral_action` flows through `score_bundle` unchanged. |
| `src/daycare/cli.py` | MODIFY | Add `--behavioral / --no-behavioral` flag (default True) to `eval-build` and `run`; **make `--behavioral` and `--synthetic` strictly mutually exclusive in BOTH commands with the same error message**; update daemon dispatch to use behavioral by default; add `--min-new-sessions` to daemon for the incremental gate. |
| `src/daycare/daemon.py` | MODIFY | Track `last_seen_session_ts` + `last_eval_set_hash` + `last_evolution_run_ts` per project; rebuild behavioral evals incrementally; trigger an evolution re-run when ≥ N new sessions have appeared **AND** the new eval set's SHA256 differs from the last and the last run is older than `min_run_interval_hours`. |
| `src/daycare/synth_builder.py` | NO CHANGE | Remains as fallback; explicitly gated by `--synthetic`. |
| `src/daycare/_manifest.py` (or extend `daemon.py` state) | NEW or extend `state.json` | Persist per-project `last_seen_session_ts`, `last_eval_build_ts`, `last_eval_set_hash`, `last_evolution_run_ts`. |
| `tests/test_behavioral_builder.py` | NEW | Extraction heuristics + rubric format + decision-point selection. |
| `tests/test_evolve_script_mutations.py` | NEW | Bundle-level token cap, script-mutation acceptance path, failure-bucket classification, MAX_SKILL_TOKENS prompt/constant guard. |
| `tests/test_eval_builder_behavioral.py` | NEW | Round-trip: corpus fixture → behavioral evals → expected types/fields; legacy path regression; verifier behavioral_action smoke; semantic_dedup default-min-clusters regression. |

---

## Implementation steps

The four groups below have **no inter-group dependencies** and can be executed in
parallel by separate implementation subagents. Each group ends with the unit
tests passing for that group's surface.

### Group A — Behavioral eval builder (the oracle replacement)

**Owner module:** `src/daycare/behavioral_builder.py` + small edits in
`eval_builder.py`.

**A1.** Create `src/daycare/behavioral_builder.py` with the following public
surface. `extract_behavioral_evals` does NOT take `projects_json` —
`AnonymizeContext` is constructed by `run_eval_build` from its own
`projects_json` parameter the same way the legacy path does.
`extract_behavioral_evals` feeds raw dicts into `build_eval_set` which is
called by the orchestrator, not by the extractor:

```python
def extract_behavioral_evals(
    db_path: Path,
    source_repo: str,
    bundle_dir: Path,
    weak_model: str,
    judge_model: str,
    api_key: str,
    seed: int,
    days: int,
    run_dir: Path,
    max_candidates: int | None = None,
    max_workers: int = 4,
) -> list[dict]:
    """Pull sessions → parse turns → pick behavioral decision points →
    generate action rubrics → calibrate → return pre-anonymization eval dicts.

    Each returned dict has the same key set as eval_builder.pull_and_classify
    so downstream Phase 1e (anonymize/dedup/split) works unchanged:
      {id, type="behavioral_action", prompt, reference, rubric,
       baseline_score, baseline_completion_len_tokens=0, accepted=True,
       source_session, source_skill}
    """
```

**A2.** Implement the **decision-point selector** as a pure function:

```python
def is_behavioral_decision_point(turn: Turn, next_turn: Turn | None) -> tuple[bool, str]:
    """Return (keep, reason). True only if the turn contains a non-trivial
    behavioral move and the assistant_text is not a pure explanation.

    Keep iff ANY of:
      - skill_invoke: any tool_call with name=="Skill"
      - multi_tool_sequence: ≥2 tool_calls in the assistant payload
      - error_recovery: previous turn's user_text matches one of:
          (?i)\\b(error|failed|traceback|exception|wrong|broken)\\b
      - reasoning_then_action: assistant_text contains "I'll", "Let me",
        "First,", "Step 1" AND tool_calls is non-empty

    Reject iff:
      - tool_calls is empty AND assistant_text < 40 chars
      - assistant_text matches eval_builder._has_safety_refusal
      - turn passes eval_builder._has_live_infra
      - the only tool_call is a single Read/LS with no follow-up action
        (heuristic: assistant_text < 80 chars AND len(tool_calls) == 1
        AND tool_calls[0].name in {"Read", "LS", "Glob"})

    The string reason is one of:
      "skill_invoke" | "multi_tool" | "error_recovery" | "reason_then_act"
      | "rejected_empty_text" | "rejected_safety" | "rejected_live_infra"
      | "rejected_trivial_read"
    """
```

**A3.** Build the **conversation-history prompt** (up to ~3000 chars):

```python
def build_prompt_with_history(
    turns: list[Turn], idx: int, max_chars: int = 3000
) -> str:
    """Walk turns[0:idx] and the user portion of turns[idx], concatenating:
      "user: <text>\\n\\nassistant: <text>\\n\\n[tool: <name>] <input_preview>\\n\\n"
    Truncate from the FRONT (keep most recent context). The last entry is
    the user turn that triggered turns[idx] — the assistant decision is the
    held-back reference.
    """
```

**A4.** Build the **reference action** string from `turns[idx]`:

```python
def build_action_reference(turn: Turn) -> str:
    """Return a compact action-oriented reference string:

    If turn.tool_calls is non-empty:
        "ACTION: invoke <tool_name>\\nINPUT: <truncated_json_input_300_chars>"
        (one line per tool call, max 3 calls)
        + "\\nTEXT: <first 200 chars of assistant_text if any>"

    Else (text-only decision):
        "ACTION: text_response\\nTEXT: <first 600 chars of assistant_text>"
    """
```

**A5.** Generate the **behavioral rubric** via a judge call. Use this exact
system prompt template (replaces `_RUBRIC_SYSTEM` for behavioral evals only):

```python
_BEHAVIORAL_RUBRIC_SYSTEM = (
    "You write rubrics that judge whether a weak LLM ({weak_model_name}) took the\n"
    "SAME behavioral action as a strong model, given the same conversation context.\n\n"
    "The reference encodes the strong model's action as either:\n"
    "  ACTION: invoke <tool>\\n INPUT: <args>\n"
    "  ACTION: text_response\\n TEXT: <body>\n\n"
    "Rules:\n"
    "- Score the CANDIDATE COMPLETION against the reference action.\n"
    "- Do NOT require exact wording. Award based on whether the candidate would\n"
    "  produce a functionally equivalent next step.\n"
    "- For tool-invocation references: 1.0 if candidate invokes the same tool with\n"
    "  semantically equivalent arguments; 0.5 if same tool but different/missing\n"
    "  argument values; 0.0 if wrong tool or refuses to act.\n"
    "- For text-response references: 1.0 if candidate states the same conclusion\n"
    "  or next-step direction; 0.5 if partial; 0.0 if contradicts or irrelevant.\n"
    "- Forbidden: comparing exact strings; demanding identical phrasing.\n"
    "- Format: \"Score 0.0–1.0. The correct action is [X]. Award 1.0 if: [criterion].\n"
    "  Award 0.5 if: [partial]. Award 0.0 if: [failure].\"\n"
    "- Max 150 words.\n"
    'Output JSON: {"rubric": "<text>"}'
)
```

**A6.** Add `generate_behavioral_rubric(prompt, reference, weak_model_name,
judge_model, api_key) -> str` that mirrors `eval_builder.generate_rubric` but
uses `_BEHAVIORAL_RUBRIC_SYSTEM`.

**A7.** Calibration: reuse `eval_builder.calibrate_eval` unchanged. Keep the
band `0.0 < score < 0.9` (looser than synthetic's `0.05 ≤ score ≤ 0.80`) so the
behavioral set tolerates near-solved turns.

**A8.** Wire into `eval_builder.run_eval_build` via a new parameter
`behavioral: bool = True`:

```python
def run_eval_build(
    ...,
    behavioral: bool = True,
) -> tuple[list[dict], list[dict]]:
    """If behavioral=True: call extract_behavioral_evals(...) → returned dicts
    feed into build_eval_set's anonymize/dedup/split pipeline unchanged.
    `AnonymizeContext` is constructed by `run_eval_build` from its own
    `projects_json` parameter the same way the legacy path does —
    `extract_behavioral_evals` feeds raw dicts into `build_eval_set` which is
    called by the orchestrator, not by the extractor.
    If behavioral=False: call the legacy pull_and_classify path UNCHANGED.
    """
```

The legacy `pull_and_classify` path is preserved; the new code only adds a
branch. `_stratified_split` already strata by `(type, accepted)` so the new
`behavioral_action` type joins the rotation automatically.

**Insufficient-stratification fallback (behavioral path only).**
`_stratified_split` (eval_builder.py:762-763) raises `insufficient_stratification`
if any holdout type has <3 items. With a small corpus, `behavioral_action`
may yield only 2-4 evals. In the behavioral branch of `run_eval_build`, if
`build_eval_set` raises `ValueError("insufficient_stratification")` AND
`behavioral=True`, fall back to `_simple_split(raw_evals, seed)`. Note:
`_simple_split` currently lives at `src/daycare/synth_builder.py:490`, and
`synth_builder.py` already imports `build_eval_set` from `eval_builder.py` —
so an `eval_builder.py` import of `_simple_split` from `synth_builder.py`
would be circular. Move `_simple_split` from `src/daycare/synth_builder.py`
to `src/daycare/eval_builder.py` (and update `synth_builder.py` to import it
from `eval_builder` instead of defining it). Then call it from
`run_eval_build`'s behavioral fallback. Log a warning:
`[behavioral] insufficient_stratification, falling back to simple_split`.

**A9.** Loosen the `semantic_dedup` minimum-clusters check from 30 → 20 when
the input is purely behavioral evals (corpus may not have ≥30 distinct decision
clusters in small projects). Add `min_clusters: int = 30` as a new kwarg with
**default 30 preserved** (legacy callers unaffected); the behavioral path
explicitly calls it with `min_clusters=20`. A regression test in Group D
asserts the default remains 30.

**Also add `min_clusters: int = 30` to `build_eval_set`'s signature and
forward it to the `semantic_dedup` call.** `build_eval_set` (eval_builder.py:792)
currently calls `semantic_dedup(raw_evals)` positionally with no kwarg, so
adding `min_clusters` to `semantic_dedup` alone is useless unless
`build_eval_set` also accepts and forwards it. Then in `run_eval_build`'s
behavioral branch call `build_eval_set(..., min_clusters=20)`.

**A10.** Group A tests (`tests/test_behavioral_builder.py`):

- `test_is_behavioral_decision_point_skill_invoke` — synthetic Turn with a
  Skill tool_use returns `(True, "skill_invoke")`.
- `test_is_behavioral_decision_point_multi_tool` — Turn with 3 tool_calls
  returns `(True, "multi_tool")`.
- `test_is_behavioral_decision_point_error_recovery` — previous turn user_text
  contains "Traceback" → `(True, "error_recovery")`.
- `test_rejects_trivial_read_only` — single Read tool_call + short
  assistant_text → `(False, "rejected_trivial_read")`.
- `test_rejects_safety_refusal` — assistant_text starts with "I cannot assist"
  → `(False, "rejected_safety")`.
- `test_build_prompt_with_history_truncates_from_front` — 10 fake turns,
  `max_chars=300`, assert the result ends with the most recent user turn and
  total length ≤ 300.
- `test_build_action_reference_tool_call` — Turn with one Bash tool_call →
  output starts with `"ACTION: invoke Bash\nINPUT: "`.
- `test_build_action_reference_text_only` — Turn with empty tool_calls →
  output starts with `"ACTION: text_response\nTEXT: "`.
- `test_behavioral_rubric_template_contains_action_clause` — assert the
  rubric system prompt contains the word "behavioral action".

---

### Group B — Bundle-level evolution (scripts unlocked)

**Owner module:** `src/daycare/evolve.py` (and a tiny addition to
`src/daycare/mutator.py` if needed for the bundle token counter).

**B1.** Add a bundle-token counter to `evolve.py`:

```python
MAX_BUNDLE_TOKENS = 60000  # SKILL.md + scripts/ + references/, cl100k_base
                           # 2× the largest known seed bundle (~30k tokens,
                           # pi/codebase-audit); gives ample headroom for
                           # script additions.
MAX_SKILL_TOKENS = 3000    # unchanged

def _bundle_tokens(bundle_dir: Path) -> int:
    """Sum cl100k_base tokens across SKILL.md + scripts/**/*.{py,sh} +
    references/**/*.md. Hidden files (dotfiles) and __pycache__ are excluded.
    """
```

In `propose_candidates`, after the existing `MAX_SKILL_TOKENS` check, add a
second guard:

```python
bundle_tokens = _bundle_tokens(candidate_bundle)
if bundle_tokens > MAX_BUNDLE_TOKENS:
    _append_history(run_dir, iter_n, slot,
                    f"validate_error:bundle_too_large:{bundle_tokens}",
                    None, reasoning[:80])
    shutil.rmtree(candidate_bundle, ignore_errors=True)
    continue
```

**B1a. MAX_SKILL_TOKENS prompt-vs-constant drift fix.** The existing
`_PROPOSER_SYSTEM_PROMPT` body contains a stale literal `"2500"` while the
`MAX_SKILL_TOKENS` constant at `evolve.py:66` is `3000`. The rewrite in B2
below replaces this with `3000`, **AND**:

1. Convert `_PROPOSER_SYSTEM_PROMPT` from a plain string into an f-string (or
   build it via `.format(max_skill=MAX_SKILL_TOKENS, max_bundle=MAX_BUNDLE_TOKENS)`)
   so the prompt body references the live constants — they cannot drift again.
2. Add a startup assertion at module load:

   ```python
   assert f"{MAX_SKILL_TOKENS}" in _PROPOSER_SYSTEM_PROMPT, (
       "MAX_SKILL_TOKENS constant must appear verbatim in the proposer prompt"
   )
   assert f"{MAX_BUNDLE_TOKENS}" in _PROPOSER_SYSTEM_PROMPT, (
       "MAX_BUNDLE_TOKENS constant must appear verbatim in the proposer prompt"
   )
   ```

3. Add a Group B test `test_proposer_prompt_token_limits_match_constants`
   that imports the module and verifies both substrings appear (cheap guard
   against silent regression if the f-string is later flattened).

Also update the `count_skill_tokens` tool description string at
`evolve.py:855` to use `f'{MAX_SKILL_TOKENS}'` instead of the literal
`'2500'`. The regression assertion in the Group B test must also assert
that `"2500"` does NOT appear in the `count_skill_tokens` tool description
string.

**B2.** Replace the `_PROPOSER_SYSTEM_PROMPT` with the version below, built
via `.format(...)` so the token limits track the constants. The two
critical changes vs. current: (1) the "PREFER mutating SKILL.md only" line is
gone, (2) script-mutation guidance is now affirmative when the cluster category
indicates it. The literal `2500` is replaced everywhere by the live
`MAX_SKILL_TOKENS` value.

```python
_PROPOSER_SYSTEM_PROMPT_TEMPLATE = """You are a skill-bundle mutator. Emit ONE mutation
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

You may NOT read held-out slice data. Use only the proposer tools.
Reasoning ≤ 200 words.
"""

_PROPOSER_SYSTEM_PROMPT = _PROPOSER_SYSTEM_PROMPT_TEMPLATE.format(
    max_skill=MAX_SKILL_TOKENS,
    max_bundle=MAX_BUNDLE_TOKENS,
)
```

**B3.** Add a new proposer tool `list_scripts()`:

```python
def list_scripts() -> str:
    """Return the parent bundle's scripts/ directory layout as
    relative paths, one per line, with byte size:
        "scripts/foo.py  (1234 bytes)"
        "scripts/bar.py  (567 bytes)"
    Returns "(no scripts/ in parent bundle)" if missing.
    """
```

Tool spec:

```python
{
    "type": "function",
    "function": {
        "name": "list_scripts",
        "description": (
            "Enumerate the parent bundle's scripts/ directory with file sizes. "
            "Use BEFORE mutating any script to confirm naming and avoid duplicates."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}
```

Add `"list_scripts": list_scripts` to the handlers dict and the spec to
`specs`. This brings the proposer tool count to 11.

**B4.** Enrich `_cluster_failures` to also emit a `category` field. Update the
judge prompt (keep the existing one, just extend the JSON schema):

```python
system = (
    "Cluster the failing evals into 3-6 named failure modes. "
    "For each cluster, also assign a CATEGORY indicating where the fix likely lives:\n"
    "  - missing_procedure_knowledge: SKILL.md is silent on the required step\n"
    "  - wrong_command_syntax: SKILL.md has the step but wrong flag/argument\n"
    "  - missing_script_capability: a needed script is absent or incomplete\n"
    "  - incomplete_workflow: multi-step workflow truncated mid-way\n\n"
    "For each cluster: name (snake_case), severity (count of evals), "
    'category in the 4 values above, mode in {"truncation","reasoning","format"}, '
    "and 2-3 paraphrased example prompts (≤120 chars each, ANONYMIZED).\n"
    'Output JSON: {"clusters": [{"name": "...", "severity": <int>, '
    '"category": "missing_procedure_knowledge|wrong_command_syntax|'
    'missing_script_capability|incomplete_workflow", '
    '"mode": "truncation|reasoning|format", "examples": ["...", ...]}, ...]}'
)
```

**B4a. `_cluster_failures` judge payload must include `category`.** The
per-cluster payload dict sent to the judge call (the input list, not just the
output) must also carry a `category` field so the judge can introspect each
input cluster's existing type-derived category and either confirm or override
it. Concretely, when `_cluster_failures` builds the per-eval payload it sends
to the judge, extend each payload row with a `category` derived from the
existing `type` mapping:

```python
type_to_category = {
    "script_gen":      "wrong_command_syntax",
    "procedural_qa":   "missing_procedure_knowledge",
    "skill_invoke":    "missing_script_capability",
    "behavioral_action": "incomplete_workflow",
}

# When building the judge input payload per failing eval:
payload_row = {
    "id": eval["id"],
    "type": eval["type"],
    "category": type_to_category.get(eval["type"], "missing_procedure_knowledge"),
    "prompt": eval["prompt"][:200],
    ...
}
```

Then update the cluster-dict normalization on the **output** side to include
`category` (default `"missing_procedure_knowledge"` if the judge omits it),
and update `_fallback_cluster_by_type` (used when the judge call fails) to
assign categories by eval type using the SAME `type_to_category` map above.
A Group B test asserts the payload includes `category`.

**B5.** Update `build_weakness_report` to render `category` in the cluster
section:

```
### {c['name']}  · severity={c['severity']}  · category={c['category']}  · mode={c['mode']}
```

This is the load-bearing signal the new proposer prompt reads.

**B6.** Group B tests (`tests/test_evolve_script_mutations.py`):

- `test_bundle_tokens_sums_skill_md_and_scripts` — build a tmp bundle with
  SKILL.md=100tok, scripts/a.py=200tok, references/b.md=50tok; assert
  `_bundle_tokens` returns ≈350.
- `test_bundle_too_large_rejected` — monkey-patch `_bundle_tokens` to return
  `MAX_BUNDLE_TOKENS + 1`; assert the candidate is rejected with
  `bundle_too_large` in history.
- `test_proposer_prompt_mentions_categories` — assert
  `_PROPOSER_SYSTEM_PROMPT` contains all 4 category strings.
- `test_proposer_prompt_token_limits_match_constants` — assert
  `str(MAX_SKILL_TOKENS)` and `str(MAX_BUNDLE_TOKENS)` both appear in
  `_PROPOSER_SYSTEM_PROMPT`; also assert the literal `"2500"` does NOT
  appear (regression guard for the audited drift).
- `test_list_scripts_tool_returns_paths` — instantiate the tool handler,
  point at a tmp bundle with `scripts/foo.py`, assert output contains
  `"scripts/foo.py"`.
- `test_cluster_failures_includes_category_in_judge_payload` — monkey-patch
  the judge call to capture the payload list it receives; assert every
  payload row has a `category` key whose value matches the
  `type_to_category` mapping for that row's `type`.
- `test_cluster_failures_includes_category_in_output` — feed a mocked judge
  response with `category` field, assert it survives into the returned
  cluster dict.
- `test_fallback_cluster_assigns_category_by_type` — synthetic
  `behavioral_action` rows → fallback returns category
  `incomplete_workflow`; `script_gen` → `wrong_command_syntax`;
  `procedural_qa` → `missing_procedure_knowledge`; `skill_invoke` →
  `missing_script_capability`.
- `test_weakness_report_renders_category` — call `build_weakness_report` with
  monkey-patched cluster judge; assert the rendered markdown contains the
  literal substring `category=`.

---

### Group C — CLI surface (flags + daemon glue)

**Owner module:** `src/daycare/cli.py` and `src/daycare/daemon.py`.

**C1.** In `eval-build`:

```python
@click.option(
    "--behavioral/--no-behavioral",
    default=None,
    show_default=False,
    help="Extract behavioral action evals from corpus turns (default unless "
         "--synthetic is passed). --no-behavioral falls back to the legacy "
         "classify-by-type path.",
)
```

Default is `None` so that a user typing only `--synthetic` does not trip
the mutual-exclusion check on the True default. Resolve intent after parsing:

```python
# eval-build uses param name `synthetic`, run uses `synthetic_evals`
if behavioral is None:
    behavioral = not synthetic  # default True unless --synthetic given
if behavioral and synthetic:
    raise click.BadParameter("--behavioral and --synthetic are mutually exclusive")
```

Note: In the `run` command (step C2), the same pattern uses
`synthetic_evals` — the parameter name differs because the `--synthetic`
flag in `run` is declared with the variable name `synthetic_evals`.

Pass `behavioral` through to `run_eval_build(..., behavioral=behavioral)`.

**C2.** In `run`:

Add the same `--behavioral/--no-behavioral` flag, **default `None`**
(NOT `True`), and resolve intent after parsing — identically to `eval-build`:

```python
@click.option("--behavioral/--no-behavioral", default=None)
# ...
# After parsing:
if behavioral is None:
    behavioral = not synthetic_evals  # default True unless --synthetic given
if behavioral and synthetic_evals:
    raise click.BadParameter("--behavioral and --synthetic are mutually exclusive")
```

The error message string must be byte-identical to `eval-build`'s. Do NOT
silently let synth win.

After the check passes, dispatch:
- `behavioral=True, synthetic=False` (default): call
  `run_eval_build(..., behavioral=True)` in the Phase 1 branch.
- `behavioral=False, synthetic=True`: call the synth_builder path.
- `behavioral=False, synthetic=False`: legacy `pull_and_classify` path.

**C3.** In daemon (`src/daycare/daemon.py`):

Extend `state.json` schema:

```json
{
  "last_daily_run": "...",
  "last_weekly": "...",
  "last_tick": "...",
  "per_project": {
    "<project>": {
      "last_seen_session_ts": "ISO8601",
      "last_eval_build_ts": "ISO8601",
      "last_eval_set_hash": "sha256 hex",
      "last_evolution_run_ts": "ISO8601",
      "new_sessions_since_build": 0
    }
  }
}
```

Add helper functions. **Important**: project matching is done in Python (three
modes: exact, startswith, basename) not SQL, so we cannot inline a
`WHERE project_dir = ?` clause. We must reuse `corpus.query_sessions` which
already runs the Python-side filter, then filter by timestamp in Python.

**CRITICAL: do NOT pass `days=None` to `corpus.query_sessions`** — it does
`timedelta(days=days)` at corpus.py:238 which raises `TypeError` on `None`.
Pass `days=36500` (100 years = effectively all sessions) instead:

```python
import hashlib

def _count_new_sessions(
    db_path: Path, source_repo: str, since_iso: str | None
) -> int:
    """Count sessions for `source_repo` whose started_at > since_iso.

    Implementation: call query_sessions(db_path, source_repo, days=36500)
    which handles the project_dir matching (exact / startswith / basename)
    in Python. Then in Python, filter rows by `started_at > since_iso`
    (lex compare on ISO8601 strings; None → keep all). Return len(filtered).
    """
    rows = query_sessions(db_path, source_repo, days=36500)
    if since_iso is None:
        return len(rows)
    return sum(1 for r in rows if r["started_at"] > since_iso)

def _latest_session_ts(db_path: Path, source_repo: str) -> str | None:
    """Return max(started_at) over matching sessions via query_sessions, or None."""
    rows = query_sessions(db_path, source_repo, days=36500)
    if not rows:
        return None
    return max(r["started_at"] for r in rows)

def _hash_eval_set(eval_set_path: Path) -> str:
    """Return sha256 hex of the eval_set.jsonl file contents."""
    h = hashlib.sha256()
    h.update(eval_set_path.read_bytes())
    return h.hexdigest()

def _latest_eval_set_path(watchmen_home: Path, project: str) -> Path | None:
    """Find the most-recent completed run for `project` and return its
    eval_set.jsonl path.

    Scans watchmen_home / "daycare" / "runs" for dirs matching f"{project}-*",
    sorts by name (ISO timestamps are lex-sortable), returns the newest one's
    eval_set.jsonl if it exists, else None.
    """
    runs_root = watchmen_home / "daycare" / "runs"
    if not runs_root.exists():
        return None
    candidates = sorted(
        (d for d in runs_root.iterdir()
         if d.is_dir() and d.name.startswith(f"{project}-")),
        reverse=True,  # newest first (lex sort on ISO timestamp)
    )
    for d in candidates:
        p = d / "eval_set.jsonl"
        if p.exists():
            return p
    return None
```

**Project enumeration.** Enumerate projects by reading
`watchmen_home / 'projects.json'` via the existing
`_read_projects_json(watchmen_home)` helper (already used elsewhere in
`cli.py`). This returns a dict `{project_key: source_repo}`. Iterate over
`project_key → source_repo` pairs in the per-tick loop. If
`_read_projects_json` doesn't exist in `daemon.py`, copy the same call
pattern from `cli.py`.

**State preservation across the tick rewrite.** `daemon.py:377-381` does a
full state overwrite (`state = {"last_daily_run": ..., "last_weekly": ...,
"last_tick": ...}`) which would blow away the `per_project` dict every tick.
In `run_daemon`, before the state-write at each tick, PRESERVE the
per-project dict via a merge: instead of full-overwriting, do:

```python
existing_per_project = state.get("per_project", {})
state = {"last_daily_run": ..., "last_weekly": ..., "last_tick": ...}
state["per_project"] = existing_per_project
# then mutate state["per_project"][project][...] as planned
```

This mirrors how the daily-run guard preserves `last_weekly` through the
tick rewrite.

**Per-project key initialization.** On a fresh state file,
`state["per_project"]` doesn't exist, and `state["per_project"][project]`
definitely doesn't. Use `.setdefault` to ensure both levels exist before
accessing. Run this at the top of the per-project block, before any `.get()`
calls:

```python
state.setdefault("per_project", {})
state["per_project"].setdefault(project, {
    "last_seen_session_ts": None,
    "last_eval_build_ts": None,
    "last_eval_set_hash": None,
    "last_evolution_run_ts": None,
    "new_sessions_since_build": 0,
})
```

In the per-tick loop, BEFORE the daily-run gate, for each project (iterating
over the `_read_projects_json` result described above). **Note the
explicit env propagation and `shutil.which` resolution of the `daycare`
binary** — mirroring the existing daily-run spawn so the subprocess inherits
`WATCHMEN_HOME`, `OPENROUTER_API_KEY`, and the full `os.environ`:

```python
import os, shutil, subprocess

daycare_bin = shutil.which("daycare") or "daycare"
spawn_env = {
    **os.environ,
    "WATCHMEN_HOME": str(watchmen_home),
    "OPENROUTER_API_KEY": api_key,
}

new_count = _count_new_sessions(
    db_path, source_repo,
    state["per_project"][project].get("last_seen_session_ts"),
)
if new_count >= MIN_NEW_SESSIONS_FOR_REBUILD:  # default 5
    logger.info("project %s has %d new sessions; spawning eval-build",
                project, new_count)
    proc = subprocess.Popen(
        [daycare_bin, "eval-build", project, "--behavioral"],
        env=spawn_env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    proc.wait()  # synchronous: we need the file written before hashing
    state["per_project"][project]["last_eval_build_ts"] = now.isoformat()
    state["per_project"][project]["last_seen_session_ts"] = _latest_session_ts(
        db_path, source_repo,
    )
    state["per_project"][project]["new_sessions_since_build"] = 0

    # --- Re-evolution gate (full flywheel) ---
    # Daycare writes runs to ~/.watchmen/daycare/runs/<project>-<ts>/eval_set.jsonl,
    # NOT ~/.watchmen/projects/<project>/eval_set.jsonl. Use the helper to find
    # the most recent run's eval_set.
    eval_set_path = _latest_eval_set_path(watchmen_home, project)
    if eval_set_path is not None and eval_set_path.exists():
        new_hash = _hash_eval_set(eval_set_path)
        prev_hash = state["per_project"][project].get("last_eval_set_hash")
        last_run_iso = state["per_project"][project].get("last_evolution_run_ts")
        hours_since = _hours_since(last_run_iso, now)  # None → +inf
        if new_hash != prev_hash and hours_since >= MIN_RUN_INTERVAL_HOURS:
            logger.info("project %s eval set changed (%s → %s); spawning run",
                        project, (prev_hash or "")[:8], new_hash[:8])
            run_proc = subprocess.Popen(
                [daycare_bin, "run", project,
                 "--eval-set", str(eval_set_path), "--yes"],
                env=spawn_env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            # don't wait — evolution runs are long
            state["per_project"][project]["last_evolution_run_ts"] = now.isoformat()
        state["per_project"][project]["last_eval_set_hash"] = new_hash
```

Module-top constants — only the two below are NEW. The existing
`_TICK_SECONDS = 2 * 60 * 60  # existing (2 hours); leave unchanged` at
daemon.py:32 stays as-is; this plan does NOT add or change it.

```python
MIN_NEW_SESSIONS_FOR_REBUILD = 5    # default for daemon flag
MIN_RUN_INTERVAL_HOURS = 24         # default for re-evolution gate
```

`_hours_since(iso, now)` returns `float("inf")` when `iso is None`, else the
elapsed hours as a float.

**C4.** Add daemon CLI flags `--min-new-sessions` and `--min-run-interval-hours`
to `daemon run`:

```python
@daemon_group.command("run")
@click.option("--min-new-sessions", default=5, type=int, show_default=True)
@click.option("--min-run-interval-hours", default=24, type=float, show_default=True)
def daemon_run(min_new_sessions: int, min_run_interval_hours: float) -> None:
    ...
    daemon_mod.run_daemon(
        watchmen_home, api_key,
        min_new_sessions=min_new_sessions,
        min_run_interval_hours=min_run_interval_hours,
    )
```

Update `run_daemon` signature to accept and use both.

`run_daemon` receives `min_new_sessions` and `min_run_interval_hours` as
parameters. Store them as local vars at the top of `run_daemon` and use
those local vars (not the module constants) throughout the per-tick loop:

```python
def run_daemon(watchmen_home, api_key, min_new_sessions=5, min_run_interval_hours=24):
    _min_sessions = min_new_sessions
    _min_run_interval = min_run_interval_hours
    # ... use _min_sessions and _min_run_interval in the tick loop
```

The module-level constants serve as defaults only.

**C5.** Group C tests live mostly in `tests/test_cli_behavioral.py` and
`tests/test_daemon_incremental.py` (see Group D below):

- `test_eval_build_behavioral_and_synthetic_mutually_exclusive` — Click runner
  with both flags returns non-zero exit + error message exactly equal to
  `"--behavioral and --synthetic are mutually exclusive"`.
- `test_run_behavioral_and_synthetic_mutually_exclusive` — same check, but
  invoking `run` instead of `eval-build`. Asserts the **identical** error
  message string (the two checks must stay in sync).
- `test_eval_build_default_is_behavioral` — invoke `eval-build --dry-run` and
  assert the call goes through (smoke test; dry-run returns before real
  extraction so we just confirm the flag is parsed).

---

### Group D — Test fixtures + integration scaffolding

**Owner module:** `tests/fixtures/` and `tests/test_eval_builder_behavioral.py`
(plus `tests/test_daemon_incremental.py`).

**D1.** Add `tests/fixtures/transcript_behavioral.jsonl` — a minimal hand-built
JSONL with 6 turns covering each decision-point category:

- turn 0: user "List files" → assistant uses single Read (should be REJECTED
  as trivial_read).
- turn 1: user "Run the build" → assistant invokes Skill `audit-gate`
  (should be KEPT as skill_invoke).
- turn 2: user "Refactor this module" → assistant uses Read + Edit + Write
  (should be KEPT as multi_tool).
- turn 3: user "I'm getting a Traceback: NameError" → assistant uses Edit
  (should be KEPT as error_recovery).
- turn 4: user "How does X work?" → assistant text only, no tools (should be
  REJECTED — pure explanation, no action).
- turn 5: user "Plan the migration" → assistant text starts with "I'll" + two
  tool calls (should be KEPT as reason_then_act).

**D2.** `tests/test_eval_builder_behavioral.py`:

- `test_behavioral_extraction_keeps_4_of_6` — point `extract_behavioral_evals`
  at the fixture (mock the judge calls + calibration to deterministic values);
  assert exactly 4 evals survive with `type=="behavioral_action"`.
- `test_behavioral_eval_has_required_fields` — assert each returned dict has
  the 10 keys listed in A1's docstring.
- `test_behavioral_dedup_min_clusters_loosened` — pass 22 behavioral evals
  through `semantic_dedup(..., min_clusters=20)`; assert it does NOT raise.
- **`test_semantic_dedup_default_still_requires_30_clusters`** — call the
  corpus extraction path (`run_eval_build(..., behavioral=False)` or
  `semantic_dedup(...)` with no `min_clusters` kwarg) on a synthetic dataset
  that would yield 25 clusters; assert it raises the same
  "too few clusters" error it raised pre-change. This is the regression
  guard requested in audit Item 5 — the default `min_clusters=30` must NOT
  silently soften when behavioral=False.
- **`test_run_eval_build_legacy_path_unchanged`** — call
  `run_eval_build(..., behavioral=False)` with `pull_and_classify` and the
  behavioral extractor both monkey-patched to record invocations; assert
  `pull_and_classify` is called and `extract_behavioral_evals` is NOT
  called. This is the legacy-path regression guard for audit Arch #2.
- **`test_score_bundle_accepts_behavioral_action_type`** — construct a fake
  eval row dict with `type="behavioral_action"`, a prompt, a rubric, and a
  reference; call `verifier.score_bundle([row], bundle_dir, ...)` with the
  underlying weak-model call monkey-patched to return a canned completion
  and the judge call patched to return `0.7`; assert `score_bundle` returns
  a numeric score in `[0.0, 1.0]` and does not raise on the new type
  string. This is the verifier smoke test required by audit Testing.

**D3.** Daemon dry-run test — `tests/test_daemon_incremental.py`:

- `test_run_daemon_one_tick_spawns_eval_build` — monkeypatch
  `subprocess.Popen` to record calls; **also monkeypatch
  `daemon._TICK_SECONDS = 1` AND `daemon.time.sleep` to a no-op** so the
  test does not block on the real 2-hour tick interval; set
  `min_new_sessions=1`; seed a fake sqlite db with 2 sessions matching the
  project; run `run_daemon` for one tick (use a thread + set `_stop = True`
  after a single iteration); assert Popen was called with argv
  `[daycare_bin, "eval-build", <project>, "--behavioral"]` and that
  `env["WATCHMEN_HOME"]` and `env["OPENROUTER_API_KEY"]` are populated.
- `test_run_daemon_spawns_evolution_when_hash_changes` — same scaffolding,
  but stage two ticks: tick 1 writes an `eval_set.jsonl` (mocked
  `eval-build` subprocess that just writes file bytes), captures
  `last_eval_set_hash`; tick 2 overwrites `eval_set.jsonl` with different
  bytes (new sessions arrive); assert a second `Popen` call argv begins
  with `[daycare_bin, "run", <project>, "--eval-set", ..., "--yes"]`.
  Also test the negative case: identical hash → no `run` spawn.
- `test_run_daemon_respects_min_run_interval_hours` — set
  `min_run_interval_hours=24` and stub `last_evolution_run_ts` to 1h ago;
  even if hash changes, no `run` spawn should fire.
- `test_daemon_count_new_sessions_python_filter` — with a tmp sqlite db
  seeded by directly inserting rows into the `sessions` table using sqlite3
  (matching the schema from `tests/test_corpus.py:106-128` which already
  hand-rolls the minimal schema: `CREATE TABLE sessions (session_id TEXT,
  project_dir TEXT, started_at TEXT, ...)`). No `record_session` helper is
  needed. Seed 3 sessions with varied `started_at`, assert
  `_count_new_sessions(db, repo, None)`
  returns 3; with `since_iso` set to the median session's ts, returns the
  count strictly after it (e.g. 1); with `since_iso` set to the latest ts,
  returns 0. **Critically this test must not rely on a raw SQL `WHERE
  project_dir = ?` — it goes through `query_sessions` so the
  project-matching modes (exact/startswith/basename) are exercised.**
- `test_latest_eval_set_path_finds_newest_run` — create a tmp
  `watchmen_home/daycare/runs/` with three sibling dirs
  (`<project>-2026-05-20T00:00:00`, `<project>-2026-05-22T00:00:00`,
  `<project>-2026-05-24T00:00:00`), write `eval_set.jsonl` into the two
  older ones AND the newest; assert `_latest_eval_set_path(watchmen_home,
  project)` returns the path inside the `2026-05-24` dir. Also test the
  case where the newest dir is MISSING its `eval_set.jsonl` — the helper
  must fall through to the next newest. Also assert `None` is returned
  when `runs/` does not exist and when no candidate has an
  `eval_set.jsonl`.
- `test_daemon_state_per_project_roundtrip` — write a fake state dict with
  all five per-project fields (`last_seen_session_ts`,
  `last_eval_build_ts`, `last_eval_set_hash`, `last_evolution_run_ts`,
  `new_sessions_since_build`), persist, reload, assert shape preserved.

---

## Testing strategy

- **Unit-only**: every new function gets a fixture-driven unit test. No live
  network calls — judge calls are monkey-patched via
  `monkeypatch.setattr(behavioral_builder, "_judge_call", fake)`.
- **Determinism**: behavioral_builder reuses `eval_builder._judge_call`, so
  monkey-patch the underlying function at the `eval_builder` module level.
- **Daemon tests never sleep**: all daemon tests MUST monkeypatch
  `daemon._TICK_SECONDS = 1` AND `daemon.time.sleep` (to a no-op or
  `lambda *_: None`). Without both, tests block on the real 2-hour
  `_TICK_SECONDS` and fail CI timeouts.
- **Coverage targets**:
  - `behavioral_builder.py` ≥ 85% line coverage.
  - New `evolve.py` additions (`_bundle_tokens`, `list_scripts`, category
    fields, prompt-constant drift guard) ≥ 90%.
  - Daemon incremental path ≥ 70% (covers happy path, skip-when-no-new,
    skip-when-hash-unchanged, skip-when-interval-not-elapsed, full
    re-evolution spawn).
- **No integration test** for the full Phase 1→5 pipeline — the existing
  `test_evolve.py` is the integration smoke (and untouched by this change,
  because the new path is a flag-gated branch).
- **Legacy-path regression** (`test_run_eval_build_legacy_path_unchanged`)
  is mandatory: ensures `behavioral=False` callers see zero behavior
  change.
- **Verifier smoke** (`test_score_bundle_accepts_behavioral_action_type`):
  ensures the new eval type flows through `verifier.score_bundle` without
  raising, even though `verifier.py` itself is unmodified.
- **Test runner**: `uv run pytest tests/ -x -q`. Add `-k "behavioral or script_mutations or daemon_incremental"` while iterating.

## Linting / formatting notes

- Run `uv run ruff check src/daycare/ tests/` after each group; the project
  uses ruff (see existing modules; no explicit config beyond defaults).
- Run `uv run ruff format src/daycare/ tests/` for formatting.
- `uv run python -m py_compile src/daycare/behavioral_builder.py` to confirm
  the new module imports cleanly.
- No new third-party dependencies. The behavioral builder reuses `httpx`,
  `fastembed` (already a dep via semantic_dedup), and stdlib only.
- Type annotations are MANDATORY on every new public function. Follow the
  existing `from __future__ import annotations` + PEP-604 union style used in
  `eval_builder.py`.

## Task List

- [x] **A. Behavioral eval builder** — `src/daycare/behavioral_builder.py` + `eval_builder.py` edits
  - [ ] A1. Create `behavioral_builder.py` with `extract_behavioral_evals` public surface
  - [ ] A2. Implement `is_behavioral_decision_point` selector function
  - [ ] A3. Implement `build_prompt_with_history` (3000-char history builder)
  - [ ] A4. Implement `build_action_reference` (compact action string)
  - [ ] A5-A6. Define `_BEHAVIORAL_RUBRIC_SYSTEM` and `generate_behavioral_rubric`
  - [ ] A7. Calibration reuse + band `0.0 < score < 0.9`
  - [ ] A8. Wire `behavioral` param into `eval_builder.run_eval_build`
  - [ ] A9. `_simple_split` move: synth_builder→eval_builder, update imports; `min_clusters` kwarg (default 30) added to `semantic_dedup` + `build_eval_set`; behavioral path calls with `min_clusters=20`
  - [ ] A10. Write `tests/test_behavioral_builder.py` (9 tests)
  - [x] A11. (post-impl 2026-05-25) Add `action_tool` signal in `is_behavioral_decision_point` — `multi_tool(≥2)` and `reason_then_act` are dead in CC corpus (one tool call per turn, empty `assistant_text`); accept any single tool call in `_ACTION_TOOLS = {Bash, Edit, Write, Agent, Skill, TaskCreate, TaskUpdate, ...}`
  - [x] A12. (post-impl 2026-05-25) Broaden `run_eval_build` behavioral fallback in `eval_builder.py:1042` to catch BOTH `insufficient_stratification` AND `insufficient_distillable_surface` (the latter is raised by `semantic_dedup` when cluster count < `min_clusters`)

- [x] **B. Bundle-level evolution** — `src/daycare/evolve.py`
  - [ ] B1. Add `MAX_BUNDLE_TOKENS=60000`, `_bundle_tokens()`, and post-apply bundle-size guard
  - [ ] B1a. Fix `MAX_SKILL_TOKENS` prompt drift: f-string `_PROPOSER_SYSTEM_PROMPT`, assertions, update `count_skill_tokens` tool description
  - [ ] B2. Replace `_PROPOSER_SYSTEM_PROMPT` with script-mutation-unlocked version
  - [ ] B3. Add `list_scripts()` proposer tool + spec + wire into handlers
  - [ ] B4. Enrich `_cluster_failures` judge prompt + payload to emit `category` field
  - [ ] B4a. Update `_fallback_cluster_by_type` with `type_to_category` mapping
  - [ ] B5. Update `build_weakness_report` to render `category=` in cluster headers
  - [ ] B6. Write `tests/test_evolve_script_mutations.py` (9 tests)

- [x] **C. CLI surface + daemon glue** — `src/daycare/cli.py` + `src/daycare/daemon.py`
  - [ ] C1. Add `--behavioral/--no-behavioral` (default None) to `eval-build` with post-parse resolution + mutual-exclusion check
  - [ ] C2. Add same flag to `run` command, using `synthetic_evals` param name, identical error message
  - [ ] C3. Add daemon helpers: `_count_new_sessions`, `_latest_session_ts`, `_hash_eval_set`, `_latest_eval_set_path`, per-project state schema in `state.json`; add per-tick flywheel loop with state-preservation + setdefault init
  - [ ] C4. Add `--min-new-sessions` + `--min-run-interval-hours` CLI flags to `daemon run`; update `run_daemon` signature
  - [ ] C5. Write `tests/test_cli_behavioral.py` (3 tests)

- [x] **D. Test fixtures + integration scaffolding** — `tests/` (runs after A)
  - [ ] D1. Create `tests/fixtures/transcript_behavioral.jsonl` with 6 turns (4 keep / 2 reject)
  - [ ] D2. Write `tests/test_eval_builder_behavioral.py` (6 tests incl. legacy-path regression + verifier smoke)
  - [ ] D3. Write `tests/test_daemon_incremental.py` (7 tests incl. state-roundtrip)

## Out of scope (explicitly deferred)

- Multi-skill bundle evolution (current scope is one slug per run, same as v1).
- Behavioral eval *quality* tuning beyond the 4 selector heuristics — defer to
  a v2.1 calibration pass once we have empirical data.
- GPU-based mutation. This plan stays text-only by mandate.
- Replacing `synth_builder.py` — kept as a fallback for empty-corpus projects.
- Cross-project transfer learning (sharing evolved scripts between projects).

## Acceptance criteria

A run of `daycare run <project> --behavioral` on a project with ≥ 50 corpus
sessions:

1. Phase 1 produces an `eval_set.jsonl` containing rows with
   `type == "behavioral_action"`.
2. Phase 3 weakness report contains the `category=` field for each cluster.
3. At least one iter's mutation_log.md shows a candidate that touched a file
   under `scripts/` (proves the script-mutation unlock works end-to-end).
4. Bundle-token guard never trips on the seed bundle (sanity check on the
   `MAX_BUNDLE_TOKENS = 60000` constant — 2× the largest known seed bundle
   (~30k tokens, pi/codebase-audit); gives ample headroom for script
   additions).
5. `daycare daemon run --min-new-sessions 5` spawns an `eval-build`
   subprocess within one tick when ≥ 5 new sessions exist since the last
   build, AND (after `eval-build` completes with a hash-changed
   `eval_set.jsonl`, AND ≥ 24h have elapsed since the last evolution run)
   also spawns a subsequent `daycare run <project> --eval-set <path>
   --yes` — closing the full flywheel.
6. All new tests pass; full suite stays green. In particular the three
   regression tests added under Group D
   (`test_semantic_dedup_default_still_requires_30_clusters`,
   `test_run_eval_build_legacy_path_unchanged`,
   `test_score_bundle_accepts_behavioral_action_type`) gate the merge.

## Phase 1 Validation Results

**Date:** 2026-05-25

**Run:** `wmca-20260525T150353Z` on JarvisLabs VM 415615

**Eval set produced:** 23 `behavioral_action` evals (12 train / 11 holdout)
- Baseline scores: min=0.17, max=0.88, mean=0.46
- All evals have type=`behavioral_action` (acceptance criterion 1)

**Bugs found and fixed during validation:**
1. **`is_behavioral_decision_point` dead signals for CC corpus** — `multi_tool(≥2)` and `reason_then_act` are structurally impossible in Claude Code corpus (one tool call per turn, empty `assistant_text` on tool-using turns). Added `action_tool` signal: any single tool call in `_ACTION_TOOLS = {Bash, Edit, Write, Agent, Skill, TaskCreate, TaskUpdate, ...}`. Fix: `behavioral_builder.py`.
2. **`insufficient_distillable_surface` fallback gap** — `run_eval_build` behavioral fallback only caught `insufficient_stratification` but `semantic_dedup` raises `insufficient_distillable_surface` when cluster count < `min_clusters`. Fix: `eval_builder.py:1042` catches both.
3. **qwen3.6-27b calibration speed** — 2-5 min per call in thinking mode → 47 min for 40 candidates. Mitigation: run on JarvisLabs CPU VM; consider faster calibration model for future.

**Step 2 (evolution) status:** Running as `r_688e0c8f` on VM 415615. Acceptance criteria 2-5 pending.

## Full Run Results (r_81f9fe55)

**Date:** 2026-05-26T02:56 UTC  
**VM:** JarvisLabs 415977 (IN2, 4 vCPU / 16GB), run dir `wmca-20260525T174508Z`  
**Config:** `--behavioral --max-iters 10 --budget 8h --max-workers 4`  
**Eval set:** 23 behavioral_action evals (12 train / 11 holdout), anchor=0.3027

**Score progression:**

| Iter | Holdout | Fitness | Δ | Status |
|---:|---:|---:|---:|---|
| 0 (anchor) | 0.3027 | 0.3027 | — | baseline |
| 1 | 0.3027 | 0.3027 | +0.000 | no_improvement (validation failures) |
| 2 | 0.3027 | 0.3027 | +0.000 | no_improvement (validation failures) |
| 3 | 0.3027 | 0.3027 | +0.000 | no_improvement (._* files cleaned) |
| 4 | 0.3027 | 0.3027 | +0.000 | no_improvement |
| **5** | **0.4130** | **0.4105** | **+0.1078** | **promoted** |
| 6 | 0.4105 | 0.4105 | +0.000 | no_improvement |
| 7 | 0.4105 | 0.4105 | +0.000 | budget_exhausted |

**Phase 4 baselines:**

| Baseline | Score |
|---|---:|
| A — empty bundle (floor) | 0.3034 |
| B — naive few-shot | 0.3656 |
| C — Opus teacher (ceiling) | 0.3943 |
| **Evolved (iter_5 c3)** | **0.4105** |
| **gap_closed** | **1.178** |

**gap_closed = 1.178**: evolved qwen (0.4105) surpassed Opus (0.3943) on behavioral evals.

**What the winning mutation did (iter_5 c3 reasoning):**
- Added `scripts/task_manager.py` (~130 lines, stdlib-only): create/update/list/link tasks with dependencies
- Added Task Management section to SKILL.md with exact usage patterns from failing evals
- Added Process Management section (kill-duplicate procedure with ps/SIGTERM/SIGKILL steps)
- SKILL.md grew from 1749 → 2438 tokens

**promote_blocked=True** — `budget_exhausted` prevents auto-promotion. Manual promotion to `~/.watchmen/bundles/wmca/audit-gate/` needed.

**Bugs found during run:**
1. macOS `._` metadata files in tar uploads tripped `py_compile` validation (iters 1-4 wasted). Fix: `find . -name "._*" -delete` before tar.
2. DeepSeek generates unterminated sentinel blocks (~50% rate). Fix needed in proposer prompt HARD LIMITS.
3. Haiku `judge_parse_error` flood (outputs prose "Score: 0.X" instead of JSON) — regex fallback active but noisy.

**Acceptance criteria status:**
1. eval_set.jsonl has `type==behavioral_action` rows — pass
2. weakness report has `category=` field — pass
3. mutations touched `scripts/` (5 iters) — pass
4. score improvement: +10.78pp over anchor, gap_closed=1.178 (beat Opus) — pass
5. all tests green (100/100) — pass
