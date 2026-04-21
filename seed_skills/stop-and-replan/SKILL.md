---
name: stop-and-replan
description: Use when the same approach has failed three or more times, when fixes keep revealing new problems elsewhere, or when you notice yourself flailing. Halt, enumerate what was tried, and pick a different angle before acting again.
version: 0.1.0
author: Hermes Agent (seed)
license: MIT
metadata:
  hermes:
    tags: [planning, anti-flailing, meta, recovery]
    category: software-development
    related_skills: [systematic-debugging, systematic-shell-debugging, patch-then-verify]
---

# Stop and Replan

## Overview

Repeated failure on the same approach is a signal that the model of the problem is wrong, not that one more tweak will work. Attempt four is almost always worse than attempt one.

**Core principle:** Three strikes, stop. Replan before acting.

## The Iron Law

```
AFTER 3 FAILED ATTEMPTS ON THE SAME APPROACH: STOP AND REPLAN
NO ATTEMPT #4 WITHOUT A DIFFERENT ANGLE
```

## Triggers

Invoke this skill when any of the following is true:

- Same command/patch has been tried 3+ times with no progress
- Each fix surfaces a new error in a different place
- You are guessing flags, paths, or names rather than checking
- The trajectory shows the same file edited 4+ times in a row
- You notice yourself writing "let me try once more" or "just one more"
- Tests pass locally but the grader keeps failing, twice in a row

## The Replan Procedure

### 1. Halt all actions

No more edits, commands, or tool calls until steps 2–4 are done.

### 2. Enumerate what was tried

Write a short list, honestly:

```
Attempt 1: <what I did> -> <what happened>
Attempt 2: <what I did> -> <what happened>
Attempt 3: <what I did> -> <what happened>
```

Ambiguous outcomes count as failures.

### 3. Extract the shared assumption

All three attempts share at least one assumption. Name it explicitly:

- "I assumed the bug was in file X."
- "I assumed this API takes a list."
- "I assumed the test runner uses pytest."

That assumption is the most likely wrong thing.

### 4. Propose a different angle

Pick one, and commit to it before acting:

- **Different layer** — if patching the caller failed, inspect the callee (or vice versa)
- **Different tool** — read the code instead of running it, or run it instead of reading
- **Different scope** — shrink the repro to the smallest failing case
- **Ask the environment** — `ask-the-environment` to check what is actually true (versions, paths, env vars)
- **Ask the user** — if assumptions cannot be verified, surface the ambiguity

### 5. Write down the new plan

Two to five concrete steps. Execute them. If this plan also fails three times, replan again — do not silently resume the old approach.

## Red Flags — You Are Flailing

- Editing the same 1–2 files for the fifth time
- Trying flag combinations without reading `--help`
- Commenting out tests to make the suite pass
- Reverting and reapplying the same change
- "Let me just try..." appearing in your reasoning

## Anti-Rationalizations

| Excuse | Reality |
|--------|---------|
| "I'm almost there" | Attempt 3 said that too. |
| "One more tweak" | The tweak is not the problem; the approach is. |
| "Replanning wastes time" | Four more failed attempts waste more time. |
| "I understand it now" | If you did, it would already work. |

## Exit Criterion

Resume normal work only after you can state, in one sentence, what the previous approach got wrong and why the new approach will not hit the same wall.
