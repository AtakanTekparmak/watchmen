---
name: patch-then-verify
description: Use when applying a code fix to an existing codebase (bug fix, SWE-bench-style patch). Reproduce the failure first, patch minimally, then verify with the reproduction test and the full suite before declaring done.
version: 0.1.0
author: Hermes Agent (seed)
license: MIT
metadata:
  hermes:
    tags: [bug-fix, patch, verification, swe-bench, regression-test]
    category: software-development
    related_skills: [systematic-debugging, test-driven-development, read-before-write]
---

# Patch Then Verify

## Overview

A patch without a reproduction is a guess. A patch without a post-verification is unfinished work. Both fail silently on SWE-bench-style graders.

**Core principle:** Reproduce, patch, verify — in that order, every time.

## The Iron Law

```
NO PATCH WITHOUT A FAILING REPRODUCTION FIRST
NO "DONE" WITHOUT THE REPRODUCTION PASSING AND THE SUITE GREEN
```

## When to Use

- Fixing a reported bug
- Implementing a SWE-bench task (issue + hidden test)
- Any change expected to alter behavior of existing code

## Steps

### 1. Locate the relevant code

- Use grep/search for identifiers from the issue (function name, error string, class)
- Read the file containing the bug fully, and its direct callers
- Read the existing tests for that module — they show intended behavior

### 2. Reproduce the failure

Build the smallest thing that fails the way the issue describes:

- If a failing test already exists, run it and confirm it fails for the reported reason
- Otherwise, write one: either a proper test in the suite, or a short `repro.py`
- Run it. Confirm it fails. Note the exact error.

If you cannot reproduce, you do not understand the bug. Go back to step 1.

### 3. Patch minimally

- Change only what the root cause requires
- No drive-by refactors, renames, formatting, or "while I'm here" cleanups
- Keep the diff small — graders and reviewers compare against a reference patch
- Preserve public API and signatures unless the issue requires changing them

### 4. Re-read the patched region

After editing, read the file again around the change. Confirm:
- The edit landed where intended
- Indentation/syntax is valid
- No stray duplicate imports or dangling code

### 5. Verify

Run in this order, and do not skip:

```bash
# a. The reproduction — must now pass
pytest path/to/test_repro.py -v      # or: python repro.py

# b. The module's own tests — no local regressions
pytest path/to/module_tests/ -v

# c. The full suite — no distant regressions
pytest -q
```

If any step fails, diagnose before patching further. Do not pile fixes.

### 6. Inspect the final diff

```bash
git diff
```

Read it end-to-end. Ask: does every hunk serve the fix? Delete anything that does not.

## Red Flags — Stop

- Patching before reproducing
- "The fix is obvious, I'll skip the repro"
- Declaring success after only the repro passes (suite not run)
- Editing files you have not read in this session (see `read-before-write`)
- Diff contains unrelated formatting or whitespace churn
- Suppressing a test to make the suite green

## Integration

- Use `systematic-debugging` during step 1–2 to find the real root cause.
- Use `read-before-write` to guarantee you have context for every edit.
- If three patch attempts fail, invoke `stop-and-replan`.
