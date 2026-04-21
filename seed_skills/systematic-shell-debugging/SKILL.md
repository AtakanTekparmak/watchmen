---
name: systematic-shell-debugging
description: Use when a shell command fails, exits non-zero, hangs, or produces unexpected output. Work the failure methodically — read the error, inspect state, form one hypothesis, test it — instead of re-running the same command or guessing flags.
version: 0.1.0
author: Hermes Agent (seed)
license: MIT
metadata:
  hermes:
    tags: [shell, terminal, debugging, troubleshooting, terminal-bench]
    category: software-development
    related_skills: [systematic-debugging, stop-and-replan, ask-the-environment]
---

# Systematic Shell Debugging

## Overview

Shell failures are cheap to diagnose and expensive to guess at. Blind retries and flag-shuffling waste turns and bury the real signal.

**Core principle:** A failing command is evidence. Read it before touching anything.

## The Iron Law

```
NO RETRY WITHOUT READING THE ERROR AND INSPECTING STATE
```

## When to Use

- Any non-zero exit code
- Command hangs or times out
- Output is empty when it should not be
- Unexpected stdout/stderr content
- "command not found", "permission denied", "no such file"
- Build, install, or test commands fail

## The Loop

### 1. Read the error completely

- Read the full stderr, not just the last line
- Note exit code, file paths, line numbers, missing names
- If output is long, scan for the first error — later ones are usually downstream

### 2. Inspect the relevant state

Pick the checks that match the error class. Do not run all of them.

```bash
pwd                      # wrong directory?
ls -la <path>            # file exists? permissions? symlink?
echo $PATH               # binary resolvable?
which <cmd>              # which version is on PATH?
env | grep <VAR>         # env var set?
cat <config>             # config contents correct?
git status               # dirty tree? wrong branch?
```

### 3. Form one hypothesis

Write it as a sentence: "The command failed because X." Be specific. If you cannot name X, gather more evidence — do not guess.

### 4. Test the hypothesis minimally

Change one thing. Re-run the command. If it still fails, the hypothesis was wrong — discard it, do not stack another fix on top.

### 5. Narrow the surface

If the command is a pipeline or script, run it in pieces:

```bash
cmd1                     # does stage 1 succeed alone?
cmd1 | head              # is stage 1's output shaped right?
bash -x script.sh        # trace script execution
<cmd> --help             # confirm flag exists and means what you think
```

## Red Flags — Stop

- Re-running the exact same command expecting a different result
- Adding flags you have not checked in `--help`
- `sudo`-ing past a permission error without understanding why
- Deleting files/dirs to "reset" before understanding what broke
- Skipping stderr because stdout "looked fine"

If you catch any of these, go back to step 1.

## Escalation

Three failed hypotheses on the same command means the model of the system is wrong. Invoke `stop-and-replan`: list what was tried, what the evidence actually says, and pick a different angle (different tool, different layer, ask the user).
