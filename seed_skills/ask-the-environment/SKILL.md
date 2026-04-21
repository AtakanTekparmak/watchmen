---
name: ask-the-environment
description: Use before making assumptions about paths, versions, installed tools, environment variables, git state, or file contents. Prefer a cheap read-only command over guessing.
version: 0.1.0
author: Hermes Agent (seed)
license: MIT
metadata:
  hermes:
    tags: [shell, inspection, grounding, terminal-bench, swe-bench]
    category: software-development
    related_skills: [systematic-shell-debugging, read-before-write, stop-and-replan]
---

# Ask the Environment

## Overview

Most wrong actions stem from wrong assumptions about the environment: which Python, which branch, which working directory, whether a file exists, what a config contains. Read-only inspection is cheap; acting on a bad assumption is expensive.

**Core principle:** When in doubt, run a read-only command. The environment will answer.

## The Iron Law

```
NO ACTION ON AN UNVERIFIED ASSUMPTION ABOUT THE ENVIRONMENT
```

If you cannot name the source of your belief ("I saw it in the output of X"), check it.

## When to Use

Before you:
- Run a build, test, or install command
- Call a binary by name
- Reference an env var
- Edit a file you think exists
- Commit, push, or branch
- Report "done"

## Inspection Menu

Pick the minimal command that answers the specific question. Do not run them all.

### Filesystem and cwd
```bash
pwd
ls -la
ls -la <dir>
stat <file>
find . -maxdepth 2 -name '<pattern>'
```

### Tooling and versions
```bash
which python python3 pip pytest node npm
python --version
pip show <pkg>
node --version
```

### Environment
```bash
env | grep -i <prefix>
echo $PATH
printenv <VAR>
```

### Repo state
```bash
git status
git branch --show-current
git log --oneline -5
git remote -v
git diff --stat
```

### Project shape
```bash
cat pyproject.toml         # or setup.py, package.json, Cargo.toml
ls tests/ test/ __tests__/ 2>/dev/null
head -n 50 README.md
```

### Runtime/process
```bash
ps -ef | grep <name>
lsof -i :<port>
```

## Usage Pattern

1. Form the specific question: "Is pytest installed?" not "Is the env set up?"
2. Pick the one command that answers it
3. Run it, read the output
4. Act on the observed reality, not the expected one

## Red Flags — Stop and Inspect

- "I'll assume the tests are under `tests/`"
- "The project probably uses pytest"
- "I think I'm on the main branch"
- "The file should already exist"
- Calling `python` without checking which Python
- Referring to `$FOO` without confirming it is set

Each of these should be replaced with one read-only command before proceeding.

## Boundaries

- Stay read-only during inspection. No installs, no writes, no network mutation.
- Do not chain ten inspection commands when one will do — that is its own flailing.
- Cache the answer in your reasoning; do not re-check within the same turn unless state might have changed.
