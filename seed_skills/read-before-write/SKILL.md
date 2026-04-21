---
name: read-before-write
description: Use before any file edit, create, or delete. Never modify a file whose current contents you have not read in this session. Re-read after the edit to confirm it landed as intended.
version: 0.1.0
author: Hermes Agent (seed)
license: MIT
metadata:
  hermes:
    tags: [file-operations, editing, safety, context]
    category: software-development
    related_skills: [patch-then-verify, ask-the-environment]
---

# Read Before Write

## Overview

Editing a file you have not read is guessing. The file on disk may differ from what you remember, what the issue describes, or what training data suggests. Unread edits corrupt files, duplicate code, and break imports.

**Core principle:** The filesystem is the source of truth. Read it first.

## The Iron Law

```
NO WRITE TO A PATH YOU HAVE NOT READ THIS SESSION
NO "DONE" WITHOUT RE-READING THE CHANGED REGION
```

## When to Use

Every time you are about to:
- Edit an existing file
- Overwrite a file
- Delete a file
- Apply a patch or diff

## Steps

### 1. Read the target

- Read the full file if it is under ~400 lines
- For larger files, read the region around your edit plus imports/top-of-file and any referenced helpers
- Note current indentation style (tabs vs spaces), quote style, and import order

### 2. Confirm the edit anchor

The exact text you plan to replace must actually exist in the file as you expect. If your edit tool requires a unique match, verify uniqueness with a search before editing.

### 3. Make the edit

- Preserve existing style (indent width, quotes, trailing newline)
- Keep the diff minimal — do not reformat untouched lines
- One logical change per edit when possible

### 4. Re-read the changed region

After the edit, read the file again around the change. Confirm:
- The new text is present and correct
- Nothing above or below got mangled
- Indentation matches the surrounding code
- No duplicate imports, functions, or braces

### 5. If creating a new file

- First check the file does not already exist (`ls` the parent dir)
- Confirm the parent directory exists
- After writing, read it back to verify contents

## Red Flags — Stop

- "I remember this file from earlier" — earlier is not this session; re-read
- Writing based on what the issue/docstring says the file contains
- Skipping the re-read because "the edit tool would have errored"
- Editing a file you only saw via `grep` snippets
- Batch-editing many files without reading each

## Rationalizations

| Excuse | Reality |
|--------|---------|
| "File is tiny, obvious what it contains" | Tiny files get edited by other agents/tools between your turns. Read it. |
| "I just read it two steps ago" | Two steps ago you also wrote to it. Read the current state. |
| "Edit tool checks uniqueness" | Uniqueness check does not catch stale context around the anchor. |
| "Re-reading wastes tokens" | One corrupted file costs more tokens to recover from. |
