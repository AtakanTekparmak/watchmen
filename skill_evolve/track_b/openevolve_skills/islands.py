"""Per-island seeding biases for Track B.

The spec calls for 3 islands, each starting from ``seed_skills/`` with a
different bias:

* **Island 0** — as-is. The baseline seed.
* **Island 1** — drop the two most generic skills (``read-before-write``
  and ``ask-the-environment``). Biases evolution toward a leaner base;
  the LLM has to rediscover or replace them if they're actually useful.
* **Island 2** — add an empty ``domain-specific-helper/`` placeholder.
  Gives the LLM an obvious slot to fill with task-specific guidance.

Rationale: these three biases sit at different points on the
"generic ↔ specific" axis, which is exactly the ``avg_specificity``
feature dimension. Each island thus has a natural starting niche on the
MAP-Elites grid, which (a) reduces wasted iterations re-exploring the
same cell and (b) encourages the populations to diverge before
migration kicks in.
"""

from __future__ import annotations

from typing import List

from .folder_artifact import FolderArtifact


_GENERIC_SKILLS_TO_DROP = ("read-before-write", "ask-the-environment")

_PLACEHOLDER_SKILL = """\
---
name: domain-specific-helper
description: Placeholder slot for a task-specific skill. Replace with concrete triggers, boundaries, and examples for the dominant failure mode on the current benchmark.
version: 0.0.1
author: Track-B island-2 seed
---

# Domain-specific helper (placeholder)

This skill is an empty slot. The evolution loop is expected to replace
its body with concrete, benchmark-specific guidance — e.g. common
terminal-bench shell patterns or SWE-bench reproduction recipes —
based on the failures surfaced in the evaluation feedback.

## When to use

(to be filled in by the evolution loop)

## How to use

(to be filled in by the evolution loop)
"""


def seed_variants(base: FolderArtifact, *, num_islands: int) -> List[FolderArtifact]:
    """Return ``num_islands`` biased copies of ``base``.

    For ``num_islands > 3`` we cycle through the biases.
    """
    builders = [_bias_asis, _bias_drop_generic, _bias_add_placeholder]
    return [builders[i % len(builders)](base) for i in range(num_islands)]


def _bias_asis(base: FolderArtifact) -> FolderArtifact:
    return FolderArtifact(files=dict(base.files))


def _bias_drop_generic(base: FolderArtifact) -> FolderArtifact:
    new = {p: c for p, c in base.files.items()
           if p.split("/", 1)[0] not in _GENERIC_SKILLS_TO_DROP}
    # If dropping would leave zero skills (shouldn't happen with our seed,
    # but defensively): fall back to the original.
    art = FolderArtifact(files=new)
    if art.num_skills() == 0:
        return _bias_asis(base)
    return art


def _bias_add_placeholder(base: FolderArtifact) -> FolderArtifact:
    new = dict(base.files)
    new["domain-specific-helper/SKILL.md"] = _PLACEHOLDER_SKILL
    return FolderArtifact(files=new)
