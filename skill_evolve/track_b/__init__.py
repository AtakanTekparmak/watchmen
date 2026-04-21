"""Track B: fork of OpenEvolve that evolves a *folder* of skill files.

Layout:
    openevolve_skills/      — forked / modified modules (FolderArtifact,
                              Program, ProgramDatabase, Evaluator wrapper,
                              PromptSampler, iteration loop).
    run.py                  — CLI entry point (``python -m skill_evolve.track_b.run``).
    tests/                  — unit + smoke tests.

Design notes (short — see BLOCKERS.md if design changes):

* What we *copied* from upstream openevolve: the conceptual dataclass for
  ``Program`` (id / parent / generation / metrics / metadata), the
  ``EvaluationResult`` shape (metrics + artifacts side-channel), the
  MAP-Elites-per-island + migration-every-N-generations idea.
* What we *did not* copy: the process-pool controller, the novelty-judge +
  embedding stack, the diff-pattern code_utils, the TemplateManager.
  Rationale: Track B's fitness signal is a heavy Docker-based eval, so we
  run single-threaded by design; the LLM ensemble / process-pool layer is
  orthogonal to the folder-artifact data-model work.

Neither ``openevolve/`` nor the other ``skill_evolve/`` modules are modified
— everything Track B needs lives in ``skill_evolve/track_b/``.
"""
