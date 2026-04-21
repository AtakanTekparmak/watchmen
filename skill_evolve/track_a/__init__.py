"""skill_evolve.track_a — autoreason-style A/B/AB evolution loop for
Hermes-agent skill folders.

The loop runs an incumbent A against a mutated proposal B and a synthesizer
AB, greedy-accepts on composite score from :mod:`skill_evolve.evaluator`,
and converges after two consecutive "A wins" passes.

Public entry point: :mod:`skill_evolve.track_a.runner` (CLI).
"""

__all__ = [
    "ops",
    "validate",
    "prompts",
    "llm",
    "folder",
    "runner",
]
