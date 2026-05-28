"""Behavioral evaluation adapter — score skill bundles via judge LLM.

Mirrors the daycare verifier flow (``daycare.verifier.score_single``) but
returns a ``skill_evolve.evaluator.EvalResult`` so the rest of the
evolution pipeline (Track B controller / iteration / archive) can
consume the same shape regardless of which eval-source was selected.
"""
