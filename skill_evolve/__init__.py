"""skill_evolve — shared substrate for evolving Hermes-agent skill folders.

Three downstream tracks (autoreason loop, openevolve fork, hybrid) all sit on
top of this package. The public entry points are:

    skill_evolve.sandbox.HermesSandbox      — per-eval HERMES_HOME isolation
    skill_evolve.benchmark.load_subset()    — load curated benchmark tasks
    skill_evolve.evaluator.evaluate(...)    — score a skills folder
"""

__all__ = ["sandbox", "benchmark", "evaluator"]
