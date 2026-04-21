"""Benchmark subset for evaluating skill folders.

The public surface is :func:`load_subset`, which reads
``manifest.json`` next to this file and hydrates each entry with the
upstream prompt (TBLite via HF; SWE-bench Verified via HF). See
``load.py`` for the implementation and ``manifest.json`` for the
curated task list.
"""

from .load import load_subset, Task  # noqa: F401

__all__ = ["load_subset", "Task"]
