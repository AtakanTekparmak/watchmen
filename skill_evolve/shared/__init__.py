"""Shared utilities for skill_evolve (daycare port).

Hosts the sentinel-block patch parser, bundle-ops helpers (apply/validate/
hash) and the leak-scanner — all moved out of daycare so the consolidated
evolution program owns the surface.
"""

from skill_evolve.shared import leak_scanner as leak_scanner  # noqa: F401  re-export

__all__ = ["leak_scanner"]
