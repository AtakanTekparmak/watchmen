"""Pytest configuration for skill_evolve tests.

Registers the ``docker`` marker so heavy integration tests are opt-in:

    pytest                          # unit tests only (no docker)
    pytest -m docker                # only docker-marked tests
    pytest -m "not docker"          # explicit exclude (default behaviour)
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "docker: integration test that needs a running docker daemon "
        "(skipped by default; run with `pytest -m docker`)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("-m"):
        # User explicitly chose markers — don't second-guess.
        return
    skip_docker = pytest.mark.skip(reason="docker tests skipped by default; "
                                          "run with `pytest -m docker`")
    for item in items:
        if "docker" in item.keywords:
            item.add_marker(skip_docker)
