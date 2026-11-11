"""Fixtures for the plotting tests (the Agg backend is set in tests/conftest.py)."""

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")
