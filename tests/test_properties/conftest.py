"""Fixtures for the L4 invariant tests."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _pinned_global_numpy_rng():
    """Seed NumPy's global RNG for each test and restore it afterwards.

    Today ``midas`` draws from the global RNG (finding A5), so without this its
    results, and the strict xfail markers that depend on them, would vary with
    test order. The seed tests still detect A5: they call ``impute`` twice
    inside one test, and the global state moves between the calls.
    """
    state = np.random.get_state()
    np.random.seed(12345)
    yield
    np.random.set_state(state)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")
