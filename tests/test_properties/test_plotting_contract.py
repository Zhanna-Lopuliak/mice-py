"""Item 7: plotting contract (C7, A8).

Planned: importing ``plotting`` has no side effects; every plot function
returns its matplotlib ``Figure``; ``xyplot`` does not show rows with an
imputed x as observed; the trace plot shows the standard deviation.
"""

import importlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

pytestmark = pytest.mark.plotting

C7_IMPORT = "C7: plotting.diagnostics seeds NumPy and sets pandas options at import"
C7_RETURN = "C7: plot functions return None instead of the Figure"
PANDAS_OPTIONS = ["display.max_columns", "mode.chained_assignment"]


@pytest.fixture(autouse=True)
def _preserve_pandas_options():
    """Re-importing ``plotting`` today changes pandas options; undo that after each test."""
    saved = {n: pd.get_option(n) for n in PANDAS_OPTIONS}
    yield
    for n, v in saved.items():
        pd.set_option(n, v)


def _reimport_plotting():
    """Import ``plotting.diagnostics`` afresh; return the restore callback."""
    saved = {k: v for k, v in sys.modules.items() if k == "plotting" or k.startswith("plotting.")}
    for k in saved:
        del sys.modules[k]

    def restore():
        for k in [k for k in sys.modules if k == "plotting" or k.startswith("plotting.")]:
            del sys.modules[k]
        sys.modules.update(saved)

    try:
        importlib.import_module("plotting.diagnostics")
    except BaseException:
        restore()
        raise
    return restore


@pytest.mark.xfail(strict=True, reason=C7_IMPORT + " (np.random.seed(42))")
def test_import_does_not_touch_numpy_global_rng():
    np.random.seed(987)
    before = np.random.get_state()
    restore = _reimport_plotting()
    try:
        after = np.random.get_state()
    finally:
        restore()
    np.testing.assert_array_equal(after[1], before[1])
    assert after[2:] == before[2:]


@pytest.mark.xfail(strict=True, reason=C7_IMPORT + " (pandas display / chained_assignment)")
def test_import_does_not_change_pandas_options():
    pd.set_option("display.max_columns", 7)
    pd.set_option("mode.chained_assignment", "warn")
    expected = {n: pd.get_option(n) for n in PANDAS_OPTIONS}
    _reimport_plotting()()
    after = {n: pd.get_option(n) for n in PANDAS_OPTIONS}
    assert after == expected


# ---------------------------------------------------------------------------
# Every plot function returns a Figure
# ---------------------------------------------------------------------------

def _plot_inputs():
    rng = np.random.default_rng(0)
    n = 30
    pattern = pd.DataFrame(
        {
            "a": np.ones(n, dtype=int),
            "b": (rng.random(n) > 0.25).astype(int),
            "c": (rng.random(n) > 0.2).astype(int),
        }
    )
    pattern.loc[0, "b"] = 0
    pattern.loc[1, "c"] = 0
    datasets = [
        pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
        for _ in range(3)
    ]
    chain_mean = {"b": rng.normal(size=(4, 3)), "c": rng.normal(size=(4, 3))}
    chain_var = {"b": rng.uniform(0.5, 2, size=(4, 3)), "c": rng.uniform(0.5, 2, size=(4, 3))}
    return datasets, pattern, chain_mean, chain_var


def _call(name):
    from plotting import diagnostics, utils

    datasets, pattern, chain_mean, chain_var = _plot_inputs()
    calls = {
        "stripplot": lambda: diagnostics.stripplot(datasets, pattern, columns=["b", "c"]),
        "bwplot": lambda: diagnostics.bwplot(datasets, pattern, columns=["b", "c"]),
        "densityplot": lambda: diagnostics.densityplot(datasets, pattern, columns=["b", "c"]),
        "densityplot_split": lambda: diagnostics.densityplot_split(datasets, pattern, "b"),
        "xyplot": lambda: diagnostics.xyplot(datasets, pattern, "a", "b"),
        "plot_chain_stats": lambda: diagnostics.plot_chain_stats(chain_mean, chain_var),
        "plot_missing_data_pattern": lambda: utils.plot_missing_data_pattern(
            utils.md_pattern_like(datasets[0].mask(pattern == 0))
        ),
    }
    return calls[name]()


PLOT_FUNCTIONS = [
    "stripplot",
    "bwplot",
    "densityplot",
    "densityplot_split",
    "xyplot",
    "plot_chain_stats",
    "plot_missing_data_pattern",
]


@pytest.mark.xfail(strict=True, reason=C7_RETURN)
@pytest.mark.parametrize("name", PLOT_FUNCTIONS)
def test_plot_function_returns_figure(name):
    assert isinstance(_call(name), Figure)


def test_saving_closes_the_figure(tmp_path):
    from plotting.diagnostics import plot_chain_stats

    _, _, chain_mean, chain_var = _plot_inputs()
    plt.close("all")
    plot_chain_stats(chain_mean, chain_var, save_path=str(tmp_path / "trace.png"))
    assert (tmp_path / "trace.png").exists()
    assert plt.get_fignums() == []


# ---------------------------------------------------------------------------
# A8
# ---------------------------------------------------------------------------

def _figure(returned):
    """The figure under test, independent of C7 (today the functions return None)."""
    return returned if isinstance(returned, Figure) else plt.gcf()


@pytest.mark.xfail(strict=True, reason="A8: trace plot shows the variance, R shows the standard deviation")
def test_trace_plot_shows_standard_deviation():
    from plotting.diagnostics import plot_chain_stats

    chain_mean = {"y": np.arange(6.0).reshape(3, 2)}
    chain_var = {"y": (np.arange(1.0, 7.0) ** 2).reshape(3, 2)}
    plt.close("all")
    fig = _figure(plot_chain_stats(chain_mean, chain_var))
    spread_axes = fig.axes[1]
    lines = [line.get_ydata() for line in spread_axes.get_lines()]
    assert len(lines) == 2
    for chain, ydata in enumerate(lines):
        np.testing.assert_allclose(ydata, np.sqrt(chain_var["y"][:, chain]))


@pytest.mark.xfail(strict=True, reason="A8: xyplot draws rows with an imputed x in the observed colour")
def test_xyplot_does_not_show_imputed_x_as_observed():
    """Only rows where both x and y are observed may appear in the observed colour."""
    from plotting.diagnostics import xyplot

    n = 12
    x = np.arange(n, dtype=float)
    y = 2 * x + 1
    x_missing = np.isin(np.arange(n), [1, 2, 3])
    y_missing = np.isin(np.arange(n), [3, 6, 7])
    pattern = pd.DataFrame({"x": (~x_missing).astype(int), "y": (~y_missing).astype(int)})
    datasets = [
        pd.DataFrame({"x": x + 100 * x_missing * (k + 1), "y": y + 100 * y_missing * (k + 1)})
        for k in range(2)
    ]
    observed_rgb = to_rgba("#0000ff")[:3]
    plt.close("all")
    fig = _figure(xyplot(datasets, pattern, "x", "y", observed_color="#0000ff", imputed_color="#ff0000"))
    both_observed = int((~x_missing & ~y_missing).sum())
    panels = [ax for ax in fig.axes if ax.get_visible() and ax.collections]
    assert panels
    for ax in panels:
        n_observed_colour = sum(
            len(c.get_offsets())
            for c in ax.collections
            if len(c.get_offsets()) and np.allclose(c.get_facecolor()[0][:3], observed_rgb)
        )
        assert n_observed_colour == both_observed, ax.get_title()
