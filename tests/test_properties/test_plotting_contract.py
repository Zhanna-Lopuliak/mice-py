"""Item 7: plotting contract (C7, A8).

Importing ``plotting`` has no side effects; every plot function returns its
matplotlib ``Figure``; ``xyplot`` groups points as R's ``xyplot.mids`` does;
the trace plot shows the standard deviation.
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

PANDAS_OPTIONS = ["display.max_columns", "mode.chained_assignment"]


@pytest.fixture(autouse=True)
def _preserve_pandas_options():
    """Restore pandas options after each test, should a re-import of ``plotting`` change them."""
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


def test_import_does_not_touch_numpy_global_rng():
    # the legacy global generator is what the test watches
    np.random.seed(987)  # noqa: NPY002
    before = np.random.get_state()  # noqa: NPY002
    restore = _reimport_plotting()
    try:
        after = np.random.get_state()  # noqa: NPY002
    finally:
        restore()
    np.testing.assert_array_equal(after[1], before[1])
    assert after[2:] == before[2:]


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
        pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)}) for _ in range(3)
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


def test_trace_plot_shows_standard_deviation():
    from plotting.diagnostics import plot_chain_stats

    chain_mean = {"y": np.arange(6.0).reshape(3, 2)}
    chain_var = {"y": (np.arange(1.0, 7.0) ** 2).reshape(3, 2)}
    plt.close("all")
    fig = plot_chain_stats(chain_mean, chain_var)
    assert isinstance(fig, Figure)
    spread_axes = fig.axes[1]
    lines = [line.get_ydata() for line in spread_axes.get_lines()]
    assert len(lines) == 2
    for chain, ydata in enumerate(lines):
        np.testing.assert_allclose(ydata, np.sqrt(chain_var["y"][:, chain]))


# xyplot. R 4.6.2 with mice 3.19.0 on this exact missingness layout (x missing in
# rows 1-3, y missing in rows 3, 6, 7; m = 2), reading the groups and the drawn
# (non-NA) points of each panel of the lattice object:
#   xyplot(imp, y ~ x | .imp)
#     .imp = 0       observed colour: rows 0, 4, 5, 8, 9, 10, 11; imputed colour: none
#     .imp = 1, 2    observed colour: rows 0, 1, 2, 4, 5, 8, 9, 10, 11; imputed colour: rows 3, 6, 7
#   xyplot(imp, y ~ x)
#     one panel      observed colour: 25 points (7 + 2 * 9); imputed colour: 6 (2 * 3)
# With its default na.groups, R colours by the missingness of y only. Rows with
# an observed y and an imputed x are therefore in the observed colour in the
# panels of the completed datasets, at the imputed x; the panel of the original
# data (.imp = 0) leaves them out because x is NA there.

OBSERVED, IMPUTED = "#0000ff", "#ff0000"


def _xy_inputs():
    n = 12
    x = np.arange(n, dtype=float)
    y = 2 * x + 1
    x_missing = np.isin(np.arange(n), [1, 2, 3])
    y_missing = np.isin(np.arange(n), [3, 6, 7])
    pattern = pd.DataFrame({"x": (~x_missing).astype(int), "y": (~y_missing).astype(int)})
    datasets = [
        pd.DataFrame({"x": x + 100 * x_missing * (k + 1), "y": y + 100 * y_missing * (k + 1)}) for k in range(2)
    ]
    return datasets, pattern, x_missing, y_missing


def _points(ax, colour):
    """Sorted (x, y) pairs drawn on ``ax`` in ``colour``."""
    rgb = to_rgba(colour)[:3]
    offsets = [
        c.get_offsets() for c in ax.collections if len(c.get_offsets()) and np.allclose(c.get_facecolor()[0][:3], rgb)
    ]
    return sorted(map(tuple, np.concatenate(offsets).tolist())) if offsets else []


def _rows(df, mask):
    return sorted(zip(df.loc[mask, "x"].tolist(), df.loc[mask, "y"].tolist()))


def test_xyplot_groups_points_as_r():
    """Observed panel: complete rows only. Panel k: coloured by whether y was imputed."""
    from plotting.diagnostics import xyplot

    datasets, pattern, x_missing, y_missing = _xy_inputs()
    plt.close("all")
    fig = xyplot(datasets, pattern, "x", "y", observed_color=OBSERVED, imputed_color=IMPUTED)
    assert isinstance(fig, Figure)
    panels = [ax for ax in fig.axes if ax.get_visible() and ax.collections]
    assert [ax.get_title() for ax in panels] == ["Observed", "Imp 1", "Imp 2"]

    # Only rows where both x and y are observed appear in the observed panel
    complete = ~x_missing & ~y_missing
    assert _points(panels[0], OBSERVED) == _rows(datasets[0], complete)
    assert len(_points(panels[0], OBSERVED)) == 7
    assert _points(panels[0], IMPUTED) == []

    for ax, df in zip(panels[1:], datasets):
        assert _points(ax, OBSERVED) == _rows(df, ~y_missing), ax.get_title()
        assert _points(ax, IMPUTED) == _rows(df, y_missing), ax.get_title()
        assert (len(_points(ax, OBSERVED)), len(_points(ax, IMPUTED))) == (9, 3)


def test_xyplot_merged_overlays_all_panels_as_r():
    from plotting.diagnostics import xyplot

    datasets, pattern, x_missing, y_missing = _xy_inputs()
    plt.close("all")
    fig = xyplot(datasets, pattern, "x", "y", merge_imputations=True, observed_color=OBSERVED, imputed_color=IMPUTED)
    assert isinstance(fig, Figure)
    (ax,) = fig.axes
    complete = ~x_missing & ~y_missing
    expected_observed = sorted(_rows(datasets[0], complete) + sum((_rows(df, ~y_missing) for df in datasets), []))
    expected_imputed = sorted(sum((_rows(df, y_missing) for df in datasets), []))
    assert _points(ax, OBSERVED) == expected_observed
    assert _points(ax, IMPUTED) == expected_imputed
    assert (len(expected_observed), len(expected_imputed)) == (25, 6)
