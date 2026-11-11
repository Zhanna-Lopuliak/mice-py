"""
Tests for diagnostic plotting functions.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from pathlib import Path

from plotting.diagnostics import stripplot, bwplot, densityplot, densityplot_split, xyplot, plot_chain_stats

N_IMPUTATIONS = 3
N_ROWS = 50


def _n_points(ax):
    """Number of scatter points drawn on ``ax``."""
    return sum(len(c.get_offsets()) for c in ax.collections)


def _points_in_colour(ax, colour):
    """Number of scatter points drawn on ``ax`` in ``colour`` (one collection may mix colours)."""
    rgb = np.array(to_rgb(colour))
    n = 0
    for c in ax.collections:
        n_points = len(c.get_offsets())
        if not n_points:
            continue
        faces = np.broadcast_to(c.get_facecolors()[:, :3], (n_points, 3))
        n += int(np.isclose(faces, rgb).all(axis=1).sum())
    return n


def _tick_labels(ax):
    return [t.get_text() for t in ax.get_xticklabels()]


def _assert_saved_and_closed(fig, save_path, tmp_dir):
    """The file exists, the figure is closed, and the returned figure can still be saved."""
    assert isinstance(fig, Figure)
    assert save_path.exists()
    assert fig.number not in plt.get_fignums()
    again = tmp_dir / "again.png"
    fig.savefig(again)
    assert again.exists()


class TestStripplot:
    """Test stripplot functionality."""

    def test_stripplot_basic(self, imputed_datasets, missing_pattern, temp_dir):
        """Test basic stripplot creation."""
        fig = stripplot(
            imputed_datasets=imputed_datasets, missing_pattern=missing_pattern, columns=["bmi"], merge_imputations=False
        )
        assert isinstance(fig, Figure)
        (ax,) = fig.axes
        assert ax.get_title() == "bmi"
        assert _tick_labels(ax) == ["Observed", "Imp 1", "Imp 2", "Imp 3"]
        # Observed values once, then all rows of each completed dataset
        n_observed = int((missing_pattern["bmi"] == 1).sum())
        assert _n_points(ax) == n_observed + N_IMPUTATIONS * N_ROWS
        plt.close("all")

    def test_stripplot_merged(self, imputed_datasets, missing_pattern):
        """Test stripplot with merged imputations."""
        fig = stripplot(
            imputed_datasets=imputed_datasets, missing_pattern=missing_pattern, columns=["bmi"], merge_imputations=True
        )
        assert isinstance(fig, Figure)
        (ax,) = fig.axes
        assert _tick_labels(ax) == ["Observed Only", "Observed + Imputed"]
        # Observed values twice, imputed values of all imputations once
        n_observed = int((missing_pattern["bmi"] == 1).sum())
        n_missing = N_ROWS - n_observed
        assert _n_points(ax) == 2 * n_observed + N_IMPUTATIONS * n_missing
        plt.close("all")

    def test_stripplot_save(self, imputed_datasets, missing_pattern, temp_dir):
        """Test stripplot saving to file."""
        save_path = temp_dir / "stripplot_test.png"

        fig = stripplot(
            imputed_datasets=imputed_datasets,
            missing_pattern=missing_pattern,
            columns=["bmi"],
            save_path=str(save_path),
        )

        _assert_saved_and_closed(fig, save_path, temp_dir)
        plt.close("all")

    def test_stripplot_custom_colors(self, imputed_datasets, missing_pattern):
        """Test stripplot with custom colors."""
        fig = stripplot(
            imputed_datasets=imputed_datasets,
            missing_pattern=missing_pattern,
            columns=["bmi"],
            observed_color="green",
            imputed_color="orange",
        )
        (ax,) = fig.axes
        n_observed = int((missing_pattern["bmi"] == 1).sum())
        assert _points_in_colour(ax, "green") == (1 + N_IMPUTATIONS) * n_observed
        assert _points_in_colour(ax, "orange") == N_IMPUTATIONS * (N_ROWS - n_observed)
        plt.close("all")

    def test_stripplot_no_missing_columns(self, imputed_datasets, missing_pattern):
        """Test stripplot behavior when no columns have missing values."""
        no_missing_pattern = missing_pattern.copy()
        no_missing_pattern[:] = 1

        with pytest.warns(UserWarning, match="No columns with missing values"):
            result = stripplot(imputed_datasets=imputed_datasets, missing_pattern=no_missing_pattern)
        assert result is None
        plt.close("all")

    def test_stripplot_many_columns_wraps_and_stays_bounded(self):
        """Ten columns: rows of at most four panels, figure at most 20 inches a side."""
        rng = np.random.default_rng(0)
        columns = [f"v{i}" for i in range(10)]
        pattern = pd.DataFrame((rng.random((30, 10)) > 0.3).astype(int), columns=columns)
        datasets = [pd.DataFrame(rng.normal(size=(30, 10)), columns=columns) for _ in range(2)]
        fig = stripplot(datasets, pattern)
        assert [ax.get_title() for ax in fig.axes] == columns
        assert len({round(ax.get_position().x0, 3) for ax in fig.axes}) == 4
        assert max(fig.get_size_inches()) <= 20
        plt.close("all")


class TestBwplot:
    """Test box-and-whisker plot functionality."""

    def test_bwplot_basic(self, imputed_datasets, missing_pattern):
        """Test basic bwplot creation."""
        fig = bwplot(imputed_datasets=imputed_datasets, missing_pattern=missing_pattern, columns=["bmi"])
        assert isinstance(fig, Figure)
        (ax,) = fig.axes
        # One box for the observed values and one per imputation, no legend
        assert len(ax.patches) == 1 + N_IMPUTATIONS
        assert _tick_labels(ax) == ["Observed", "Imp 1", "Imp 2", "Imp 3"]
        assert ax.get_legend() is None
        plt.close("all")

    def test_bwplot_save(self, imputed_datasets, missing_pattern, temp_dir):
        """Test bwplot saving to file."""
        save_path = temp_dir / "bwplot_test.png"

        fig = bwplot(
            imputed_datasets=imputed_datasets,
            missing_pattern=missing_pattern,
            columns=["bmi"],
            save_path=str(save_path),
        )

        _assert_saved_and_closed(fig, save_path, temp_dir)
        plt.close("all")

    @pytest.mark.parametrize("merge", [False, True])
    def test_bwplot_shows_outliers_and_no_legend(self, imputed_datasets, missing_pattern, merge):
        """Points beyond the whiskers are drawn, as R does; no panel has a legend."""
        datasets = [df.copy() for df in imputed_datasets]
        observed_row = missing_pattern.index[missing_pattern["bmi"] == 1][0]
        for df in datasets:
            df.loc[observed_row, "bmi"] = 1000.0
        fig = bwplot(datasets, missing_pattern, columns=["bmi", "cholesterol"], merge_imputations=merge)
        assert len(fig.axes) == 2
        assert all(ax.get_legend() is None for ax in fig.axes)
        fliers = [
            y
            for line in fig.axes[0].lines
            if line.get_marker() == "o" and line.get_linestyle() == "None"
            for y in line.get_ydata()
        ]
        assert 1000.0 in fliers
        plt.close("all")


class TestDensityplot:
    """Test density plot functionality."""

    def test_densityplot_basic(self, imputed_datasets, missing_pattern):
        """Test basic densityplot creation."""
        fig = densityplot(imputed_datasets=imputed_datasets, missing_pattern=missing_pattern, columns=["bmi"])
        assert isinstance(fig, Figure)
        (ax,) = fig.axes
        # One curve for the observed values and one per imputation
        lines = ax.get_lines()
        assert len(lines) == 1 + N_IMPUTATIONS
        assert to_rgb(lines[0].get_color()) == to_rgb("blue")
        assert [to_rgb(line.get_color()) for line in lines[1:]] == [to_rgb("red")] * N_IMPUTATIONS
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Observed", "Imputed"]
        plt.close("all")

    def test_densityplot_merged(self, imputed_datasets, missing_pattern):
        """All imputations are overlaid in one panel per column; only the first panel has a legend."""
        fig = densityplot(
            imputed_datasets=imputed_datasets, missing_pattern=missing_pattern, columns=["bmi", "cholesterol"]
        )
        assert [ax.get_title() for ax in fig.axes] == ["bmi", "cholesterol"]
        for ax in fig.axes:
            assert len(ax.get_lines()) == 1 + N_IMPUTATIONS
        assert fig.axes[0].get_legend() is not None
        assert fig.axes[1].get_legend() is None
        plt.close("all")

    def test_densityplot_split(self, imputed_datasets, missing_pattern):
        """Test split density plot."""
        fig = densityplot_split(imputed_datasets=imputed_datasets, missing_pattern=missing_pattern, column="bmi")
        assert isinstance(fig, Figure)
        assert [ax.get_title() for ax in fig.axes] == ["Observed", "Imp 1", "Imp 2", "Imp 3"]
        assert all(len(ax.get_lines()) == 1 for ax in fig.axes)
        # Common axes, so the densities can be compared
        first = fig.axes[0]
        for ax in fig.axes[1:]:
            assert first.get_shared_x_axes().joined(first, ax)
            assert first.get_shared_y_axes().joined(first, ax)
        plt.close("all")

    def test_densityplot_split_save(self, imputed_datasets, missing_pattern, temp_dir):
        """Test split density plot saving."""
        save_path = temp_dir / "densityplot_split_test.png"

        fig = densityplot_split(
            imputed_datasets=imputed_datasets, missing_pattern=missing_pattern, column="bmi", save_path=str(save_path)
        )

        _assert_saved_and_closed(fig, save_path, temp_dir)
        plt.close("all")

    def test_densityplot_zero_variance(self, imputed_datasets, missing_pattern):
        """Imputed values without variance give a vertical line instead of a density."""
        datasets = [df.copy() for df in imputed_datasets]
        missing = missing_pattern["bmi"] == 0
        datasets[1].loc[missing, "bmi"] = 21.5
        fig = densityplot(datasets, missing_pattern, columns=["bmi"])
        (ax,) = fig.axes
        lines = ax.get_lines()
        assert len(lines) == 1 + N_IMPUTATIONS
        vertical = [line for line in lines if list(line.get_xdata()) == [21.5, 21.5]]
        assert len(vertical) == 1
        assert to_rgb(vertical[0].get_color()) == to_rgb("red")
        plt.close("all")

    def test_densityplot_split_single_missing_value(self, imputed_datasets, missing_pattern):
        """One missing cell: each imputation is a single value, drawn as a vertical line."""
        pattern = missing_pattern.copy()
        pattern["bmi"] = 1
        pattern.loc[0, "bmi"] = 0
        fig = densityplot_split(imputed_datasets, pattern, "bmi")
        for k, ax in enumerate(fig.axes[1:]):
            value = imputed_datasets[k].loc[0, "bmi"]
            assert ax.get_title() == f"Imp {k + 1} (constant value: {value:.2f})"
            (line,) = ax.get_lines()
            assert list(line.get_xdata()) == [value, value]
        plt.close("all")


class TestXyplot:
    """Test scatter plot (xyplot) functionality."""

    def test_xyplot_basic(self, imputed_datasets, missing_pattern):
        """Test basic xyplot creation."""
        fig = xyplot(imputed_datasets=imputed_datasets, missing_pattern=missing_pattern, x="age", y="bmi")
        assert isinstance(fig, Figure)
        assert [ax.get_title() for ax in fig.axes] == ["Observed", "Imp 1", "Imp 2", "Imp 3"]
        # age is complete: the observed panel holds the rows with an observed bmi,
        # each imputation panel all rows
        n_observed = int((missing_pattern["bmi"] == 1).sum())
        assert _n_points(fig.axes[0]) == n_observed
        for ax in fig.axes[1:]:
            assert _points_in_colour(ax, "blue") == n_observed
            assert _points_in_colour(ax, "red") == N_ROWS - n_observed
        first = fig.axes[0]
        for ax in fig.axes[1:]:
            assert first.get_shared_x_axes().joined(first, ax)
            assert first.get_shared_y_axes().joined(first, ax)
        plt.close("all")

    def test_xyplot_merged(self, imputed_datasets, missing_pattern):
        """Test xyplot with merged imputations."""
        fig = xyplot(
            imputed_datasets=imputed_datasets, missing_pattern=missing_pattern, x="age", y="bmi", merge_imputations=True
        )
        assert isinstance(fig, Figure)
        (ax,) = fig.axes
        n_observed = int((missing_pattern["bmi"] == 1).sum())
        assert _points_in_colour(ax, "blue") == (1 + N_IMPUTATIONS) * n_observed
        assert _points_in_colour(ax, "red") == N_IMPUTATIONS * (N_ROWS - n_observed)
        plt.close("all")

    def test_xyplot_save(self, imputed_datasets, missing_pattern, temp_dir):
        """Test xyplot saving to file."""
        save_path = temp_dir / "xyplot_test.png"

        fig = xyplot(
            imputed_datasets=imputed_datasets,
            missing_pattern=missing_pattern,
            x="age",
            y="bmi",
            save_path=str(save_path),
        )

        _assert_saved_and_closed(fig, save_path, temp_dir)
        plt.close("all")

    def test_xyplot_missing_columns(self, imputed_datasets, missing_pattern):
        """Test xyplot error handling for missing columns."""
        # This should handle the error gracefully (warn and return None)
        with pytest.warns(UserWarning, match="nonexistent_x"):
            result = xyplot(
                imputed_datasets=imputed_datasets, missing_pattern=missing_pattern, x="nonexistent_x", y="nonexistent_y"
            )
        assert result is None
        plt.close("all")

    def test_xyplot_many_imputations_stays_bounded(self, imputed_datasets, missing_pattern):
        """Twenty imputations: at most four panels per row, figure at most 20 inches a side."""
        fig = xyplot(imputed_datasets * 7, missing_pattern, "age", "bmi")
        assert len(fig.axes) == 1 + 21
        assert len({round(ax.get_position().x0, 3) for ax in fig.axes}) == 4
        assert max(fig.get_size_inches()) <= 20
        plt.close("all")


class TestPlotChainStats:
    """Test chain statistics plotting."""

    def test_plot_chain_stats_basic(self, temp_dir):
        """Test basic chain statistics plotting."""
        # Create mock chain statistics
        n_iter, n_chains = 10, 3
        chain_mean = {"bmi": np.random.randn(n_iter, n_chains), "cholesterol": np.random.randn(n_iter, n_chains)}
        chain_var = {"bmi": np.random.rand(n_iter, n_chains), "cholesterol": np.random.rand(n_iter, n_chains)}

        fig = plot_chain_stats(chain_mean=chain_mean, chain_var=chain_var, columns=["bmi"])
        assert isinstance(fig, Figure)
        ax_mean, ax_sd = fig.axes
        for k, line in enumerate(ax_mean.get_lines()):
            np.testing.assert_allclose(line.get_ydata(), chain_mean["bmi"][:, k])
            np.testing.assert_allclose(line.get_xdata(), np.arange(1, n_iter + 1))
        assert len(ax_sd.get_lines()) == n_chains
        for k, line in enumerate(ax_sd.get_lines()):
            np.testing.assert_allclose(line.get_ydata(), np.sqrt(chain_var["bmi"][:, k]))
        assert ax_sd.get_ylabel() == "Standard deviation"
        assert ax_sd.get_title() == "bmi – SD"
        plt.close("all")

    def test_plot_chain_stats_save(self, temp_dir):
        """Test chain statistics plotting with save."""
        n_iter, n_chains = 10, 3
        chain_mean = {"bmi": np.random.randn(n_iter, n_chains)}
        chain_var = {"bmi": np.random.rand(n_iter, n_chains)}

        save_path = temp_dir / "chain_stats_test.png"

        fig = plot_chain_stats(chain_mean=chain_mean, chain_var=chain_var, save_path=str(save_path))

        _assert_saved_and_closed(fig, save_path, temp_dir)
        plt.close("all")

    def test_plot_chain_stats_no_columns(self):
        """Test chain statistics plotting with no columns."""
        chain_mean = {}
        chain_var = {}
        plt.close("all")

        result = plot_chain_stats(chain_mean=chain_mean, chain_var=chain_var)
        assert result is None
        assert plt.get_fignums() == []
        plt.close("all")

    def test_plot_chain_stats_missing_column(self):
        """Test chain statistics plotting with missing column data."""
        chain_mean = {"bmi": np.random.randn(10, 3)}
        chain_var = {}
        plt.close("all")

        result = plot_chain_stats(chain_mean=chain_mean, chain_var=chain_var, columns=["bmi"])
        assert result is None
        assert plt.get_fignums() == []
        plt.close("all")

    def test_plot_chain_stats_height_is_bounded(self):
        """Thirty variables: the figure stays at most 20 inches high."""
        rng = np.random.default_rng(0)
        chain_mean = {f"v{i}": rng.normal(size=(5, 2)) for i in range(30)}
        chain_var = {f"v{i}": rng.uniform(size=(5, 2)) for i in range(30)}
        fig = plot_chain_stats(chain_mean, chain_var)
        assert len(fig.axes) == 60
        assert fig.get_size_inches()[1] <= 20
        plt.close("all")

    def test_plot_chain_stats_unequal_chain_counts(self):
        """A later column with more chains than the first still gets a colour per chain."""
        chain_mean = {"a": np.zeros((4, 2)), "b": np.zeros((4, 5))}
        chain_var = {"a": np.ones((4, 2)), "b": np.ones((4, 5))}
        fig = plot_chain_stats(chain_mean, chain_var)
        assert [len(ax.get_lines()) for ax in fig.axes] == [2, 2, 5, 5]
        plt.close("all")
