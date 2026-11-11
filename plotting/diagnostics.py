"""Diagnostic plots of imputed datasets and of the convergence of the chains, after R mice's plots."""

from __future__ import annotations

import logging
import math
import os
import warnings
from collections.abc import Hashable, Mapping, Sequence
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from .utils import _save_or_show

logger = logging.getLogger(__name__)

# Size of one panel in inches, and the largest figure side. Figures with many
# panels shrink their panels rather than grow without bound.
_PANEL_INCHES = 5.0
_MAX_FIGURE_INCHES = 20.0
# Panels per row in a grid of panels
_MAX_PANEL_COLUMNS = 4


def _figure_size(n_rows: int, n_cols: int) -> tuple[float, float]:
    """Figure size for a grid of square panels, scaled down to fit _MAX_FIGURE_INCHES."""
    width, height = _PANEL_INCHES * n_cols, _PANEL_INCHES * n_rows
    scale = min(1.0, _MAX_FIGURE_INCHES / width, _MAX_FIGURE_INCHES / height)
    return (width * scale, height * scale)


def _panel_grid(n_panels: int, n_rows: int, n_cols: int, **subplot_kws: Any) -> tuple[Figure, list[Axes]]:
    """
    Create a grid of panels and remove the slots beyond ``n_panels``.

    Returns the figure and a flat list of the ``n_panels`` axes.
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=_figure_size(n_rows, n_cols), squeeze=False, **subplot_kws)
    axes = list(axes.flatten())
    for i in range(n_panels, len(axes)):
        axes[i].remove()
        # With a shared x-axis only the bottom row shows tick labels; the
        # panel above a removed slot is now the bottom of its column
        if 0 <= i - n_cols < n_panels:
            axes[i - n_cols].xaxis.set_tick_params(labelbottom=True)
    return fig, axes[:n_panels]


def _variable_grid(n_panels: int, **subplot_kws: Any) -> tuple[Figure, list[Axes]]:
    """One panel per variable, in rows of at most _MAX_PANEL_COLUMNS panels."""
    n_cols = min(n_panels, _MAX_PANEL_COLUMNS)
    return _panel_grid(n_panels, math.ceil(n_panels / n_cols), n_cols, **subplot_kws)


def _imputation_grid(n_panels: int, **subplot_kws: Any) -> tuple[Figure, list[Axes]]:
    """One panel for the observed data and one per imputation."""
    if n_panels <= 3:
        n_rows, n_cols = 1, n_panels
    else:
        n_cols = min(_MAX_PANEL_COLUMNS, math.ceil(math.sqrt(n_panels)))
        n_rows = math.ceil(n_panels / n_cols)
    return _panel_grid(n_panels, n_rows, n_cols, **subplot_kws)


def _columns_to_plot(missing_pattern: pd.DataFrame, columns: Sequence[Hashable] | None) -> list[Hashable]:
    """The requested columns (default: all) that have at least one missing value."""
    if columns is None:
        columns = missing_pattern.columns[missing_pattern.eq(0).any()]
    return [col for col in columns if col in missing_pattern.columns and missing_pattern[col].eq(0).any()]


def _density(ax: Axes, values: Any, color: str, label: str | None = None, **line_kws: Any) -> Any:
    """
    Kernel density of ``values`` on ``ax``.

    A kernel density needs at least two distinct values; otherwise a dashed
    vertical line marks the single value. Returns that value, or None when a
    density (or nothing, for no values) was drawn.
    """
    values = pd.Series(values).dropna()
    if values.nunique() <= 1:
        if values.empty:
            return None
        value = values.iloc[0]
        ax.axvline(x=value, color=color, label=label, alpha=0.6, linestyle="--", **line_kws)
        return value
    sns.kdeplot(data=values, ax=ax, color=color, label=label, alpha=0.6, **line_kws)
    return None


def stripplot(
    imputed_datasets: Sequence[pd.DataFrame],
    missing_pattern: pd.DataFrame,
    columns: Sequence[Hashable] | None = None,
    merge_imputations: bool = False,
    observed_color: str = "blue",
    imputed_color: str = "red",
    save_path: str | os.PathLike[str] | None = None,
) -> Figure | None:
    """
    Strip plots of the observed and imputed values, as R mice's ``stripplot``.

    Per column, the first strip shows the observed values; then, for each
    imputation, the observed and the imputed values in different colours.

    Parameters
    ----------
    imputed_datasets : sequence of pd.DataFrame
        List of DataFrames containing imputed values
    missing_pattern : pd.DataFrame
        DataFrame indicating missing values (0 where missing, 1 where observed)
    columns : list of str, optional
        List of column names to plot. If None, plots all columns with missing values.
    merge_imputations : bool, default False
        If True, shows two columns: one with only observed values and another with observed and imputed values overlaid.
        If False, shows separate plots for each imputation.
    observed_color : str, default 'blue'
        Color for observed values
    imputed_color : str, default 'red'
        Color for imputed values
    save_path : str or path, optional
        If given, save the figure there (creating missing directories) and
        close it, instead of showing it.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure, or None (with a warning) when no column has missing values.
    """
    columns = _columns_to_plot(missing_pattern, columns)

    if not columns:
        warnings.warn("No columns with missing values to plot", UserWarning, stacklevel=2)
        return None

    # One panel per column, at most four per row
    fig, axes = _variable_grid(len(columns))

    # Plot each column
    for ax, col in zip(axes, columns):
        # Get observed values (where missing_pattern is 1)
        observed_mask = missing_pattern[col] == 1
        observed = imputed_datasets[0].loc[observed_mask, col]

        # Create data for stripplot
        plot_data = []

        if merge_imputations:
            # Add observed-only data
            plot_data.append(
                pd.DataFrame(
                    {
                        "value": observed.values,
                        "type": ["Observed"] * len(observed),
                        "imputation": ["Observed Only"] * len(observed),
                    }
                )
            )

            # Add observed data again for the second column
            plot_data.append(
                pd.DataFrame(
                    {
                        "value": observed.values,
                        "type": ["Observed"] * len(observed),
                        "imputation": ["Observed + Imputed"] * len(observed),
                    }
                )
            )

            # Combine all imputed values
            all_imp_values = []
            for df in imputed_datasets:
                imp_values = df.loc[~observed_mask, col]
                all_imp_values.extend(imp_values.values)

            # Add all imputed values to the second column
            plot_data.append(
                pd.DataFrame(
                    {
                        "value": all_imp_values,
                        "type": ["Imputed"] * len(all_imp_values),
                        "imputation": ["Observed + Imputed"] * len(all_imp_values),
                    }
                )
            )
        else:
            # Add observed data
            plot_data.append(
                pd.DataFrame(
                    {
                        "value": observed.values,
                        "type": ["Observed"] * len(observed),
                        "imputation": ["Observed"] * len(observed),
                    }
                )
            )

            # Add each imputation
            for i, df in enumerate(imputed_datasets, 1):
                # Get imputed values for this imputation
                imp_values = df.loc[~observed_mask, col]

                # Add observed and imputed values
                imp_data = pd.DataFrame(
                    {
                        "value": np.concatenate([observed.values, imp_values.values]),  # Use same observed values
                        "type": ["Observed"] * len(observed) + ["Imputed"] * len(imp_values),
                        "imputation": [f"Imp {i}"] * (len(observed) + len(imp_values)),
                    }
                )
                plot_data.append(imp_data)

        # Combine all data
        plot_data = pd.concat(plot_data, ignore_index=True)

        # Create stripplot
        sns.stripplot(
            data=plot_data,
            x="imputation",
            y="value",
            hue="type",
            ax=ax,
            jitter=True,
            palette={"Observed": observed_color, "Imputed": imputed_color},
            legend=False,
            alpha=0.6,
            linewidth=0.5,
        )

        # Customize plot
        ax.set_title(f"{col}")
        ax.set_ylabel("Value")
        ax.set_xlabel("")

        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    return _save_or_show(fig, save_path)


def bwplot(
    imputed_datasets: Sequence[pd.DataFrame],
    missing_pattern: pd.DataFrame,
    columns: Sequence[Hashable] | None = None,
    merge_imputations: bool = False,
    observed_color: str = "blue",
    imputed_color: str = "red",
    save_path: str | os.PathLike[str] | None = None,
) -> Figure | None:
    """
    Box-and-whisker plots of the observed and imputed values, as R mice's ``bwplot``.

    Per column, the first box shows the observed values, then one box per
    imputation shows its imputed values only.

    As in R's ``bwplot.mids``, values beyond the whiskers (1.5 times the
    interquartile range from the box) are drawn as outliers.

    Parameters
    ----------
    imputed_datasets : sequence of pd.DataFrame
        List of DataFrames containing imputed values
    missing_pattern : pd.DataFrame
        DataFrame indicating missing values (0 where missing, 1 where observed)
    columns : list of str, optional
        List of column names to plot. If None, plots all columns with missing values.
    merge_imputations : bool, default False
        If True, combines all imputed values into a single boxplot. If False,
        shows separate boxplots for each imputation.
    observed_color : str, default 'blue'
        Color for observed values
    imputed_color : str, default 'red'
        Color for imputed values
    save_path : str or path, optional
        If given, save the figure there (creating missing directories) and
        close it, instead of showing it.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure, or None (with a warning) when no column has missing values.
    """
    columns = _columns_to_plot(missing_pattern, columns)

    if not columns:
        warnings.warn("No columns with missing values to plot", UserWarning, stacklevel=2)
        return None

    # One panel per column, at most four per row
    fig, axes = _variable_grid(len(columns))

    # Plot each column
    for ax, col in zip(axes, columns):
        # Get observed values (where missing_pattern is 1)
        observed_mask = missing_pattern[col] == 1
        observed = imputed_datasets[0].loc[observed_mask, col]

        # Create data for boxplot
        plot_data = []

        if merge_imputations:
            # Add observed data
            plot_data.append(
                pd.DataFrame(
                    {
                        "value": observed.values,
                        "type": ["Observed"] * len(observed),
                        "imputation": ["Observed"] * len(observed),
                    }
                )
            )

            # Combine all imputed values
            all_imp_values = []
            for df in imputed_datasets:
                imp_values = df.loc[~observed_mask, col]
                all_imp_values.extend(imp_values.values)

            # Add all imputed values together
            plot_data.append(
                pd.DataFrame(
                    {
                        "value": all_imp_values,
                        "type": ["Imputed"] * len(all_imp_values),
                        "imputation": ["Imputed"] * len(all_imp_values),
                    }
                )
            )
        else:
            # Add observed data
            plot_data.append(
                pd.DataFrame(
                    {
                        "value": observed.values,
                        "type": ["Observed"] * len(observed),
                        "imputation": ["Observed"] * len(observed),
                    }
                )
            )

            # Add each imputation
            for i, df in enumerate(imputed_datasets, 1):
                # Get only imputed values for this imputation
                imp_values = df.loc[~observed_mask, col]

                # Add only imputed values
                imp_data = pd.DataFrame(
                    {
                        "value": imp_values.values,
                        "type": ["Imputed"] * len(imp_values),
                        "imputation": [f"Imp {i}"] * len(imp_values),
                    }
                )
                plot_data.append(imp_data)

        # Combine all data
        plot_data = pd.concat(plot_data, ignore_index=True)

        # Create boxplot: dashed whiskers, outliers shown, no legend (the
        # x-axis labels already say which box is observed and which imputed)
        sns.boxplot(
            data=plot_data,
            x="imputation",
            y="value",
            hue="type",
            ax=ax,
            palette={"Observed": observed_color, "Imputed": imputed_color},
            width=0.8,
            showfliers=True,
            showcaps=True,
            showmeans=False,
            legend=False,
            medianprops={"visible": False},
            boxprops={"alpha": 0.6},
            whiskerprops={"alpha": 0.6, "linestyle": "--"},
            capprops={"alpha": 0.6},
            flierprops={"marker": "o", "alpha": 0.6},
        )

        # Add median points
        if merge_imputations:
            # Add observed data point
            observed_data = plot_data[plot_data["imputation"] == "Observed"]
            if not observed_data.empty:
                median_val = observed_data["value"].median()
                ax.plot(0, median_val, "o", color=observed_color, alpha=0.6, markersize=6, zorder=3)

            # Add imputed data point
            imp_data = plot_data[plot_data["imputation"] == "Imputed"]
            if not imp_data.empty:
                median_val = imp_data["value"].median()
                ax.plot(1, median_val, "o", color=imputed_color, alpha=0.6, markersize=6, zorder=3)
        else:
            # Add observed data point
            observed_data = plot_data[plot_data["imputation"] == "Observed"]
            if not observed_data.empty:
                median_val = observed_data["value"].median()
                ax.plot(0, median_val, "o", color=observed_color, alpha=0.6, markersize=6, zorder=3)

            # Then add imputed data points
            for i in range(1, len(imputed_datasets) + 1):
                imp_data = plot_data[(plot_data["imputation"] == f"Imp {i}") & (plot_data["type"] == "Imputed")]
                if not imp_data.empty:
                    median_val = imp_data["value"].median()
                    ax.plot(i, median_val, "o", color=imputed_color, alpha=0.6, markersize=6, zorder=3)

        # Customize plot
        ax.set_title(f"{col}")
        ax.set_ylabel("Value")
        ax.set_xlabel("")

        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    return _save_or_show(fig, save_path)


def densityplot(
    imputed_datasets: Sequence[pd.DataFrame],
    missing_pattern: pd.DataFrame,
    columns: Sequence[Hashable] | None = None,
    observed_color: str = "blue",
    imputed_color: str = "red",
    save_path: str | os.PathLike[str] | None = None,
) -> Figure | None:
    """
    Kernel density plots of the observed and imputed values, as R mice's ``densityplot``.

    Per column, one panel with the density of the observed values and one
    density of the imputed values per imputation.

    Values without variance (for example a single imputed cell) have no
    kernel density; they are drawn as a dashed vertical line at the value.

    Parameters
    ----------
    imputed_datasets : sequence of pd.DataFrame
        List of DataFrames containing imputed values
    missing_pattern : pd.DataFrame
        DataFrame indicating missing values (0 where missing, 1 where observed)
    columns : list of str, optional
        List of column names to plot. If None, plots all columns with missing values.
    observed_color : str, default 'blue'
        Color for observed values
    imputed_color : str, default 'red'
        Color for imputed values
    save_path : str or path, optional
        If given, save the figure there (creating missing directories) and
        close it, instead of showing it.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure, or None (with a warning) when no column has missing values.
    """
    columns = _columns_to_plot(missing_pattern, columns)

    if not columns:
        warnings.warn("No columns with missing values to plot", UserWarning, stacklevel=2)
        return None

    # One panel per column, at most four per row
    fig, axes = _variable_grid(len(columns))

    # Plot each column
    for ax, col in zip(axes, columns):
        # Get observed values (where missing_pattern is 1)
        observed_mask = missing_pattern[col] == 1
        observed = imputed_datasets[0].loc[observed_mask, col]

        # Plot observed data KDE
        _density(ax, observed, observed_color, label="Observed", linewidth=2.5)

        # Plot imputed data KDE for each imputation
        for i, df in enumerate(imputed_datasets, 1):
            # Get only imputed values
            imp_values = df.loc[~observed_mask, col]
            # Only add label for the first imputation
            label = "Imputed" if i == 1 else None
            _density(ax, imp_values, imputed_color, label=label)

        # Customize plot
        ax.set_title(f"{col}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")

        # Handle legend - only for the first plot
        if col == columns[0]:
            handles, labels = ax.get_legend_handles_labels()
            if handles:  # If we have any handles
                ax.legend(handles, labels, title="")

    fig.tight_layout()
    return _save_or_show(fig, save_path)


def densityplot_split(
    imputed_datasets: Sequence[pd.DataFrame],
    missing_pattern: pd.DataFrame,
    column: Hashable,
    observed_color: str = "blue",
    imputed_color: str = "red",
    save_path: str | os.PathLike[str] | None = None,
) -> Figure | None:
    """
    Kernel density plots of one column, one panel for the observed values and one per imputation.

    The observed values are drawn in `observed_color`, the imputed values
    of each imputation in `imputed_color`. All panels share their axes.

    Parameters
    ----------
    imputed_datasets : sequence of pd.DataFrame
        List of DataFrames containing imputed values
    missing_pattern : pd.DataFrame
        DataFrame indicating missing values (0 where missing, 1 where observed)
    column : str
        Name of the column to plot
    observed_color : str, default 'blue'
        Color for observed values
    imputed_color : str, default 'red'
        Color for imputed values
    save_path : str or path, optional
        If given, save the figure there (creating missing directories) and
        close it, instead of showing it.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure, or None (with a warning) when the column is not in
        ``missing_pattern`` or has no missing values.
    """
    if column not in missing_pattern.columns:
        warnings.warn(f"Column {column} not found in the data", UserWarning, stacklevel=2)
        return None

    if not missing_pattern[column].eq(0).any():
        warnings.warn(f"No missing values in column {column}", UserWarning, stacklevel=2)
        return None

    # Get observed values (where missing_pattern is 1)
    observed_mask = missing_pattern[column] == 1
    observed = imputed_datasets[0].loc[observed_mask, column]

    # One panel for the observed data and one per imputation, on common axes
    n_plots = len(imputed_datasets) + 1  # +1 for observed data
    fig, axes = _imputation_grid(n_plots, sharex=True, sharey=True)

    # Plot observed data; with no variance a dashed vertical line marks the value
    value = _density(axes[0], observed, observed_color, linewidth=2.5)
    axes[0].set_title("Observed" if value is None else f"Observed (constant value: {value:.2f})")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Density")

    # Plot each imputation
    for i, df in enumerate(imputed_datasets, 1):
        # Get only imputed values
        imp_values = df.loc[~observed_mask, column]

        value = _density(axes[i], imp_values, imputed_color)
        axes[i].set_title(f"Imp {i}" if value is None else f"Imp {i} (constant value: {value:.2f})")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Density")

    # Add overall title
    fig.suptitle(f"Density plots for {column}", y=1.02)

    fig.tight_layout()
    return _save_or_show(fig, save_path)


def xyplot(
    imputed_datasets: Sequence[pd.DataFrame],
    missing_pattern: pd.DataFrame,
    x: str,
    y: str,
    merge_imputations: bool = False,
    observed_color: str = "blue",
    imputed_color: str = "red",
    save_path: str | os.PathLike[str] | None = None,
) -> Figure | None:
    """
    Create scatter plots of two columns, showing observed and imputed values.

    Points are coloured by whether ``y`` was imputed, as in R's
    ``xyplot.mids`` with its default ``na.groups``:

    - The first panel ("Observed") is the incomplete data: only the rows
      where both ``x`` and ``y`` are observed.
    - Panel ``Imp k`` is the k-th completed dataset, every row: in the
      imputed colour when ``y`` was imputed, in the observed colour
      otherwise. A row with an observed ``y`` but an imputed ``x`` is
      therefore drawn in the observed colour, at the ``x`` of that
      imputation.

    All panels share their axes.

    Parameters
    ----------
    imputed_datasets : sequence of pd.DataFrame
        List of DataFrames containing imputed values
    missing_pattern : pd.DataFrame
        DataFrame indicating missing values (0 where missing, 1 where observed)
    x : str
        Name of the column to plot on x-axis
    y : str
        Name of the column to plot on y-axis
    merge_imputations : bool, default False
        If True, overlays all those panels in a single plot, as R's
        ``xyplot(imp, y ~ x)`` without conditioning on ``.imp``. If False,
        shows n+1 plots: first plot with only observed data, followed by one
        plot for each imputation.
    observed_color : str, default 'blue'
        Color for observed values
    imputed_color : str, default 'red'
        Color for imputed values
    save_path : str or path, optional
        If given, save the figure there (creating missing directories) and
        close it, instead of showing it.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure, or None (with a warning) when ``x`` or ``y`` is not a
        column of ``missing_pattern``.
    """
    # Check if columns exist and provide specific error messages
    missing_cols = []
    if x not in missing_pattern.columns:
        missing_cols.append(f"x-axis column '{x}'")
    if y not in missing_pattern.columns:
        missing_cols.append(f"y-axis column '{y}'")

    if missing_cols:
        warnings.warn(
            f"The following columns are not found in the data: {', '.join(missing_cols)}. "
            f"Available columns are: {', '.join(map(str, missing_pattern.columns))}",
            UserWarning,
            stacklevel=2,
        )
        return None

    # Rows with an observed y, and rows with both x and y observed
    y_observed = missing_pattern[y] == 1
    complete = y_observed & (missing_pattern[x] == 1)
    first = imputed_datasets[0]

    if merge_imputations:
        # Create a single plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Observed colour: the complete rows of the incomplete data, and the
        # rows with an observed y of every completed dataset
        observed_x = pd.concat([first.loc[complete, x]] + [df.loc[y_observed, x] for df in imputed_datasets])
        observed_y = pd.concat([first.loc[complete, y]] + [df.loc[y_observed, y] for df in imputed_datasets])
        ax.scatter(observed_x, observed_y, color=observed_color, alpha=0.6, label="Observed")

        # Imputed colour: the rows with an imputed y of every completed dataset
        imp_x = pd.concat([df.loc[~y_observed, x] for df in imputed_datasets])
        imp_y = pd.concat([df.loc[~y_observed, y] for df in imputed_datasets])
        ax.scatter(imp_x, imp_y, color=imputed_color, alpha=0.6, label=f"Imputed ({y})")

        # Customize plot
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.legend()
        ax.set_title(f"Scatter plot of {x} vs {y} (all imputations)")

    else:
        # One panel for the observed data and one per imputation, on common axes
        n_plots = len(imputed_datasets) + 1  # +1 for observed data
        fig, axes = _imputation_grid(n_plots, sharex=True, sharey=True)

        # First panel: the incomplete data, so only rows with x and y observed
        axes[0].scatter(
            first.loc[complete, x], first.loc[complete, y], color=observed_color, alpha=0.6, label="Observed"
        )
        axes[0].set_xlabel(x)
        axes[0].set_ylabel(y)
        axes[0].legend()
        axes[0].set_title("Observed")

        # Plot each imputation
        for i, df in enumerate(imputed_datasets, 1):
            # Rows with an observed y (their x may be imputed)
            axes[i].scatter(
                df.loc[y_observed, x], df.loc[y_observed, y], color=observed_color, alpha=0.6, label=f"Observed ({y})"
            )

            # Rows with an imputed y
            axes[i].scatter(
                df.loc[~y_observed, x], df.loc[~y_observed, y], color=imputed_color, alpha=0.6, label=f"Imputed ({y})"
            )

            # Customize plot
            axes[i].set_xlabel(x)
            axes[i].set_ylabel(y)
            axes[i].legend()
            axes[i].set_title(f"Imp {i}")

        # Add overall title
        fig.suptitle(f"Scatter plots of {x} vs {y}", y=1.02)

    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_chain_stats(
    chain_mean: Mapping[Hashable, ArrayLike],
    chain_var: Mapping[Hashable, ArrayLike],
    columns: Sequence[Hashable] | None = None,
    figsize: tuple[float, float] = (10, 5),
    save_path: str | os.PathLike[str] | None = None,
) -> Figure | None:
    """
    Plot per-iteration chain means and standard deviations for the given columns.

    As in R's ``plot.mids``, the right-hand panels show the standard
    deviation of the imputed values: the square root of ``chain_var``.

    Parameters
    ----------
    chain_mean : dict of str to np.ndarray
        Dictionary where each key is a column name and each value is a 2-D
        array of shape (n_iter, n_imputations) containing the means of the
        newly imputed values.
    chain_var : Dict[str, np.ndarray]
        Same structure as `chain_mean` but for the variance of the imputed
        values (``MICE.chain_var``). The plot shows its square root.
    columns : list of str, optional
        Columns to plot. If None, plots all keys present in `chain_mean`.
    figsize : tuple, default (10, 5)
        Base size of a single row (width, height). The final figure will be
        scaled according to the number of rows, up to a height of 20 inches
        (or ``figsize[1]`` if that is larger); beyond it the rows get shorter.
    save_path : str, optional
        If provided, save the plot to this path and close it, instead of displaying it.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure, or None (with a logged warning) when there is nothing to plot.
    """
    logger.info("Generating chain statistics plots.")

    # If columns are not specified, use all available from chain_mean
    if columns is None:
        columns = list(chain_mean.keys())

    if not columns:
        logger.warning("No columns specified or found to plot.")
        return None

    # Filter columns to only those that exist in both chain_mean and chain_var
    valid_columns = []
    for col in columns:
        if col not in chain_mean or col not in chain_var:
            logger.warning(f"Statistics for column '{col}' not found in both chain_mean and chain_var. Skipping.")
            continue
        valid_columns.append(col)

    if not valid_columns:
        logger.warning("No valid columns found to plot.")
        return None

    n_rows = len(valid_columns)
    height = min(figsize[1] * n_rows, max(_MAX_FIGURE_INCHES, figsize[1]))
    fig, axes = plt.subplots(n_rows, 2, figsize=(figsize[0], height), squeeze=False)

    # One colour per chain, enough for the column with the most chains
    n_chains = max(np.shape(chain_mean[col])[1] for col in valid_columns)
    palette = sns.color_palette("husl", n_colors=n_chains)

    for row_idx, col in enumerate(valid_columns):
        mean_mat = np.asarray(chain_mean[col], dtype=float)
        sd_mat = np.sqrt(np.asarray(chain_var[col], dtype=float))

        n_iter = mean_mat.shape[0]
        x = np.arange(1, n_iter + 1)

        # Plot means
        ax_mean = cast(Axes, axes[row_idx, 0])
        for chain_idx in range(mean_mat.shape[1]):
            ax_mean.plot(
                x,
                mean_mat[:, chain_idx],
                marker="o",
                alpha=0.7,
                label=f"Chain {chain_idx + 1}",
                color=palette[chain_idx],
            )
        ax_mean.set_title(f"{col} – Mean")
        ax_mean.set_xlabel("Iteration")
        ax_mean.set_ylabel("Mean")
        if row_idx == 0:
            ax_mean.legend(loc="best")

        # Plot standard deviations
        ax_sd = cast(Axes, axes[row_idx, 1])
        for chain_idx in range(sd_mat.shape[1]):
            ax_sd.plot(
                x, sd_mat[:, chain_idx], marker="o", alpha=0.7, label=f"Chain {chain_idx + 1}", color=palette[chain_idx]
            )
        ax_sd.set_title(f"{col} – SD")
        ax_sd.set_xlabel("Iteration")
        ax_sd.set_ylabel("Standard deviation")

    fig.tight_layout()
    return _save_or_show(fig, save_path)
