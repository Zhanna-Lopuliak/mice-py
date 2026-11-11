"""The missing-data pattern, as R mice's ``md.pattern``, and its plot."""

from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def _ensure_directory(file_path: str | os.PathLike[str]) -> None:
    """
    Create the parent directories of a file path if they do not exist.

    Parameters
    ----------
    file_path : str or path
        Path to a file (not a directory).
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Created directory: {directory}")


def _save_or_show(fig: Figure, save_path: str | os.PathLike[str] | None) -> Figure:
    """
    Save and close ``fig`` when ``save_path`` is given, otherwise show it.

    Closing a saved figure keeps loops that save many plots from piling up
    open figures. The figure object itself stays usable, for example for
    another ``savefig``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save or show.
    save_path : str, path or None
        Where to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        ``fig``.
    """
    if save_path:
        _ensure_directory(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return fig


def md_pattern_like(df: pd.DataFrame) -> pd.DataFrame:
    """
    The missing-data patterns of a DataFrame, as R mice's ``md.pattern``.

    Each row is one pattern, with 1 for observed and 0 for missing cells;
    the index holds the number of rows with that pattern, the last column
    (``#miss_row``) the number of missing cells in it, and the last row
    (``#miss_col``) the number of missing values per column.

    Columns are ordered by their number of missing values (ties keep the
    data order). Rows are ordered as in R's ``md.pattern``: by the pattern
    of missing cells read as a string of 0s (observed) and 1s (missing)
    over the sorted columns. Patterns with the same number of missing cells
    are therefore not necessarily next to each other.

    Parameters
    ----------
    df : pd.DataFrame
        The incomplete data, with at least two columns.

    Returns
    -------
    pd.DataFrame
        Integer table of the patterns and counts described above.

    Raises
    ------
    ValueError
        If `df` is not a DataFrame or has fewer than two columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    if df.shape[1] < 2:
        raise ValueError("Data must have at least two columns")

    missing = df.isna()

    col_missing_counts = missing.sum()
    sorted_cols = col_missing_counts.sort_values(ascending=True, kind="stable").index.tolist()

    # np.unique sorts the rows lexicographically, which for 0/1 rows of equal
    # length is R's order(pat) on the pattern strings
    patterns, pattern_counts = np.unique(missing[sorted_cols].to_numpy(dtype=int), axis=0, return_counts=True)

    pattern_matrix = pd.DataFrame(1 - patterns, columns=sorted_cols)
    pattern_matrix["#miss_row"] = patterns.sum(axis=1)
    pattern_matrix.index = pd.Index([str(c) for c in pattern_counts], name="#rows")

    col_missing_row = col_missing_counts[sorted_cols].to_list()
    col_missing_row.append(sum(col_missing_row))

    summary_df = pattern_matrix.astype(int)
    summary_df.loc["#miss_col"] = col_missing_row

    return summary_df


def plot_missing_data_pattern(
    pattern_df: pd.DataFrame,
    figsize: tuple[float, float] = (8, 5),
    title: str = "Missing Data Pattern",
    rotate_names: bool = False,
    save_path: str | os.PathLike[str] | None = None,
) -> Figure:
    """
    Plot a missing-data pattern, as R mice's ``md.pattern`` does.

    Parameters
    ----------
    pattern_df : pd.DataFrame
        The pattern table of :func:`md_pattern_like`.
    figsize : tuple of float, default (8, 5)
        Figure size in inches (width, height).
    title : str, default "Missing Data Pattern"
        Title of the plot.
    rotate_names : bool, default False
        Whether to rotate the column names by 90 degrees.
    save_path : str or path, optional
        If given, save the figure there (creating missing directories) and
        close it, instead of showing it.

    Returns
    -------
    matplotlib.figure.Figure
        The figure. With `save_path` it is saved and then closed.
    """
    data_only = pattern_df.iloc[:-1, :-1]
    row_counts = pattern_df.index[:-1]
    row_miss = pattern_df.iloc[:-1, -1]
    col_miss = pattern_df.loc["#miss_col"].iloc[:-1]

    cmap = ListedColormap(
        [
            (101 / 255, 155 / 255, 213 / 255),  # blue for present
            (205 / 255, 100 / 255, 140 / 255),  # pink for missing
        ]
    )

    # Create figure with adjusted size based on rotation
    if rotate_names:
        figsize = (figsize[0], figsize[1] + 0.5)  # Add extra height for rotated labels
    fig, ax = plt.subplots(figsize=figsize)

    # Create the heatmap; fixed limits keep the colours right when every
    # cell is observed or every cell is missing
    sns.heatmap(
        1 - data_only.astype(int),
        cmap=cmap,
        cbar=False,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="black",
        square=True,
        ax=ax,
    )

    # Add row counts on the left
    for i, count in enumerate(row_counts):
        ax.text(-0.7, i + 0.5, f"{count}", va="center", ha="right", fontsize=10)

    # Add missing counts on the right
    for i, miss in enumerate(row_miss):
        ax.text(data_only.shape[1] + 0.1, i + 0.5, f"{int(miss)}", va="center", ha="left", fontsize=10)

    # Add column missing counts at the bottom
    for j, miss in enumerate(col_miss):
        ax.text(j + 0.5, data_only.shape[0] + 0.5, f"{int(miss)}", ha="center", va="top", fontsize=10)

    # Add column names on top
    for j, col_name in enumerate(data_only.columns):
        if rotate_names:
            ax.text(j + 0.5, -0.5, col_name, ha="right", va="top", rotation=90, fontsize=10)
        else:
            ax.text(j + 0.5, -0.5, col_name, ha="center", va="top", fontsize=10)

    # Remove default x and y axis labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Add total missing count at bottom right
    ax.text(
        data_only.shape[1] + 0.1,
        data_only.shape[0] + 0.5,
        f"{int(pattern_df.loc['#miss_col'].iloc[-1])}",
        ha="left",
        va="top",
        fontsize=10,
    )

    ax.set_title(title)
    fig.tight_layout(pad=0.4)

    return _save_or_show(fig, save_path)
