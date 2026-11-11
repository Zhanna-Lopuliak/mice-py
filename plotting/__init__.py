"""
Diagnostic plots for multiple imputation, after R's ``mice`` plots.

Plots of imputed datasets (:func:`stripplot`, :func:`bwplot`,
:func:`densityplot`, :func:`densityplot_split`, :func:`xyplot`), the trace
plot of the chains (:func:`plot_chain_stats`) and the missing-data pattern
(:func:`md_pattern_like`, :func:`plot_missing_data_pattern`). Every plot
function returns its ``matplotlib.figure.Figure``.

The modules log to loggers named after them (``plotting.diagnostics``,
``plotting.utils``); the package only adds a ``logging.NullHandler`` to the
``plotting`` logger.
"""

import logging
from importlib.metadata import PackageNotFoundError, version

from .diagnostics import bwplot, densityplot, densityplot_split, plot_chain_stats, stripplot, xyplot
from .utils import md_pattern_like, plot_missing_data_pattern

try:
    __version__ = version("mice-py")
except PackageNotFoundError:  # not installed, e.g. imported from a source tree
    __version__ = "0+unknown"

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "stripplot",
    "bwplot",
    "densityplot",
    "densityplot_split",
    "xyplot",
    "plot_chain_stats",
    "md_pattern_like",
    "plot_missing_data_pattern",
]
