"""
Plotting module for MICE (Multiple Imputation by Chained Equations)

This module combines diagnostic plots for analyzing imputed datasets 
and utilities for visualizing missing data patterns.
"""

from .diagnostics import (
    stripplot,
    bwplot, 
    densityplot,
    densityplot_split,
    xyplot,
    plot_chain_stats
)

from .utils import (
    md_pattern_like,
    plot_missing_data_pattern
)

__all__ = [
    'stripplot',
    'bwplot',
    'densityplot', 
    'densityplot_split',
    'xyplot',
    'plot_chain_stats',
    'md_pattern_like',
    'plot_missing_data_pattern'
] 