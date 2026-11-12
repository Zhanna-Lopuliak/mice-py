"""Unconditional mean imputation (``mean``), as R mice's ``mice.impute.mean``."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .norm import _masks, _numeric_target


def mean(
    y: ArrayLike,
    id_obs: ArrayLike,
    x: Any = None,
    id_mis: ArrayLike | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Impute the mean of the observed values, as R mice's ``mice.impute.mean``.

    Every imputed cell gets the same value, which understates the variance
    and the correlations of the completed data. As R's help page warns,
    mean imputation is almost never appropriate (Little and Rubin 2002,
    pp. 61-62).

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Numeric target. Only the entries where `id_obs` is True are used.
        Boolean, categorical and text targets raise ``ValueError``.
    id_obs : array-like of bool of shape (n_samples,)
        True where `y` is observed.
    x : optional
        Not used; accepted for the common signature of the imputers.
    id_mis : array-like of bool of shape (n_samples,), optional
        The cells to impute (R's ``wy``). If None, those where `id_obs` is
        False.
    rng : np.random.Generator, optional
        Not used (no random numbers are drawn).

    Returns
    -------
    np.ndarray of float
        The mean of the observed values, once per cell in `id_mis`.

    Raises
    ------
    ValueError
        If `y` is not numeric, or if it has no observed value and some
        value is to be imputed (R imputes NaN there).

    Examples
    --------
    >>> y = np.array([1.0, np.nan, 2.0, 6.0, np.nan])
    >>> mean(y, ~np.isnan(y))
    array([3., 3.])
    """
    id_obs, id_mis = _masks(id_obs, id_mis, "mean")
    if not id_obs.any():  # and nothing to impute (else _masks raised)
        return np.empty(0)
    ynum = _numeric_target(y, "mean")
    return np.full(int(id_mis.sum()), ynum[id_obs].mean())
