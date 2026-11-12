"""
Imputation by linear regression (``norm``, ``norm.nob``, ``norm.boot``, ``norm.predict``).

The ports of R mice's ``mice.impute.norm``, ``mice.impute.norm.nob``,
``mice.impute.norm.boot`` and ``mice.impute.norm.predict``. All four fit the
linear regression of the observed target on ``[1, x]`` by
:func:`imputation.sampler.estimice` and impute new values (floats), not
observed ones. They differ in how they treat the uncertainty of the fit:

- ``norm``: Bayesian draw of the coefficients and the residual standard
  deviation (Rubin 1987, p. 167), then a random residual. Proper.
- ``norm.boot``: least squares on a bootstrap sample of the observed rows,
  then a random residual. Proper.
- ``norm.nob``: least squares on the observed rows, then a random residual;
  the parameter uncertainty is ignored. Improper.
- ``norm.predict``: the predicted value, no randomness at all ("regression
  imputation"). Improper.

The target must be numeric.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .design import _column_kind, obtain_design
from .sampler import estimice, norm_draw

# How the target kinds of `design._column_kind` are named in error messages
_KIND_NAMES = {"bool": "boolean", "factor": "categorical or text", "ordered": "ordered categorical"}


def norm(
    y: ArrayLike,
    id_obs: ArrayLike,
    x: ArrayLike | pd.DataFrame,
    id_mis: ArrayLike | None = None,
    ls_meth: str = "qr",
    ridge: float = 1e-5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Imputation by Bayesian linear regression, as R mice's ``mice.impute.norm``.

    The coefficients and the residual standard deviation are drawn from
    their posterior distribution given the observed rows
    (:func:`imputation.sampler.norm_draw`, Rubin 1987, p. 167), and each
    imputation is the prediction with the drawn coefficients plus a normal
    residual with the drawn standard deviation.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Numeric target. Only the entries where `id_obs` is True are used, so
        the missing entries may hold anything (inside MICE they hold the
        current imputations). Boolean, categorical and text targets raise
        ``ValueError``.
    id_obs : array-like of bool of shape (n_samples,)
        True where `y` is observed.
    x : array-like or pd.DataFrame of shape (n_samples, n_features)
        Predictors without the intercept, without missing values; zero
        columns (or None) for the intercept-only model. A DataFrame with
        categorical, object, string or boolean columns is dummy-coded first
        by :func:`imputation.design.obtain_design`, as MICE does.
    id_mis : array-like of bool of shape (n_samples,), optional
        The cells to impute (R's ``wy``). If None, those where `id_obs` is
        False.
    ls_meth : {"qr", "ridge", "svd"}, default="qr"
        Least squares method of :func:`imputation.sampler.estimice`.
    ridge : float, default=1e-5
        Ridge penalty of :func:`imputation.sampler.estimice`.
    rng : np.random.Generator, optional
        Random number generator. If None, a fresh generator is used.

    Returns
    -------
    np.ndarray of float
        The imputations of the cells in `id_mis`, in their order.

    Raises
    ------
    ValueError
        If `y` is not numeric, if `y` has no observed value and some value
        is to be imputed, or if `x` is not numeric or has missing values.

    Notes
    -----
    The random numbers are drawn in R's order: one chi-square variate with
    ``n_obs - q`` degrees of freedom (at least 1) and ``q`` standard normal
    variates for the parameters (``q`` coefficients, intercept included),
    then one standard normal variate per imputed cell. Coefficients of
    columns that are linear combinations of earlier ones are set to 0, as R
    does.

    References
    ----------
    Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*.
    New York: Wiley, p. 167.

    Examples
    --------
    >>> y = np.array([2.1, 3.9, np.nan, 8.2, 9.8, 12.3, np.nan, 15.7, 18.4, 19.9])
    >>> id_obs = ~np.isnan(y)
    >>> x = np.arange(10.0).reshape(-1, 1)  # one predictor
    >>> y_imp = norm(y, id_obs, x, rng=np.random.default_rng(1))
    >>> y_imp.shape  # one value per missing cell
    (2,)
    """
    ynum, id_obs, id_mis, design = _model_inputs(y, id_obs, x, id_mis, "norm")
    if not id_obs.any():  # and nothing to impute (else _model_inputs raised)
        return np.empty(0)
    if rng is None:
        rng = np.random.default_rng()
    # chi-square, then one normal per coefficient (R's .norm.draw)
    parm = norm_draw(ynum, id_obs, design, rng=rng, ls_meth=ls_meth, ridge=ridge)
    noise = rng.standard_normal(int(id_mis.sum()))
    return design[id_mis] @ parm["beta"] + noise * parm["sigma"]


def norm_nob(
    y: ArrayLike,
    id_obs: ArrayLike,
    x: ArrayLike | pd.DataFrame,
    id_mis: ArrayLike | None = None,
    ls_meth: str = "qr",
    ridge: float = 1e-5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Imputation by linear regression without parameter uncertainty, as R mice's ``mice.impute.norm.nob``.

    Each imputation is the least-squares prediction plus a normal residual
    with the estimated residual standard deviation. The coefficients are not
    drawn, so the method is not proper in Rubin's sense and understates the
    variability of the imputations in small samples. R's help page offers it
    mainly to compare proper and improper imputation, and for large data,
    where the difference is small.

    Parameters
    ----------
    y, id_obs, x, id_mis : array-like
        As in :func:`norm`.
    ls_meth, ridge, rng
        As in :func:`norm`.

    Returns
    -------
    np.ndarray of float
        The imputations of the cells in `id_mis`, in their order.

    Raises
    ------
    ValueError
        As :func:`norm`; also if there are not more than ``q + 1`` observed
        values, ``q`` the number of coefficients (intercept included).

    Notes
    -----
    The residual standard deviation is ``sqrt(sum(r ** 2) / (n_obs - q - 1))``,
    with one degree of freedom less than the usual ``n_obs - q``: that is
    what R mice computes, and it is kept. Where ``n_obs <= q + 1`` R imputes
    infinite or undefined values; this port raises ``ValueError`` instead.
    Coefficients of columns that are linear combinations of earlier ones are
    set to 0 (R imputes NA).

    Random numbers: one standard normal variate per imputed cell.

    Examples
    --------
    >>> y = np.array([2.1, 3.9, np.nan, 8.2, 9.8, 12.3, np.nan, 15.7, 18.4, 19.9])
    >>> x = np.arange(10.0).reshape(-1, 1)
    >>> norm_nob(y, ~np.isnan(y), x, rng=np.random.default_rng(1)).shape
    (2,)
    """
    ynum, id_obs, id_mis, design = _model_inputs(y, id_obs, x, id_mis, "norm.nob")
    if not id_obs.any():
        return np.empty(0)
    n_obs, n_coef = int(id_obs.sum()), design.shape[1]
    _require_residual_df(n_obs, n_coef, "norm.nob")
    if rng is None:
        rng = np.random.default_rng()
    p = estimice(design[id_obs], ynum[id_obs], ls_meth=ls_meth, ridge=ridge)
    sigma = _residual_sd(p["r"], n_obs, n_coef)
    noise = rng.standard_normal(int(id_mis.sum()))
    return design[id_mis] @ _zero_aliased(p["c"]) + noise * sigma


def norm_boot(
    y: ArrayLike,
    id_obs: ArrayLike,
    x: ArrayLike | pd.DataFrame,
    id_mis: ArrayLike | None = None,
    ls_meth: str = "qr",
    ridge: float = 1e-5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Imputation by linear regression on a bootstrap sample, as R mice's ``mice.impute.norm.boot``.

    The observed rows are resampled with replacement; the least-squares fit
    on that sample gives the coefficients and the residual standard
    deviation, and each imputation is the prediction plus a normal residual
    with that standard deviation. The bootstrap stands in for the
    parameter draw of :func:`norm`.

    Parameters
    ----------
    y, id_obs, x, id_mis : array-like
        As in :func:`norm`.
    ls_meth, ridge, rng
        As in :func:`norm`.

    Returns
    -------
    np.ndarray of float
        The imputations of the cells in `id_mis`, in their order.

    Raises
    ------
    ValueError
        As :func:`norm_nob`.

    Notes
    -----
    Random numbers, in R's order: ``n_obs`` bootstrap indices
    (``rng.integers(0, n_obs, size=n_obs)``, R's
    ``sample(n1, n1, replace = TRUE)``), then one standard normal variate
    per imputed cell.

    The residual standard deviation of the bootstrap fit is
    ``sqrt(sum(r ** 2) / (n_obs - q - 1))``, as R computes it (see
    :func:`norm_nob`). Where R fails, this port goes on:

    - without predictors R stops with an error (``'x' must be a matrix``);
      here the intercept-only model is fitted;
    - where ``n_obs <= q + 1`` R imputes infinite or undefined values; here
      ``ValueError`` is raised;
    - coefficients of columns that are linear combinations of earlier ones
      in the bootstrap sample are set to 0 (R imputes NA).

    Examples
    --------
    >>> y = np.array([2.1, 3.9, np.nan, 8.2, 9.8, 12.3, np.nan, 15.7, 18.4, 19.9])
    >>> x = np.arange(10.0).reshape(-1, 1)
    >>> norm_boot(y, ~np.isnan(y), x, rng=np.random.default_rng(1)).shape
    (2,)
    """
    ynum, id_obs, id_mis, design = _model_inputs(y, id_obs, x, id_mis, "norm.boot")
    if not id_obs.any():
        return np.empty(0)
    n_obs, n_coef = int(id_obs.sum()), design.shape[1]
    _require_residual_df(n_obs, n_coef, "norm.boot")
    if rng is None:
        rng = np.random.default_rng()
    rows = _bootstrap_sample(n_obs, rng)
    p = estimice(design[id_obs][rows], ynum[id_obs][rows], ls_meth=ls_meth, ridge=ridge)
    sigma = _residual_sd(p["r"], n_obs, n_coef)
    noise = rng.standard_normal(int(id_mis.sum()))
    return design[id_mis] @ _zero_aliased(p["c"]) + noise * sigma


def norm_predict(
    y: ArrayLike,
    id_obs: ArrayLike,
    x: ArrayLike | pd.DataFrame,
    id_mis: ArrayLike | None = None,
    ls_meth: str = "qr",
    ridge: float = 1e-5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Imputation by the predicted value of a linear regression, as R mice's ``mice.impute.norm.predict``.

    Each imputation is the least-squares prediction: regression imputation,
    without any random component. It understates the variance and inflates
    the correlations of the completed data.

    Parameters
    ----------
    y, id_obs, x, id_mis : array-like
        As in :func:`norm`.
    ls_meth, ridge
        As in :func:`norm`.
    rng : np.random.Generator, optional
        Not used (no random numbers are drawn); accepted for the common
        signature of the imputers.

    Returns
    -------
    np.ndarray of float
        The imputations of the cells in `id_mis`, in their order.

    Raises
    ------
    ValueError
        As :func:`norm`.

    Notes
    -----
    Coefficients of columns that are linear combinations of earlier ones
    are set to 0 (R imputes NA). The result depends only on the data, but
    inside MICE the predictors of a column hold imputations of other columns,
    which are random.

    Examples
    --------
    >>> y = np.array([1.0, 3.0, np.nan, 7.0])
    >>> x = np.arange(4.0).reshape(-1, 1)
    >>> norm_predict(y, ~np.isnan(y), x)
    array([5.])
    """
    ynum, id_obs, id_mis, design = _model_inputs(y, id_obs, x, id_mis, "norm.predict")
    if not id_obs.any():
        return np.empty(0)
    p = estimice(design[id_obs], ynum[id_obs], ls_meth=ls_meth, ridge=ridge)
    return design[id_mis] @ _zero_aliased(p["c"])


# ---------------------------------------------------------------------------
# Helpers, shared with imputation.mean
# ---------------------------------------------------------------------------


def _masks(id_obs: ArrayLike, id_mis: ArrayLike | None, method: str) -> tuple[np.ndarray, np.ndarray]:
    """`id_obs` and `id_mis` as boolean arrays; `id_mis` defaults to the complement of `id_obs`.

    Raises ``ValueError`` if nothing is observed but something is to be imputed.
    """
    id_obs = np.asarray(id_obs, dtype=bool)
    id_mis = ~id_obs if id_mis is None else np.asarray(id_mis, dtype=bool)
    if not id_obs.any() and id_mis.any():
        raise ValueError(
            f"{method}: the target has no observed value in the rows used for the model, so there is nothing to fit."
        )
    return id_obs, id_mis


def _numeric_target(y: Any, method: str) -> np.ndarray:
    """`y` as a float array with NaN for missing entries; it must be numeric.

    The norm methods and mean model the target as a number, so, unlike pmm,
    they cannot impute a boolean, categorical or text target (R's versions
    impute NA there, with warnings). Such a target raises ``ValueError``,
    as do object columns (which count as categorical in this package).
    """
    series = pd.Series(y)
    try:
        kind = _column_kind(series)
    except TypeError:  # complex or another dtype that is no column type
        kind = str(series.dtype)
    if kind != "numeric":
        raise ValueError(
            f"Target y must be numeric for {method}, got {_KIND_NAMES.get(kind, kind)} values. {method} "
            f"imputes numbers (R's mice.impute.{method} imputes NA here); impute a boolean column or one with "
            "two categories with logreg or logreg.boot, a categorical or text column with polyreg (polr if "
            "it is ordered), or any of them with pmm, cart, rf or sample."
        )
    return series.to_numpy(dtype=float, na_value=np.nan)


def _model_inputs(
    y: Any, id_obs: ArrayLike, x: Any, id_mis: ArrayLike | None, method: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validated inputs of a norm method: (y as float, id_obs, id_mis, design matrix ``[1, x]``)."""
    id_obs, id_mis = _masks(id_obs, id_mis, method)
    ynum = _numeric_target(y, method)
    return ynum, id_obs, id_mis, _design_matrix(x, len(ynum), method)


def _design_matrix(x: Any, n_rows: int, method: str) -> np.ndarray:
    """``[1, x]`` as a float array; `x` must be numeric (or dummy-coded) and complete.

    A DataFrame with non-numeric columns is dummy-coded by
    :func:`imputation.design.obtain_design`, as MICE does. None or zero
    columns give the intercept-only design.
    """
    if x is None:
        x = np.empty((n_rows, 0))
    if isinstance(x, pd.DataFrame):
        if not x.select_dtypes(exclude=[np.number]).columns.empty:
            x = obtain_design(x, list(x.columns))
        missing_cols = x.columns[x.isna().any()].tolist()
        if missing_cols:
            raise ValueError(
                f"Predictors must not contain missing values for {method}. Columns with missing values: {missing_cols}"
            )
        values = x.to_numpy(dtype=float)
    else:
        try:
            values = np.asarray(x, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(
                f"Predictors must be numeric for {method}. Could not convert 'x' to numeric array."
            ) from None
        if np.isnan(values).any():
            raise ValueError(f"Predictors must not contain missing values for {method}.")
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.shape[0] != n_rows:
        raise ValueError(f"{method}: x has {values.shape[0]} rows, but y has {n_rows} entries.")
    return np.c_[np.ones(n_rows), values]


def _require_residual_df(n_obs: int, n_coef: int, method: str) -> None:
    """Raise ``ValueError`` unless ``n_obs - n_coef - 1``, R's residual df of norm.nob and norm.boot, is positive."""
    if n_obs <= n_coef + 1:
        raise ValueError(
            f"{method} needs more than {n_coef + 1} observed values with {n_coef} coefficients (intercept "
            f"included), got {n_obs}: its residual standard deviation divides by n_obs - n_coef - 1, as in R, "
            "where the imputations would be infinite or undefined. Use fewer predictors, or norm or pmm."
        )


def _residual_sd(residuals: np.ndarray, n_obs: int, n_coef: int) -> float:
    """Residual standard deviation of norm.nob and norm.boot, ``sqrt(RSS / (n_obs - n_coef - 1))`` as in R."""
    return float(np.sqrt(np.sum(residuals**2) / (n_obs - n_coef - 1)))


def _bootstrap_sample(n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` row indices drawn with replacement from ``0, ..., n - 1``.

    R's ``sample(n, n, replace = TRUE)``, minus 1.
    """
    return rng.integers(0, n, size=n)


def _zero_aliased(coef: np.ndarray) -> np.ndarray:
    """The coefficients with NaN (columns aliased in the least-squares fit) set to 0."""
    return np.nan_to_num(coef, nan=0.0)
