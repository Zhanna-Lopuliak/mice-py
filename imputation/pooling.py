"""
Standalone pooling module for multiple imputation results.

This module provides functions to pool descriptive statistics and model estimates
from multiple imputed datasets using Rubin's rules, without requiring coupling
to any specific imputation framework.
"""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

logger = logging.getLogger(__name__)

# Columns of the table returned by `pool_estimates`, in this order. They are
# those of R's `summary(pool(fit), type = "all", conf.int = TRUE)` with
# Python names.
POOL_TABLE_COLUMNS = [
    "m",
    "estimate",
    "ubar",
    "b",
    "t",
    "dfcom",
    "df",
    "riv",
    "lambda",
    "fmi",
    "std_error",
    "statistic",
    "p_value",
    "conf_low",
    "conf_high",
]


def barnard_rubin(m: ArrayLike, b: ArrayLike, t: ArrayLike, dfcom: ArrayLike = np.inf) -> float | np.ndarray:
    """
    Barnard-Rubin degrees of freedom of a pooled scalar estimate.

    Same rule as R mice's ``barnard.rubin``: with ``lambda = (1 + 1/m) b / t``
    the large-sample value is ``(m - 1) / lambda**2``; for finite ``dfcom``
    it is combined with the observed-data df of Barnard and Rubin (1999).

    Parameters
    ----------
    m : int or array_like
        Number of imputations.
    b : float or array_like
        Between-imputation variance.
    t : float or array_like
        Total variance, ``ubar + (1 + 1/m) * b``.
    dfcom : float or array_like, default=np.inf
        Residual degrees of freedom of the complete-data analysis.

    Returns
    -------
    float or ndarray
        The degrees of freedom; infinite when ``b = 0`` and ``dfcom`` is
        infinite.

    References
    ----------
    Barnard, J. and Rubin, D.B. (1999). Small-sample degrees of freedom with
    multiple imputation. Biometrika, 86, 948-955.
    """
    m, b, t, dfcom = (np.asarray(a, dtype=float) for a in (m, b, t, dfcom))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        lam = (1.0 + 1.0 / m) * b / t
        dfold = (m - 1.0) / lam**2
        tmp = (1.0 - lam) * (1.0 + dfcom) * dfcom
        df_br = (m - 1.0) * tmp / ((dfcom + 3.0) * (m - 1.0) + lam**2 * tmp)
        df = np.where(np.isinf(dfcom), dfold, df_br)
    return df[()] if df.ndim == 0 else df


def _pool_table(
    m: int,
    qbar: np.ndarray,
    ubar: np.ndarray,
    b: np.ndarray | None,
    dfcom: float,
    param_names: Sequence[Hashable] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Build the pooled table from the per-term summaries of Rubin's rules.

    With ``m = 1`` nothing is pooled: ``b`` cannot be estimated (NaN), the
    total variance is ``ubar`` and the reference is a t with ``dfcom``
    degrees of freedom, which is the single complete-data analysis.
    """
    qbar = np.asarray(qbar, dtype=float)
    ubar = np.asarray(ubar, dtype=float)
    dfcom = float(dfcom)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if m > 1:
            b = np.asarray(b, dtype=float)
            t = ubar + (1.0 + 1.0 / m) * b
            df = barnard_rubin(m, b, t, dfcom)
            riv = (1.0 + 1.0 / m) * b / ubar
            lam = (1.0 + 1.0 / m) * b / t
            fmi = (riv + 2.0 / (df + 3.0)) / (riv + 1.0)
        else:
            b = np.full_like(qbar, np.nan)
            t = ubar.copy()
            df = np.full_like(qbar, dfcom)
            riv = lam = fmi = np.full_like(qbar, np.nan)
        std_error = np.sqrt(t)
        statistic = qbar / std_error
        # R floors df at 0.001 for the p-value, not for the interval.
        p_value = 2.0 * stats.t.sf(np.abs(statistic), np.maximum(df, 0.001))
        conf_low = qbar + stats.t.ppf(alpha / 2.0, df) * std_error
        conf_high = qbar + stats.t.ppf(1.0 - alpha / 2.0, df) * std_error

    p = qbar.shape[0]
    index = pd.Index(range(p) if param_names is None else list(param_names), name="term")
    return pd.DataFrame(
        {
            "m": np.full(p, m, dtype=int),
            "estimate": qbar,
            "ubar": ubar,
            "b": b,
            "t": t,
            "dfcom": np.full(p, dfcom),
            "df": df,
            "riv": riv,
            "lambda": lam,
            "fmi": fmi,
            "std_error": std_error,
            "statistic": statistic,
            "p_value": p_value,
            "conf_low": conf_low,
            "conf_high": conf_high,
        },
        index=index,
        columns=POOL_TABLE_COLUMNS,
    )


def pool_estimates(
    estimates: ArrayLike,
    variances: ArrayLike,
    dfcom: float = np.inf,
    param_names: Sequence[Hashable] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Pool scalar estimates with Rubin's rules and Barnard-Rubin inference.

    Gives the same numbers as R mice's ``pool()`` followed by
    ``summary(type = "all", conf.int = TRUE)``. For each term::

        ubar   = mean of the m within-imputation variances
        b      = variance of the m estimates
        t      = ubar + (1 + 1/m) * b
        riv    = (1 + 1/m) * b / ubar
        lambda = (1 + 1/m) * b / t
        df     = barnard_rubin(m, b, t, dfcom)
        fmi    = (riv + 2 / (df + 3)) / (riv + 1)

    Intervals and p-values use the t distribution with ``df`` degrees of
    freedom.

    Parameters
    ----------
    estimates : array_like, shape (m, p)
        Estimate of each of the ``p`` terms in each of the ``m`` imputed
        datasets. A 1-D array holds the ``m`` estimates of a single term.
    variances : array_like, shape (m, p)
        Squared standard errors matching ``estimates``.
    dfcom : float, default=np.inf
        Residual degrees of freedom of the complete-data analysis, for
        example ``n - p`` in a linear regression. ``np.inf`` assumes a large
        sample.
    param_names : list of str, optional
        Names of the terms, used as the index. Defaults to ``0 .. p-1``.
    alpha : float, default=0.05
        One minus the confidence level of the interval.

    Returns
    -------
    pd.DataFrame
        One row per term (index ``term``) with the columns ``m, estimate,
        ubar, b, t, dfcom, df, riv, lambda, fmi, std_error, statistic,
        p_value, conf_low, conf_high``.

    Notes
    -----
    With ``m = 1`` there is nothing to pool. As R's ``pool()``, the function
    warns and reports the single analysis: ``b``, ``riv``, ``lambda`` and
    ``fmi`` are NaN, ``t = ubar`` and ``df = dfcom``.
    """
    est = np.asarray(estimates, dtype=float)
    var = np.asarray(variances, dtype=float)
    if est.ndim == 1:
        est = est[:, None]
    if var.ndim == 1:
        var = var[:, None]
    if est.ndim != 2 or est.shape != var.shape:
        raise ValueError(
            f"estimates and variances must be arrays of the same shape (m, p); "
            f"got {np.shape(estimates)} and {np.shape(variances)}"
        )
    m, p = est.shape
    if m < 1 or p < 1:
        raise ValueError("estimates must hold at least one imputation and one term")
    if param_names is not None and len(param_names) != p:
        raise ValueError(f"param_names has {len(param_names)} entries for {p} terms")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie strictly between 0 and 1, got {alpha}")
    if np.isnan(dfcom) or dfcom < 0:
        raise ValueError(f"dfcom must be non-negative, got {dfcom}")

    if m == 1:
        warnings.warn(
            "Number of multiple imputations m = 1. No pooling done: the "
            "between-imputation variance cannot be estimated, so inference "
            "uses the single analysis with dfcom degrees of freedom.",
            UserWarning,
            stacklevel=2,
        )
        b = None
    else:
        b = est.var(axis=0, ddof=1)
        # Identical estimates in every imputation: b is exactly zero, as in R.
        b[np.all(est == est[0], axis=0)] = 0.0

    return _pool_table(m, est.mean(axis=0), var.mean(axis=0), b, dfcom, param_names, alpha)


def _residual_df(fit_result: Any) -> float:
    """Residual degrees of freedom of a fitted model, as R's ``get.dfcom``.

    Uses ``df_resid`` when the result has it, else ``max(nobs - p, 1)``,
    else infinity.
    """
    df = getattr(fit_result, "df_resid", None)
    if df is not None and np.isfinite(df):
        return float(df)
    nobs = getattr(fit_result, "nobs", None)
    if nobs is not None and np.isfinite(nobs):
        return float(max(nobs - len(np.atleast_1d(fit_result.params)), 1))
    return np.inf


@dataclass
class PoolingResult:
    """
    Container for pooled multiple imputation results.

    Attributes
    ----------
    estimates : np.ndarray
        Pooled parameter estimates (q_bar)
    variances : np.ndarray
        Total variances for each parameter (t)
    within_variance : np.ndarray
        Average within-imputation variance (u_bar)
    between_variance : np.ndarray
        Between-imputation variance (b)
    frac_miss_info : np.ndarray
        Fraction of missing information for each parameter, R's ``fmi``
    param_names : list of str
        Names of the pooled parameters
    n_imputations : int
        Number of imputations used
    sample_size : int
        Sample size of each imputed dataset. The complete-data degrees of
        freedom are ``sample_size - 1``.
    """

    estimates: np.ndarray
    variances: np.ndarray
    within_variance: np.ndarray
    between_variance: np.ndarray
    frac_miss_info: np.ndarray
    param_names: list[str]
    n_imputations: int
    sample_size: int

    @property
    def dfcom(self) -> float:
        """Complete-data degrees of freedom of a mean or proportion, ``n - 1``."""
        return float(self.sample_size - 1)

    def table(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Return the full pooled table, as `pool_estimates` does.

        Parameters
        ----------
        alpha : float, default=0.05
            One minus the confidence level of the interval.

        Returns
        -------
        pd.DataFrame
            One row per parameter with the columns of `pool_estimates`.
        """
        return _pool_table(
            self.n_imputations,
            self.estimates,
            self.within_variance,
            self.between_variance,
            self.dfcom,
            self.param_names,
            alpha,
        )

    def summary(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Return a summary DataFrame with pooled statistics.

        The interval uses the t distribution with the Barnard-Rubin degrees
        of freedom of each parameter (``DF``), computed with
        ``dfcom = sample_size - 1``. ``Lambda`` is the proportion of the total
        variance due to missingness and ``FMI`` the fraction of missing
        information, as in R mice.

        Parameters
        ----------
        alpha : float, default=0.05
            One minus the confidence level of the interval.

        Returns
        -------
        pd.DataFrame
            Summary table with estimates, standard errors, and diagnostics
        """
        tab = self.table(alpha=alpha)

        summary_df = pd.DataFrame(
            {
                "Parameter": self.param_names,
                "Estimate": tab["estimate"].to_numpy(),
                "Std_Error": tab["std_error"].to_numpy(),
                "DF": tab["df"].to_numpy(),
                "CI_Lower": tab["conf_low"].to_numpy(),
                "CI_Upper": tab["conf_high"].to_numpy(),
                "Within_Var": tab["ubar"].to_numpy(),
                "Between_Var": tab["b"].to_numpy(),
                "Total_Var": tab["t"].to_numpy(),
                "Lambda": tab["lambda"].to_numpy(),
                "FMI": tab["fmi"].to_numpy(),
            }
        )

        return summary_df

    def __str__(self) -> str:
        return f"PoolingResult(n_params={len(self.param_names)}, n_imputations={self.n_imputations})"

    def __repr__(self) -> str:
        return self.__str__()


def validate_imputed_datasets(datasets: list[pd.DataFrame]) -> None:
    """
    Validate that the input datasets are suitable for pooling.

    Parameters
    ----------
    datasets : list of pd.DataFrame
        List of imputed datasets to validate

    Raises
    ------
    ValueError
        If datasets are invalid for pooling
    """
    if not datasets:
        raise ValueError("No datasets provided for pooling")

    if len(datasets) < 1:
        raise ValueError("At least one dataset is required for pooling")

    # Check that all datasets are DataFrames
    for i, df in enumerate(datasets):
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Dataset {i} is not a pandas DataFrame")

    # Check consistent shapes and columns
    first_df = datasets[0]
    n_rows, n_cols = first_df.shape
    columns = first_df.columns

    for i, df in enumerate(datasets[1:], 1):
        if df.shape != (n_rows, n_cols):
            raise ValueError(f"Dataset {i} has shape {df.shape}, expected {(n_rows, n_cols)}")

        if not df.columns.equals(columns):
            raise ValueError(f"Dataset {i} has different columns than the first dataset")

    # Check for missing values (should not exist in imputed datasets)
    for i, df in enumerate(datasets):
        if df.isna().any().any():
            logger.warning(f"Dataset {i} contains missing values. Pooling assumes complete datasets.")


def apply_rubins_rules(
    estimates: np.ndarray, variances: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Rubin's rules to combine estimates and variances across imputations.

    Kept for existing callers; `pool_estimates` gives the full table with
    degrees of freedom, intervals and R's fraction of missing information.

    Parameters
    ----------
    estimates : np.ndarray
        Array of shape (n_imputations, n_parameters) with parameter estimates
    variances : np.ndarray
        Array of shape (n_imputations, n_parameters) with within-imputation variances

    Returns
    -------
    tuple
        (pooled_estimates, total_variances, within_variance, between_variance,
        lambda). The last element is ``(1 + 1/m) * b / t``, the proportion of
        the total variance due to missingness (R's ``lambda``). Before
        version 0.2 it was documented as the fraction of missing
        information, which in R is ``fmi``, a different quantity.
    """
    m = estimates.shape[0]  # number of imputations

    # Pooled estimates (q_bar)
    q_bar = np.nanmean(estimates, axis=0)

    # Average within-imputation variance (u_bar)
    u_bar = np.nanmean(variances, axis=0)

    # Between-imputation variance (b)
    if m > 1:
        b = np.nansum((estimates - q_bar) ** 2, axis=0) / (m - 1)
    else:
        b = np.zeros_like(q_bar)

    # Total variance (t)
    t = u_bar + (1.0 + 1.0 / max(m, 1)) * b

    # Fraction of missing information
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_miss_info = ((1.0 + 1.0 / max(m, 1)) * b) / t
        frac_miss_info = np.where(np.isfinite(frac_miss_info), frac_miss_info, np.nan)

    return q_bar, t, u_bar, b, frac_miss_info


def pool_descriptive_statistics(
    datasets: list[pd.DataFrame], include_numeric: bool = True, include_categorical: bool = True
) -> PoolingResult:
    """
    Pool descriptive statistics across multiple imputed datasets using Rubin's rules.

    For numeric columns, pools the sample mean and its variance.
    For categorical columns, pools the per-level proportions and their variances.
    Pooling follows `pool_estimates` with ``dfcom = n - 1``, so intervals use
    the t distribution with Barnard-Rubin degrees of freedom.

    Parameters
    ----------
    datasets : list of pd.DataFrame
        List of complete imputed datasets. All datasets must have the same
        shape and column names.
    include_numeric : bool, default=True
        Whether to include numeric columns in pooling
    include_categorical : bool, default=True
        Whether to include categorical columns in pooling

    Returns
    -------
    PoolingResult
        Object containing pooled estimates, variances, and diagnostic statistics

    Raises
    ------
    ValueError
        If datasets are invalid or no columns are available for pooling
    """
    logger.info(f"Starting pooling of {len(datasets)} imputed datasets")

    # Validate inputs
    validate_imputed_datasets(datasets)

    m = len(datasets)
    n = datasets[0].shape[0]

    logger.debug(f"Number of imputations: {m}, Sample size: {n}")

    # Identify column types from the first dataset
    first_df = datasets[0]
    numeric_cols = []
    categorical_cols = []

    if include_numeric:
        numeric_cols = first_df.select_dtypes(include=[np.number]).columns.tolist()

    if include_categorical:
        categorical_cols = first_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if not numeric_cols and not categorical_cols:
        raise ValueError("No numeric or categorical columns available for pooling.")

    logger.debug(f"Found {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns")

    # Build parameter vectors per imputed dataset
    param_names: list[str] = []
    estimates_list: list[list[float]] = [[] for _ in range(m)]
    variances_list: list[list[float]] = [[] for _ in range(m)]

    # 1) Numeric columns: mean and its within-imputation variance
    for col in numeric_cols:
        param_names.append(col)
        for j, df in enumerate(datasets):
            series = df[col]
            estimate = float(series.mean())
            # Within-imputation variance of the mean: var / n
            variance = float(series.var(ddof=1)) / n if n > 0 else np.nan
            estimates_list[j].append(estimate)
            variances_list[j].append(variance)

    # 2) Categorical columns: per-level proportions and their within-imputation variance
    for col in categorical_cols:
        # Determine stable set of levels across imputations
        all_levels = []
        for df in datasets:
            all_levels.extend(pd.unique(df[col]))

        # Create ordered, unique levels while preserving first occurrence order
        seen = set()
        levels: list[object] = []
        for lvl in all_levels:
            if lvl not in seen:
                seen.add(lvl)
                levels.append(lvl)

        for lvl in levels:
            lvl_name = f"{col}[{str(lvl)}]"
            param_names.append(lvl_name)
            for j, df in enumerate(datasets):
                # Proportion of rows equal to this level
                col_vals = df[col].to_numpy()
                p = float(np.mean(col_vals == lvl)) if n > 0 else np.nan
                # Variance of proportion: p(1-p)/n
                variance = p * (1.0 - p) / n if n > 0 else np.nan
                estimates_list[j].append(p)
                variances_list[j].append(variance)

    # Convert to numpy arrays
    estimates = np.asarray(estimates_list, dtype=float)
    variances = np.asarray(variances_list, dtype=float)

    # Apply Rubin's rules; a mean or a proportion has n - 1 complete-data df
    logger.debug("Applying Rubin's rules for pooling")
    table = pool_estimates(estimates, variances, dfcom=n - 1, param_names=param_names)

    # Log pooling statistics
    for param_name, row in table.iterrows():
        logger.debug(f"Pooling statistics for '{param_name}':")
        logger.debug(f"  - Pooled estimate: {row['estimate']:.4f}")
        logger.debug(f"  - Total variance: {row['t']:.4f}")
        logger.debug(f"  - Fraction of missing information: {row['fmi']:.4f}")

    logger.info("Pooling completed successfully")

    return PoolingResult(
        estimates=table["estimate"].to_numpy(),
        variances=table["t"].to_numpy(),
        within_variance=table["ubar"].to_numpy(),
        between_variance=table["b"].to_numpy(),
        frac_miss_info=table["fmi"].to_numpy(),
        param_names=param_names,
        n_imputations=m,
        sample_size=n,
    )


def pool_from_files(
    file_paths: Sequence[str | os.PathLike[str]], read_kwargs: dict[str, Any] | None = None, **pooling_kwargs: Any
) -> PoolingResult:
    """
    Pool descriptive statistics from datasets stored in files.

    Parameters
    ----------
    file_paths : sequence of str or path
        List of file paths to imputed datasets
    read_kwargs : dict, optional
        Keyword arguments to pass to pd.read_csv()
    **pooling_kwargs
        Additional arguments to pass to pool_descriptive_statistics()

    Returns
    -------
    PoolingResult
        Pooled results from the datasets
    """
    if read_kwargs is None:
        read_kwargs = {}

    logger.info(f"Loading {len(file_paths)} datasets from files")

    datasets = []
    for i, file_path in enumerate(file_paths):
        try:
            df = pd.read_csv(file_path, **read_kwargs)
            datasets.append(df)
            logger.debug(f"Loaded dataset {i + 1}: {file_path} (shape: {df.shape})")
        except Exception as e:
            logger.error(f"Failed to load dataset from {file_path}: {e}")
            raise

    return pool_descriptive_statistics(datasets, **pooling_kwargs)


def pool_subset(
    datasets: list[pd.DataFrame], columns: list[Hashable] | None = None, **pooling_kwargs: Any
) -> PoolingResult:
    """
    Pool descriptive statistics for a subset of columns.

    Parameters
    ----------
    datasets : list of pd.DataFrame
        List of complete imputed datasets
    columns : list of str, optional
        List of column names to include in pooling. If None, uses all columns.
    **pooling_kwargs
        Additional arguments to pass to pool_descriptive_statistics()

    Returns
    -------
    PoolingResult
        Pooled results for the specified columns
    """
    if columns is not None:
        # Validate that all specified columns exist
        first_df = datasets[0]
        missing_cols = [col for col in columns if col not in first_df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in datasets: {missing_cols}")

        # Subset datasets to specified columns
        datasets = [df[columns].copy() for df in datasets]
        logger.info(f"Pooling subset of {len(columns)} columns")

    return pool_descriptive_statistics(datasets, **pooling_kwargs)
