"""
Imputation of a target with two categories by logistic regression (``logreg``, ``logreg.boot``).

The ports of R mice's ``mice.impute.logreg`` and ``mice.impute.logreg.boot``.
Both fit a logistic regression of the observed target on ``[1, x]`` with
:func:`imputation.fitting.logit_fit` (R's ``glm.fit``) and draw each
imputation from the fitted probability of the second category. They differ
in how they treat the uncertainty of the fit:

- ``logreg``: the data are first augmented with a few weighted rows
  (:func:`augment`, White, Daniel and Royston 2010), which keeps the fit
  finite when the categories are separated by the predictors; then the
  coefficients are drawn from their approximate posterior. Proper.
- ``logreg.boot``: the fit on a bootstrap sample of the observed rows, no
  augmentation, no draw of the coefficients. Proper.

The target may be boolean, categorical, text, or numeric with two distinct
values; the imputations are its labels. :func:`target_categories` decides
which category is the first (the reference of the regression), with the
rule R follows after the augmentation: an ordered Categorical uses its
declared order, every other target its sorted labels.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .fitting import LogitFit, logit_fit
from .norm import _bootstrap_sample, _design_matrix, _masks, _zero_aliased
from .sampler import sym

#: R's ``augment`` refuses more categories than this (its argument ``maxcat``).
MAX_CATEGORIES = 50


class Categories(NamedTuple):
    """The categories of a target, as :func:`target_categories` orders them.

    Attributes
    ----------
    levels : list
        The categories, first to last.
    codes : np.ndarray of int
        For every entry of the target, the position of its category in
        `levels`; -1 where the target is missing.
    labels : np.ndarray
        `levels` as an array of the kind the imputers return for the
        target (the dtype of ``np.asarray(y)``), so that ``labels[codes]``
        gives imputations that MICE can write back into the column.
    """

    levels: list
    codes: np.ndarray
    labels: np.ndarray


class Augmented(NamedTuple):
    """Data augmented by :func:`augment`: the input rows, then the added rows.

    Attributes
    ----------
    y : np.ndarray of int
        Category codes.
    id_obs : np.ndarray of bool
        True where `y` is observed; the added rows count as observed.
    x : np.ndarray of float
        Predictors, without the intercept.
    id_mis : np.ndarray of bool
        The cells to impute; never an added row.
    weights : np.ndarray of float
        1 for the input rows, ``(p + 1) / (2 p k)`` for the added rows.
    """

    y: np.ndarray
    id_obs: np.ndarray
    x: np.ndarray
    id_mis: np.ndarray
    weights: np.ndarray


class _Steps(NamedTuple):
    """What `logreg` or `logreg_boot` computed; the parity tests compare it with R."""

    labels: np.ndarray  # the imputations
    fit: LogitFit | None  # None where no model was fitted (one category)
    beta: np.ndarray | None  # the coefficients used for the prediction (0 for aliased columns)
    prob: np.ndarray | None  # probability of the second category in each imputed cell


def logreg(
    y: ArrayLike,
    id_obs: ArrayLike,
    x: ArrayLike | pd.DataFrame,
    id_mis: ArrayLike | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Imputation by Bayesian logistic regression, as R mice's ``mice.impute.logreg``.

    The data are augmented with weighted rows (:func:`augment`), the
    logistic regression of the target on ``[1, x]`` is fitted to the
    observed and the added rows (:func:`imputation.fitting.logit_fit`), the
    coefficients are drawn from a normal distribution around the estimates
    with covariance ``(X' W X)^-1``, and each cell gets the second category
    with the probability these coefficients predict.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Target with two categories: boolean, categorical (``pd.Categorical``,
        possibly ordered), text, or numeric with two distinct values.
        Inside MICE the missing entries hold the current imputations; they
        enter only through the categories they hold (see :func:`augment`),
        not the fit.
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
    rng : np.random.Generator, optional
        Random number generator. If None, a fresh generator is used.

    Returns
    -------
    np.ndarray
        The imputations of the cells in `id_mis`, in their order: labels of
        `y` (``target_categories(y).labels``), in the dtype of
        ``np.asarray(y)``.

    Raises
    ------
    ValueError
        If `y` has more than two categories, if `y` has no observed value
        and some value is to be imputed, or if `x` is not numeric or has
        missing values.

    Notes
    -----
    Which category is the first, and so which label a random number maps
    to, follows :func:`target_categories`: the declared order of an ordered
    Categorical, the sorted labels otherwise. That is R's order after the
    augmentation, which rebuilds an unordered factor with sorted levels.

    Random numbers, in R's order: one standard normal variate per
    coefficient that is not aliased (``rng.standard_normal``), then one
    uniform number per imputed cell (``rng.random``); a cell gets the second
    category where its number is at most the predicted probability.

    The covariance of the coefficient draw is ``summary.glm(fit)$cov.unscaled``
    of R's quasibinomial fit, not scaled by the estimated dispersion, and it
    comes from the working weights of the last least-squares step, computed
    before the last update of the coefficients (see
    :func:`imputation.fitting.logit_fit`).

    Where R fails or misbehaves, this port goes on:

    - more than two categories: R stops (``invalid 'labels'``); here a
      ``ValueError`` names ``polyreg``;
    - a numeric target with two values other than 0 and 1: R stops
      (``y values must be 0 <= y <= 1``); here the smaller value is the
      first category;
    - a single category present in `y`: R imputes invented labels such as
      ``a2``; here every cell gets that category, without a fit and without
      random numbers;
    - columns that are aliased in the fit (for example a constant column,
      which the augmentation keeps constant): R stops (non-conformable
      arguments); here their coefficients are 0 and only the others are
      drawn;
    - an ordered Categorical with an unused declared level: R stops
      (``invalid 'labels'``); here the levels present are used.

    R adds the rows of a factor in the order of its declared levels; here
    they come in the order of :func:`target_categories`. That changes the
    fit only by rounding.

    References
    ----------
    White, I. R., Daniel, R. and Royston, P. (2010). Avoiding bias due to
    perfect prediction in multiple imputation of incomplete categorical
    variables. *Computational Statistics and Data Analysis*, 54, 2267-2275.

    Examples
    --------
    >>> y = np.array(["no", "yes", None, "no", "yes", "yes", None, "no"], dtype=object)
    >>> id_obs = pd.notna(y)
    >>> x = np.array([0.1, 1.2, 0.8, -0.4, 0.9, 2.0, -1.1, 0.3])
    >>> imputed = logreg(y, id_obs, x, rng=np.random.default_rng(1))
    >>> imputed.shape  # one label per missing cell
    (2,)
    >>> bool(np.isin(imputed, ["no", "yes"]).all())
    True
    """
    return _logreg(y, id_obs, x, id_mis, rng).labels


def logreg_boot(
    y: ArrayLike,
    id_obs: ArrayLike,
    x: ArrayLike | pd.DataFrame,
    id_mis: ArrayLike | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Imputation by logistic regression on a bootstrap sample, as R mice's ``mice.impute.logreg.boot``.

    The observed rows are resampled with replacement, the logistic
    regression of the target on ``[1, x]`` is fitted to that sample
    (:func:`imputation.fitting.logit_fit`), and each cell gets the second
    category with the probability the fitted coefficients predict. The
    bootstrap stands in for the coefficient draw of :func:`logreg`; there is
    no augmentation.

    Parameters
    ----------
    y, id_obs, x, id_mis, rng
        As in :func:`logreg`.

    Returns
    -------
    np.ndarray
        The imputations of the cells in `id_mis`, in their order, as in
        :func:`logreg`.

    Raises
    ------
    ValueError
        As :func:`logreg`.

    Notes
    -----
    Random numbers, in R's order: ``n_obs`` bootstrap indices
    (``rng.integers(0, n_obs, size=n_obs)``, R's
    ``sample(n1, n1, replace = TRUE)``), then one uniform number per imputed
    cell. A bootstrap sample whose categories are separated by the
    predictors, or that holds a single category, is fitted anyway, as in R:
    the fit stops at the iteration limit or with large coefficients, and the
    probabilities are close to 0 or 1.

    The categories are ordered as in :func:`logreg`. R's ``logreg.boot``,
    which does not augment, takes the declared level order of an unordered
    factor instead; the distribution of the imputations is the same, only
    the label a given random number maps to can differ.

    Where R fails or misbehaves, this port goes on:

    - more than two categories, two numeric values other than 0 and 1, a
      single category: as :func:`logreg`;
    - without predictors and with more than one cell to impute R stops
      (non-conformable arguments); here the intercept-only model is fitted;
    - columns aliased in the bootstrap sample: R imputes NA; here their
      coefficients are 0.

    Examples
    --------
    >>> y = np.array([False, True, None, False, True, True, None, False], dtype=object)
    >>> id_obs = pd.notna(y)
    >>> x = np.array([0.1, 1.2, 0.8, -0.4, 0.9, 2.0, -1.1, 0.3])
    >>> logreg_boot(y, id_obs, x, rng=np.random.default_rng(1)).shape
    (2,)
    """
    return _logreg_boot(y, id_obs, x, id_mis, rng).labels


def target_categories(y: Any, keep_unused: bool = False) -> Categories:
    """
    The categories of a target in the order the categorical imputers use.

    - An ordered ``pd.Categorical`` (or a Series of that dtype): its
      declared categories in declared order. Unused categories are dropped
      unless `keep_unused` is True.
    - Anything else (an unordered Categorical, booleans, text, numbers):
      the distinct values present in `y`, sorted (``False < True``, numbers
      by value, text by code point); unused categories of an unordered
      Categorical are always dropped.

    This is the order R's ``mice:::augment`` leaves a factor in: it keeps
    the levels of an ordered factor and rebuilds an unordered one from its
    labels, which sorts them and drops unused levels. The first category is
    the reference category of the regressions.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        The target. Missing entries (None, NaN, ``pd.NA``) are not
        categories. All entries count, also the missing cells that hold
        current imputations inside MICE.
    keep_unused : bool, default=False
        Keep the declared categories of an ordered Categorical that do not
        occur in `y` (R's ``polr`` counts every declared level).

    Returns
    -------
    Categories
        ``levels``, ``codes`` (-1 for missing entries) and ``labels``.

    Raises
    ------
    TypeError
        If the values of `y` are of mixed types that cannot be sorted.

    Examples
    --------
    >>> target_categories(np.array(["yes", "no", None, "yes"], dtype=object)).levels
    ['no', 'yes']
    >>> target_categories(np.array(["yes", "no", None, "yes"], dtype=object)).codes
    array([ 1,  0, -1,  1])
    >>> grade = pd.Categorical(["mid", "low", "mid"], categories=["low", "mid", "high"], ordered=True)
    >>> target_categories(grade).levels, target_categories(grade, keep_unused=True).levels
    (['low', 'mid'], ['low', 'mid', 'high'])
    """
    values = np.asarray(y)
    if isinstance(getattr(y, "dtype", None), pd.CategoricalDtype):
        categorical = pd.Categorical(y)
        declared = list(categorical.categories)
        declared_codes = np.asarray(categorical.codes, dtype=np.intp)
        used = np.unique(declared_codes[declared_codes >= 0])
        if categorical.ordered:
            chosen = np.arange(len(declared)) if keep_unused else used
        else:
            chosen = np.array(sorted(used, key=lambda i: _SortKey(declared[i])), dtype=np.intp)
        position = np.full(len(declared), -1, dtype=np.intp)
        position[chosen] = np.arange(len(chosen))
        codes = np.where(declared_codes >= 0, position[declared_codes], -1)
        levels = [declared[i] for i in chosen]
    else:
        missing = np.asarray(pd.isna(values), dtype=bool)
        levels = _sorted(pd.unique(values[~missing]))
        codes = np.full(len(values), -1, dtype=np.intp)
        codes[~missing] = np.asarray(pd.Categorical(values[~missing], categories=levels).codes, dtype=np.intp)
    return Categories(levels=levels, codes=codes, labels=_label_array(levels, values.dtype))


def augment(
    y: ArrayLike,
    id_obs: ArrayLike,
    x: ArrayLike,
    id_mis: ArrayLike,
    max_categories: int = MAX_CATEGORIES,
) -> Augmented:
    """
    Augment the data of a categorical imputation model, as R mice's ``mice:::augment``.

    White, Daniel and Royston (2010) add a few weighted rows to the data so
    that the categories are never separated by the predictors and the fit
    stays finite. With ``p`` predictors and the ``k`` categories present in
    `y`, ``2 p k`` rows are appended, in ``p`` blocks of ``2 k`` rows. In
    block ``j``, predictor ``j`` is ``mean_j + sd_j / 2`` and
    ``mean_j - sd_j / 2`` in turn, clamped to the range of predictor ``j``;
    the other predictors are at their means; the targets are the categories
    in increasing code order, each twice. Means, standard deviations
    (divisor ``n - 1``) and ranges are computed over all rows of `x`. The
    added rows count as observed, are not imputed, and have the weight
    ``(p + 1) / (2 p k)``; the input rows have weight 1.

    Parameters
    ----------
    y : array-like of int of shape (n_samples,)
        Category codes, for example ``target_categories(y).codes``; a
        negative code marks a missing entry. Every entry with a code counts
        for the categories present, also the cells to impute.
    id_obs : array-like of bool of shape (n_samples,)
        True where `y` is observed.
    x : array-like of shape (n_samples, n_features)
        Predictors without the intercept, without missing values.
    id_mis : array-like of bool of shape (n_samples,)
        The cells to impute (R's ``wy``).
    max_categories : int, default=50
        More categories than this raise ``ValueError`` (R's ``maxcat``).

    Returns
    -------
    Augmented
        The augmented ``y``, ``id_obs``, ``x``, ``id_mis`` and the weights.
        Without predictors, or when exactly one entry of `y` is not
        observed, the input is returned unchanged with all weights 1, as R
        does.

    Raises
    ------
    ValueError
        If `y` has more than `max_categories` categories.

    References
    ----------
    White, I. R., Daniel, R. and Royston, P. (2010). Avoiding bias due to
    perfect prediction in multiple imputation of incomplete categorical
    variables. *Computational Statistics and Data Analysis*, 54, 2267-2275.

    Examples
    --------
    >>> codes = np.array([0, 1, 1, 0, 1])
    >>> id_obs = np.array([True, True, False, True, False])
    >>> x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    >>> aug = augment(codes, id_obs, x, ~id_obs)
    >>> aug.x[5:, 0].round(4)  # mean 3, sd 1.5811: 3 +- 0.7906 for each category
    array([3.7906, 2.2094, 3.7906, 2.2094])
    >>> aug.y[5:], aug.weights[5:]
    (array([0, 0, 1, 1]), array([0.5, 0.5, 0.5, 0.5]))
    """
    y = np.asarray(y, dtype=np.intp).ravel()
    id_obs = np.asarray(id_obs, dtype=bool)
    id_mis = np.asarray(id_mis, dtype=bool)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    categories = np.unique(y[y >= 0])
    n_categories = categories.size
    if n_categories > max_categories:
        raise ValueError(
            f"augment: the target has {n_categories} categories, more than the maximum of {max_categories} "
            f"(R: 'Maximum number of categories ({max_categories}) exceeded')."
        )
    n_rows, n_predictors = x.shape
    if _augmentation_skipped(n_predictors, int((~id_obs).sum())):
        return Augmented(y=y, id_obs=id_obs, x=x, id_mis=id_mis, weights=np.ones(n_rows))

    mean = x.mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):  # one row: sd is NaN, as R's NA
        sd = np.sqrt(np.sum((x - mean) ** 2, axis=0) / (n_rows - 1))
    n_added = 2 * n_predictors * n_categories
    # shifts of +-1/2 sd on the block diagonal, 0 elsewhere (R's matrix b)
    shift = np.zeros((n_added, n_predictors))
    half = np.tile([0.5, -0.5], n_categories)
    for j in range(n_predictors):
        shift[2 * n_categories * j : 2 * n_categories * (j + 1), j] = half
    with np.errstate(invalid="ignore"):
        added = _clamp(mean + shift * sd, x.min(axis=0), x.max(axis=0))
    return Augmented(
        y=np.concatenate([y, np.tile(np.repeat(categories, 2), n_predictors)]),
        id_obs=np.concatenate([id_obs, np.ones(n_added, dtype=bool)]),
        x=np.vstack([x, added]),
        id_mis=np.concatenate([id_mis, np.zeros(n_added, dtype=bool)]),
        weights=np.concatenate([np.ones(n_rows), np.full(n_added, (n_predictors + 1) / max(n_added, 1))]),
    )


# ---------------------------------------------------------------------------
# The two methods, step by step
# ---------------------------------------------------------------------------


def _logreg(y: Any, id_obs: ArrayLike, x: Any, id_mis: ArrayLike | None, rng: np.random.Generator | None) -> _Steps:
    """`logreg`, returning its intermediate results too."""
    id_obs, id_mis = _masks(id_obs, id_mis, "logreg")
    categories = _two_categories(y, id_obs, "logreg")
    design = _design_matrix(x, len(categories.codes), "logreg")
    if not id_obs.any() or len(categories.levels) == 1:
        return _single_category(categories, id_mis)
    if rng is None:
        rng = np.random.default_rng()
    aug = augment(categories.codes, id_obs, design[:, 1:], id_mis)
    augmented = np.c_[np.ones(len(aug.y)), aug.x]
    fit = logit_fit(augmented[aug.id_obs], aug.y[aug.id_obs], weights=aug.weights[aug.id_obs])
    beta = _coefficient_draw(fit, rng)
    prob = _probabilities(augmented[aug.id_mis], beta)
    codes = _draw_codes(rng.random(int(id_mis.sum())), prob)
    return _Steps(labels=categories.labels[codes], fit=fit, beta=beta, prob=prob)


def _logreg_boot(
    y: Any, id_obs: ArrayLike, x: Any, id_mis: ArrayLike | None, rng: np.random.Generator | None
) -> _Steps:
    """`logreg_boot`, returning its intermediate results too."""
    id_obs, id_mis = _masks(id_obs, id_mis, "logreg.boot")
    categories = _two_categories(y, id_obs, "logreg.boot")
    design = _design_matrix(x, len(categories.codes), "logreg.boot")
    if not id_obs.any() or len(categories.levels) == 1:
        return _single_category(categories, id_mis)
    if rng is None:
        rng = np.random.default_rng()
    rows = _bootstrap_sample(int(id_obs.sum()), rng)
    fit = logit_fit(design[id_obs][rows], categories.codes[id_obs][rows])
    beta = _zero_aliased(fit.coef)
    prob = _probabilities(design[id_mis], beta)
    codes = _draw_codes(rng.random(int(id_mis.sum())), prob)
    return _Steps(labels=categories.labels[codes], fit=fit, beta=beta, prob=prob)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_categories(y: Any, id_obs: np.ndarray, method: str) -> Categories:
    """The categories of `y`; at most two, and every observed entry must have one."""
    categories = target_categories(y)
    if len(categories.levels) > 2:
        shown = ", ".join(repr(_plain(level)) for level in categories.levels[:5])
        more = ", ..." if len(categories.levels) > 5 else ""
        raise ValueError(
            f"{method} imputes a target with two categories, but y has {len(categories.levels)} "
            f"({shown}{more}). Use polyreg for more than two categories, or a donor method "
            "(pmm, cart, rf, sample)."
        )
    if np.any(categories.codes[id_obs] < 0):
        raise ValueError(f"{method}: y is missing in entries that id_obs marks as observed.")
    return categories


def _single_category(categories: Categories, id_mis: np.ndarray) -> _Steps:
    """Every cell in `id_mis` gets the one category present (R invents labels here); no fit, no draws."""
    codes = np.zeros(int(id_mis.sum()), dtype=np.intp)
    return _Steps(labels=categories.labels[codes], fit=None, beta=None, prob=None)


def _augmentation_skipped(n_predictors: int, n_not_observed: int) -> bool:
    """R's ``augment`` leaves the data alone without predictors or with exactly one entry not observed."""
    return n_predictors == 0 or n_not_observed == 1


def _clamp(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """`values` clamped to ``[low, high]`` column by column; NaN becomes `low` (R's ``pmax(..., na.rm = TRUE)``)."""
    return np.fmin(np.fmax(values, low), high)


def _coefficient_draw(fit: LogitFit, rng: np.random.Generator) -> np.ndarray:
    """The coefficients drawn from ``N(coef, cov_unscaled)``; 0 for aliased columns.

    R: ``coef + t(chol(sym(cov.unscaled))) %*% rnorm(q)``, with one standard
    normal variate per column that is not aliased.
    """
    kept = ~fit.aliased
    z = rng.standard_normal(int(kept.sum()))
    beta = np.zeros(fit.coef.size)
    beta[kept] = fit.coef[kept] + np.linalg.cholesky(sym(fit.cov_unscaled)) @ z
    return beta


def _probabilities(design: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Probability of the second category, ``1 / (1 + exp(-design @ beta))`` as R computes it."""
    with np.errstate(over="ignore"):  # exp overflows to inf: probability 0
        return 1 / (1 + np.exp(-(design @ beta)))


def _draw_codes(u: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """Category codes from uniform numbers: 1 (the second category) where ``u <= prob``, else 0.

    R's ``runif(n) <= p``: a number equal to the probability gives the
    second category.
    """
    return (np.asarray(u) <= np.asarray(prob)).astype(np.intp)


class _SortKey:
    """Sort key that reports values of mixed, unorderable types clearly."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __lt__(self, other: _SortKey) -> bool:
        try:
            return bool(self.value < other.value)
        except TypeError as err:
            raise TypeError(
                f"the target has values of mixed types that cannot be ordered as categories: "
                f"{self.value!r} and {other.value!r}"
            ) from err


def _sorted(values: Any) -> list:
    """The values sorted, as R sorts the labels of a factor (see `_SortKey`)."""
    return sorted(values, key=_SortKey)


def _plain(value: Any) -> Any:
    """A NumPy scalar as the Python scalar it holds (for messages); anything else unchanged."""
    return value.item() if isinstance(value, np.generic) else value


def _label_array(levels: list, dtype: np.dtype) -> np.ndarray:
    """`levels` as a 1-D array of `dtype`, one element per level."""
    labels = np.empty(len(levels), dtype=dtype)
    for i, level in enumerate(levels):
        labels[i] = level
    return labels
