"""
Imputation of an unordered categorical target by multinomial logistic regression (``polyreg``).

The port of R mice's ``mice.impute.polyreg``. The data are augmented with a
few weighted rows (:func:`imputation.logreg.augment`, White, Daniel and
Royston 2010), the multinomial logistic regression of the target on
``[1, x]`` is fitted to the observed and the added rows as R's
``nnet::multinom`` fits it (:func:`imputation.fitting.multinom_fit`), and
each cell gets a category drawn from the probabilities it predicts. There
is no draw of the coefficients: like R's ``polyreg``, the method ignores
the uncertainty of the fit apart from the augmentation.

The target may be categorical (ordered or not), text, boolean or numeric,
with any number of categories up to 50; the imputations are its labels.
:func:`imputation.logreg.target_categories` orders the categories as R does
after the augmentation: an ordered Categorical in its declared order, every
other target by its sorted labels.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .fitting import MultinomFit, multinom_fit
from .logreg import Categories, augment, target_categories
from .norm import _design_matrix, _masks


class _Steps(NamedTuple):
    """What `polyreg` computed; the parity tests compare it with R."""

    labels: np.ndarray  # the imputations
    fit: MultinomFit | None  # None where no model was fitted (one category)
    prob: np.ndarray | None  # probabilities of the categories in each imputed cell, (n_imp, n_categories)


def polyreg(
    y: ArrayLike,
    id_obs: ArrayLike,
    x: ArrayLike | pd.DataFrame,
    id_mis: ArrayLike | None = None,
    maxit: int = 100,
    max_weights: int = 1500,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Imputation by multinomial logistic regression, as R mice's ``mice.impute.polyreg``.

    The data are augmented with weighted rows
    (:func:`imputation.logreg.augment`), the multinomial logistic regression
    of the target on ``[1, x]`` is fitted to the observed and the added rows
    with nnet's minimiser and stopping rules
    (:func:`imputation.fitting.multinom_fit`), and each cell gets a category
    drawn with the probabilities the fit predicts for it.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Categorical target: a ``pd.Categorical`` (possibly ordered), text,
        booleans, or numbers, with at most 50 categories. Inside MICE the
        missing entries hold the current imputations; they enter only
        through the categories they hold (see
        :func:`imputation.logreg.augment`), not the fit.
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
    maxit : int, default=100
        Maximal number of iterations of the fit (R's ``nnet.maxit``). The
        fit stops there without an error, as in R.
    max_weights : int, default=1500
        Maximal number of weights of the model (R's ``nnet.MaxNWts``):
        ``k (q + 1)`` for ``k`` categories and ``q`` columns of ``[1, x]``,
        ``q + 1`` for two categories.
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
        If `y` has more than 50 categories (R: "Maximum number of
        categories (50) exceeded"), if the model has more than
        `max_weights` weights (R: "too many (N) weights"), if `y` has no
        observed value and some value is to be imputed, or if `x` is not
        numeric or has missing values.

    Notes
    -----
    Random numbers, in R's order: one uniform number per imputed cell
    (``rng.random``), drawn after the augmentation and before anything is
    fitted, so also when a single category is observed. A cell gets the
    category ``k`` (counting from 0) for which its number exceeds ``k`` of
    the cumulative probabilities (R's ``un > cumsum``): a number equal to a
    cumulative probability gives the lower category. The fit itself draws
    nothing (all coefficients start at 0).

    If the observed rows, the added ones included, hold a single category,
    every cell gets it, without a fit.

    Which category is the first, and so which label a random number maps
    to, follows :func:`imputation.logreg.target_categories`: the declared
    order of an ordered Categorical, the sorted labels otherwise. That is
    R's order after the augmentation; where R skips the augmentation
    (without predictors, or with exactly one value missing) it keeps the
    declared order of an unordered factor, which changes no distribution,
    only the label a given random number maps to.

    Without predictors the model matrix is ``[1, 1]``, as mice adds a
    constant column ``int`` next to the intercept; the probabilities are
    those of the intercept-only model, reached on R's path.

    Where R fails or misbehaves, this port goes on:

    - two categories and exactly one cell to impute: R stops
      (``dim(X) must have a positive length``); here the cell is imputed;
    - an ordered Categorical with an unused declared level: R stops
      (``dims ... do not match``); here the levels present are used;
    - a category held only by cells that are not observed, where the
      augmentation is skipped: R's ``multinom`` drops it and R then stops;
      here the categories with observed rows are fitted and that category
      gets probability 0;
    - a uniform number above the last cumulative probability, which is 1
      up to rounding: R imputes NA; here the last category (the last one
      with a positive probability).

    References
    ----------
    White, I. R., Daniel, R. and Royston, P. (2010). Avoiding bias due to
    perfect prediction in multiple imputation of incomplete categorical
    variables. *Computational Statistics and Data Analysis*, 54, 2267-2275.

    Examples
    --------
    >>> y = np.array(["b", "a", None, "c", "a", "b", None, "c", "c", "a"], dtype=object)
    >>> id_obs = pd.notna(y)
    >>> x = np.array([0.4, -1.2, 0.8, 1.1, -0.9, 0.2, -1.4, 1.6, 0.9, -0.3])
    >>> imputed = polyreg(y, id_obs, x, rng=np.random.default_rng(1))
    >>> imputed.shape  # one label per missing cell
    (2,)
    >>> bool(np.isin(imputed, ["a", "b", "c"]).all())
    True
    """
    return _polyreg(y, id_obs, x, id_mis, maxit, max_weights, rng).labels


def _polyreg(
    y: Any,
    id_obs: ArrayLike,
    x: Any,
    id_mis: ArrayLike | None,
    maxit: int,
    max_weights: int,
    rng: np.random.Generator | None,
) -> _Steps:
    """`polyreg`, returning its intermediate results too."""
    id_obs, id_mis = _masks(id_obs, id_mis, "polyreg")
    categories = _observed_categories(y, id_obs)
    design = _design_matrix(x, len(categories.codes), "polyreg")
    if not id_obs.any():  # and nothing to impute (_masks raises otherwise)
        return _Steps(labels=categories.labels[:0], fit=None, prob=None)
    if rng is None:
        rng = np.random.default_rng()
    aug = augment(categories.codes, id_obs, design[:, 1:], id_mis)
    u = _uniforms(rng, int(id_mis.sum()))
    observed = aug.y[aug.id_obs]
    present = np.unique(observed)
    if present.size == 1:
        codes = np.full(u.size, present[0], dtype=np.intp)
        return _Steps(labels=categories.labels[codes], fit=None, prob=None)
    model = np.c_[np.ones(len(aug.y)), aug.x]
    if aug.x.shape[1] == 0:
        model = np.c_[model, np.ones(len(aug.y))]  # mice's constant column `int`
    fit = multinom_fit(
        model[aug.id_obs],
        observed,
        weights=aug.weights[aug.id_obs],
        n_classes=len(categories.levels),
        maxit=maxit,
        max_weights=max_weights,
    )
    prob = fit.predict(model[aug.id_mis])
    codes = _draw_categories(u, prob)
    return _Steps(labels=categories.labels[codes], fit=fit, prob=prob)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _observed_categories(y: Any, id_obs: np.ndarray) -> Categories:
    """The categories of `y` (:func:`imputation.logreg.target_categories`); every observed entry must have one."""
    categories = target_categories(y)
    if np.any(categories.codes[id_obs] < 0):
        raise ValueError("polyreg: y is missing in entries that id_obs marks as observed.")
    return categories


def _uniforms(rng: np.random.Generator, n: int) -> np.ndarray:
    """One uniform number per imputed cell, R's ``runif(sum(wy))``.

    R draws them before it looks at the categories or fits anything.
    """
    return rng.random(n)


def _draw_categories(u: ArrayLike, prob: ArrayLike) -> np.ndarray:
    """Category codes from uniform numbers and the probabilities of each cell (one row per cell).

    R's ``1 + colSums(un > apply(post, 1, cumsum))``: the code of a cell is
    the number of its cumulative probabilities that its number exceeds, so
    a number equal to a cumulative probability gives the lower category.
    A number above the last cumulative probability (1 up to rounding),
    where R imputes NA, gives the last category with a positive
    probability.
    """
    u = np.asarray(u, dtype=float)
    prob = np.asarray(prob, dtype=float)
    cumulative = np.cumsum(prob, axis=1)
    codes = np.sum(u[:, None] > cumulative, axis=1)
    last_positive = prob.shape[1] - 1 - np.argmax(prob[:, ::-1] > 0, axis=1)
    return np.minimum(codes, last_positive).astype(np.intp)
