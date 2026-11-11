"""Imputation by random forests (``rf``), as R mice's ``mice.impute.rf``."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .cart import _design_matrix, _draw_pooled_donors, _is_numeric_target, _label_codes

logger = logging.getLogger(__name__)

# ranger's default min.node.size: a node is split only if it holds more
# observations than this (5 for regression, 1 for classification). The
# defaults of `min_samples_split` below follow the settings that matched
# ranger in the layer-2 comparisons: 5 for regression, 2 for classification.
_MIN_SAMPLES_SPLIT = {"regression": 5, "classification": 2}


def rf(
    y: pd.Series | np.ndarray,
    id_obs: np.ndarray,
    x: pd.DataFrame | np.ndarray,
    id_mis: np.ndarray | None = None,
    n_estimators: int = 10,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    min_samples_split: int | None = None,
    forest_params: dict | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Impute missing values using Random Forests with donor sampling.

    This function is designed to be compatible with the MICE framework and
    follows R's ``mice.impute.rf``, which grows the forest with ``ranger``.

    Parameters
    ----------
    y : pd.Series or np.ndarray
        Target variable with missing values
    id_obs : np.ndarray
        Boolean mask of observed values in y (True = observed, False = missing)
    x : pd.DataFrame or np.ndarray
        Predictor variables (should be the current completed columns). A
        DataFrame with categorical, object, string or boolean columns is
        dummy-coded first by :func:`imputation.design.obtain_design`, as MICE
        does. Without any predictor, the imputations are draws from the
        observed values, as with ``cart``.
    id_mis : np.ndarray, optional
        Boolean mask of missing values. If None, uses ~id_obs.
    n_estimators : int, default=10
        Number of trees in the forest (R mice's ``ntree``).
    max_depth : int, optional
        Maximum depth of the trees; None grows them until the leaves are pure
        or reach the node size limits, as ranger does.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node (ranger's
        ``min.bucket``).
    min_samples_split : int, optional
        Minimum number of samples in a node for a split to be tried. None
        means 5 for a numeric target and 2 for any other, after ranger's
        ``min.node.size`` (see Notes).
    forest_params : dict, optional
        Further arguments for scikit-learn's ``RandomForestRegressor`` or
        ``RandomForestClassifier``, e.g. ``{"n_jobs": -1}`` to grow the trees
        in parallel. It must not set ``n_estimators``, ``max_depth``,
        ``min_samples_leaf``, ``min_samples_split`` or ``random_state``.
        ``max_features`` defaults to ``"sqrt"`` (ranger's ``mtry``, the
        square root of the number of predictors, rounded down, at least 1);
        ``n_jobs`` is left at scikit-learn's default (one job).
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None, a fresh generator is used.

    Returns
    -------
    np.ndarray
        Imputed values for missing positions only. A numeric target gets a
        regression forest. Any other target (categorical, object, string,
        boolean) gets a classification forest, and its imputations are
        observed labels of `y`, so booleans stay booleans.

    Raises
    ------
    ValueError
        If `y` has no observed value and some value is to be imputed.

    Notes
    -----
    Algorithm (Doove et al., 2014), as R's ``mice.impute.rf``:

    1. Fit a random forest on the observed rows; each tree is grown on a
       bootstrap sample of them.
    2. For each missing row, find its leaf in every tree.
    3. Pool, over all trees, the observed values that share that leaf (all
       observed rows, whether in the tree's bootstrap sample or not; a row
       counts once per tree), and draw one value from the pool.

    ranger counts a node's size with the bootstrap duplicates and splits a
    node when it holds more than ``min.node.size`` observations;
    scikit-learn counts distinct rows. The defaults here are the settings
    that agreed with ranger in distribution on the layer-2 inputs; the
    forests are not the same.
    """
    logger.debug("Starting Random Forest imputation.")

    forest_params = dict(forest_params or {})
    reserved = {"n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "random_state"}
    if reserved & set(forest_params):
        raise TypeError(
            f"forest_params must not set {sorted(reserved & set(forest_params))}: pass "
            "n_estimators, max_depth, min_samples_leaf and min_samples_split as arguments "
            "of rf; the random state is drawn from rng."
        )

    if rng is None:
        rng = np.random.default_rng()

    x = _design_matrix(x)
    id_obs = np.asarray(id_obs, bool)
    if id_mis is None:
        id_mis = ~id_obs
    id_mis = np.asarray(id_mis, bool)

    is_numeric = _is_numeric_target(y)
    x_obs, y_obs = x[id_obs], np.asarray(y)[id_obs]
    x_mis = x[id_mis]
    if x_mis.shape[0] == 0:
        return np.array([])
    if len(y_obs) == 0:
        raise ValueError(
            "rf: the target has no observed value in the rows used for the model, so there is no donor to draw from."
        )

    if x.shape[1] == 0:
        # No predictor: every tree is a single leaf holding all observed
        # values, so the imputation is a draw from them (as cart does)
        logger.debug("No predictors: drawing from the observed values.")
        return rng.choice(y_obs, size=x_mis.shape[0])

    if is_numeric:
        logger.debug("Performing regression random forest imputation.")
        target = y_obs.astype(float)
        forest_class = RandomForestRegressor
        kind = "regression"
    else:
        logger.debug("Performing classification random forest imputation.")
        # fitted on label codes, so that any label type works (e.g. booleans)
        target = _label_codes(y_obs)
        if target.max() == 0:  # a single observed label
            return np.repeat(y_obs[:1], x_mis.shape[0])
        forest_class = RandomForestClassifier
        kind = "classification"

    forest = forest_class(
        n_estimators=max(1, int(n_estimators)),
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=_MIN_SAMPLES_SPLIT[kind] if min_samples_split is None else min_samples_split,
        random_state=int(rng.integers(2**31 - 1)),
        **{"max_features": "sqrt", **forest_params},
    )
    forest.fit(x_obs, target)

    imputed_values = _draw_pooled_donors(forest.apply(x_obs), forest.apply(x_mis), y_obs, rng)
    logger.debug(f"Random Forest imputation complete. Imputed {len(imputed_values)} values.")
    return imputed_values
