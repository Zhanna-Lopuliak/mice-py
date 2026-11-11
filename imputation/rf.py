import numpy as np
import pandas as pd
from typing import Union, Optional
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import logging

from .cart import _design_matrix, _is_numeric_target, _label_codes

# Get a logger for the current module.
# This will be a child of the 'imputation' logger configured in MICE.py
logger = logging.getLogger('imputation.rf')

def rf(
    y: Union[pd.Series, np.ndarray],
    id_obs: np.ndarray,
    x: Union[pd.DataFrame, np.ndarray],
    id_mis: Optional[np.ndarray] = None,
    n_estimators: int = 10,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
    forest_params: Optional[dict] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Impute missing values using Random Forests with donor sampling.
    
    This function is designed to be compatible with the MICE framework,
    following the same interface as PMM, midas, CART, and sample methods.
    
    Parameters
    ----------
    y : Union[pd.Series, np.ndarray]
        Target variable with missing values
    id_obs : np.ndarray
        Boolean mask of observed values in y (True = observed, False = missing)
    x : Union[pd.DataFrame, np.ndarray]
        Predictor variables (should be the current completed columns). A
        DataFrame with categorical, object, string or boolean columns is
        dummy-coded first by :func:`imputation.design.obtain_design`, as MICE
        does. Without any predictor, the imputations are draws from the
        observed values, as with ``cart``.
    id_mis : np.ndarray, optional
        Boolean mask of missing values. If None, uses ~id_obs.
    n_estimators : int, default=10
        Number of trees in the forest
    max_depth : int, optional
        Maximum depth of the trees; None grows them until the leaves are pure
        or reach ``min_samples_leaf``.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node
    forest_params : dict, optional
        Further arguments for scikit-learn's ``RandomForestRegressor`` or
        ``RandomForestClassifier``, e.g. ``{"max_features": 0.5}``. It must not
        set ``n_estimators``, ``max_depth``, ``min_samples_leaf`` or
        ``random_state``. ``n_jobs`` defaults to -1.
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None, a fresh generator is used.
        
    Returns
    -------
    np.ndarray
        Imputed values for missing positions only. A numeric target gets a
        regression forest. Any other target (categorical, object, string,
        boolean) gets a classification forest, and its imputations are
        observed labels of `y`, so booleans stay booleans.
        
    Notes
    -----
    Algorithm (Doove et al., 2014; mirrors R mice):
    1. Fit a random forest on observed data.
    2. For each missing case, find terminal nodes across all trees.
    3. For each tree, collect donors (observed cases in same node).
    4. Randomly sample one donor per tree.
    5. Take final imputation as a random draw from those donor predictions.
    
    Bootstrapping is inherent to Random Forest (bagging), so no additional
    bootstrap is applied (matching R mice behavior). Each tree is already
    built on a bootstrap sample of the data.
    """
    logger.debug("Starting Random Forest imputation.")

    forest_params = dict(forest_params or {})
    reserved = {"n_estimators", "max_depth", "min_samples_leaf", "random_state"}
    if reserved & set(forest_params):
        raise TypeError(
            f"forest_params must not set {sorted(reserved & set(forest_params))}: pass "
            "n_estimators, max_depth and min_samples_leaf as arguments of rf; the random "
            "state is drawn from rng."
        )
    forest_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "n_jobs": -1,
        **forest_params,
    }
    
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
    
    if len(y_obs) < 2:
        logger.warning("Not enough observed data for RF. Using fallback imputation.")
        if is_numeric:
            return np.full(np.sum(id_mis), np.mean(y_obs))
        else:
            from collections import Counter
            most_frequent = Counter(y_obs).most_common(1)[0][0]
            return np.full(np.sum(id_mis), most_frequent)
    
    if x.shape[1] == 0:
        # No predictor: every tree is a single leaf holding all observed
        # values, so the imputation is a draw from them (as cart does)
        logger.debug("No predictors: drawing from the observed values.")
        return rng.choice(y_obs, size=x_mis.shape[0])

    if is_numeric:
        logger.debug("Performing regression random forest imputation.")
        imputed_values = _rf_regression_impute(x_obs, x_mis, y_obs, rng, forest_params)
    else:
        logger.debug("Performing classification random forest imputation.")
        imputed_values = _rf_classification_impute(x_obs, x_mis, y_obs, rng, forest_params)
    
    logger.debug(f"Random Forest imputation complete. Imputed {len(imputed_values)} values.")
    return imputed_values

def _rf_regression_impute(x_obs, x_mis, y_obs, rng, forest_params):
    """Helper for regression random forest imputation."""
    rf_seed = int(rng.integers(2**31 - 1))
    rf = RandomForestRegressor(random_state=rf_seed, **forest_params)
    rf.fit(x_obs, y_obs)
    
    nodes_obs = np.array([t.apply(x_obs) for t in rf.estimators_]).T
    nodes_mis = np.array([t.apply(x_mis) for t in rf.estimators_]).T
    
    imputed = np.empty(x_mis.shape[0])
    fallback_count = 0
    
    for i in range(x_mis.shape[0]):
        donors_all = []
        for j, _ in enumerate(rf.estimators_):
            same = nodes_obs[:, j] == nodes_mis[i, j]
            if same.any():
                donors_all.append(rng.choice(y_obs[same]))
        if donors_all:
            imputed[i] = rng.choice(donors_all)
        else:
            imputed[i] = y_obs.mean()
            fallback_count += 1
    
    if fallback_count:
        logger.debug(f"RF regression donor fallback used {fallback_count}/{len(imputed)} times.")
    return imputed

def _rf_classification_impute(x_obs, x_mis, y_obs, rng, forest_params):
    """Helper for classification random forest imputation."""
    # fitted on label codes, so that any label type works (e.g. booleans)
    codes = _label_codes(y_obs)
    if codes.max() == 0:  # a single observed label
        return np.repeat(y_obs[:1], x_mis.shape[0])
    
    rf_seed = int(rng.integers(2**31 - 1))
    rf = RandomForestClassifier(random_state=rf_seed, **forest_params)
    rf.fit(x_obs, codes)
    
    nodes_obs = np.array([t.apply(x_obs) for t in rf.estimators_]).T
    nodes_mis = np.array([t.apply(x_mis) for t in rf.estimators_]).T
    
    imputed = np.empty(x_mis.shape[0], dtype=y_obs.dtype)
    fallback_count = 0
    
    for i in range(x_mis.shape[0]):
        donors_all = []
        for j, _ in enumerate(rf.estimators_):
            same = nodes_obs[:, j] == nodes_mis[i, j]
            if same.any():
                donors_all.append(rng.choice(y_obs[same]))
        if donors_all:
            imputed[i] = rng.choice(donors_all)
        else:
            counts = np.bincount(codes)
            first = np.unique(codes, return_index=True)[1]  # a label per code
            imputed[i] = y_obs[first[rng.choice(np.flatnonzero(counts == counts.max()))]]
            fallback_count += 1
    
    if fallback_count:
        logger.debug(f"RF classification donor fallback used {fallback_count}/{len(imputed)} times.")
    return imputed