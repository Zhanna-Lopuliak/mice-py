import numpy as np
import pandas as pd
from typing import Union, Optional
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import logging

# Get a logger for the current module.
# This will be a child of the 'imputation' logger configured in MICE.py
logger = logging.getLogger('imputation.rf')

def rf(
    y: Union[pd.Series, np.ndarray],
    id_obs: np.ndarray,
    x: Union[pd.DataFrame, np.ndarray],
    id_mis: Optional[np.ndarray] = None,
    n_estimators: int = 10,
    rng: Optional[np.random.Generator] = None,
    **kwargs
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
        Predictor variables (should be the current completed columns)
    id_mis : np.ndarray, optional
        Boolean mask of missing values. If None, uses ~id_obs.
    n_estimators : int, default=10
        Number of trees in the forest
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None, a fresh generator is used.
    **kwargs : dict
        Additional parameters passed to the random forest model.
        
    Returns
    -------
    np.ndarray
        Imputed values for missing positions only.
        
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
    
    if rng is None:
        rng = np.random.default_rng()
    
    # One-hot encode categoricals
    if isinstance(x, pd.DataFrame) and (x.select_dtypes(include=["object", "category"]).shape[1] > 0):
        logger.debug("One-hot encoding categorical predictors.")
        x = pd.get_dummies(x, drop_first=True)
    
    x = np.asarray(x)
    id_obs = np.asarray(id_obs, bool)
    if id_mis is None:
        id_mis = ~id_obs
    
    x_obs, y_obs = x[id_obs], np.asarray(y)[id_obs]
    x_mis = x[id_mis]
    if x_mis.shape[0] == 0:
        return np.array([])
    
    if len(y_obs) < 2:
        logger.warning("Not enough observed data for RF. Using fallback imputation.")
        if pd.api.types.is_numeric_dtype(y_obs):
            return np.full(np.sum(id_mis), np.mean(y_obs))
        else:
            from collections import Counter
            most_frequent = Counter(y_obs).most_common(1)[0][0]
            return np.full(np.sum(id_mis), most_frequent)
    
    is_numeric = pd.api.types.is_numeric_dtype(y_obs)
    if is_numeric:
        logger.debug("Performing regression random forest imputation.")
        imputed_values = _rf_regression_impute(x_obs, x_mis, y_obs, n_estimators, rng, **kwargs)
    else:
        logger.debug("Performing classification random forest imputation.")
        imputed_values = _rf_classification_impute(x_obs, x_mis, y_obs, n_estimators, rng, **kwargs)
    
    logger.debug(f"Random Forest imputation complete. Imputed {len(imputed_values)} values.")
    return imputed_values

def _rf_regression_impute(x_obs, x_mis, y_obs, n_estimators, rng, **kwargs):
    """Helper for regression random forest imputation."""
    rf_seed = int(rng.integers(2**31 - 1))
    kwargs.pop("random_state", None)
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=kwargs.pop("max_depth", None),
        min_samples_leaf=kwargs.pop("min_samples_leaf", 1),
        random_state=rf_seed,
        n_jobs=-1,
        **kwargs
    )
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

def _rf_classification_impute(x_obs, x_mis, y_obs, n_estimators, rng, **kwargs):
    """Helper for classification random forest imputation."""
    unique_cats = np.unique(y_obs)
    if len(unique_cats) == 1:
        return np.repeat(unique_cats[0], x_mis.shape[0])
    
    rf_seed = int(rng.integers(2**31 - 1))
    kwargs.pop("random_state", None)
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=kwargs.pop("max_depth", None),
        min_samples_leaf=kwargs.pop("min_samples_leaf", 1),
        random_state=rf_seed,
        n_jobs=-1,
        **kwargs
    )
    rf.fit(x_obs, y_obs)
    
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
            vals, counts = np.unique(y_obs, return_counts=True)
            imputed[i] = vals[rng.choice(np.flatnonzero(counts == counts.max()))]
            fallback_count += 1
    
    if fallback_count:
        logger.debug(f"RF classification donor fallback used {fallback_count}/{len(imputed)} times.")
    return imputed