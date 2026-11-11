import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from typing import Union, Optional
import logging

from .design import obtain_design

logger = logging.getLogger('imputation.cart')


def _design_matrix(x) -> np.ndarray:
    """Predictors as a 2-D float array.

    A DataFrame with categorical, object, string or boolean columns is
    dummy-coded by :func:`imputation.design.obtain_design`, as MICE does;
    numeric input is only converted to float.
    """
    if isinstance(x, pd.DataFrame) and not x.select_dtypes(exclude=[np.number]).columns.empty:
        x = obtain_design(x, list(x.columns))
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def _is_numeric_target(y) -> bool:
    """True if `y` has a numeric dtype that is neither boolean nor categorical.

    Such a target gets a regression tree; any other target (categorical,
    object, string, boolean) a classification tree.
    """
    dtype = getattr(y, "dtype", None)
    if dtype is None:
        dtype = np.asarray(y).dtype
    return pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)


def _label_codes(y_obs: np.ndarray) -> np.ndarray:
    """Integer codes of the observed labels, in sorted label order if they sort.

    scikit-learn's classifiers accept numbers and strings as labels but not,
    for example, an object array of booleans. The trees are fitted on these
    codes; the imputations are drawn from the labels themselves. Sorted codes
    give the same tree as fitting on sortable labels directly.
    """
    try:
        codes, _ = pd.factorize(y_obs, sort=True)
    except TypeError:  # labels that cannot be ordered: order of appearance
        codes, _ = pd.factorize(y_obs, sort=False)
    return np.asarray(codes)

def cart(
    y: Union[pd.Series, np.ndarray],
    id_obs: np.ndarray,
    x: Union[pd.DataFrame, np.ndarray],
    id_mis: Optional[np.ndarray] = None,
    min_samples_leaf: int = 5,
    ccp_alpha: float = 1e-4,
    tree_params: Optional[dict] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Impute missing values using Classification and Regression Trees (CART).
    
    This function is designed to be compatible with the MICE framework.
    
    Parameters
    ----------
    y : Union[pd.Series, np.ndarray]
        Target variable with missing values
    id_obs : np.ndarray
        Boolean mask of observed values in y (True for observed, False for missing)
    x : Union[pd.DataFrame, np.ndarray]
        Predictor variables (must be fully observed). A DataFrame with
        categorical, object, string or boolean columns is dummy-coded first by
        :func:`imputation.design.obtain_design`, as MICE does. Without any
        predictor, the imputations are draws from the observed values.
    id_mis : np.ndarray, optional
        Boolean mask of missing values to impute. If None, uses ~id_obs
    min_samples_leaf : int, default=5
        Minimum number of samples required to be at a leaf node
    ccp_alpha : float, default=1e-4
        Complexity parameter for pruning
    tree_params : dict, optional
        Further arguments for scikit-learn's ``DecisionTreeRegressor`` or
        ``DecisionTreeClassifier``, e.g. ``{"max_depth": 10}``. It must not set
        ``min_samples_leaf``, ``ccp_alpha`` or ``random_state``.
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None, a fresh generator is used.
        
    Returns
    -------
    np.ndarray
        Imputed values for missing positions only (matching R implementation).
        A numeric target gets a regression tree. Any other target (categorical,
        object, string, boolean) gets a classification tree, and its
        imputations are observed labels of `y`, so booleans stay booleans.
        
    Notes
    -----
    The procedure follows R's mice CART implementation:
    1. Bootstrap the observed cases (sample with replacement)
    2. Fit a classification or regression tree on the bootstrap sample
    3. For each missing value, find the terminal node it would end up in
    4. Make a random draw from the ORIGINAL observed values in that node
    
    This adds stochasticity through both bootstrapping and donor sampling.
    """
    logger.debug("Starting CART imputation.")

    tree_params = dict(tree_params or {})
    reserved = sorted({"min_samples_leaf", "ccp_alpha", "random_state"} & set(tree_params))
    if reserved:
        raise TypeError(
            f"tree_params must not set {reserved}: pass min_samples_leaf and ccp_alpha "
            "as arguments of cart; the random state is drawn from rng."
        )
    
    x = _design_matrix(x)
    id_obs = np.asarray(id_obs, dtype=bool)
    
    if id_mis is None:
        id_mis = ~id_obs
    id_mis = np.asarray(id_mis, dtype=bool)

    if rng is None:
        rng = np.random.default_rng()
    
    min_samples_leaf = max(1, min_samples_leaf)
    
    if x.shape[1] == 0:
        x = np.ones((len(x), 1))
    
    x_obs = x[id_obs].copy()
    x_mis = x[id_mis].copy()
    is_numeric = _is_numeric_target(y)
    y_obs = np.asarray(y)[id_obs]
    
    if len(x_mis) == 0:
        return np.array([])
    
    if len(y_obs) < 2:
        logger.warning("Not enough observed data to fit a tree. Using fallback imputation.")
        if is_numeric:
            mean_val = np.mean(y_obs)
            imputed_values = np.full(np.sum(id_mis), mean_val)
        else:
            from collections import Counter
            most_frequent = Counter(np.asarray(y_obs)).most_common(1)[0][0]
            imputed_values = np.full(np.sum(id_mis), most_frequent)
        
        return imputed_values
    
    if is_numeric:
        logger.debug("Performing regression tree imputation with bootstrap (matching R mice).")
        
        n_obs = len(y_obs)
        boot_idx = rng.choice(n_obs, size=n_obs, replace=True)
        x_boot = x_obs[boot_idx]
        y_boot = np.asarray(y_obs)[boot_idx]
        
        tree_seed = int(rng.integers(2**31 - 1))
        tree = DecisionTreeRegressor(
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
            random_state=tree_seed,
            **tree_params
        )
        
        tree.fit(x_boot, y_boot)
        
        leaf_nodes = tree.apply(x_obs)
        mis_leaf_nodes = tree.apply(x_mis)
        
        imputed_values = np.zeros(np.sum(id_mis))
        y_obs_arr = np.asarray(y_obs)
        fallback_count = 0
        
        for i, leaf in enumerate(mis_leaf_nodes):
            leaf_values = y_obs_arr[leaf_nodes == leaf]
            
            if len(leaf_values) > 0:
                imputed_values[i] = rng.choice(leaf_values)
            else:
                imputed_values[i] = rng.choice(y_obs_arr)
                fallback_count += 1
        
        if fallback_count > 0:
            logger.debug(f"CART regression fallback used {fallback_count}/{len(imputed_values)} times (empty leaf nodes).")
            
    else:
        logger.debug("Performing classification tree imputation with bootstrap (matching R mice).")
        
        codes = _label_codes(y_obs)
        if codes.max() == 0:  # a single observed label
            return np.repeat(y_obs[:1], np.sum(id_mis))
        
        n_obs = len(y_obs)
        boot_idx = rng.choice(n_obs, size=n_obs, replace=True)
        x_boot = x_obs[boot_idx]
        y_boot = codes[boot_idx]
        
        tree_seed = int(rng.integers(2**31 - 1))
        tree = DecisionTreeClassifier(
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
            random_state=tree_seed,
            **tree_params
        )
        
        tree.fit(x_boot, y_boot)
        
        leaf_nodes = tree.apply(x_obs)
        mis_leaf_nodes = tree.apply(x_mis)
        
        imputed_values = np.empty(np.sum(id_mis), dtype=y_obs.dtype)
        y_obs_arr = np.asarray(y_obs)
        fallback_count = 0
        
        for i, leaf in enumerate(mis_leaf_nodes):
            leaf_values = y_obs_arr[leaf_nodes == leaf]
            
            if len(leaf_values) > 0:
                imputed_values[i] = rng.choice(leaf_values)
            else:
                # Fallback: if node contains no observed donors (rare),
                # borrow from the overall distribution (matching R behavior)
                imputed_values[i] = rng.choice(y_obs_arr)
                fallback_count += 1
        
        if fallback_count > 0:
            logger.debug(f"CART classification fallback used {fallback_count}/{len(imputed_values)} times (empty leaf nodes).")
    
    logger.debug(f"CART imputation complete. Imputed {len(imputed_values)} values.")
    return imputed_values
