import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from typing import Union, Optional
import logging
logger = logging.getLogger('imputation.cart')

def cart(
    y: Union[pd.Series, np.ndarray],
    id_obs: np.ndarray,
    x: Union[pd.DataFrame, np.ndarray],
    id_mis: Optional[np.ndarray] = None,
    min_samples_leaf: int = 5,
    ccp_alpha: float = 1e-4,
    rng: Optional[np.random.Generator] = None,
    **kwargs
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
        Predictor variables (must be fully observed)
    id_mis : np.ndarray, optional
        Boolean mask of missing values to impute. If None, uses ~id_obs
    min_samples_leaf : int, default=5
        Minimum number of samples required to be at a leaf node
    ccp_alpha : float, default=1e-4
        Complexity parameter for pruning
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None, a fresh generator is used.
    **kwargs : dict
        Additional parameters passed to the tree model
        
    Returns
    -------
    np.ndarray
        Imputed values for missing positions only (matching R implementation).
        
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
    
    if isinstance(x, pd.DataFrame) and (x.select_dtypes(include=['object', 'category']).shape[1] > 0):
        logger.debug("One-hot encoding categorical predictors for column %s.", x.select_dtypes(include=['object', 'category']).columns[0])
        x = pd.get_dummies(x, drop_first=True)
    
    x = np.asarray(x)
    id_obs = np.asarray(id_obs, dtype=bool)
    
    if id_mis is None:
        id_mis = ~id_obs

    if rng is None:
        rng = np.random.default_rng()
    
    min_samples_leaf = max(1, min_samples_leaf)
    
    if x.shape[1] == 0:
        x = np.ones((len(x), 1))
    
    x_obs = x[id_obs].copy()
    x_mis = x[id_mis].copy()
    y_obs = y[id_obs]
    
    if len(x_mis) == 0:
        return np.array([])
    
    if len(y_obs) < 2:
        logger.warning("Not enough observed data to fit a tree. Using fallback imputation.")
        is_numeric = pd.api.types.is_numeric_dtype(y_obs)
        if is_numeric:
            mean_val = np.mean(y_obs)
            imputed_values = np.full(np.sum(id_mis), mean_val)
        else:
            from collections import Counter
            most_frequent = Counter(np.asarray(y_obs)).most_common(1)[0][0]
            imputed_values = np.full(np.sum(id_mis), most_frequent)
        
        return imputed_values
    
    is_numeric = pd.api.types.is_numeric_dtype(y_obs)
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
            **kwargs
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
        
        unique_cats, _ = np.unique(y_obs, return_counts=True)
        if len(unique_cats) == 1:
            return np.repeat(unique_cats[0], np.sum(id_mis))
        
        n_obs = len(y_obs)
        boot_idx = rng.choice(n_obs, size=n_obs, replace=True)
        x_boot = x_obs[boot_idx]
        y_boot = np.asarray(y_obs)[boot_idx]
        
        tree_seed = int(rng.integers(2**31 - 1))
        tree = DecisionTreeClassifier(
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
            random_state=tree_seed,
            **kwargs
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
