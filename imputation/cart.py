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
    
    # Pre-process x to handle categorical predictors
    if isinstance(x, pd.DataFrame) and (x.select_dtypes(include=['object', 'category']).shape[1] > 0):
        logger.debug("One-hot encoding categorical predictors for column %s.", x.select_dtypes(include=['object', 'category']).columns[0])
        # One-hot encode categorical features, which is necessary for scikit-learn.
        # This mimics R's ability to handle factors.
        x = pd.get_dummies(x, drop_first=True)
    
    # Convert inputs to numpy arrays for consistency, y is handled later
    x = np.asarray(x)
    id_obs = np.asarray(id_obs, dtype=bool)
    
    # Set default id_mis if not provided
    if id_mis is None:
        id_mis = ~id_obs

    # Create random generator if not provided (matching R behavior)
    if rng is None:
        rng = np.random.default_rng()
    
    # Ensure minimum samples per leaf is at least 1
    min_samples_leaf = max(1, min_samples_leaf)
    
    # Add intercept if no predictors (matching R behavior)
    if x.shape[1] == 0:
        x = np.ones((len(x), 1))
    
    # Split data into observed and missing
    x_obs = x[id_obs].copy()
    x_mis = x[id_mis].copy()
    y_obs = y[id_obs]
    
    # Check if we have any missing values to impute
    if len(x_mis) == 0:
        # No missing values to impute, return empty array
        return np.array([])
    
    # Check if we have enough observed data to fit the model
    if len(y_obs) < 2:
        logger.warning("Not enough observed data to fit a tree. Using fallback imputation.")
        # Not enough observed data, use mean/sample for imputation
        is_numeric = pd.api.types.is_numeric_dtype(y_obs)
        if is_numeric:
            # Numeric case - use mean
            mean_val = np.mean(y_obs)
            imputed_values = np.full(np.sum(id_mis), mean_val)
        else:
            # Categorical case - use most frequent
            from collections import Counter
            # np.asarray is needed in case y_obs is a pandas Series
            most_frequent = Counter(np.asarray(y_obs)).most_common(1)[0][0]
            imputed_values = np.full(np.sum(id_mis), most_frequent)
        
        return imputed_values
    
    # Handle numeric and categorical variables differently
    is_numeric = pd.api.types.is_numeric_dtype(y_obs)
    if is_numeric:
        logger.debug("Performing regression tree imputation with bootstrap (matching R mice).")
        # Regression case with bootstrapping
        
        # Bootstrap the observed cases (matching R mice CART behavior)
        n_obs = len(y_obs)
        boot_idx = rng.choice(n_obs, size=n_obs, replace=True)
        x_boot = x_obs[boot_idx]
        y_boot = np.asarray(y_obs)[boot_idx]
        
        # Use rng to create a random seed for sklearn (which requires int seed)
        tree_seed = int(rng.integers(2**31 - 1))
        tree = DecisionTreeRegressor(
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
            random_state=tree_seed,
            **kwargs
        )
        
        # Fit the tree on bootstrap sample
        tree.fit(x_boot, y_boot)
        
        # Get leaf nodes for ORIGINAL observed data (not bootstrap)
        # This is important: we sample from original values, not bootstrap
        leaf_nodes = tree.apply(x_obs)
        
        # Get leaf nodes for missing data
        mis_leaf_nodes = tree.apply(x_mis)
        
        # For each missing value, sample from the same leaf node
        imputed_values = np.zeros(np.sum(id_mis))
        y_obs_arr = np.asarray(y_obs)
        fallback_count = 0
        
        for i, leaf in enumerate(mis_leaf_nodes):
            # Get all observed values in the same leaf
            leaf_values = y_obs_arr[leaf_nodes == leaf]
            
            if len(leaf_values) > 0:
                # Randomly sample one donor from the same leaf using rng
                imputed_values[i] = rng.choice(leaf_values)
            else:
                # Fallback: if node contains no observed donors (rare),
                # borrow from the overall distribution (matching R behavior)
                imputed_values[i] = rng.choice(y_obs_arr)
                fallback_count += 1
        
        if fallback_count > 0:
            logger.debug(f"CART regression fallback used {fallback_count}/{len(imputed_values)} times (empty leaf nodes).")
            
    else:
        logger.debug("Performing classification tree imputation with bootstrap (matching R mice).")
        # Classification case with bootstrapping
        
        # Check if all observed values are in one category (matching R behavior)
        unique_cats, _ = np.unique(y_obs, return_counts=True)
        if len(unique_cats) == 1:
            return np.repeat(unique_cats[0], np.sum(id_mis))
        
        # Bootstrap the observed cases (matching R mice CART behavior)
        n_obs = len(y_obs)
        boot_idx = rng.choice(n_obs, size=n_obs, replace=True)
        x_boot = x_obs[boot_idx]
        y_boot = np.asarray(y_obs)[boot_idx]
        
        # Use rng to create a random seed for sklearn (which requires int seed)
        tree_seed = int(rng.integers(2**31 - 1))
        tree = DecisionTreeClassifier(
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
            random_state=tree_seed,
            **kwargs
        )
        
        # Fit the tree on bootstrap sample
        tree.fit(x_boot, y_boot)
        
        # Get leaf nodes for ORIGINAL observed data (not bootstrap)
        # This is important: we sample from original values, not bootstrap
        leaf_nodes = tree.apply(x_obs)
        
        # Get leaf nodes for missing data
        mis_leaf_nodes = tree.apply(x_mis)
        
        # For each missing value, sample from donors in the same leaf node
        # (matching R implementation - donor sampling, not probability-based)
        imputed_values = np.empty(np.sum(id_mis), dtype=y_obs.dtype)
        y_obs_arr = np.asarray(y_obs)
        fallback_count = 0
        
        for i, leaf in enumerate(mis_leaf_nodes):
            # Get all observed values in the same leaf (donors)
            leaf_values = y_obs_arr[leaf_nodes == leaf]
            
            if len(leaf_values) > 0:
                # Randomly sample one donor from the same leaf using rng
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
