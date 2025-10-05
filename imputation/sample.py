import numpy as np
import pandas as pd
from typing import Union, Optional

def sample(
    y: Union[pd.Series, np.ndarray],
    id_obs: np.ndarray,
    x: Union[pd.DataFrame, np.ndarray],
    id_mis: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    **kwargs
) -> np.ndarray:
    """
    Impute missing values by random sampling from observed values.
    
    This function is designed to be compatible with the MICE framework,
    following the same interface as PMM, midas, and CART imputation methods.
    
    Parameters
    ----------
    y : Union[pd.Series, np.ndarray]
        Target variable with missing values
    id_obs : np.ndarray
        Boolean mask of observed values in y (True for observed, False for missing)
    x : Union[pd.DataFrame, np.ndarray]
        Predictor variables (not used in this method, but kept for consistency)
    id_mis : np.ndarray, optional
        Boolean mask of missing values to impute. If None, uses ~id_obs
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None, a fresh generator is used.
    **kwargs : dict
        Additional arguments (not used in this method)
        
    Returns
    -------
    np.ndarray
        Imputed values for missing positions only (matching R implementation).
        
    Notes
    -----
    This is the simplest imputation method that:
    1. Takes all observed values in the target variable
    2. Randomly samples from them to fill in missing values
    3. No modeling is involved, just random sampling with replacement
    
    This method ignores the predictor variables (x) and only uses the observed
    values of the target variable for imputation.
    
    Edge cases handled (matching R implementation):
    - If no observed values: returns random normal values for numeric data,
      None values for categorical data
    - If only one observed value: duplicates it to allow sampling
    """
    # Convert boolean masks to numpy arrays, but preserve y's original type
    id_obs = np.asarray(id_obs, dtype=bool)
    
    # Set default id_mis if not provided
    if id_mis is None:
        id_mis = ~id_obs
    
    # Create random generator if not provided (matching R behavior)
    if rng is None:
        rng = np.random.default_rng()
    
    # Get observed values
    y_obs = y[id_obs]
    
    # Handle edge cases (matching R implementation)
    if len(y_obs) < 1:
        # If no observed values, handle based on data type
        n_mis = np.sum(id_mis)
        if hasattr(y, 'dtype') and y.dtype == 'object':
            # For categorical/string data, we can't generate meaningful values
            # Return None values that will need to be handled by the caller
            imputed_values = np.full(n_mis, None, dtype=object)
        else:
            # For numeric data, return random normal values using rng
            imputed_values = rng.normal(0, 1, n_mis)
    elif len(y_obs) == 1:
        # If only one observed value, duplicate it to allow sampling
        n_mis = np.sum(id_mis)
        imputed_values = np.full(n_mis, y_obs[0])
    else:
        # Normal case: sample from observed values using rng
        n_mis = np.sum(id_mis)
        imputed_values = rng.choice(y_obs, size=n_mis, replace=True)
    
    return imputed_values 