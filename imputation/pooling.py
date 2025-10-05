"""
Standalone pooling module for multiple imputation results.

This module provides functions to pool descriptive statistics and model estimates
from multiple imputed datasets using Rubin's rules, without requiring coupling
to any specific imputation framework.
"""

import warnings
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

logger = logging.getLogger('imputation.pooling')


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
        Fraction of missing information for each parameter
    param_names : List[str]
        Names of the pooled parameters
    n_imputations : int
        Number of imputations used
    sample_size : int
        Sample size of each imputed dataset
    """
    estimates: np.ndarray
    variances: np.ndarray
    within_variance: np.ndarray
    between_variance: np.ndarray
    frac_miss_info: np.ndarray
    param_names: List[str]
    n_imputations: int
    sample_size: int
    
    def summary(self) -> pd.DataFrame:
        """
        Return a summary DataFrame with pooled statistics.
        
        Returns
        -------
        pd.DataFrame
            Summary table with estimates, standard errors, and diagnostics
        """
        std_errors = np.sqrt(self.variances)
        
        # Calculate confidence intervals (95% by default)
        alpha = 0.05
        z_score = 1.96  # For large samples, t-distribution approaches normal
        ci_lower = self.estimates - z_score * std_errors
        ci_upper = self.estimates + z_score * std_errors
        
        summary_df = pd.DataFrame({
            'Parameter': self.param_names,
            'Estimate': self.estimates,
            'Std_Error': std_errors,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'Within_Var': self.within_variance,
            'Between_Var': self.between_variance,
            'Total_Var': self.variances,
            'FMI': self.frac_miss_info
        })
        
        return summary_df
    
    def __str__(self) -> str:
        return f"PoolingResult(n_params={len(self.param_names)}, n_imputations={self.n_imputations})"
    
    def __repr__(self) -> str:
        return self.__str__()


def validate_imputed_datasets(datasets: List[pd.DataFrame]) -> None:
    """
    Validate that the input datasets are suitable for pooling.
    
    Parameters
    ----------
    datasets : List[pd.DataFrame]
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
        if df.isnull().any().any():
            logger.warning(f"Dataset {i} contains missing values. "
                          "Pooling assumes complete datasets.")


def apply_rubins_rules(
    estimates: np.ndarray, 
    variances: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Rubin's rules to combine estimates and variances across imputations.
    
    Parameters
    ----------
    estimates : np.ndarray
        Array of shape (n_imputations, n_parameters) with parameter estimates
    variances : np.ndarray
        Array of shape (n_imputations, n_parameters) with within-imputation variances
        
    Returns
    -------
    tuple
        (pooled_estimates, total_variances, within_variance, between_variance)
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
    with np.errstate(divide='ignore', invalid='ignore'):
        frac_miss_info = ((1.0 + 1.0 / max(m, 1)) * b) / t
        frac_miss_info = np.where(np.isfinite(frac_miss_info), frac_miss_info, np.nan)
    
    return q_bar, t, u_bar, b, frac_miss_info


def pool_descriptive_statistics(
    datasets: List[pd.DataFrame],
    include_numeric: bool = True,
    include_categorical: bool = True
) -> PoolingResult:
    """
    Pool descriptive statistics across multiple imputed datasets using Rubin's rules.
    
    For numeric columns, pools the sample mean and its variance.
    For categorical columns, pools the per-level proportions and their variances.
    
    Parameters
    ----------
    datasets : List[pd.DataFrame]
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
    
    if m == 1:
        warnings.warn("Number of multiple imputations m = 1. "
                     "Pooling will not reflect between-imputation uncertainty.")
    
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
    param_names: List[str] = []
    estimates_list: List[List[float]] = [[] for _ in range(m)]
    variances_list: List[List[float]] = [[] for _ in range(m)]
    
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
        levels: List[object] = []
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
    
    # Apply Rubin's rules
    logger.debug("Applying Rubin's rules for pooling")
    q_bar, t, u_bar, b, frac_miss_info = apply_rubins_rules(estimates, variances)
    
    # Log pooling statistics
    for i, param_name in enumerate(param_names):
        logger.debug(f"Pooling statistics for '{param_name}':")
        logger.debug(f"  - Pooled estimate: {q_bar[i]:.4f}")
        logger.debug(f"  - Total variance: {t[i]:.4f}")
        logger.debug(f"  - Fraction of missing information: {frac_miss_info[i]:.4f}")
    
    logger.info("Pooling completed successfully")
    
    return PoolingResult(
        estimates=q_bar,
        variances=t,
        within_variance=u_bar,
        between_variance=b,
        frac_miss_info=frac_miss_info,
        param_names=param_names,
        n_imputations=m,
        sample_size=n
    )


def pool_from_files(
    file_paths: List[str],
    read_kwargs: Optional[Dict] = None,
    **pooling_kwargs
) -> PoolingResult:
    """
    Pool descriptive statistics from datasets stored in files.
    
    Parameters
    ----------
    file_paths : List[str]
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
            logger.debug(f"Loaded dataset {i+1}: {file_path} (shape: {df.shape})")
        except Exception as e:
            logger.error(f"Failed to load dataset from {file_path}: {e}")
            raise
    
    return pool_descriptive_statistics(datasets, **pooling_kwargs)


def pool_subset(
    datasets: List[pd.DataFrame],
    columns: Optional[List[str]] = None,
    **pooling_kwargs
) -> PoolingResult:
    """
    Pool descriptive statistics for a subset of columns.
    
    Parameters
    ----------
    datasets : List[pd.DataFrame]
        List of complete imputed datasets
    columns : List[str], optional
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
