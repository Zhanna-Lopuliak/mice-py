import warnings
import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, Union, Optional, List
from datetime import datetime
import os
import statsmodels.formula.api as smf

# Get a logger for this module using the proper package hierarchy
# This will inherit configuration from the package logger when configured
logger = logging.getLogger('imputation.mice')

# Check if logging has been configured; if not, provide helpful guidance
def _check_logging_configured():
    """Check if package logging has been configured and provide guidance if not."""
    package_logger = logging.getLogger('imputation')
    
    # Check if the package logger has any handlers (other than NullHandler)
    has_real_handlers = any(
        not isinstance(handler, logging.NullHandler) 
        for handler in package_logger.handlers
    )
    
    if not has_real_handlers and not package_logger.propagate:
        # Only show warning once per session
        if not hasattr(_check_logging_configured, '_warned'):
            logger.warning(
                "No logging configured for imputation package. "
                "Call imputation.configure_logging() to enable logging, "
                "or imputation.disable_logging() to suppress this warning."
            )
            _check_logging_configured._warned = True


from .validators import (
    validate_dataframe,
    validate_columns,
    check_n_imputations,
    check_maxit,
    check_method,
    check_initial_method,
    validate_predictor_matrix,
    check_visit_sequence,
    validate_formula,
)
from .constants import (
    InitialMethod,
    DEFAULT_METHOD,
    DEFAULT_INITIAL_METHOD,
    VisitSequence,
)

# Import concrete imputation functions
from .utils import get_imputer_func
# External helpers
from .mice_result import MICEresult
from .pooling import pool_descriptive_statistics

class MICE:
    """
    Multiple Imputation by Chained Equations (MICE) class.
    
    This class implements the MICE algorithm for handling missing data through
    multiple imputations using chained equations.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data with missing values. Must be a pandas DataFrame.
        
    Attributes
    ----------
    data : pd.DataFrame
        The validated and cleaned input data
    id_obs : Dict[str, np.ndarray]
        Dictionary mapping column names to indices of observed values
    id_mis : Dict[str, np.ndarray]
        Dictionary mapping column names to indices of missing values
    """
    
    def __init__(self, data):
        """
        Initialize the MICE object.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data with missing values. Must be a pandas DataFrame.
            
        Raises
        ------
        ValueError
            If data is not a pandas DataFrame or contains duplicate column names
        """
        # Check if logging has been configured and provide guidance if needed
        _check_logging_configured()
        
        logger.info("Initializing MICE object")
        self.data = validate_dataframe(data)
        self.data = validate_columns(self.data)
        
        logger.debug(f"Input data shape: {self.data.shape}")
        self.id_obs = {}
        self.id_mis = {}
        missing_stats = {}
        
        for col in self.data.columns:
            notna = self.data[col].notna()
            self.id_obs[col] = notna
            self.id_mis[col] = ~notna
            missing_stats[col] = {
                'missing_count': (~notna).sum(),
                'missing_percentage': (~notna).mean() * 100
            }
            
        logger.debug("Missing value statistics:")
        for col, stats in missing_stats.items():
            logger.debug(f"  {col}: {stats['missing_count']} values ({stats['missing_percentage']:.2f}%) missing")
            
        # Container for pooled results
        self.result = None  # Will hold the pooled `MICEresult` instance
        self.run_output_dir = None
        
        # For storing analysis model results
        self.model_results = []

        # Required by statsmodels result wrappers
        self.nobs = self.data.shape[0]
        logger.info("MICE object initialized successfully")

    def impute(
        self,
        n_imputations: int = 5,
        maxit: int = 10,
        predictor_matrix: Optional[pd.DataFrame] = None,
        initial: str = DEFAULT_INITIAL_METHOD,
        method: Optional[Union[str, Dict[str, str]]] = None,
        visit_sequence: Union[str, List[str]] = "monotone",
        **kwargs
    ) -> None:
        """
        Perform multiple imputation by chained equations.
        
        Parameters
        ----------
        n_imputations : int, default=5
            Number of imputations to perform
            
        maxit : int, default=10
            Maximum number of iterations for each imputation cycle.
            Must be a positive integer.
            
        predictor_matrix : pd.DataFrame, optional
            Binary matrix indicating which variables should be used as predictors
            for each target variable. Should have column names as both index and columns.
            A 1 indicates that the column variable is used as predictor for the index variable.
            If None, a predictor matrix is estimated using `_quickpred`.
            
        initial : str, default=DEFAULT_INITIAL_METHOD
            Initial imputation method. Must be one of SUPPORTED_INITIAL_METHODS.
            
        method : Union[str, Dict[str, str]], optional
            Imputation method(s) to use:
            - str: use the same method for all columns
            - Dict[str, str]: dictionary mapping column names to their methods
            - None: use default method for all columns
            Must be one of SUPPORTED_METHODS.
            
        visit_sequence : Union[str, List[str]], default="monotone"
            Sequence in which variables should be visited during imputation:
            - str: "monotone" for monotone missing data pattern
            - List[str]: list of column names specifying the order to visit variables
            
        **kwargs : dict
            Additional keyword arguments.
            - `output_dir` (str, optional): Directory to save outputs for this run.
              If not provided, a timestamped folder is created in `output_figures`.
            
            Parameters for specific imputation methods can also be passed. These should
            be prefixed with the method name and an underscore, e.g., `pmm_donors=5` to pass
            `donors=5` to the `pmm` imputer.
            
            When `predictor_matrix` is not specified, the following can be passed for `_quickpred`:
            - `min_cor` (float, default=0.1): Minimum correlation for a predictor.
            - `min_puc` (float, default=0.0): Minimum proportion of usable cases.
            - `include` (list, optional): Columns to always include as predictors.
            - `exclude` (list, optional): Columns to always exclude as predictors.
            - `correlation_method` (str, default="pearson"): Correlation method used to
              compute the correlation matrix inside `_quickpred`.
        """
        logger.info("Starting imputation process")
        logger.debug(f"Parameters: n_imputations={n_imputations}, maxit={maxit}, "
                    f"initial={initial}, method={method}, visit_sequence={visit_sequence}")

        start_time = time.time()

        check_n_imputations(n_imputations)
        check_maxit(maxit)
        check_initial_method(initial)
        
        if predictor_matrix is None:
            min_cor = kwargs.pop('min_cor', 0.1)
            min_puc = kwargs.pop('min_puc', 0.0)
            include = kwargs.pop('include', None)
            exclude = kwargs.pop('exclude', None)
            correlation_method = kwargs.pop('correlation_method', 'pearson')
            predictor_matrix = self._quickpred(
                min_cor=min_cor, 
                min_puc=min_puc, 
                include=include, 
                exclude=exclude, 
                method=correlation_method
            )
        else:
            predictor_matrix = validate_predictor_matrix(predictor_matrix, list(self.data.columns), self.data)
            logger.debug("Predictor matrix validated successfully")
        
        if method is not None:
            self.method = check_method(method, list(self.data.columns))
        else:
            self.method = check_method(DEFAULT_METHOD, list(self.data.columns))
        logger.debug(f"Using imputation methods: {self.method}")

        # Store imputation parameters before using them
        self.imputation_params = kwargs

        # Warn if user provided method-specific parameters for methods not used
        if self.imputation_params:
            provided_prefixes = set()
            for key in self.imputation_params.keys():
                if '_' in key:
                    provided_prefixes.add(key.split('_', 1)[0])
            used_methods = set(self.method.values())
            unused_provided = provided_prefixes - used_methods
            if unused_provided:
                logger.warning(
                    "Method-specific parameters were provided for unused methods: %s. "
                    "These parameters will be ignored.",
                    sorted(list(unused_provided))
                )
        
        self.n_imputations = n_imputations
        self.maxit = maxit
        self.predictor_matrix = predictor_matrix
        self.initial = initial
        self.imputation_params = kwargs

        self._set_visit_sequence(visit_sequence)
        logger.debug(f"Visit sequence set to: {self.visit_sequence}")

        # Prepare chain statistics containers 
        # Only track statistics for numeric columns that will be imputed (i.e., have missing values)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        numeric_cols_to_impute = [col for col in self.visit_sequence if col in numeric_cols]
        
        self.chain_mean = {
            col: np.full((self.maxit, self.n_imputations), np.nan, dtype=float)
            for col in numeric_cols_to_impute
        }
        self.chain_var = {
            col: np.full((self.maxit, self.n_imputations), np.nan, dtype=float)
            for col in numeric_cols_to_impute
        }

        self.imputed_datasets = []
        individual_times = []

        for chain_idx in range(self.n_imputations):
            chain_start_time = time.time()
            logger.info(f"Starting imputation chain {chain_idx + 1}/{self.n_imputations}")
            self.imputed_datasets.append(self._impute_once(chain_idx))
            chain_end_time = time.time()
            chain_duration = chain_end_time - chain_start_time
            individual_times.append(chain_duration)
            logger.info(f"Completed imputation chain {chain_idx + 1} in {chain_duration:.2f} seconds")
        
        end_time = time.time()
        total_duration = end_time - start_time
        avg_chain_time = sum(individual_times) / len(individual_times)
        
        logger.info(f"All {self.n_imputations} imputations completed in {total_duration:.2f} seconds")
        logger.info(f"Average time per imputation chain: {avg_chain_time:.2f} seconds")
        logger.debug(f"Individual chain times: {[f'{t:.2f}s' for t in individual_times]}")
        
        logger.debug("Final imputation statistics:")
        logger.debug(f"  - Number of imputations: {self.n_imputations}")
        logger.debug(f"  - Maximum iterations: {self.maxit}")
        logger.debug(f"  - Initial method: {self.initial}")
        logger.debug(f"  - Method: {self.method}")
        logger.debug(f"  - Visit sequence: {self.visit_sequence}")
        logger.debug(f"  - Predictor matrix provided: {self.predictor_matrix is not None}")

        # Create a simple result object to hold the imputed datasets for backward compatibility
        class ImputationResult:
            def __init__(self, imputed_datasets):
                self.imputed_datasets = imputed_datasets
        
        self.result = ImputationResult(self.imputed_datasets)
        logger.debug("Created result object with imputed datasets")

        return self.imputed_datasets

    def _quickpred(
        self, 
        min_cor: float = 0.1, 
        min_puc: float = 0.0, 
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        method: str = "pearson"
    ) -> pd.DataFrame:
        """
        Generate a predictor matrix based on correlation and proportion of usable cases.
        
        This method is inspired by the `quickpred` function from the R `mice` package.
        
        Parameters
        ----------
        min_cor : float, default=0.1
            The minimum absolute correlation required to be included as a predictor.
        min_puc : float, default=0.0
            The minimum proportion of usable cases for correlation calculation.
        include : list of str, optional
            Columns to always include as predictors.
        exclude : list of str, optional
            Columns to always exclude as predictors.
        method : str, default="pearson"
            The correlation method to use ('pearson', 'kendall', 'spearman').
        
        Returns
        -------
        pd.DataFrame
            A square binary matrix indicating predictor relationships.
        """
        logger.info(f"Estimating predictor matrix with min_cor={min_cor}, min_puc={min_puc}, method='{method}'")
        
        predictor_matrix = pd.DataFrame(0, index=self.data.columns, columns=self.data.columns)
        
        # Calculate correlation matrix only for numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        cor_matrix = self.data[numeric_cols].corr(method=method)

        for target_col in self.data.columns:
            # Skip targets with no missing values
            if self.id_obs[target_col].all():
                continue

            for predictor_col in self.data.columns:
                if target_col == predictor_col:
                    continue

                # Proportion of usable cases
                puc = self.data[[target_col, predictor_col]].notna().all(axis=1).mean()

                if puc >= min_puc:
                    # Only use correlation if both columns are numeric
                    if target_col in cor_matrix.index and predictor_col in cor_matrix.columns:
                        correlation = cor_matrix.loc[target_col, predictor_col]
                        if abs(correlation) >= min_cor:
                            predictor_matrix.loc[target_col, predictor_col] = 1
                    else:
                        # For non-numeric columns, use them as predictors
                        predictor_matrix.loc[target_col, predictor_col] = 1
        
        # Handle include and exclude lists with validation for unknown columns
        if include:
            unknown_includes = [c for c in include if c not in predictor_matrix.columns]
            if unknown_includes:
                raise ValueError(f"_quickpred include contains unknown columns: {unknown_includes}")
            predictor_matrix.loc[:, include] = 1
        if exclude:
            unknown_excludes = [c for c in exclude if c not in predictor_matrix.columns]
            if unknown_excludes:
                raise ValueError(f"_quickpred exclude contains unknown columns: {unknown_excludes}")
            predictor_matrix.loc[:, exclude] = 0
            
        # Ensure diagonal is zero
        np.fill_diagonal(predictor_matrix.values, 0)
        
        logger.debug(f"Estimated predictor matrix:\n{predictor_matrix}")
        return predictor_matrix

    #def pool(self, summ: bool = False):
        # """Pool descriptive estimates across ``self.imputed_datasets`` using Rubin's rules.

        # This method is a convenience wrapper around the standalone pooling module.
        # For more flexibility, consider using ``imputation.pooling.pool_descriptive_statistics`` directly.

        # What is pooled
        # --------------
        # - Numeric columns: the sample mean per column.
        # - Categorical columns (object/category): the per-level proportions for each column.

        # Within-imputation variance
        # --------------------------
        # - Numeric: ``Var(mean) = s^2 / n`` (with ``ddof=1`` for ``s^2``).
        # - Categorical level proportion ``p``: ``Var(p) = p(1-p)/n``.

        # Notes
        # -----
        # - Cross-parameter covariances are ignored and a diagonal covariance matrix is constructed.
        # - Degrees of freedom small-sample adjustments are not applied.
        # - Categorical level parameter names are formatted as ``<column>[<level>]``.

        # Parameters
        # ----------
        # summ : bool, optional
        #     If True, return ``self.result.summary()``.
        # """
        # logger.info("Starting pooling of imputed datasets using standalone pooling module")

        # if not self.imputed_datasets:
        #     msg = "No imputed datasets found – run `.impute()` first."
        #     logger.error(msg)
        #     raise ValueError(msg)

        # # Use standalone pooling module
        # pooling_result = pool_descriptive_statistics(self.imputed_datasets)
        
        # # Convert standalone result to MICE-compatible result for backward compatibility
        # logger.debug("Converting pooling result to MICE-compatible format")
        
        # # Build diagonal covariance matrix
        # cov_params = np.diag(pooling_result.variances)
        
        # # Make parameter names available for summaries
        # self.exog_names = pooling_result.param_names

        # # Create results object compatible with existing MICE interface
        # logger.debug("Creating MICEresult object")
        # self.result = MICEresult(self, pooling_result.estimates, cov_params)
        # self.result.scale = 1.0
        # self.result.frac_miss_info = pooling_result.frac_miss_info
        
        # # Store the standalone pooling result for advanced users
        # self.pooling_result = pooling_result

        # logger.info("Pooling completed successfully using standalone module")

        # if summ:
        #     logger.debug("Generating summary")
        #     return self.result.summary()
    
    def fit(self, formula: str) -> None:
        """
        Fit a statistical model to each imputed dataset using the specified formula.
        
        This method fits the specified statistical model to each dataset in 
        self.imputed_datasets and stores the results in self.model_results.
        
        Parameters
        ----------
        formula : str
            A formula string in patsy syntax for statsmodels (e.g., 'y ~ x1 + x2')
            
        Raises
        ------
        ValueError
            If no imputed datasets are available or if variables in formula are not in data
            
        Examples
        --------
        >>> mice_obj = MICE(data)
        >>> mice_obj.impute(n_imputations=5)
        >>> mice_obj.fit('outcome ~ predictor1 + predictor2')
        """
        logger.info(f"Starting analysis with formula: {formula}")
        
        # Check if imputation has been performed
        if not hasattr(self, 'imputed_datasets') or not self.imputed_datasets:
            msg = "No imputed datasets found. Please run .impute() first."
            logger.error(msg)
            raise ValueError(msg)
        
        # Validate formula
        validate_formula(formula, list(self.data.columns))
        
        # Clear any previous model results
        self.model_results = []
        
        # Fit model to each imputed dataset
        n_datasets = len(self.imputed_datasets)
        logger.info(f"Fitting model to {n_datasets} imputed datasets")
        
        for i, dataset in enumerate(self.imputed_datasets):
            logger.debug(f"Fitting model to dataset {i + 1}/{n_datasets}")
            
            try:
                # Fit OLS model using statsmodels
                model = smf.ols(formula, data=dataset)
                fitted_model = model.fit()
                self.model_results.append(fitted_model)
                
                logger.debug(f"Successfully fitted model to dataset {i + 1}")
                
            except Exception as e:
                logger.error(f"Error fitting model to dataset {i + 1}: {str(e)}")
                raise RuntimeError(f"Failed to fit model to dataset {i + 1}: {str(e)}")
        
        # Store formula for potential later use
        self.formula = formula
        
        logger.info(f"Analysis completed successfully. Fitted models to {len(self.model_results)} datasets")
        logger.debug(f"Model results stored in self.model_results with {len(self.model_results)} entries")
        return self.model_results

    def pool(self, summ: bool = False):
        """
        Pool parameter estimates from fitted models using Rubin's rules.
        
        This method combines parameter estimates and their uncertainties from 
        multiple imputed datasets according to Rubin's (1987) rules for 
        multiple imputation inference.
        
        Parameters
        ----------
        summ : bool, default=False
            If True, returns a summary of the pooled results
            
        Returns
        -------
        MICEresult or summary
            If summ=False, returns a MICEresult object containing pooled estimates.
            If summ=True, returns a summary table of the pooled results.
            
        Raises
        ------
        ValueError
            If no model results are available from analysis
            
        Notes
        -----
        Rubin's pooling rules combine:
        - Point estimates: average across imputations
        - Within-imputation variance: average of individual model variances  
        - Between-imputation variance: variance of point estimates across imputations
        - Total variance: within + (1 + 1/m) * between
        - Fraction of missing information (FMI): proportion of uncertainty due to missingness
        
        References
        ----------
        Rubin, D.B. (1987). Multiple Imputation for Nonresponse in Surveys. 
        New York: John Wiley and Sons.
        """
        logger.info("Starting pooling of model results using Rubin's rules")
        
        # Check if analysis has been performed
        if not hasattr(self, 'model_results') or not self.model_results:
            msg = "No model results found. Please run .fit() first."
            logger.error(msg)
            raise ValueError(msg)
        
        # Check if formula was stored (should be set by fit())
        if not hasattr(self, 'formula'):
            logger.warning("No formula found. This may indicate .fit() was not called properly.")
        
        m = len(self.model_results)  # Number of imputations
        logger.info(f"Pooling estimates from {m} fitted models")
        
        # Extract parameters, covariances, and scales from each model
        params_list = []
        cov_within_list = []
        scale_list = []
        
        for i, model_result in enumerate(self.model_results):
            logger.debug(f"Extracting results from model {i + 1}")
            
            # Extract parameter estimates
            params_list.append(model_result.params.values)
            
            # Extract covariance matrix (within-imputation variance)
            cov_within_list.append(model_result.cov_params().values)
            
            # Extract scale (residual variance)
            scale_list.append(model_result.scale)
        
        # Convert to numpy arrays for easier computation
        params_array = np.array(params_list)  # Shape: (m, p) where p = number of parameters
        cov_within_array = np.array(cov_within_list)  # Shape: (m, p, p)
        scale_array = np.array(scale_list)
        
        logger.debug(f"Parameter array shape: {params_array.shape}")
        logger.debug(f"Covariance array shape: {cov_within_array.shape}")
        
        # Apply Rubin's pooling rules
        # 1. Pooled point estimates (qbar): average of individual estimates
        pooled_params = np.mean(params_array, axis=0)
        logger.debug(f"Computed pooled parameter estimates: {pooled_params}")
        
        # 2. Within-imputation variance (ubar): average of individual covariances
        cov_within = np.mean(cov_within_array, axis=0)
        
        # 3. Between-imputation variance (b): covariance of parameter estimates across imputations
        if m > 1:
            cov_between = np.cov(params_array, rowvar=False, ddof=1)
        else:
            cov_between = np.zeros_like(cov_within)
            logger.warning("Only one imputation available. Between-imputation variance set to zero.")
        
        # 4. Total covariance matrix using Rubin's rules
        # Total variance = within + (1 + 1/m) * between
        f = 1.0 + 1.0 / m  # Adjustment factor
        cov_total = cov_within + f * cov_between
        
        # 5. Fraction of missing information (FMI)
        # FMI = (1 + 1/m) * diag(between) / diag(total)
        if m > 1:
            fmi = f * np.diag(cov_between) / np.diag(cov_total)
            # Ensure FMI is between 0 and 1
            fmi = np.clip(fmi, 0.0, 1.0)
        else:
            fmi = np.zeros(len(pooled_params))
        
        # 6. Pooled scale (average of individual scales)
        pooled_scale = np.mean(scale_array)
        
        logger.debug(f"Computed within-imputation variance diagonal: {np.diag(cov_within)}")
        logger.debug(f"Computed between-imputation variance diagonal: {np.diag(cov_between)}")
        logger.debug(f"Computed total variance diagonal: {np.diag(cov_total)}")
        logger.debug(f"Computed fraction of missing information: {fmi}")
        logger.debug(f"Computed pooled scale: {pooled_scale}")
        
        # Create parameter names (use from first model)
        param_names = list(self.model_results[0].params.index)
        logger.debug(f"Parameter names: {param_names}")
        
        # Store results for backward compatibility
        self.exog_names = param_names
        if hasattr(self.model_results[0], 'model') and hasattr(self.model_results[0].model, 'endog_names'):
            self.endog_names = self.model_results[0].model.endog_names
        
        # Create MICEresult object
        logger.debug("Creating MICEresult object")
        from .mice_result import MICEresult
        
        # The MICEresult expects normalized covariance params (divided by scale)
        normalized_cov_params = cov_total / pooled_scale
        
        pooled_result = MICEresult(self, pooled_params, normalized_cov_params)
        pooled_result.scale = pooled_scale
        pooled_result.frac_miss_info = fmi
        
        # Store additional pooling diagnostics
        pooled_result.cov_within = cov_within
        pooled_result.cov_between = cov_between
        pooled_result.cov_total = cov_total
        pooled_result.m = m
        
        # Store the result
        self.pooled_result = pooled_result
        
        logger.info("Pooling completed successfully using Rubin's rules")
        logger.debug(f"Pooled estimates: {dict(zip(param_names, pooled_params))}")
        logger.debug(f"Fraction of missing information: {dict(zip(param_names, fmi))}")
        
        if summ:
            logger.debug("Generating summary")
            return pooled_result.summary()
        
        # Return comprehensive results for analysis
        comprehensive_result = {
            'pooled_result': pooled_result,
            'pooled_params': pooled_params,
            'pooled_covariance': cov_total,
            'within_covariance': cov_within,
            'between_covariance': cov_between,
            'fraction_missing_info': fmi,
            'pooled_scale': pooled_scale,
            'n_imputations': m,
            'parameter_names': param_names,
            'formula': getattr(self, 'formula', None)
        }
        
        return comprehensive_result
    
    def _impute_once(self, chain_idx: int):
        """
        Perform one complete imputation cycle.
        
        Returns
        -------
        pd.DataFrame
            A copy of the data with one complete imputation cycle applied
        """
        logger.debug(f"Starting imputation cycle for chain {chain_idx}")
        current_data = self.data.copy(deep=True)
        
        logger.debug("Performing initial imputation")
        # Create ONE RNG for this entire chain (matching R behavior)
        # The RNG state will advance through all iterations
        rng = np.random.default_rng(42 + chain_idx)
        self._initial_imputation(current_data, rng)

        for iter_idx in range(self.maxit):
            logger.debug(f"Starting iteration {iter_idx + 1}/{self.maxit} for chain {chain_idx}")
            # Pass the same RNG to each iteration - it will advance with each call
            current_data = self._iterate(current_data, iter_idx, chain_idx, rng)
            logger.debug(f"Completed iteration {iter_idx + 1}")
        
        logger.debug(f"Completed imputation cycle for chain {chain_idx}")
        return current_data
    
    def _iterate(self, data: pd.DataFrame, iter_idx: int, chain_idx: int, rng: np.random.Generator):
        """
        Perform one iteration of the imputation cycle.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to iterate over
        iter_idx : int
            Current iteration index
        chain_idx : int
            Current chain index
        rng : np.random.Generator
            Random number generator for this chain (state advances across iterations)

        Returns
        -------
        pd.DataFrame
            A copy of the data with one iteration of the imputation cycle applied
        """
        updated_data = data
        iteration_start_time = time.time()

        for col in self.visit_sequence:
            logger.debug(f"Processing column '{col}' (iteration {iter_idx + 1}, chain {chain_idx})")
            method_name = self.method[col]

            # Determine predictors
            if self.predictor_matrix is not None:
                predictor_flags = self.predictor_matrix.loc[col]
                predictor_cols = predictor_flags[predictor_flags == 1].index.tolist()
                predictor_cols = [c for c in predictor_cols if c != col]
            else:
                predictor_cols = [c for c in updated_data.columns if c != col]

            logger.debug(f"Using {len(predictor_cols)} predictors for column '{col}'")
            predictors = updated_data[predictor_cols]

            # Prepare arrays/masks
            y = updated_data[col].to_numpy()
            id_obs_mask = self.id_obs[col]
            id_mis_mask = self.id_mis[col]
            id_obs = id_obs_mask.to_numpy()
            id_mis = id_mis_mask.to_numpy()

            # Get imputer function and perform imputation
            imputer_func = get_imputer_func(method_name)
            logger.debug(f"Using imputation method '{method_name}' for column '{col}'")

            # Extract method-specific parameters from kwargs
            method_params = {}
            prefix = f"{method_name}_"
            for key, value in self.imputation_params.items():
                if key.startswith(prefix):
                    param_name = key[len(prefix):]
                    method_params[param_name] = value
            
            # Only pass rng if not using PMM
            if method_name != 'pmm':
                method_params['rng'] = rng
            
            if method_params:
                logger.debug(f"Passing parameters to imputer: {method_params}")

            imputed_values = imputer_func(y=y, id_obs=id_obs, id_mis=id_mis, x=predictors, **method_params)
            logger.debug(f"Successfully imputed {len(imputed_values)} values for column '{col}'")

            # Assign imputed values
            updated_data.loc[id_mis_mask, col] = imputed_values

            # Record chain statistics (only for numeric columns)
            if id_mis.sum() > 0 and col in self.chain_mean:
                imputed_arr = np.asarray(imputed_values, dtype=float)
                mean_val = np.nanmean(imputed_arr)
                self.chain_mean[col][iter_idx, chain_idx] = mean_val
                    
                if imputed_arr.size > 1:
                    var_val = np.nanvar(imputed_arr, ddof=1)
                    self.chain_var[col][iter_idx, chain_idx] = var_val
                    logger.debug(f"Chain statistics for '{col}': mean={mean_val:.4f}, variance={var_val:.4f}")
                else:
                    self.chain_var[col][iter_idx, chain_idx] = np.nan
                    logger.debug(f"Chain statistics for '{col}': mean={mean_val:.4f}, variance=N/A (single value)")

        iteration_time = time.time() - iteration_start_time
        logger.debug(f"Iteration {iter_idx + 1} completed in {iteration_time:.2f} seconds")
        return updated_data
  
    # def _initial_imputation(self, data):
    #     """
    #     Initialize missing values based on the initial method.
        
    #     Parameters
    #     ----------
    #     data : pd.DataFrame
    #         Data to initialize missing values in
    #     """
    #     if self.initial == InitialMethod.SAMPLE.value:
    #         for col in data.columns:
    #             if data[col].isna().any():
    #                 observed_values = data.loc[self.id_obs[col], col].values
    #                 # data.loc[self.id_mis[col], col] = np.random.choice(observed_values, size=self.id_mis[col].sum())
    #                 seed = 42
    #                 rng = np.random.default_rng(seed)  # independent generator
    #                 data.loc[self.id_mis[col], col] = rng.choice(
    #                     observed_values,
    #                     size=self.id_mis[col].sum()
    # )
    def _initial_imputation(self, data, rng=None):
        """
        Initialize missing values based on the initial method.
        """
        if rng is None:
            rng = np.random.default_rng()  # fresh random generator

        if self.initial == InitialMethod.SAMPLE.value:
            for col in data.columns:
                if data[col].isna().any():
                    observed_values = data.loc[self.id_obs[col], col].values
                    data.loc[self.id_mis[col], col] = rng.choice(
                        observed_values,
                        size=self.id_mis[col].sum()
                    )
                    
        elif self.initial == InitialMethod.MEANOBS.value:
            for col in data.columns:
                if data[col].isna().any():
                    col_mean = data[col].mean()
                    observed_values = data.loc[self.id_obs[col], col]
                    closest_idx = (observed_values - col_mean).abs().idxmin()
                    closest_value = data.loc[closest_idx, col]
                    data.loc[self.id_mis[col], col] = closest_value

    def _set_visit_sequence(self, visit_sequence):
        """
        Set the visit sequence for imputation based on the input parameter.
        
        Parameters
        ----------
        visit_sequence : Union[str, List[str]]
            Visit sequence specification. Can be:
            - str: "monotone" or "random" for predefined sequences
            - List[str]: list of column names specifying the order to visit variables
        """
        check_visit_sequence(visit_sequence, list(self.data.columns))
        
        if isinstance(visit_sequence, list):
            self.visit_sequence = visit_sequence
        else:
            columns_with_missing = [col for col in self.data.columns if self.data[col].isna().any()]
            
            if visit_sequence == VisitSequence.RANDOM.value:
                self.visit_sequence = list(np.random.permutation(columns_with_missing))
            elif visit_sequence == VisitSequence.MONOTONE.value:
                nmis = np.array([self.id_mis[col].sum() for col in columns_with_missing])
                ii = np.argsort(nmis)
                self.visit_sequence = [columns_with_missing[i] for i in ii]