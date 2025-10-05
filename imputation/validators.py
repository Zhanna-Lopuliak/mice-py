import pandas as pd
import warnings
from typing import Dict, Union, List
from .constants import ImputationMethod, SUPPORTED_METHODS, DEFAULT_METHOD, InitialMethod, SUPPORTED_INITIAL_METHODS, DEFAULT_INITIAL_METHOD, VisitSequence, SUPPORTED_VISIT_SEQUENCES
import numpy as np
import re

def check_n_imputations(n_imputations: int) -> None:
    """
    Check if the number of imputations is valid and provide a warning if it's high.
    
    Parameters
    ----------
    n_imputations : int
        Number of imputations to perform
        
    Raises
    ------
    ValueError
        If n_imputations is not a positive integer
    """
    if not isinstance(n_imputations, int):
        raise ValueError("n_imputations must be a positive integer")
    
    if n_imputations <= 0:
        raise ValueError("n_imputations must be a positive integer")
        
    if n_imputations > 100:
        print(f"Warning: {n_imputations} imputations is a large number. This might take a while to compute.")

def check_maxit(maxit: int) -> None:
    """
    Check if the maximum iterations parameter is valid and provide a warning if it's high.
    
    Parameters
    ----------
    maxit : int
        Maximum number of iterations for each imputation cycle
        
    Raises
    ------
    ValueError
        If maxit is not a positive integer
    """
    if not isinstance(maxit, int):
        raise ValueError("maxit must be an integer")
    
    if maxit <= 0:
        raise ValueError("maxit must be a positive integer")
        
    if maxit > 50:
        warnings.warn("maxit is greater than 50, imputations will take a lot of time", UserWarning)

def check_method(method: Union[str, Dict[str, str]], columns: List[str]) -> Dict[str, str]:
    """
    Check and process the method parameter for MICE imputation.
    
    Parameters
    ----------
    method : Union[str, Dict[str, str]]
        Method specification. Can be:
        - str: use the same method for all columns
        - Dict[str, str]: dictionary mapping column names to their methods
    columns : List[str]
        List of column names in the data
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping each column to its imputation method
        
    Raises
    ------
    ValueError
        If method is invalid or references non-existent columns
    """
    # If method is a string, validate and use for all columns
    if isinstance(method, str):
        if not ImputationMethod.is_valid_method(method):
            raise ValueError(f"Unsupported method: {method}. Supported methods are: {SUPPORTED_METHODS}")
        return {col: method for col in columns}
    
    # If method is a dictionary, validate each entry
    if isinstance(method, dict):
        # Check if all specified columns exist
        invalid_cols = [col for col in method.keys() if col not in columns]
        if invalid_cols:
            raise ValueError(f"Columns not found in data: {invalid_cols}")
        
        # Check if all methods are supported
        invalid_methods = {col: m for col, m in method.items() if not ImputationMethod.is_valid_method(m)}
        if invalid_methods:
            raise ValueError(f"Unsupported methods: {invalid_methods}. Supported methods are: {SUPPORTED_METHODS}")
        
        # Create result dict with default method for unspecified columns
        # TODO: make default method dependent on column type or just handle it otherwise, 
        # e.g. not let method for a column be not specified
        result = {col: DEFAULT_METHOD for col in columns}  # Default method
        result.update(method)  # Override with specified methods
        return result
    
    raise ValueError("method must be either a string or a dictionary")

def check_initial_method(initial_method: str) -> None:
    """
    Check if the initial imputation method is valid.
    
    Parameters
    ----------
    initial_method : str
        Initial imputation method to validate
        
    Raises
    ------
    ValueError
        If initial_method is not a valid initial imputation method
    """
    if not isinstance(initial_method, str):
        raise ValueError("initial_method must be a string")
    
    if not InitialMethod.is_valid_method(initial_method):
        raise ValueError(f"Unsupported initial method: {initial_method}. Supported initial methods are: {SUPPORTED_INITIAL_METHODS}")

def check_visit_sequence(visit_sequence: Union[str, List[str]], columns: List[str]) -> None:
    """
    Check and process the visit sequence parameter for MICE imputation.
    
    Parameters
    ----------
    visit_sequence : Union[str, List[str]]
        Visit sequence specification. Can be:
        - str: "monotone" or "random" for predefined sequences
        - List[str]: list of column names specifying the order to visit variables
    columns : List[str]
        List of column names in the data
        
    Raises
    ------
    ValueError
        If visit_sequence is invalid or references non-existent columns
    """

    if isinstance(visit_sequence, str):
        if not VisitSequence.is_valid_sequence(visit_sequence):
            raise ValueError(f"Unsupported visit sequence: {visit_sequence}. Supported visit sequences are: {SUPPORTED_VISIT_SEQUENCES}")
        return

    if isinstance(visit_sequence, list):
        invalid_cols = [col for col in visit_sequence if col not in columns]
        if invalid_cols:
            raise ValueError(f"Visit sequence contains columns not in data: {invalid_cols}")
        
        missing_cols = [col for col in columns if col not in visit_sequence]
        if missing_cols:
            raise ValueError(f"Missing columns in visit sequence: {missing_cols}. All data columns must be included.")
        
        return
    
    raise ValueError("visit_sequence must be either a string or a list of strings")

def validate_predictor_matrix(predictor_matrix: pd.DataFrame, data_columns: List[str], data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate predictor matrix for MICE imputation.
    
    Parameters
    ----------
    predictor_matrix : pd.DataFrame
        Binary matrix indicating which variables should be used as predictors
        for each target variable. Rows represent target variables, columns represent predictors.
        A 1 indicates that the column variable is used as predictor for the index variable.
    data_columns : List[str]
        List of column names in the data to validate against
    data : pd.DataFrame
        The data to check for missing values
        
    Returns
    -------
    pd.DataFrame
        Validated predictor matrix
        
    Raises
    ------
    ValueError
        If predictor_matrix has invalid structure or column names don't match data
    """
    if not isinstance(predictor_matrix, pd.DataFrame):
        raise ValueError("predictor_matrix must be a pandas DataFrame")
    
    # Check if all column names in predictor matrix exist in data
    predictor_cols = list(predictor_matrix.columns)
    missing_cols = [col for col in predictor_cols if col not in data_columns]
    if missing_cols:
        raise ValueError(f"predictor_matrix contains columns not in data: {missing_cols}")
    
    # Check if all row names (target variables) exist in data
    target_vars = list(predictor_matrix.index)
    missing_targets = [var for var in target_vars if var not in data_columns]
    if missing_targets:
        raise ValueError(f"predictor_matrix contains target variables not in data: {missing_targets}")
    
    # Check if matrix contains only 0s and 1s
    if not predictor_matrix.isin([0, 1]).all().all():
        raise ValueError("predictor_matrix must contain only 0s and 1s")
    
    # Check for columns without missing data that are being imputed (warning)
    complete_targets = [var for var in target_vars if not data[var].isna().any()]
    if complete_targets:
        warnings.warn(f"Target variables without missing data are being imputed: {complete_targets}. This is unnecessary but allowed.")
    
    # Check for columns with missing data that are used as predictors but not imputed (error)
    predictor_only_vars = [col for col in predictor_cols if col not in target_vars]
    incomplete_predictors = [var for var in predictor_only_vars if data[var].isna().any()]
    if incomplete_predictors:
        raise ValueError(f"Predictor variables with missing data are not being imputed: {incomplete_predictors}. All predictors must be complete or included as targets.")
    
    return predictor_matrix

def validate_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean columns in the DataFrame.
    
    Checks for columns with only NaN values and drops them with appropriate warnings.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame to validate
        
    Returns
    -------
    pd.DataFrame
        DataFrame with invalid columns removed
        
    Warns
    -----
    UserWarning
        If columns with only NaN values are found and dropped
        
    Notes
    -----
    Missing data values that are treated as NaN:
    - pandas NaN (numpy.nan)
    """
    # Check for columns with only NaN values
    nan_only_cols = []
    for col in data.columns:
        if data[col].isna().all():
            nan_only_cols.append(col)
    
    # Drop columns with only NaN values and print warning
    if nan_only_cols:
        warnings.warn(f"Found columns with only NaN values: {nan_only_cols}. These columns will be dropped as they cannot be imputed.")
        print(f"Dropping {len(nan_only_cols)} columns with only NaN values: {nan_only_cols}")
        data = data.drop(columns=nan_only_cols)
    
    return data

def validate_dataframe(data) -> pd.DataFrame:
    """
    Check and validate input data for MICE imputation.
    
    Parameters
    ----------
    data : Any
        Input data to be checked and converted to DataFrame
        
    Returns
    -------
    pd.DataFrame
        Validated and cleaned DataFrame
        
    Raises
    ------
    ValueError
        If data cannot be converted to DataFrame or has duplicate column names
        
    Notes
    -----
    Missing data values that are treated as NaN:
    - pandas NaN (numpy.nan)
    """
    # Try to convert to DataFrame if it's not already one
    try:
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
    except Exception as e:
        raise ValueError(f"Input data cannot be converted to DataFrame: {str(e)}")
    
    # Check for duplicate column names
    duplicate_cols = data.columns[data.columns.duplicated()].tolist()
    if duplicate_cols:
        print(f"Found duplicate column names: {duplicate_cols}. Please make sure that the column names are unique.")
        raise ValueError("DataFrame contains duplicate column names")
    
    # Check for fully empty rows
    n_rows_before = len(data)
    data = data.dropna(how='all')
    n_rows_after = len(data)
    n_dropped = n_rows_before - n_rows_after
    
    if n_dropped > 0:
        print(f"Dropped {n_dropped} fully empty rows")
    
    # Check for columns with no values
    empty_cols = data.columns[data.isna().all()].tolist()
    if empty_cols:
        warnings.warn(f"Found columns with no values: {empty_cols}. These columns will be dropped as they cannot be imputed.")
        print(f"Dropping {len(empty_cols)} columns with no values: {empty_cols}")
        data = data.drop(columns=empty_cols)
    
    return data

def validate_formula(formula: str, columns: List[str]) -> None:
    """
    Validate that all variables in the formula exist in the dataset columns.
    
    Parameters
    ----------
    formula : str
        The formula string to validate
    columns : List[str]
        List of column names in the dataset
        
    Raises
    ------
    ValueError
        If any variables in the formula are not found in the columns
    """
    if not isinstance(formula, str):
        raise ValueError("formula must be a string")
    
    if not isinstance(columns, list):
        raise ValueError("columns must be a list")
    
    # Extract variable names from formula using regex
    # This pattern matches valid Python identifiers that could be column names
    variable_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
    variables_in_formula = set(re.findall(variable_pattern, formula))
    
    # Remove common statsmodels/patsy keywords that are not variables
    keywords_to_ignore = {
        'I', 'Q', 'C', 'np', 'pd', 'log', 'exp', 'sqrt', 'abs', 'sin', 'cos', 'tan',
        'int', 'float', 'str', 'bool', 'True', 'False', 'None'
    }
    variables_in_formula = variables_in_formula - keywords_to_ignore
    
    # Check which variables exist in the dataset
    available_columns = set(columns)
    missing_variables = variables_in_formula - available_columns
    
    if missing_variables:
        raise ValueError(f"The following variables in the formula are not present in the dataset: {missing_variables}")
    
    return None