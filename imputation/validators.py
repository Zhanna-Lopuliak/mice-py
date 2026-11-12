"""Checks of the arguments of :class:`imputation.MICE`; they raise ``ValueError`` or ``TypeError`` or warn."""

from __future__ import annotations

import logging
import numbers
import re
import warnings
from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .constants import (
    DEFAULT_METHOD,
    DEFAULT_METHODS,
    SUPPORTED_INITIAL_METHODS,
    SUPPORTED_METHODS,
    SUPPORTED_VISIT_SEQUENCES,
    ImputationMethod,
    InitialMethod,
    VisitSequence,
)
from .design import _column_kind
from .logreg import MAX_CATEGORIES, _plain
from .norm import _KIND_NAMES

logger = logging.getLogger(__name__)

# Methods that impute numbers: the column must be numeric
_NUMERIC_METHODS = frozenset(
    method.value
    for method in (
        ImputationMethod.NORM,
        ImputationMethod.NORM_NOB,
        ImputationMethod.NORM_BOOT,
        ImputationMethod.NORM_PREDICT,
        ImputationMethod.MEAN,
    )
)
# Methods for a column with two categories
_TWO_CATEGORY_METHODS = frozenset((ImputationMethod.LOGREG.value, ImputationMethod.LOGREG_BOOT.value))
# Methods for categorical columns: they model at most MAX_CATEGORIES categories
_CATEGORICAL_METHODS = _TWO_CATEGORY_METHODS | {ImputationMethod.POLYREG.value, ImputationMethod.POLR.value}
# Donor methods that impute a column of any type
_DONORS = "a donor method (pmm, cart, rf, sample)"


def _is_integer(value: Any) -> bool:
    """True for Python and NumPy integers; False for booleans."""
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))


def _is_valid_method(method: Any) -> bool:
    """A supported method, or "" for a column that is not imputed (as in R)."""
    return method == "" or ImputationMethod.is_valid_method(method)


def check_n_imputations(n_imputations: int) -> None:
    """
    Check if the number of imputations is valid and provide a warning if it's high.

    Parameters
    ----------
    n_imputations : int
        Number of imputations to perform (a Python or NumPy integer)

    Raises
    ------
    ValueError
        If n_imputations is not a positive integer
    """
    if not _is_integer(n_imputations):
        raise ValueError("n_imputations must be a positive integer")

    if n_imputations <= 0:
        raise ValueError("n_imputations must be a positive integer")

    if n_imputations > 100:
        warnings.warn(
            f"{n_imputations} imputations is a large number. This might take a while to compute.",
            UserWarning,
            stacklevel=3,
        )


def check_maxit(maxit: int) -> None:
    """
    Check if the maximum iterations parameter is valid and provide a warning if it's high.

    Parameters
    ----------
    maxit : int
        Maximum number of iterations for each imputation cycle (a Python or
        NumPy integer)

    Raises
    ------
    ValueError
        If maxit is not a positive integer
    """
    if not _is_integer(maxit):
        raise ValueError("maxit must be an integer")

    if maxit <= 0:
        raise ValueError("maxit must be a positive integer")

    if maxit > 50:
        warnings.warn("maxit is greater than 50, imputations will take a lot of time", UserWarning, stacklevel=3)


def check_method(
    method: str | dict[str, str], columns: list[str], defaults: Mapping[Any, str] | None = None
) -> dict[str, str]:
    """
    Check and process the method parameter for MICE imputation.

    Only the names are checked here; whether a method suits the type of its
    column is checked by `check_method_types`.

    Parameters
    ----------
    method : str or dict of str to str
        Method specification:

        - str: use the same method for all columns;
        - dict: the method of each listed column; the other columns get
          their method in `defaults`.

        A method is one of SUPPORTED_METHODS, or "" for a column that is
        not imputed.
    columns : list of str
        List of column names in the data
    defaults : mapping of column to str, optional
        The method of each column that a dict `method` does not list, as
        `imputation.make_method` returns it (the default of the column's
        type). If None, DEFAULT_METHOD (``"pmm"``) for every column.

    Returns
    -------
    dict of str to str
        Dictionary mapping each column to its imputation method

    Raises
    ------
    ValueError
        If method is invalid or references non-existent columns
    """
    # If method is a string, validate and use for all columns
    if isinstance(method, str):
        if not _is_valid_method(method):
            raise ValueError(f"Unsupported method: {method}. Supported methods are: {SUPPORTED_METHODS}")
        return {col: method for col in columns}

    # If method is a dictionary, validate each entry
    if isinstance(method, dict):
        # Check if all specified columns exist
        invalid_cols = [col for col in method.keys() if col not in columns]
        if invalid_cols:
            raise ValueError(f"Columns not found in data: {invalid_cols}")

        # Check if all methods are supported
        invalid_methods = {col: m for col, m in method.items() if not _is_valid_method(m)}
        if invalid_methods:
            raise ValueError(f"Unsupported methods: {invalid_methods}. Supported methods are: {SUPPORTED_METHODS}")

        # Columns not in the dict get their default method
        if defaults is None:
            result = {col: DEFAULT_METHOD for col in columns}
        else:
            result = {col: defaults[col] for col in columns}
        result.update(method)  # Override with specified methods
        return result

    raise ValueError("method must be either a string or a dictionary")


def check_default_method(default_method: Any) -> tuple[str, ...]:
    """
    Check the default methods by column type (R's ``defaultMethod``).

    Parameters
    ----------
    default_method : list or tuple of str
        Four supported method names: for numeric columns, for columns with
        two categories (and boolean ones), for unordered categorical (and
        text) columns with more than two categories, and for ordered
        categorical columns with more than two categories.

    Returns
    -------
    tuple of str
        The four names.

    Raises
    ------
    ValueError
        If `default_method` is not a list or tuple of four names, or names a
        method that is not supported.

    Examples
    --------
    >>> check_default_method(["pmm", "logreg", "polyreg", "polr"])
    ('pmm', 'logreg', 'polyreg', 'polr')
    """
    if not isinstance(default_method, (list, tuple)) or len(default_method) != 4:
        raise ValueError(
            "default_method must be a list or tuple of four method names: the methods of numeric columns, of "
            "columns with two categories, and of unordered and of ordered categorical columns with more "
            f"(R's defaultMethod; the default is {DEFAULT_METHODS}). Got {default_method!r}."
        )
    invalid = [method for method in default_method if not ImputationMethod.is_valid_method(method)]
    if invalid:
        raise ValueError(
            f"default_method contains unsupported methods: {invalid}. Supported methods are: {SUPPORTED_METHODS}"
        )
    return tuple(default_method)


def _shown_categories(series: pd.Series) -> str:
    """The first five distinct observed values of a column (sorted if they can be), for a message."""
    values = list(pd.unique(series.dropna()))
    try:
        values = sorted(values)
    except TypeError:  # labels of mixed types: in order of appearance
        pass
    shown = ", ".join(repr(_plain(value)) for value in values[:5])
    return shown + (", ..." if len(values) > 5 else "")


def _suitable_methods(kind: str, n_categories: int) -> str:
    """The methods that can impute a non-numeric column, for an error message."""
    if n_categories > MAX_CATEGORIES:
        return _DONORS
    if kind == "bool" or n_categories <= 2:
        return f"logreg, logreg.boot, or {_DONORS}"
    if kind == "ordered":
        return f"polr, polyreg, or {_DONORS}"
    return f"polyreg, or {_DONORS}"


def check_method_types(data: pd.DataFrame, method: Mapping[Hashable, str]) -> None:
    """
    Check that each method suits the type of the column it imputes, as R's ``check.method``.

    R only warns about a mismatch, and the imputation then fails later or
    imputes missing values. Here the pairings that cannot work raise before
    any imputation is drawn, with the column named:

    - a method for numeric columns (``"norm"``, ``"norm.nob"``,
      ``"norm.boot"``, ``"norm.predict"``, ``"mean"``) on a boolean,
      categorical or text column;
    - ``"logreg"`` or ``"logreg.boot"`` on a column with more than two
      distinct observed values;
    - ``"logreg"``, ``"logreg.boot"``, ``"polyreg"`` or ``"polr"`` on a
      column with more than 50 distinct observed values (R's limit, for
      example a text column of names or identifiers).

    A categorical method on a numeric column that passes these checks is
    allowed, with R's warning. ``"midas"`` is checked by its imputer.

    Parameters
    ----------
    data : pd.DataFrame
        The data.
    method : mapping of column to str
        The method of each column; columns with ``""`` are not checked.

    Raises
    ------
    ValueError
        For an impossible pairing; the message names the column and the
        methods that suit it.

    Warns
    -----
    UserWarning
        For ``"logreg"``, ``"logreg.boot"``, ``"polyreg"`` or ``"polr"`` on
        a numeric column, with R's text: "Type mismatch for variable(s):
        <column>\\nImputation method <method> is for categorical data."
    """
    for col, meth in method.items():
        if meth not in _NUMERIC_METHODS and meth not in _CATEGORICAL_METHODS:
            continue
        series = data[col]
        kind = _column_kind(series)
        if meth in _NUMERIC_METHODS:
            if kind != "numeric":
                suitable = _suitable_methods(kind, int(series.nunique(dropna=True)))
                raise ValueError(
                    f"Column {col!r} is {_KIND_NAMES.get(kind, kind)} (dtype {series.dtype}), but {meth} imputes "
                    f"numeric columns only (R's mice.impute.{meth} imputes NA here). Methods for this column: "
                    f"{suitable}."
                )
            continue
        # The categories the imputer will model: the distinct observed values
        n_categories = int(series.nunique(dropna=True))
        if n_categories > MAX_CATEGORIES:
            raise ValueError(
                f"Column {col!r} has {n_categories} distinct observed values, more categories than {meth} can "
                f"model: R's mice allows at most {MAX_CATEGORIES} ('Maximum number of categories "
                f"({MAX_CATEGORIES}) exceeded'). Text columns of names or identifiers often look like this, and "
                "the default method of a text column with more than two values is polyreg. Impute the column "
                f"with a donor method instead, for example method={{{col!r}: 'pmm'}}, or remove it from the data "
                "if it identifies the rows."
            )
        if meth in _TWO_CATEGORY_METHODS and n_categories > 2:
            if kind == "numeric":
                alternatives = "polyreg for more than two categories, or a method for numeric columns such as pmm"
            elif kind == "ordered":
                alternatives = f"polr or polyreg for more than two categories, or {_DONORS}"
            else:
                alternatives = f"polyreg for more than two categories, or {_DONORS}"
            raise ValueError(
                f"Column {col!r}: {meth} imputes a target with two categories, but the column has {n_categories} "
                f"distinct observed values ({_shown_categories(series)}). Use {alternatives}."
            )
        if kind == "numeric":
            warnings.warn(
                f"Type mismatch for variable(s): {col}\nImputation method {meth} is for categorical data.",
                UserWarning,
                stacklevel=3,
            )


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
        raise ValueError(
            f"Unsupported initial method: {initial_method}. Supported initial methods are: {SUPPORTED_INITIAL_METHODS}"
        )


def check_visit_sequence(
    visit_sequence: str | list[str], columns: list[str], columns_with_missing: list[str] | None = None
) -> tuple[list[str] | None, list[str]]:
    """
    Check and process the visit sequence parameter for MICE imputation.

    Parameters
    ----------
    visit_sequence : str or list of str
        Visit sequence specification:

        - str: one of SUPPORTED_VISIT_SEQUENCES ("roman", "arabic",
          "monotone", "revmonotone", "random");
        - list of str: column names specifying the order to visit variables.
    columns : list of str
        List of all column names in the data
    columns_with_missing : list of str, optional
        List of columns that have missing values. If provided, will validate
        that all these columns are included in a custom visit sequence.

    Returns
    -------
    tuple
        (validated_sequence, columns_without_missing) where:

        - validated_sequence: the processed visit sequence (only for list input, None for string)
        - columns_without_missing: list of columns in sequence that don't have missing values

    Raises
    ------
    ValueError
        If visit_sequence is invalid or references non-existent columns

    Notes
    -----
    For string visit sequences, the actual sequence will be generated in
    MICE._set_visit_sequence() based on the data.

    For list visit sequences, this function validates that:

    1. All columns exist in the data
    2. No duplicate columns
    3. All columns with missing values are included (if columns_with_missing provided)
    """

    if isinstance(visit_sequence, str):
        if not VisitSequence.is_valid_sequence(visit_sequence):
            raise ValueError(
                f"Unsupported visit sequence: {visit_sequence}. "
                f"Supported visit sequences are: {SUPPORTED_VISIT_SEQUENCES}"
            )
        return None, []

    if isinstance(visit_sequence, list):
        invalid_cols = [col for col in visit_sequence if col not in columns]
        if invalid_cols:
            raise ValueError(f"Visit sequence contains columns not in data: {invalid_cols}")

        if len(visit_sequence) != len(set(visit_sequence)):
            duplicates = [col for col in visit_sequence if visit_sequence.count(col) > 1]
            raise ValueError(f"Visit sequence contains duplicate columns: {set(duplicates)}")

        if columns_with_missing is not None:
            missing_from_sequence = [col for col in columns_with_missing if col not in visit_sequence]
            if missing_from_sequence:
                raise ValueError(
                    f"Visit sequence is missing columns with missing values: {missing_from_sequence}. "
                    f"All columns with missing values must be included in the visit sequence. "
                    f"Columns with missing values: {columns_with_missing}"
                )

            cols_without_missing = [col for col in visit_sequence if col not in columns_with_missing]

            validated_sequence = [col for col in visit_sequence if col in columns_with_missing]

            return validated_sequence, cols_without_missing

        return visit_sequence, []

    raise ValueError("visit_sequence must be either a string or a list of strings")


def validate_predictor_matrix(
    predictor_matrix: pd.DataFrame, data_columns: list[str], data: pd.DataFrame
) -> pd.DataFrame:
    """
    Validate predictor matrix for MICE imputation.

    Parameters
    ----------
    predictor_matrix : pd.DataFrame
        Binary matrix indicating which variables should be used as predictors
        for each target variable. Rows represent target variables, columns represent predictors.
        A 1 indicates that the column variable is used as predictor for the index variable.
        It may cover a subset of the columns; absent entries count as 0. Every
        incomplete column needs a row; rows of complete columns are unused.
    data_columns : list of str
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
        If predictor_matrix has invalid structure, column names don't match
        data, an incomplete column is a predictor without being a target, or
        an incomplete column has no row
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

    # Rows of complete variables are allowed and unused, as in R: a complete
    # variable is never imputed.

    # Check for columns with missing data that are used as predictors but not imputed (error)
    predictor_only_vars = [col for col in predictor_cols if col not in target_vars]
    incomplete_predictors = [var for var in predictor_only_vars if data[var].isna().any()]
    if incomplete_predictors:
        raise ValueError(
            f"Predictor variables with missing data are not being imputed: {incomplete_predictors}. "
            "All predictors must be complete or included as targets."
        )

    # Every incomplete column needs a row: it says which predictors impute it
    without_row = [col for col in data_columns if col not in target_vars and data[col].isna().any()]
    if without_row:
        raise ValueError(
            f"predictor_matrix has no row for the incomplete columns {without_row}. "
            "Add a row for each (all zeros to impute it without predictors)."
        )

    return predictor_matrix


def _is_supported_dtype(dtype: Any) -> bool:
    """Numeric (not complex), boolean, categorical, object or string."""
    if isinstance(dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_bool_dtype(dtype):
        return True
    if pd.api.types.is_complex_dtype(dtype):
        return False
    if pd.api.types.is_numeric_dtype(dtype):
        return True
    return pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)


def validate_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Check that every column has a dtype that can be imputed.

    Supported are numeric columns (including pandas' nullable ``Int64`` and
    ``Float64``), ``bool`` and ``boolean``, ``category``, ``object`` and
    ``string``. Object and string columns are treated as categorical
    variables. Columns that are entirely missing are kept: they are neither
    imputed nor used as predictors (see :meth:`imputation.MICE.impute`).

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame to validate

    Returns
    -------
    pd.DataFrame
        The same DataFrame

    Raises
    ------
    TypeError
        If a column has another dtype (for example datetime, timedelta,
        period or complex); the message names the column and its dtype.

    Notes
    -----
    Missing data values that are treated as NaN:

    - pandas NaN (numpy.nan), None and pd.NA
    """
    unsupported = [(col, data[col].dtype) for col in data.columns if not _is_supported_dtype(data[col].dtype)]
    if unsupported:
        details = ", ".join(f"{col!r} has dtype {dtype}" for col, dtype in unsupported)
        raise TypeError(
            f"Unsupported column dtype: {details}. Columns must be numeric (int, float, "
            "Int64, Float64), bool, boolean, category, object or string."
        )

    empty_cols = [col for col in data.columns if data[col].isna().all()]
    if empty_cols:
        logger.info("Columns without any observed value (not imputed, not used as predictors): %s", empty_cols)

    return data


def validate_dataframe(data: Any) -> pd.DataFrame:
    """
    Check and validate input data for MICE imputation.

    Parameters
    ----------
    data : Any
        Input data to be checked and converted to DataFrame

    Returns
    -------
    pd.DataFrame
        The data as a DataFrame. Rows and columns that are entirely missing
        are kept.

    Raises
    ------
    ValueError
        If data cannot be converted to DataFrame or has duplicate column names

    Warns
    -----
    UserWarning
        If some rows are entirely missing. They are kept and imputed, but
        their imputations carry no information from the row itself.

    Notes
    -----
    Missing data values that are treated as NaN:

    - pandas NaN (numpy.nan), None and pd.NA
    """
    # Try to convert to DataFrame if it's not already one
    try:
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
    except Exception as e:
        raise ValueError(f"Input data cannot be converted to DataFrame: {str(e)}") from e

    # Check for duplicate column names
    duplicate_cols = data.columns[data.columns.duplicated()].tolist()
    if duplicate_cols:
        raise ValueError(
            f"DataFrame contains duplicate column names: {duplicate_cols}. "
            "Please make sure that the column names are unique."
        )

    # Fully empty rows are kept and imputed
    n_empty_rows = int(data.isna().all(axis=1).sum()) if data.shape[1] else 0
    if n_empty_rows > 0:
        warnings.warn(
            f"{n_empty_rows} row(s) have no observed value. They are kept and imputed, "
            "but their imputations use no information from the row itself.",
            UserWarning,
            stacklevel=3,
        )

    return data


def validate_formula(formula: str, columns: list[str]) -> None:
    """
    Validate that all variables in the formula exist in the dataset columns.

    Parameters
    ----------
    formula : str
        The formula string to validate
    columns : list of str
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
    variable_pattern = r"\b[a-zA-Z_][a-zA-Z0-9_]*\b"
    variables_in_formula = set(re.findall(variable_pattern, formula))

    # Remove common statsmodels/patsy keywords that are not variables
    keywords_to_ignore = {
        "I",
        "Q",
        "C",
        "np",
        "pd",
        "log",
        "exp",
        "sqrt",
        "abs",
        "sin",
        "cos",
        "tan",
        "int",
        "float",
        "str",
        "bool",
        "True",
        "False",
        "None",
    }
    variables_in_formula = variables_in_formula - keywords_to_ignore

    # Check which variables exist in the dataset
    available_columns = set(columns)
    missing_variables = variables_in_formula - available_columns

    if missing_variables:
        raise ValueError(f"The following variables in the formula are not present in the dataset: {missing_variables}")

    return None
