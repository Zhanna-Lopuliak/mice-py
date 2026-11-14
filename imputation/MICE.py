"""The MICE class: multiple imputation by chained equations, as R's ``mice``."""

from __future__ import annotations

import functools
import inspect
import logging
import time
import warnings
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .constants import (
    DEFAULT_INITIAL_METHOD,
    DEFAULT_METHODS,
    DEFAULT_VISIT_SEQUENCE,
    PASSIVE_METHOD,
    ImputationMethod,
    InitialMethod,
    VisitSequence,
)
from .design import _as_numeric, _column_kind, _remove_lindep, find_collinear, obtain_design
from .mice_result import MICEresult
from .pooling import _residual_df, pool_estimates
from .utils import get_imputer_func, get_imputer_params, make_method, method_prefix, split_method_param
from .validators import (
    _PASSIVE_RESULT,
    _POST_RESULT,
    _holds_booleans,
    _values_per_entry,
    check_default_method,
    check_ignore,
    check_initial_method,
    check_maxit,
    check_method,
    check_method_types,
    check_n_imputations,
    check_passive,
    check_post,
    check_visit_sequence,
    check_where,
    validate_columns,
    validate_dataframe,
    validate_formula,
    validate_predictor_matrix,
)

if TYPE_CHECKING:
    from statsmodels.iolib.summary2 import Summary

logger = logging.getLogger(__name__)

# Keyword arguments of `MICE.impute` removed in version 0.4, and the hint
# that the error for each of them gives
_QUICKPRED_HINT = (
    "It was removed in version 0.4: build the predictor matrix with imputation.quickpred(data, mincor=..., "
    "minpuc=..., include=..., exclude=..., method=...) and pass it as predictor_matrix."
)
_REMOVED_ARGUMENTS = {
    "min_cor": _QUICKPRED_HINT,
    "min_puc": _QUICKPRED_HINT,
    "include": _QUICKPRED_HINT,
    "exclude": _QUICKPRED_HINT,
    "correlation_method": _QUICKPRED_HINT,
    "output_dir": "It was removed in version 0.4 and never had an effect.",
}

# Columns of `MICE.logged_events`, as R's `loggedEvents`
_LOG_COLUMNS = ("it", "im", "dep", "meth", "out")

# A variable whose variance is below this is constant (R: 1000 * .Machine$double.eps)
_CONSTANT_VARIANCE = 1000 * np.finfo(float).eps

# Methods whose imputations are new floats, not observed values
_FLOAT_METHODS = frozenset(
    method.value
    for method in (
        ImputationMethod.NORM,
        ImputationMethod.NORM_NOB,
        ImputationMethod.NORM_BOOT,
        ImputationMethod.NORM_PREDICT,
        ImputationMethod.MEAN,
    )
)


def _full_predictor_matrix(columns) -> pd.DataFrame:
    """All ones with a zero diagonal: R's default predictor matrix."""
    matrix = pd.DataFrame(1, index=columns, columns=columns)
    np.fill_diagonal(matrix.values, 0)
    return matrix


def _empty_log() -> pd.DataFrame:
    """An empty `logged_events` frame with R's columns."""
    return pd.DataFrame(
        {
            "it": pd.Series(dtype="int64"),
            "im": pd.Series(dtype="int64"),
            "dep": pd.Series(dtype=object),
            "meth": pd.Series(dtype=object),
            "out": pd.Series(dtype=object),
        }
    )


def _variance(series: pd.Series) -> float:
    """Sample variance of the observed values, categories counted by their codes.

    NaN when fewer than two values are observed, as R's ``var(na.rm = TRUE)``.
    """
    values = _as_numeric(series)
    values = values[~np.isnan(values)]
    if values.size < 2:
        return np.nan
    return float(np.var(values, ddof=1))


def _target_values(series: pd.Series):
    """The column passed to an imputer as its target `y`.

    A categorical column stays a ``pd.Categorical``, so that the imputer sees
    its type even when its categories are numbers. pandas' nullable numeric
    columns become float arrays with NaN. Everything else becomes the NumPy
    array of the column.
    """
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return series.array
    if (
        isinstance(dtype, pd.api.extensions.ExtensionDtype)
        and pd.api.types.is_numeric_dtype(dtype)
        and not pd.api.types.is_bool_dtype(dtype)
    ):
        return series.to_numpy(dtype=float, na_value=np.nan)
    return series.to_numpy()


def _is_nullable_integer(dtype) -> bool:
    """True for pandas' integer extension dtypes (``Int64``, ``UInt8``, ...), which hold missing values."""
    return isinstance(dtype, pd.api.extensions.ExtensionDtype) and pd.api.types.is_integer_dtype(dtype)


def _nullable_dtype(dtype):
    """The pandas dtype that holds missing values for a NumPy integer or boolean dtype, else None.

    ``int64`` gives ``Int64`` (each integer dtype the nullable one of its
    width and sign, ``uint8`` gives ``UInt8``), ``bool`` gives ``boolean``.
    """
    if not isinstance(dtype, np.dtype):
        return None
    if dtype.kind == "b":
        return pd.BooleanDtype()
    if dtype.kind in "iu":
        return pd.api.types.pandas_dtype(f"{'UInt' if dtype.kind == 'u' else 'Int'}{8 * dtype.itemsize}")
    return None


def _in_float_dtype(values, dtype):
    """Float imputations in the float dtype of their column, so that writing them keeps it.

    A ``float32`` column gets ``float32`` values (pandas would otherwise
    upcast the column, with a warning); a ``Float64`` or ``Float32`` column
    gets floats of its NumPy dtype. Any other dtype leaves the values as
    they are.
    """
    if not pd.api.types.is_float_dtype(dtype):
        return values
    return np.asarray(values, dtype=getattr(dtype, "numpy_dtype", dtype))


def _is_number_dtype(dtype) -> bool:
    """True for a dtype of real numbers: integer or float, NumPy or nullable; not boolean or complex."""
    return (
        pd.api.types.is_numeric_dtype(dtype)
        and not pd.api.types.is_bool_dtype(dtype)
        and not pd.api.types.is_complex_dtype(dtype)
    )


def _shown(values: pd.Series) -> str:
    """The first three distinct values of `values`, for a message."""
    distinct = list(pd.unique(values.to_numpy(dtype=object)))
    return ", ".join(repr(value) for value in distinct[:3]) + (", ..." if len(distinct) > 3 else "")


def _in_column_dtype(values: pd.Series, dtype, context: str):
    """`values`, which a function of `post` or `passive` returned, as an array of `dtype`, the dtype of their column.

    A float column holds numbers; an integer column whole numbers within
    the range of its dtype; a boolean column booleans; a categorical column
    its categories; a string column strings; an object column anything.
    Missing values are written as missing (a NumPy integer or boolean
    column, which cannot hold them, does not occur among the columns with
    where-cells). pandas would change the dtype of the column, or truncate,
    where the column cannot hold a value; here that raises ValueError,
    starting with `context` and naming the dtype.

    Parameters
    ----------
    values : pd.Series
        One value per entry, in the order in which they are written.
    dtype : dtype
        The dtype of the column in the chains.
    context : str
        The start of the message ("Column 'bmi', passive (iteration 1,
        imputation 1)").

    Returns
    -------
    np.ndarray or pandas array
        The values in `dtype`, without index.
    """
    present = values.notna().to_numpy()
    given = values[present]

    def cannot_hold(what: str):
        shown = "" if what == "missing values" else f": {_shown(given)}"
        return ValueError(
            f"{context}: the function returned {what}, which the column (dtype {dtype}) cannot hold{shown}."
        )

    if isinstance(dtype, pd.CategoricalDtype):
        unknown = given[dtype.categories.get_indexer(given.to_numpy(dtype=object)) < 0]
        if len(unknown):
            raise ValueError(
                f"{context}: the function returned values that are not categories of the column (dtype {dtype}): "
                f"{_shown(unknown)}."
            )
        return pd.Categorical(values.to_numpy(dtype=object), dtype=dtype)
    if pd.api.types.is_bool_dtype(dtype):
        if len(given) and not _holds_booleans(given.to_numpy(), given.dtype):
            raise cannot_hold("values that are not booleans")
        if isinstance(dtype, np.dtype):
            if not present.all():
                raise cannot_hold("missing values")
            return values.to_numpy(dtype=bool)
        return pd.array(values.to_numpy(dtype=object), dtype=dtype)
    if pd.api.types.is_numeric_dtype(dtype):
        numbers = given
        if numbers.dtype == object:
            # Python or NumPy integers and floats; not booleans, not text that looks like a number
            if not all(
                isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))
                for value in numbers
            ):
                raise cannot_hold("values that are not numbers")
            numbers = pd.to_numeric(numbers)
        if not _is_number_dtype(numbers.dtype):
            raise cannot_hold("values that are not numbers")
        numpy_dtype = getattr(dtype, "numpy_dtype", dtype)
        if pd.api.types.is_float_dtype(dtype):
            floats = np.full(len(values), np.nan, dtype=numpy_dtype)
            floats[present] = numbers.to_numpy(dtype=float)
            return floats if isinstance(dtype, np.dtype) else pd.array(floats, dtype=dtype)
        if pd.api.types.is_integer_dtype(numbers.dtype):
            whole = numbers.to_numpy(dtype=getattr(numbers.dtype, "numpy_dtype", numbers.dtype))
        else:
            whole = numbers.to_numpy(dtype=float)
            if not (np.isfinite(whole) & (whole == np.round(whole))).all():
                raise cannot_hold("numbers that are not whole")
        limits = np.iinfo(numpy_dtype)
        if whole.size and (whole.min() < limits.min or whole.max() > limits.max):
            raise cannot_hold(f"numbers outside the range of {numpy_dtype}")
        integers = np.zeros(len(values), dtype=numpy_dtype)
        integers[present] = whole.astype(numpy_dtype)
        if isinstance(dtype, np.dtype):
            if not present.all():
                raise cannot_hold("missing values")
            return integers
        return pd.arrays.IntegerArray(integers, ~present)
    if isinstance(dtype, pd.StringDtype):
        if not all(isinstance(value, str) for value in given):
            raise cannot_hold("values that are not strings")
        return pd.array(values.to_numpy(dtype=object), dtype=dtype)
    return values.to_numpy(dtype=object)


@dataclass
class _ChainState:
    """One chain between two iterations, as R keeps it in ``imp`` and ``.Random.seed``.

    Attributes
    ----------
    imp : dict of column label to pd.Series
        For each column with where-cells (cells that get an imputation),
        the current imputation of each of them: index their row labels in
        row order, dtype that of the column in the chains. NaN where a cell
        could not be imputed, and in every where-cell of a column without a
        method.
    rng : np.random.Generator
        The generator of the chain; its state advances through the start
        and every iteration.
    """

    imp: dict[Hashable, pd.Series]
    rng: np.random.Generator


def _values(series: pd.Series):
    """The values of `series` in its own dtype, without the index, so that writing them does not align labels."""
    return series.to_numpy() if isinstance(series.dtype, np.dtype) else series.array


def _fit_rows(cx: np.ndarray, observed: np.ndarray, ignore: np.ndarray) -> np.ndarray:
    """The rows the model of a column is fitted on, R's ``ry``.

    The predictors of the row are complete in the working data (`cx`), the
    cell of the column is observed and the row is not ignored.
    """
    return cx & observed & ~ignore


def _rows_to_impute(cx: np.ndarray, where: np.ndarray) -> np.ndarray:
    """The rows the imputer of a column fills, R's ``wy``: where-cells whose predictors are complete (`cx`)."""
    return cx & where


def _store_imputations(imp: pd.Series, imputable: np.ndarray, values) -> None:
    """Write the imputations of one visit of a column into its `imp`, in place.

    `imputable` marks the where-cells that the imputer filled, those of
    `_rows_to_impute` (R's ``cc <- wy[where]``, one entry per where-cell);
    they get `values`, in row order. Every other where-cell gets NaN, also
    over its start value, as R's ``imputes[!cc] <- NA``.
    """
    if not imputable.all():
        imp.iloc[np.flatnonzero(~imputable)] = np.nan
    if imputable.any():
        imp.iloc[np.flatnonzero(imputable)] = values


def _refresh_working_column(
    working: pd.DataFrame, column: Hashable, imp: pd.Series, where: np.ndarray, observed: np.ndarray
) -> None:
    """Write the imputations of the missing where-cells of one column into the working data, in place.

    As R's ``data[(!ry) & wy, j] <- imp[[j]][(!ry)[wy], i]``: an observed
    where-cell keeps its observed value in the working data, so that the
    other columns are predicted from it.

    Parameters
    ----------
    working : pd.DataFrame
        The working data of a chain.
    column : label
        The column.
    imp : pd.Series
        The imputations of the where-cells of the column, in row order.
    where, observed : np.ndarray of bool
        One entry per row: the where-cells and the observed cells of the column.
    """
    missing = ~observed[where]
    if missing.any():
        working.loc[where & ~observed, column] = _values(imp)[missing]


def _replace_imputations(imp: pd.Series, values) -> None:
    """Every where-cell of `imp` gets its value from `values` (one per where-cell, in row order), in place.

    For the values of a ``post`` function and of a passive column, which
    cover all where-cells of the column.
    """
    imp.iloc[:] = values


def _post_values(function: Callable, imp: pd.Series, data: pd.DataFrame, context: str):
    """What the ``post`` function of a column returns for its imputations `imp`, in the dtype of `imp`.

    `function` is called as ``function(imp, data)`` with a copy of `imp`
    and `data` (the working data in the rows of the where-cells, a copy).
    It must return one value per where-cell: a Series with the index of
    `imp`, or a sequence of its length, of values the column can hold
    (`_in_column_dtype`). Raises ValueError, starting with `context`,
    otherwise, and for an exception inside `function` (the original is the
    cause).
    """
    try:
        result = function(imp.copy(), data)
    except Exception as err:
        raise ValueError(f"{context}: the function raised {type(err).__name__}: {err}") from err
    values = _values_per_entry(result, imp.index, context, _POST_RESULT, "where-cells of the column")
    return _in_column_dtype(values, imp.dtype, context)


def _passive_last(columns: list, method: Mapping[Hashable, str]) -> list:
    """`columns` with the passive ones moved to the end, each part in the order given.

    R's order without a visit sequence (``mice.edit.setup``: the active
    columns, then the passive ones), so that a passive column is computed
    from the imputations of its inputs in the same iteration.
    """
    return [col for col in columns if method[col] != PASSIVE_METHOD] + [
        col for col in columns if method[col] == PASSIVE_METHOD
    ]


def _passive_dtype(column: pd.Series, result: pd.Series, starts_from_draws: bool):
    """The dtype of a passive column in the chains where it differs from its dtype in the data, else None.

    Parameters
    ----------
    column : pd.Series
        The passive column, after a NumPy integer or boolean column with
        where-cells became nullable.
    result : pd.Series
        What its function returned on the data (`check_passive`).
    starts_from_draws : bool
        True if the column has no observed value in the rows that are not
        ignored, so that it starts from standard normal draws (its function
        returns numbers, see `MICE._check_passive_start`).

    Returns
    -------
    dtype or None
        For a column without any observed value, the dtype of `result` if
        it holds floats, ``Float64`` if it holds integers (the start values
        are floats); for an integer column whose function returns floats,
        or that starts from draws, ``Float64``, as for the norm methods;
        otherwise None: the column keeps its dtype.
    """
    if column.isna().all():
        return result.dtype if pd.api.types.is_float_dtype(result.dtype) else pd.Float64Dtype()
    if _is_nullable_integer(column.dtype) and (pd.api.types.is_float_dtype(result.dtype) or starts_from_draws):
        return pd.Float64Dtype()
    return None


def _complete(data: pd.DataFrame, imp: dict[Hashable, pd.Series], where: pd.DataFrame) -> pd.DataFrame:
    """A completed dataset, as R's ``complete``: a copy of `data` with every where-cell replaced by `imp`.

    An observed where-cell gets its imputation too, and a where-cell whose
    imputation is missing becomes missing.
    """
    completed = data.copy(deep=True)
    for column, values in imp.items():
        completed.loc[where[column].to_numpy(), column] = _values(values)
    return completed


def _monotone(where: pd.DataFrame, columns: list) -> list:
    """`columns` by increasing number of where-cells, ties in the order given, as R's ``order(nimp)``."""
    counts = where[columns].sum().to_numpy()
    return [columns[i] for i in np.argsort(counts, kind="stable")]


def _monotone_predictor_matrix(predictor_matrix: pd.DataFrame, where: pd.DataFrame) -> pd.DataFrame:
    """The predictor matrix as R's ``mice.edit.predictorMatrix`` edits it for a monotone pattern.

    R's ``"monotone"`` visit sequence holds every column, by increasing
    number of where-cells (`_monotone`), also the columns without any. Each
    column keeps as predictors only the columns before it in that order:
    R sets ``predictorMatrix[vis[i], vis[i:length(vis)]]`` to 0 for each
    position ``i``. Returns a new frame.

    Parameters
    ----------
    predictor_matrix : pd.DataFrame
        Square, rows and columns in data order.
    where : pd.DataFrame
        The where-cells (see `MICE.where`).
    """
    order = _monotone(where, list(where.columns))
    edited = predictor_matrix.copy()
    for i, col in enumerate(order):
        edited.loc[col, order[i:]] = 0
    return edited


def _sample_start(observed_values: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """`size` start values drawn with replacement from `observed_values`.

    As R's ``mice.impute.sample`` in ``initialize.imp``.
    """
    return rng.choice(observed_values, size=size)


def _with_rows(statistics: np.ndarray, n_rows: int) -> np.ndarray:
    """`statistics` with `n_rows` rows of NaN appended for the iterations to come.

    `statistics` has one row per iteration and one column per chain (see
    `MICE.chain_mean`).
    """
    return np.vstack([statistics, np.full((n_rows, statistics.shape[1]), np.nan)])


def _method_param_names() -> dict[str, list[str]]:
    """The parameters a user may set for each supported method (see `get_imputer_params`)."""
    return {method: get_imputer_params(method) for method in ImputationMethod.get_all_methods()}


def _check_method_params(params):
    """
    Check that every name in `params` has the form `<prefix>_<param>`.

    `<prefix>` must be the name of a supported method with its dots replaced
    by underscores (see `imputation.utils.method_prefix`) and `<param>` a
    parameter of its imputer that MICE does not supply itself (see
    `get_imputer_params`). A name that several methods match goes to the
    longest prefix with a valid parameter (see
    `imputation.utils.split_method_param`).

    Raises
    ------
    TypeError
        For the first invalid name; the message lists the valid names, or,
        for an argument removed in version 0.4, says what replaces it.
    """
    valid = _method_param_names()
    for key in params:
        match = split_method_param(key, valid)
        if match is not None and match[1] in valid[match[0]]:
            continue
        method = match[0] if match is not None else None
        if key in _REMOVED_ARGUMENTS:
            hint = _REMOVED_ARGUMENTS[key]
        elif method is not None and not valid[method]:
            hint = f"Method '{method}' takes no parameters."
        elif method is not None:
            prefix = method_prefix(method)
            hint = f"Valid parameters for method '{method}': {', '.join(f'{prefix}_{p}' for p in valid[method])}."
        else:
            hint = (
                "Method parameters take the form <method>_<parameter>; valid ones: "
                f"{', '.join(f'{method_prefix(m)}_{p}' for m, names in valid.items() for p in names)}."
            )
        raise TypeError(f"impute() got an unexpected keyword argument '{key}'. {hint}")


class MICE:
    """
    Multiple imputation by chained equations, as R's ``mice``.

    Create the object with the incomplete data, then call `impute`. If the
    chains have not converged, `continue_chains` runs more iterations of
    them (R's ``mice.mids``). To analyse the completed datasets, call `fit`
    with a model formula and combine the fits with `pool` (Rubin's rules,
    as R's ``pool``).

    By default, as in R, each incomplete column is imputed with the method
    of its type (`imputation.make_method`): predictive mean matching
    (``"pmm"``) for numeric columns, logistic regression (``"logreg"``)
    for boolean columns and columns with two categories, multinomial
    logistic regression (``"polyreg"``) for unordered categorical and text
    columns with more categories, and proportional-odds regression
    (``"polr"``) for ordered categorical columns with more categories.

    Parameters
    ----------
    data : pd.DataFrame
        The incomplete data. Anything else is converted with
        ``pd.DataFrame(data)``. Columns may be numeric, boolean, categorical,
        object or string; object and string columns are treated as
        unordered categorical.

    Attributes
    ----------
    data : pd.DataFrame
        The validated input data.
    id_obs : dict of str to pd.Series
        Per column, a boolean Series that is True where the value is observed.
    id_mis : dict of str to pd.Series
        Per column, a boolean Series that is True where the value is missing.
    imputed_datasets : list of pd.DataFrame
        The completed datasets of the last call of `impute` or
        `continue_chains`: copies of the data with every where-cell (see
        `where`) replaced by its imputation, as R's ``complete``. A
        where-cell that could not be imputed is missing, also where it was
        observed (see `impute`).
    where : pd.DataFrame
        The cells that the last `impute` imputed (where-cells), R's
        ``where``: a boolean frame with the index and the columns of
        `data`, True for a where-cell. The `where` argument of `impute` in
        this form; by default ``data.isna()``. ``~mice.where`` marks the
        cells that were not imputed, the frame of observed cells that the
        diagnostic plots of `plotting` take.
    ignore : np.ndarray of bool
        The rows that the last `impute` left out of the model fits and of
        the start of the chains, R's ``ignore``: one entry per row of
        `data`, True for a row left out (its where-cells were imputed all
        the same). The `ignore` argument of `impute` in this form; by
        default all False. `continue_chains` uses it too.
    post : dict of column label to callable
        The functions of the argument `post` of the last `impute` that are
        in effect, R's ``imp$post``: those of the columns that are imputed.
        As in R, the entry of a column that has no method, or whose method
        the setup step removed, is dropped. `continue_chains` uses them.
    passive : dict of column label to callable
        The functions of the passive columns of the last `impute` that are
        in effect (the argument `passive`): those of the passive columns
        with where-cells that the setup step kept, whose entry in `method`
        is ``"passive"``. `continue_chains` uses them.
    iteration : int
        The number of iterations each chain has run, R's ``imp$iteration``:
        the `maxit` of the last `impute` plus that of every `continue_chains`
        since.
    method : dict of str to str
        The imputation method of each column used by the last `impute`
        (``""`` for a column that is not imputed, such as a column without
        where-cells; ``"passive"`` for a passive column), as R's
        ``imp$method``: after the setup step, which removes the method of
        a constant or collinear column.
    predictor_matrix : pd.DataFrame
        The predictor matrix used by the last `impute`, after the setup step
        and, for ``visit_sequence="monotone"`` with ``maxit=1``, after R's
        edit for a monotone pattern (see `impute`); square, in data order;
        rows are targets, columns predictors.
    visit_sequence : list of str
        The order in which the last `impute` visited the columns, a column
        named twice included twice.
    chain_mean, chain_var : dict of str to np.ndarray
        For each imputed numeric column (passive ones included), the mean
        and variance of the imputations of all its where-cells (observed
        ones included; missing imputations skipped) at the end of each
        iteration, after `post`, shape
        ``(iteration, n_imputations)``: one row per iteration, one column
        per chain.
        `continue_chains` appends rows; after ``impute(maxit=0)`` they have
        none.
    logged_events : pd.DataFrame
        What the last call of `impute`, and every `continue_chains` since,
        changed in the imputation model, with R's columns ``it`` (iteration,
        0 for the setup step), ``im`` (imputation), ``dep`` (the target),
        ``meth`` and ``out``. Empty when nothing was logged. See `impute`.
    model_results : list
        The fitted models of the last call of `fit`, one per dataset.
        `impute` and `continue_chains` empty it, so that `pool` cannot
        combine fits of earlier datasets.
    pooled_result : MICEresult
        The pooled estimates of the last call of `pool`; `impute` and
        `continue_chains` remove it.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from imputation import MICE
    >>> rng = np.random.default_rng(1)
    >>> df = pd.DataFrame({"x": rng.normal(size=50)})
    >>> df["y"] = 2 * df["x"] + rng.normal(size=50)
    >>> df.loc[::5, "y"] = np.nan
    >>> mice = MICE(df)
    >>> datasets = mice.impute(n_imputations=5, seed=1)
    >>> len(datasets), int(datasets[0].isna().sum().sum())
    (5, 0)
    >>> _ = mice.fit("y ~ x")
    >>> table = mice.pool()["table"]  # estimate, df, fmi, intervals, ... per term

    A passive column, kept equal to a function of the others, and bounds on
    the imputations of a column (R's ``post``):

    >>> df["y2"] = df["y"] ** 2
    >>> df["z"] = (df["x"] + rng.normal(size=50)).clip(-1.5, 1.5)
    >>> df.loc[1::5, "z"] = np.nan
    >>> predictors = 1 - pd.DataFrame(np.eye(4, dtype=int), index=df.columns, columns=df.columns)
    >>> predictors.loc["y", "y2"] = 0  # y2 is computed from y and does not predict it
    >>> mice = MICE(df)
    >>> datasets = mice.impute(
    ...     n_imputations=2,
    ...     seed=1,
    ...     predictor_matrix=predictors,
    ...     passive={"y2": lambda d: d["y"] ** 2},
    ...     post={"z": lambda imp, data: imp.clip(-1.5, 1.5)},
    ... )
    >>> mice.method["y2"], mice.visit_sequence
    ('passive', ['y', 'z', 'y2'])
    >>> all((d["y2"] == d["y"] ** 2).all() and d["z"].between(-1.5, 1.5).all() for d in datasets)
    True
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialize the MICE object.

        Parameters
        ----------
        data : pd.DataFrame
            The incomplete data.

        Raises
        ------
        ValueError
            If `data` cannot be converted to a DataFrame or has duplicate
            column names.
        TypeError
            If a column has a dtype that cannot be imputed (for example
            datetime or complex); the message names the column and the dtype

        Notes
        -----
        Rows and columns that are entirely missing are kept. Such a row is
        imputed like any other; such a column is not imputed and not used as
        a predictor (see `impute`).
        """
        logger.info("Initializing MICE object")
        self.data = validate_dataframe(data)
        self.data = validate_columns(self.data)

        logger.debug(f"Input data shape: {self.data.shape}")
        self.id_obs: dict[Hashable, pd.Series] = {}
        self.id_mis: dict[Hashable, pd.Series] = {}
        missing_stats = {}

        for col in self.data.columns:
            notna = self.data[col].notna()
            self.id_obs[col] = notna
            self.id_mis[col] = ~notna
            missing_stats[col] = {"missing_count": (~notna).sum(), "missing_percentage": (~notna).mean() * 100}

        logger.debug("Missing value statistics:")
        for col, stats in missing_stats.items():
            logger.debug(f"  {col}: {stats['missing_count']} values ({stats['missing_percentage']:.2f}%) missing")

        # For storing analysis model results
        self.model_results: list[Any] = []

        # Changes to the imputation model made by `impute` (R's loggedEvents)
        self.logged_events = _empty_log()
        self._log: list[tuple[int, int, Hashable, str, str]] = []
        self._not_imputed: list[Hashable] = []

        # Required by statsmodels result wrappers
        self.nobs = self.data.shape[0]
        logger.info("MICE object initialized successfully")

    def impute(
        self,
        n_imputations: int = 5,
        maxit: int = 5,
        predictor_matrix: pd.DataFrame | None = None,
        initial: str = DEFAULT_INITIAL_METHOD,
        method: str | dict[str, str] | None = None,
        visit_sequence: str | list[str] | None = DEFAULT_VISIT_SEQUENCE,
        seed: int | Sequence[int] | None = None,
        default_method: Sequence[str] = DEFAULT_METHODS,
        where: str | pd.DataFrame | np.ndarray | None = None,
        ignore: Sequence[bool] | np.ndarray | pd.Series | None = None,
        post: Mapping[Hashable, Callable[[pd.Series, pd.DataFrame], Any]] | None = None,
        passive: Mapping[Hashable, Callable[[pd.DataFrame], Any]] | None = None,
        **kwargs: Any,
    ) -> list[pd.DataFrame]:
        """
        Perform multiple imputation by chained equations.

        Parameters
        ----------
        n_imputations : int, default=5
            Number of imputations (R's ``m``), a positive integer.
        maxit : int, default=5
            Number of iterations of each chain, a non-negative integer. With
            0, as in R, no iteration runs: the completed datasets hold the
            start of each chain (see `initial`), and `chain_mean` and
            `chain_var` have no rows. Everything decided before the chains
            start (methods, setup step, visit sequence, checks) is the same
            for every `maxit`. `continue_chains` runs further iterations.
        predictor_matrix : pd.DataFrame, optional
            0/1 matrix with column labels as index (targets) and columns
            (predictors); a 1 means that the column is used as a predictor of
            the row. It may cover only some columns (missing entries count as
            0), but every column that is imputed (with where-cells, see
            `where`, and a method other than ``""``) needs a row; a column
            that is not imputed may be a predictor without a row. If None,
            every other column predicts every column (all ones with a zero
            diagonal), as in R. With ``visit_sequence="monotone"`` and
            ``maxit=1``, as in R, each column then keeps only the predictors
            that come before it in the monotone order of all columns (by
            number of where-cells, the columns without any first), so that
            one iteration imputes a monotone pattern without using start
            values. The setup step described in the Notes then removes
            constant and collinear predictors.
        initial : {"sample", "meanobs"}, default="sample"
            Initial imputation: random draws from the observed values (as in
            R), or the observed value closest to the mean of a numeric column
            and the most frequent observed value of any other column.
        method : str or dict of str to str, optional
            Imputation method, by R's name:

            - donor methods, which impute observed values: ``"pmm"``,
              ``"midas"``, ``"cart"``, ``"rf"``, ``"sample"``;
            - linear regression for numeric columns, which imputes new
              values: ``"norm"`` (Bayesian), ``"norm.boot"`` (bootstrap),
              ``"norm.nob"`` (without parameter uncertainty),
              ``"norm.predict"`` (the predicted value), and ``"mean"`` (the
              observed mean). A nullable integer column (``Int64`` ...)
              imputed by one of them becomes ``Float64``, as R turns an
              integer column into double; float columns keep their dtype;
            - logistic regression for columns with two categories (boolean,
              categorical, text, or numeric with two distinct values), which
              imputes their labels: ``"logreg"`` (Bayesian, on data
              augmented as in R) and ``"logreg.boot"`` (bootstrap);
            - multinomial logistic regression for categorical columns with
              any number of categories (up to 50), which imputes their
              labels: ``"polyreg"`` (on data augmented as in R, fitted as
              R's ``nnet::multinom``);
            - proportional-odds logistic regression for ordered
              categorical columns, which imputes their labels: ``"polr"``
              (on data augmented as in R, fitted as R's ``MASS::polr``).
              Text and unordered categorical columns are taken as ordered
              by their sorted labels; where the model cannot be fitted
              (boolean or numeric columns, two categories, start values
              that cannot be found), it falls back to the fit of
              ``"polyreg"``, as in R;
            - ``""`` for a column that is not imputed.

            A passive column (R's method ``"~ ..."``) is not given here but
            with `passive`; ``"passive"``, its entry in `MICE.method`, is
            not a method name that `method` takes.

            The argument is:

            - str: the same method for all columns (the passive ones
              excepted);
            - dict: the method of each listed column; the columns not in it
              get the default method of their type (see None). It must not
              list a passive column;
            - None: the default method of each column's type, as R's
              ``make.method`` chooses it (`imputation.make_method`):
              ``"pmm"`` for a numeric column, ``"logreg"`` for a boolean
              column or one with two categories, ``"polyreg"`` for an
              unordered categorical, object or string column with more
              categories, ``"polr"`` for an ordered categorical column with
              more categories. `default_method` changes these four.

            As in R, a column without where-cells (see `where`; by default a
            complete column) always gets ``""``, whatever `method` says, and
            a column with where-cells gets a method, also a complete one
            under `where`. A column with method ``""`` keeps its missing
            values; where it is used as a predictor, the rows in which it is
            missing are not used to fit the imputation model and are not
            imputed. As in R, the imputer still receives these rows; only
            ``"logreg"``, ``"polyreg"`` and ``"polr"`` use them, for the
            categories of the target and for the rows they add to the data
            (R's ``augment``). Before the chains start, each method is
            checked against the type of its column (see Raises and Warns).
        visit_sequence : str, list of str or None, default None
            Order in which the columns are visited in each iteration. Only
            the columns that are imputed are visited: with where-cells (by
            default the incomplete columns) and a method other than ``""``,
            passive columns included.

            - None: data order with the passive columns last, as R's default
              (``visitSequence = NULL``), so that a passive column is
              computed from the imputations of its inputs in the same
              iteration. Without passive columns this is "roman";
            - "roman": data order;
            - "arabic": reverse data order;
            - "monotone": increasing number of where-cells (by default of
              missing values), ties in data order; with ``maxit=1`` it also
              edits the predictor matrix, as in R (see `predictor_matrix`);
            - "revmonotone": the reverse of "monotone";
            - "random": a random order, drawn once per call from `seed`;
            - a list of column labels: that order. It must contain every
              column that is imputed, passive ones included; other columns
              in it are skipped. A column may appear more than once and is
              then visited each time, as in R: ``["hgt", "bmi", "wgt",
              "bmi"]`` computes the passive ``bmi`` after each of its
              inputs.

            A keyword or a list places the passive columns where it puts
            them, as in R: under "roman" a passive column before its inputs
            is computed from their values of the previous iteration.
            `MICE.visit_sequence` is the order used, repeats included.
        seed : int, sequence of int, or None, default=None
            Seed for the random number generators. It seeds one
            ``np.random.SeedSequence``, from which independent streams are
            spawned: one for the random visit sequence and one per chain.
            Chain ``k`` always receives the same stream, whatever
            ``n_imputations`` is. The same seed gives identical results;
            ``None`` draws fresh entropy from the operating system.
        default_method : list or tuple of str, default ("pmm", "logreg", "polyreg", "polr")
            The default methods by column type, R's ``defaultMethod``: of
            numeric columns, of boolean columns and columns with two
            categories, of unordered categorical (and text) columns with
            more than two categories, and of ordered categorical columns
            with more than two categories. They apply where `method` is
            None, and to the columns a `method` dict leaves out. Version 0.2
            used ``"pmm"`` for every column; ``method="pmm"`` restores that
            for a call without `method`, ``default_method=("pmm",) * 4`` for
            a call with a partial `method` dict.
        where : str, pd.DataFrame or np.ndarray, optional
            The cells that get an imputation (where-cells), R's ``where``:

            - None or ``"missing"``: the missing cells (the default, as in R);
            - ``"all"``: every cell; ``"observed"``: the observed cells;
              ``"none"``: no cell, so nothing is imputed and the completed
              datasets are copies of the data (R's keywords);
            - a DataFrame of booleans with the index of the data (the same
              labels in the same order) and its column labels in any order,
              aligned by label (R takes a data frame by position);
            - a NumPy array of booleans of the shape of the data, its columns
              in data order.

            An observed where-cell gets an imputation in the completed
            datasets, but keeps its observed value in the data the chains
            work on, so that the other columns are predicted from it, as in
            R. A missing cell that `where` leaves out stays missing. See
            `imputation.validators.check_where` and Notes; the boolean frame
            used is the attribute ``MICE.where``.
        ignore : sequence of bool, optional
            The rows that the imputation models leave out, R's ``ignore``:
            one entry per row, True for a row that is left out of the fit of
            every column and of the values the chains start from, while its
            where-cells are still imputed. None (the default) leaves out no
            row. A list, a tuple, or a NumPy or pandas array in the order of
            the rows, or a Series with the index of the data (the same
            labels in the same order); booleans only (see
            `imputation.validators.check_ignore`). For example, the rows of
            a test set, imputed with models fitted on the training rows only
            (what R's ``mice.mids(newdata=)`` does). See Notes; the array
            used is the attribute ``MICE.ignore``.
        post : mapping of column label to callable, optional
            Functions that change the imputations of a column right after
            they are drawn, R's ``post``: ``{column: f}``. After each
            univariate imputation of the column, in every iteration and
            chain, ``f(imp, data)`` is called with

            - `imp`: the current imputations of the where-cells of the
              column, a Series whose index holds their row labels in row
              order, missing where a cell could not be imputed;
            - `data`: the data the chains work on, in the same rows, all
              columns; in the column itself the missing where-cells hold the
              new imputations and the observed where-cells their observed
              values.

            Both are copies. `f` returns the values to keep, one per entry
            of `imp`: a Series with the index of `imp`, or a sequence of its
            length, of values the column can hold. They replace the
            imputations, and the next columns are imputed from them. For
            example bounds ``{"tv": lambda imp, data: imp.clip(1, 25)}``
            (R's ``squeeze``), a shift for a sensitivity analysis
            ``{"y": lambda imp, data: imp + 2}``, or a rule that reads
            another column ``{"g": lambda imp, data: imp.where(data["age"]
            >= 8, "G1")}``. `f` is not called for a column that is not
            imputed (no where-cells, method ``""``, or removed by the setup
            step), as in R, and must not be given for a passive column.
            `f` gets no random generator: a function that draws random
            numbers must own a seeded one, and the results then depend on
            how the iterations are split between `impute` and
            `continue_chains` unless it draws the same numbers each time.
            The mapping in effect is the attribute ``MICE.post``.
        passive : mapping of column label to callable, optional
            Passive columns, R's passive imputation (a method such as
            ``"~ I(wgt / (hgt / 100)^2)"``): ``{column: f}``, for a column
            that is a function of other columns, such as ``{"bmi": lambda
            d: d["wgt"] / (d["hgt"] / 100) ** 2}``. At each visit of the
            column, ``f(data)`` gets a copy of the data the chains work on
            (all rows, with the current imputations) and returns one value
            per row: a Series with the index of the data, or a sequence of
            its length. Its values go to the where-cells of the column
            (all of them in the completed data; in the data the chains work
            on, as for any column, only the missing ones); nothing is
            drawn and nothing is logged. ``MICE.method`` shows
            ``"passive"`` for the column. As in R, a passive column starts
            from sampled observed values, not from its function; is visited
            after the imputed columns unless `visit_sequence` says
            otherwise; gets ``""`` when it has no where-cells; is not
            checked for being constant but is checked for being collinear
            (see Notes); and counts for the chain statistics. Whether it
            predicts its inputs is set in `predictor_matrix` (R's
            practice: set those entries to 0); it needs no row there.
            Unlike R, which evaluates the formula on the rows being
            imputed, `f` gets all rows: a function that is not row-wise,
            such as ``lambda d: d["x"] - d["x"].mean()``, sees the mean of
            the whole column. Each function is called once on the data
            before the chains start (see Raises), and the dtype of the
            column is fixed from what it returns (see Notes). The mapping in
            effect is the attribute ``MICE.passive``.

        **kwargs
            Parameters of the imputation methods, as
            ``<prefix>_<parameter>``, where the prefix is the method name
            with every dot replaced by an underscore
            (`imputation.utils.method_prefix`): ``pmm_donors=3`` passes
            ``donors=3`` to `imputation.PMM.pmm`, ``norm_boot_ridge=1e-4``
            passes ``ridge=1e-4`` to `imputation.norm.norm_boot`,
            ``polyreg_maxit=200`` passes ``maxit=200`` to
            `imputation.polyreg.polyreg` (R's ``nnet.maxit``),
            ``polr_log_fallback=True`` passes ``log_fallback=True`` to
            `imputation.polr.polr` (R's ``polr.to.loggedEvents``). The
            parameter must exist in that imputer's signature; any other
            keyword argument raises ``TypeError``.

        Returns
        -------
        list of pd.DataFrame
            The `n_imputations` completed datasets: copies of the data in
            which every where-cell holds its imputation, also an observed
            one, and a where-cell that could not be imputed is missing (see
            Notes); every other cell is as in the data. All have the same
            dtypes. They are also stored in `imputed_datasets`; `where` is
            the frame of where-cells used, `ignore` the rows left out of the
            fits, `post` and `passive` the functions in effect, `iteration`
            is set to `maxit`,
            `chain_mean` and `chain_var` get one row per iteration,
            `logged_events` holds the events of this call (see Notes), and
            the state of each chain is kept for `continue_chains`.

        Raises
        ------
        ValueError
            For an invalid argument (number of imputations or iterations,
            method, default method, initial method, visit sequence,
            predictor matrix, `where`: another keyword or type, another
            shape, other column labels or another index than the data,
            values that are not booleans or missing values, see
            `imputation.validators.check_where`; `ignore`: another type or
            number of entries, a Series with another index than the data,
            values that are not booleans or missing values, see
            `imputation.validators.check_ignore`; `post` or `passive`:
            not a mapping, a key that is not a column, a value that cannot
            be called, see `imputation.validators.check_post` and
            `imputation.validators.check_passive`). A `post` entry for a
            passive column (R ignores it). The name ``"passive"`` in
            `method`, and a `method` dict that names a passive column. A
            predictor matrix without a row for a column that is imputed (a
            passive column needs none). Before the chains start, when a
            passive function, called once on the data, raises (the original
            exception is the cause) or does not return one value per row
            of values its column can hold (R finds such a formula in the
            first iteration; with two terms it warns and uses the first).
            Before the chains start, after the setup step, for a column
            that is imputed and has no observed value in the rows that
            `ignore` keeps, with the column named (R starts such a column
            from random normal values and stops in the first iteration with
            "0 (non-NA) cases"); for a passive column without observed value
            there whose function does not return numbers (R starts it from
            random normal values); and for a method that cannot impute the
            type of its column, with the column named
            (`imputation.validators.check_method_types`):
            ``"norm"``, ``"norm.nob"``, ``"norm.boot"``, ``"norm.predict"``
            or ``"mean"`` on a boolean, categorical or text column;
            ``"logreg"`` or ``"logreg.boot"`` on a column with more than two
            distinct observed values; ``"logreg"``, ``"logreg.boot"``,
            ``"polyreg"`` or ``"polr"`` on a column with more than 50
            distinct observed values (R's limit; for example a text column
            of names, which gets ``"polyreg"`` by default); the message says
            to remove such a column from the data if it identifies the rows,
            and otherwise to impute it with a donor method, under which the
            column stays a predictor of the other columns, with one dummy
            column per value, unless the predictor matrix excludes it.
            Inside a chain, when an imputer cannot impute a column (for
            example ``"midas"`` on a categorical column, a model with more
            weights than ``polyreg_max_weights``, ``"norm.nob"`` with too
            few observed values for its predictors, or a column without
            observed value in the rows usable for its model); the message
            starts with the column, the method, the iteration and the
            imputation, followed by the imputer's message. Inside a chain,
            when a `post` or `passive` function raises (the message names
            the column, ``post`` or ``passive``, the iteration and the
            imputation, and the original exception is the cause), or
            returns something else than one value per where-cell (`post`)
            or per row (`passive`): a scalar, a DataFrame, a sequence of
            another length, a Series with another index, or values that the
            column cannot hold (a number that is not whole in an integer
            column, a label that is not a category, text in a numeric
            column, ...).
        TypeError
            For an unknown keyword argument. For one of the arguments
            removed in version 0.4 (`min_cor`, `min_puc`, `include`,
            `exclude`, `correlation_method`, `output_dir`), the message says
            what replaces it: `imputation.quickpred`, whose result is passed
            as `predictor_matrix`, or nothing for `output_dir`, which never
            had an effect.

        Warns
        -----
        UserWarning
            When the imputation model was changed (see Notes), with the
            number of logged events, as R's "Number of logged events"; also
            for more than 100 imputations or 50 iterations, and for
            ``"logreg"``, ``"logreg.boot"``, ``"polyreg"`` or ``"polr"`` on a
            numeric column, with R's text "Type mismatch for variable(s):
            <column>\\nImputation method <method> is for categorical data."
            When observed where-cells are missing in the completed datasets
            (see Notes), once, with the number of observed values lost in
            each column and the reason: no method, predictors incomplete
            in these rows, or (for a passive column) the function missing in
            these rows. R loses them silently. When `ignore` is given
            and leaves fewer than 10 rows for the fits, with R's text
            "Fewer than 10 rows for fitting the imputation model. Are you
            sure?"; the imputation goes on, as in R.

        Notes
        -----
        What is imputed, as in R: each column with where-cells and a method
        is visited in each iteration. Its model is fitted on the rows where
        the column is observed, its predictors are complete in the data the
        chains work on and `ignore` does not leave the row out, and the
        imputer fills the where-cells whose predictors are complete there.
        These data hold the imputations of the missing where-cells only: an
        observed where-cell keeps its observed value, and a missing cell
        that `where` leaves out stays missing. So a where-cell becomes
        missing (in the completed data too, also where it was observed)
        when its column has no method (``""``, set by the user or by the
        setup step), or when one of its predictors is missing in its row at
        the visit: a missing cell left out of `where`, a where-cell that
        could not be imputed itself, or the missing cell of a predictor
        without a method. Such missing values spread to the columns that
        the column predicts. The chain statistics are taken over all
        where-cells of a column at the end of each iteration, so after
        `post`, and for a column visited twice in an iteration after its
        last visit. A NumPy integer or boolean column with
        where-cells (possible only under `where`, since it is complete)
        becomes ``Int64`` (the nullable integer dtype of its width) or
        ``boolean`` in the completed datasets, because a where-cell can end
        up missing; then the rule for the norm methods and ``"mean"`` (see
        `method`) applies.

        A passive column (`passive`) is computed, not imputed: at each
        visit its where-cells get the value its function gives for their
        row, from the data the chains work on, so a passive value is
        missing where the function returns a missing value (for example
        where an input keeps a missing value), and that spreads to the
        columns it predicts. Under the default `where` every completed
        dataset holds the function of its own row in the missing cells of
        the column, as long as the passive column is visited after its
        inputs (the default `visit_sequence`). An observed where-cell of a
        passive column gets the function value in the completed data and
        keeps its observed value in the data the chains work on, as for any
        column. Its dtype is fixed before the chains start from what the
        function returned on the data: an integer column (``Int64`` and
        the like, or a NumPy integer column with where-cells) whose
        function returns floats becomes ``Float64``; an entirely missing
        column takes the dtype of the result (``Float64`` for integers,
        since it starts from standard normal draws, as does an integer
        column without observed value in the rows that `ignore` keeps);
        any other column keeps its dtype, and a value it cannot hold raises
        ``ValueError``. The setup step (below) does not check a passive
        column for being constant, so an entirely missing one is kept, but
        checks it for being collinear: a passive copy of another column
        gets ``""``, keeps its missing values and is logged.

        A row that `ignore` leaves out, as in R, is left out of the fit
        rows of every column in every iteration, and its observed values
        are not drawn as start values (``initial="meanobs"`` takes its
        values from the other rows too); its where-cells are imputed like
        any other. Everything decided before the chains start uses all
        rows: the methods, the setup step, the visit sequence. The checks
        of each univariate imputation ("df set to 1" and the dropped
        predictors, see below) look at the fit rows, so without the ignored
        rows. The imputer still receives the ignored rows, in neither of its
        masks where their cell is observed: ``"logreg"``, ``"polyreg"`` and
        ``"polr"`` take the categories of the target and the statistics of
        the rows they add (R's ``augment``) over them too.

        As R's ``mice``, the imputation model is edited before the chains
        start and before every univariate imputation; each change is a row
        of `logged_events`:

        - Setup (``it`` 0): a variable that is constant (variance below
          ``1000 * eps``; also when fewer than two values are observed, so
          also an entirely missing variable) is no longer used as a
          predictor (``meth`` "constant"); if it has where-cells, it is not
          imputed: its missing values are kept, and its observed
          where-cells become missing too. The same holds for a
          variable that (nearly) duplicates another one
          (`imputation.design.find_collinear`, ``meth`` "collinear").
          Object and string columns count as categorical variables here.
          As in R, a passive column is checked for being collinear only,
          and the `post` function of a column removed here is not applied
          (it is not in ``MICE.post``).
        - Each univariate imputation: the predictors are dummy-coded by
          `imputation.design.obtain_design`, and columns that are constant
          or linearly dependent on the fit rows are dropped
          (`imputation.design.remove_lindep`); ``out`` names the dropped
          columns. In the first iteration (not again in `continue_chains`),
          a model with fewer fit rows (R's "observed cases") than
          parameters logs "df set to 1".
        - An imputer may log events itself: with ``polr_log_fallback=True``
          each fallback of ``"polr"`` to the multinomial model logs
          "polr falls back to multinom" (``meth`` "polr"), as R's
          ``polr.to.loggedEvents = TRUE``.
        """
        logger.info("Starting imputation process")
        logger.debug(
            f"Parameters: n_imputations={n_imputations}, maxit={maxit}, "
            f"initial={initial}, method={method}, visit_sequence={visit_sequence}, default_method={default_method}"
        )

        start_time = time.time()

        check_n_imputations(n_imputations)
        check_maxit(maxit)
        check_initial_method(initial)
        default_method = check_default_method(default_method)
        n_imputations = int(n_imputations)
        maxit = int(maxit)

        # One seed sequence per call, split into independent streams: the
        # first for the random visit sequence, then one per chain. Chain k
        # always gets child k + 1, so its stream does not depend on n_imputations.
        seed_sequence = np.random.SeedSequence(seed)
        visit_stream = seed_sequence.spawn(1)[0]
        chain_streams = seed_sequence.spawn(n_imputations)

        _check_method_params(kwargs)
        # The cells that get an imputation, R's where, and the rows the
        # models leave out, R's ignore
        where_cells = check_where(where, self.data)
        ignored_rows = check_ignore(ignore, self.data)
        # The functions of post, and those of the passive columns with what
        # each returns on the data (it fixes their dtypes below)
        post_functions = check_post(post, self.data)
        passive_values = check_passive(passive, self.data)
        both = [col for col in post_functions if col in passive_values]
        if both:
            raise ValueError(
                f"post names the passive columns {both}: post is not applied to a passive column (R ignores it "
                "silently). Remove them from post, or put the change into their function in passive."
            )
        columns = list(self.data.columns)

        # The settings of the last run change from here on: its chains cannot
        # be continued any more, also if this call stops with an error
        self._chain_states = []
        self.where: pd.DataFrame = where_cells
        self.ignore: np.ndarray = ignored_rows

        # The default method of each column's type (R's make.method), for
        # method=None and for the columns a method dict leaves out
        if method is None:
            self.method = make_method(self.data, default_method, where=where_cells)
        elif isinstance(method, dict):
            defaults = make_method(self.data, default_method, where=where_cells)
            self.method = check_method(method, columns, defaults=defaults, passive=passive_values)
        else:
            self.method = check_method(method, columns, passive=passive_values)
        # A passive column is computed by its function, R's passive method
        for col in passive_values:
            self.method[col] = PASSIVE_METHOD
        # As in R, a column without where-cells has no imputation method
        for col in columns:
            if not where_cells[col].any():
                self.method[col] = ""
        logger.debug(f"Using imputation methods: {self.method}")

        if predictor_matrix is None:
            predictor_matrix = _full_predictor_matrix(columns)
        else:
            # A passive column has no predictors: it needs no row
            imputed_columns = [col for col in columns if self.method[col] not in ("", PASSIVE_METHOD)]
            predictor_matrix = validate_predictor_matrix(predictor_matrix, columns, self.data, imputed_columns)
            logger.debug("Predictor matrix validated successfully")
        # Square over all columns, in data order; absent entries are 0
        predictor_matrix = predictor_matrix.reindex(index=columns, columns=columns, fill_value=0).astype(int)
        # One iteration of a monotone sequence: each column is predicted only
        # by the columns visited before it, as R's mice.edit.predictorMatrix
        if maxit == 1 and isinstance(visit_sequence, str) and visit_sequence == VisitSequence.MONOTONE.value:
            predictor_matrix = _monotone_predictor_matrix(predictor_matrix, where_cells)

        self.imputation_params = kwargs
        param_names = _method_param_names()

        # Warn if user provided method-specific parameters for methods not used
        if self.imputation_params:
            provided_methods = set()
            for key in self.imputation_params.keys():
                match = split_method_param(key, param_names)
                if match is not None:
                    provided_methods.add(match[0])
            used_methods = set(self.method.values())
            unused_provided = provided_methods - used_methods
            if unused_provided:
                logger.warning(
                    "Method-specific parameters were provided for unused methods: %s. "
                    "These parameters will be ignored.",
                    sorted(list(unused_provided)),
                )

        self.n_imputations = n_imputations
        self.maxit = maxit
        self.initial = initial

        # Edit the model before the chains start, as R's mice.edit.setup
        self._log = []
        self.predictor_matrix = self._edit_setup(predictor_matrix)
        # Each method must suit the type of the column it imputes (R's check.method)
        check_method_types(
            self.data, {col: meth for col, meth in self.method.items() if meth not in ("", PASSIVE_METHOD)}
        )
        # Each column that is imputed needs a row to fit on and start from
        self._check_rows_not_ignored()
        # A passive column without any needs a function that returns numbers
        self._check_passive_start(passive_values)
        # The functions in effect: as in R, none for a column without a
        # method, also one whose method the setup step removed
        self.post = {col: function for col, function in post_functions.items() if self.method[col] != ""}
        self.passive = {
            col: function for col, function in (passive or {}).items() if self.method[col] == PASSIVE_METHOD
        }

        self._set_visit_sequence(visit_sequence, np.random.default_rng(visit_stream))
        logger.debug(f"Visit sequence set to: {self.visit_sequence}")

        # The data the chains start from, with the dtypes of every completed
        # dataset. A NumPy integer or boolean column with where-cells (it is
        # complete, so only under where) becomes nullable, because a
        # where-cell can end up missing. The norm methods and mean impute
        # floats: a nullable integer column they impute becomes Float64, as R
        # turns an integer column into double. A passive column gets its
        # dtype from what its function returned on the data (_passive_dtype)
        self._chain_data = self.data.copy(deep=True)
        for col in columns:
            nullable = _nullable_dtype(self._chain_data[col].dtype)
            if nullable is not None and where_cells[col].any():
                self._chain_data[col] = self._chain_data[col].astype(nullable)
        for col in self.visit_sequence:
            if self.method[col] in _FLOAT_METHODS and _is_nullable_integer(self._chain_data[col].dtype):
                self._chain_data[col] = self._chain_data[col].astype("Float64")
        for col in self.passive:
            starts_from_draws = not (self.id_obs[col].to_numpy() & ~self.ignore).any()
            dtype = _passive_dtype(self._chain_data[col], passive_values[col], starts_from_draws)
            if dtype is not None:
                self._chain_data[col] = self._chain_data[col].astype(dtype)
            # What the function returned on the data must fit the column
            _in_column_dtype(
                passive_values[col],
                self._chain_data[col].dtype,
                f"Column {col!r}, passive (called once on the data before the chains start)",
            )

        # Chain statistics of the numeric columns that are imputed (each
        # once, also if visited twice): one row per iteration, appended as
        # the chains run
        numeric_cols = self._chain_data.select_dtypes(include=[np.number]).columns
        numeric_cols_to_impute = list(dict.fromkeys(col for col in self.visit_sequence if col in numeric_cols))
        self.chain_mean = {col: np.empty((0, self.n_imputations), dtype=float) for col in numeric_cols_to_impute}
        self.chain_var = {col: np.empty((0, self.n_imputations), dtype=float) for col in numeric_cols_to_impute}

        # Fixed for the whole run: the predictors of each target, in data
        # order, and the parameters of each method (a passive column has
        # neither)
        self._predictors = {
            col: [c for c in columns if c != col and self.predictor_matrix.at[col, c] != 0]
            for col in self.visit_sequence
            if self.method[col] != PASSIVE_METHOD
        }
        self._method_params: dict[str, dict[str, Any]] = {
            method_name: {}
            for method_name in set(self.method[col] for col in self.visit_sequence)
            if method_name != PASSIVE_METHOD
        }
        for key, value in self.imputation_params.items():
            match = split_method_param(key, param_names)
            if match is not None and match[0] in self._method_params:
                self._method_params[match[0]][match[1]] = value

        self.imputed_datasets: list[pd.DataFrame] = []

        # The start of each chain, from its own generator (R's initialize.imp)
        states: list[_ChainState] = []
        for chain_idx in range(self.n_imputations):
            chain_rng = np.random.default_rng(chain_streams[chain_idx])
            states.append(_ChainState(imp=self._initial_imputation(chain_rng), rng=chain_rng))
        self._chain_states = states
        self.iteration = 0

        self._run_chains(range(maxit))
        logger.info(f"impute completed in {time.time() - start_time:.2f} seconds")
        logger.debug("Final imputation statistics:")
        logger.debug(f"  - Number of imputations: {self.n_imputations}")
        logger.debug(f"  - Maximum iterations: {self.maxit}")
        logger.debug(f"  - Initial method: {self.initial}")
        logger.debug(f"  - Method: {self.method}")
        logger.debug(f"  - Visit sequence: {self.visit_sequence}")

        self._report_logged_events(warn=True)
        self._warn_lost_observed_values()

        return self.imputed_datasets

    def continue_chains(self, maxit: int = 1) -> list[pd.DataFrame]:
        """
        Run more iterations of the chains of the last `impute`, as R's ``mice.mids``.

        Each chain goes on from where it stopped: from its current
        imputations, with its own generator in the state it reached. Nothing
        is decided again: the start, the visit sequence (also a ``"random"``
        one), the methods, the predictor matrix, the setup step, the
        where-cells, the rows left out by `ignore` and the functions of
        `post` and `passive` (`MICE.post`, `MICE.passive`) are those of the
        last `impute`, and the check of the first iteration ("df set to 1",
        see `impute`) does not run again. The chains run one after the
        other, as in `impute`.

        Parameters
        ----------
        maxit : int, default=1
            Number of further iterations of each chain, a non-negative
            integer. 0 runs nothing and returns the datasets as they are.

        Returns
        -------
        list of pd.DataFrame
            The `n_imputations` completed datasets after the new iterations,
            also stored in `imputed_datasets`.

        Raises
        ------
        ValueError
            If `impute` was not called first, or if the last call of `impute`
            or `continue_chains` stopped with an error (its chains cannot be
            continued); for an invalid `maxit`; inside a chain, as in
            `impute`, also for a `post` or `passive` function.

        Warns
        -----
        UserWarning
            When the new iterations logged an event, with the number of all
            logged events, as `impute` warns (R's ``mice.mids`` prints
            nothing); for more than 50 iterations; when observed where-cells
            are missing in the completed datasets, as `impute` warns.

        Notes
        -----
        `iteration` goes up by `maxit`. The rows of the new iterations are
        appended to `chain_mean` and `chain_var`, and their events to
        `logged_events`, numbered by the iteration they happened in. As
        `impute`, the method empties `model_results` and removes
        `pooled_result`: call `fit` again.

        ``impute(maxit=a + b, seed=s)`` and ``impute(maxit=a, seed=s)``
        followed by ``continue_chains(b)`` give identical results: the
        completed datasets, `chain_mean`, `chain_var`, `logged_events` and
        `iteration`. This holds for ``a = 0`` and for a continuation split
        into several calls, also with `post` and `passive`, as long as their
        functions give the same values for the same input (a `post`
        function that draws from a generator of its own does not).

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from imputation import MICE
        >>> rng = np.random.default_rng(1)
        >>> df = pd.DataFrame({"x": rng.normal(size=50)})
        >>> df["y"] = 2 * df["x"] + rng.normal(size=50)
        >>> df.loc[::5, "y"] = np.nan
        >>> mice = MICE(df)
        >>> _ = mice.impute(n_imputations=3, maxit=2, seed=1)
        >>> datasets = mice.continue_chains(3)
        >>> mice.iteration, mice.chain_mean["y"].shape
        (5, (5, 3))
        >>> longer = MICE(df).impute(n_imputations=3, maxit=5, seed=1)
        >>> all(a.equals(b) for a, b in zip(datasets, longer))
        True
        """
        if not getattr(self, "_chain_states", None):
            raise ValueError(
                "There are no chains to continue: call impute() first. "
                "A run that stopped with an error cannot be continued."
            )
        check_maxit(maxit)
        maxit = int(maxit)
        if maxit == 0:
            return self.imputed_datasets

        start_time = time.time()
        n_logged = len(self._log)
        self._run_chains(range(self.iteration, self.iteration + maxit))
        logger.info(f"continue_chains completed in {time.time() - start_time:.2f} seconds")
        # As impute, but only when the new iterations logged an event
        self._report_logged_events(warn=len(self._log) > n_logged)
        self._warn_lost_observed_values()
        return self.imputed_datasets

    def _run_chains(self, iterations: range) -> None:
        """
        Run the iterations `iterations` of every chain from its state; shared by `impute` and `continue_chains`.

        The chains run one after the other (`_impute_once`), each with its
        own generator. The chain statistics get one row per iteration, the
        events go to the log after those of earlier calls, and the fitted
        models of earlier datasets are removed. Sets `imputed_datasets` and
        `iteration`. If a chain raises, the chains are dropped, because some
        of them went on and others did not: the run cannot be continued.

        Parameters
        ----------
        iterations : range
            The indices of the iterations to run (0 for R's iteration 1):
            from the number of iterations run so far, in steps of 1.
        """
        for col in self.chain_mean:
            self.chain_mean[col] = _with_rows(self.chain_mean[col], len(iterations))
            self.chain_var[col] = _with_rows(self.chain_var[col], len(iterations))
        # Fits of the earlier datasets must not be pooled any more
        self.model_results = []
        if hasattr(self, "pooled_result"):
            del self.pooled_result

        datasets = []
        individual_times = []
        try:
            for chain_idx, state in enumerate(self._chain_states):
                chain_start_time = time.time()
                logger.info(f"Starting imputation chain {chain_idx + 1}/{self.n_imputations}")
                datasets.append(self._impute_once(state, chain_idx, iterations))
                individual_times.append(time.time() - chain_start_time)
                logger.info(f"Completed imputation chain {chain_idx + 1} in {individual_times[-1]:.2f} seconds")
        except BaseException:
            # Some chains went on and others did not
            self._chain_states = []
            raise
        self.imputed_datasets = datasets
        self.iteration = iterations.stop

        logger.info(f"{len(iterations)} iterations of {len(datasets)} chains completed, {self.iteration} in all")
        logger.debug(f"Individual chain times: {[f'{t:.2f}s' for t in individual_times]}")

    def _log_event(self, out, meth, it=0, im=0, dep=""):
        """Record one change of the imputation model, as R's ``updateLog``."""
        self._log.append((it, im, dep, meth, out))

    def _report_logged_events(self, warn: bool) -> None:
        """Fill `logged_events` from the log and, with `warn`, warn once if anything was logged.

        R runs the iterations in the outer loop and the imputations in the
        inner one; the chains here run one after the other. Sorting by
        iteration and imputation (stable) gives R's order, also when
        `continue_chains` has added the events of later iterations to the log.
        """
        if not self._log:
            self.logged_events = _empty_log()
            return
        events = pd.DataFrame(self._log, columns=list(_LOG_COLUMNS))
        events = events.sort_values(["it", "im"], kind="stable").reset_index(drop=True)
        self.logged_events = events.astype({"it": "int64", "im": "int64"})
        if not warn:
            return
        message = f"Number of logged events: {len(events)}."
        if self._not_imputed:
            # Under where, observed where-cells of such a column become missing too
            if any((self.where[col] & self.id_obs[col]).any() for col in self._not_imputed):
                kept = "their where-cells are missing in the completed data, the observed ones too"
            else:
                kept = "their missing values are kept"
            message += f" Not imputed because constant or collinear, {kept}: {self._not_imputed}."
        message += " See MICE.logged_events."
        logger.info(message)
        warnings.warn(message, UserWarning, stacklevel=3)

    def _warn_lost_observed_values(self) -> None:
        """Warn once when observed where-cells are missing in the completed datasets.

        R's ``complete`` writes every where-cell, also one without an
        imputation, silently: a cell whose column has no method (set by the
        user, or by the setup step), whose predictors are incomplete in its
        row when its column is visited, or, in a passive column, where its
        function returned a missing value. The warning gives, per column,
        the number of observed values replaced by missing values and why.
        Without where-cells on observed cells nothing is lost.
        """
        observed_where = self.where & self.data.notna()
        lost = {}
        for col in self.where.columns[observed_where.any().to_numpy()]:
            cells = observed_where[col].to_numpy()
            missing = np.zeros(int(cells.sum()), dtype=bool)
            for dataset in self.imputed_datasets:
                missing |= dataset[col].isna().to_numpy()[cells]
            if missing.any():
                if self.method[col] == "":
                    reason = "no method"
                elif self.method[col] == PASSIVE_METHOD:
                    reason = "passive function missing in these rows"
                else:
                    reason = "predictors incomplete in these rows"
                lost[col] = f"{col!r}: {int(missing.sum())} ({reason})"
        if not lost:
            return
        message = (
            "Observed values replaced by missing values in the completed data, because where marks them and they "
            f"could not be imputed (as in R); per column: {', '.join(lost.values())}. See MICE.where."
        )
        logger.info(message)
        warnings.warn(message, UserWarning, stacklevel=3)

    def _edit_setup(self, predictor_matrix: pd.DataFrame) -> pd.DataFrame:
        """Remove constant and collinear variables, as R's ``mice.edit.setup``.

        A variable that is constant, or (nearly) collinear with another
        variable that is used as a predictor, stops being a predictor. If it
        has a method, the method becomes ``""``: it is not imputed, keeps its
        missing values (its observed where-cells, under `where`, become
        missing in the completed data) and leaves the visit sequence. Each
        variable affected is logged once (``meth`` "constant" or "collinear").
        As in R, a passive column is not checked for being constant (so an
        entirely missing one stays), but is checked for being collinear: a
        passive copy of another column loses its method too.
        Updates `self.method`; returns the edited predictor matrix.

        Notes
        -----
        R stops when this leaves no predictor at all; here the imputation
        goes on (targets without predictors are imputed from their observed
        values alone), and the logged events say what was removed.
        """
        columns = list(self.data.columns)
        pred = predictor_matrix.to_numpy(copy=True)
        self._not_imputed = []

        def remove(j, reason):
            col = columns[j]
            logged = False
            if pred[:, j].any():
                pred[:, j] = 0
                self._log_event(out=str(col), meth=reason)
                logged = True
            if self.method[col] != "":
                pred[j, :] = 0
                if not logged:
                    self._log_event(out=str(col), meth=reason)
                self.method[col] = ""
                self._not_imputed.append(col)

        for j, col in enumerate(columns):
            if self.method[col] == PASSIVE_METHOD:
                # As in R: a passive column is not checked for being constant
                continue
            variance = _variance(self.data[col])
            if not variance >= _CONSTANT_VARIANCE:  # NaN: fewer than two observed values
                remove(j, "constant")

        is_predictor = pred.any(axis=0)
        if is_predictor.any():
            for col in find_collinear(self.data.loc[:, is_predictor]):
                remove(columns.index(col), "collinear")

        if self._not_imputed:
            logger.info("Not imputed because constant or collinear: %s", self._not_imputed)
        return pd.DataFrame(pred, index=predictor_matrix.index, columns=predictor_matrix.columns)

    def _check_rows_not_ignored(self) -> None:
        """Raise ValueError if a column that is imputed has no observed value in the rows that `ignore` keeps.

        Its model would have no row to be fitted on, and its start no value
        to draw. R starts such a column from random normal values and stops
        in the first iteration ("0 (non-NA) cases"); here the run stops
        before the chains start. Called after the setup step: a column
        without a method needs no such row, also one whose method the setup
        step removed (a column observed in fewer than two rows is constant).
        A passive column has no model; `_check_passive_start` checks its
        start.
        """
        without = [
            col
            for col in self.data.columns
            if self.method[col] not in ("", PASSIVE_METHOD) and not (self.id_obs[col].to_numpy() & ~self.ignore).any()
        ]
        if not without:
            return
        if len(without) == 1:
            columns, its = f"the column {without[0]!r}", "its"
        else:
            columns, its = f"the columns {without}", "their"
        raise ValueError(
            f"Every observed value of {columns} lies in a row that ignore leaves out: {its} imputation model has no "
            f"row to be fitted on and {its} start no value to draw (R stops in the first iteration with "
            "'0 (non-NA) cases'). Leave out fewer rows, or give the method '' to these columns."
        )

    def _check_passive_start(self, passive_values: dict[Hashable, pd.Series]) -> None:
        """Raise ValueError for a passive column that cannot start: no observed value, and a function without numbers.

        As in R, a passive column starts from its observed values in the
        rows that `ignore` keeps, and from standard normal draws when it has
        none there. Such draws suit a column whose function returns numbers
        (what it returned on the data, `passive_values`); for another one
        (booleans, categories, text) the run stops before the chains start.
        Called after the setup step, for the passive columns in effect.
        """
        for col, values in passive_values.items():
            if self.method[col] != PASSIVE_METHOD or (self.id_obs[col].to_numpy() & ~self.ignore).any():
                continue
            if not _is_number_dtype(values.dtype):
                rows = "in the rows that ignore keeps" if self.ignore.any() else "at all"
                raise ValueError(
                    f"The passive column {col!r} has no observed value {rows} to start from, and its function "
                    f"returns values of dtype {values.dtype}, not numbers: R starts such a column from standard "
                    "normal draws, which only a numeric column can take. Give it observed values to start from, or "
                    "impute it with a method instead."
                )

    def fit(
        self,
        formula: str,
        model: Callable[..., Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        **fit_kwargs: Any,
    ) -> list[Any]:
        """
        Fit a statistical model to each imputed dataset using the specified formula.

        This method fits the specified statistical model to each dataset in
        self.imputed_datasets and stores the results in self.model_results.
        It corresponds to R's ``with(imp, <model>)``.

        Parameters
        ----------
        formula : str
            A formula string in patsy syntax for statsmodels (e.g., 'y ~ x1 + x2')
        model : callable, optional
            A statsmodels formula model such as ``statsmodels.formula.api.ols``,
            ``logit``, ``poisson`` or ``glm``, called as
            ``model(formula, data=dataset, **model_kwargs)``. A statsmodels
            model class (for example ``sm.GLM``) is used through its
            ``from_formula``. Default: ``statsmodels.formula.api.ols``.
        model_kwargs : dict, optional
            Keyword arguments for the model, for example
            ``{"family": sm.families.Binomial()}`` for ``smf.glm``.
        **fit_kwargs
            Keyword arguments passed to the model's ``.fit()``, for example
            ``disp=False`` for ``smf.logit``.

        Returns
        -------
        list
            The fitted results, one per imputed dataset (also stored in
            ``self.model_results``).

        Raises
        ------
        ValueError
            If no imputed datasets are available or if variables in formula are not in data
        RuntimeError
            If the model cannot be fitted to one of the imputed datasets

        Examples
        --------
        >>> import statsmodels.api as sm
        >>> import statsmodels.formula.api as smf
        >>> mice_obj = MICE(data)  # doctest: +SKIP
        >>> mice_obj.impute(n_imputations=5)  # doctest: +SKIP
        >>> mice_obj.fit('outcome ~ predictor1 + predictor2')  # doctest: +SKIP
        >>> mice_obj.fit('event ~ age + dose', model=smf.logit, disp=False)  # doctest: +SKIP
        >>> mice_obj.fit('count ~ age', model=smf.glm,
        ...              model_kwargs={'family': sm.families.Poisson()})  # doctest: +SKIP
        """
        logger.info(f"Starting analysis with formula: {formula}")

        # Check if imputation has been performed
        if not hasattr(self, "imputed_datasets") or not self.imputed_datasets:
            msg = "No imputed datasets found. Please run .impute() first."
            logger.error(msg)
            raise ValueError(msg)

        # Validate formula
        validate_formula(formula, list(self.data.columns))

        if model is None:
            model = smf.ols
        elif hasattr(model, "from_formula"):
            model = model.from_formula
        if not callable(model):
            raise ValueError(f"model must be a statsmodels formula model such as smf.ols, got {model!r}")
        model_kwargs = dict(model_kwargs or {})

        # Clear any previous model results
        self.model_results = []

        # Fit model to each imputed dataset
        n_datasets = len(self.imputed_datasets)
        model_name = getattr(model, "__qualname__", repr(model))
        logger.info(f"Fitting {model_name} to {n_datasets} imputed datasets")

        for i, dataset in enumerate(self.imputed_datasets):
            logger.debug(f"Fitting model to dataset {i + 1}/{n_datasets}")

            try:
                fitted_model = model(formula, data=dataset, **model_kwargs).fit(**fit_kwargs)
                self.model_results.append(fitted_model)

                logger.debug(f"Successfully fitted model to dataset {i + 1}")

            except Exception as e:
                logger.error(f"Error fitting model to dataset {i + 1}: {str(e)}")
                raise RuntimeError(f"Failed to fit model to dataset {i + 1}: {str(e)}") from e

        # Store formula for potential later use
        self.formula = formula

        logger.info(f"Analysis completed successfully. Fitted models to {len(self.model_results)} datasets")
        logger.debug(f"Model results stored in self.model_results with {len(self.model_results)} entries")
        return self.model_results

    def pool(self, summ: bool = False) -> dict[str, Any] | Summary:
        """
        Pool parameter estimates from fitted models using Rubin's rules.

        This method combines parameter estimates and their uncertainties from
        multiple imputed datasets according to Rubin's (1987) rules for
        multiple imputation inference, with the small-sample degrees of
        freedom of Barnard and Rubin (1999), as R's ``pool()`` does.

        Parameters
        ----------
        summ : bool, default=False
            If True, return the statsmodels summary of the pooled result
            instead of the dict.

        Returns
        -------
        dict or summary
            If summ=False, a dict with the keys ``pooled_result`` (a
            MICEresult), ``pooled_params``, ``pooled_covariance``,
            ``within_covariance``, ``between_covariance``,
            ``fraction_missing_info`` (R's ``fmi``), ``pooled_scale``,
            ``n_imputations``, ``parameter_names``, ``formula`` and ``table``
            (the DataFrame of `imputation.pooling.pool_estimates`, one row
            per parameter, with R's columns ``df``, ``riv``, ``lambda``,
            ``fmi``, intervals and p-values).
            If summ=True, the statsmodels summary of the MICEresult, which
            shows the df and FMI of each parameter.

        Raises
        ------
        ValueError
            If `fit` has not been called.

        Warns
        -----
        UserWarning
            With a single imputation, where nothing can be pooled.

        Notes
        -----
        Rubin's pooling rules combine, per parameter:

        - Point estimates: average across imputations
        - Within-imputation variance (ubar): average of individual model variances
        - Between-imputation variance (b): variance of point estimates across imputations
        - Total variance (t): ubar + (1 + 1/m) * b
        - Degrees of freedom (df): Barnard-Rubin, with ``dfcom`` the residual
          degrees of freedom of the first fitted model
        - riv = (1 + 1/m) * b / ubar, lambda = (1 + 1/m) * b / t, and the
          fraction of missing information fmi = (riv + 2/(df + 3)) / (riv + 1)

        Standard errors, t statistics, p-values and confidence intervals of
        the MICEresult use the t distribution with each parameter's df.
        With a single imputation nothing is pooled (as in R): the
        between-imputation variance is NaN and the single analysis is
        reported with ``dfcom`` degrees of freedom.

        References
        ----------
        Rubin, D.B. (1987). Multiple Imputation for Nonresponse in Surveys.
        New York: John Wiley and Sons.

        Barnard, J. and Rubin, D.B. (1999). Small-sample degrees of freedom
        with multiple imputation. Biometrika, 86, 948-955.
        """
        logger.info("Starting pooling of model results using Rubin's rules")

        # Check if analysis has been performed
        if not hasattr(self, "model_results") or not self.model_results:
            msg = "No model results found. Please run .fit() first."
            logger.error(msg)
            raise ValueError(msg)

        # Check if formula was stored (should be set by fit())
        if not hasattr(self, "formula"):
            logger.warning("No formula found. This may indicate .fit() was not called properly.")

        m = len(self.model_results)  # Number of imputations
        logger.info(f"Pooling estimates from {m} fitted models")

        # Extract parameters, covariances, and scales from each model
        params_list = []
        cov_within_list = []
        scale_list = []

        for i, model_result in enumerate(self.model_results):
            logger.debug(f"Extracting results from model {i + 1}")

            # Extract parameter estimates. Multi-equation models (MNLogit)
            # hold one column per equation; their covariance stacks the
            # equations, which is column-major order.
            params_list.append(np.asarray(model_result.params, dtype=float).ravel(order="F"))

            # Extract covariance matrix (within-imputation variance)
            cov_within_list.append(np.asarray(model_result.cov_params(), dtype=float))

            # Extract scale (residual variance); not every model has one
            try:
                scale_list.append(float(model_result.scale))
            except (AttributeError, TypeError, ValueError, NotImplementedError):
                scale_list.append(np.nan)

        # Convert to numpy arrays for easier computation
        params_array = np.array(params_list)  # Shape: (m, p) where p = number of parameters
        cov_within_array = np.array(cov_within_list)  # Shape: (m, p, p)
        scale_array = np.array(scale_list)

        logger.debug(f"Parameter array shape: {params_array.shape}")
        logger.debug(f"Covariance array shape: {cov_within_array.shape}")

        # Create parameter names (use from first model)
        first_params = self.model_results[0].params
        if np.ndim(first_params) == 2:
            # Multi-equation model: '<equation>:<term>', as its covariance labels them
            labels = getattr(self.model_results[0].cov_params(), "index", range(params_array.shape[1]))
            param_names = [":".join(map(str, k)) if isinstance(k, tuple) else k for k in labels]
        else:
            param_names = list(getattr(first_params, "index", range(params_array.shape[1])))
        logger.debug(f"Parameter names: {param_names}")

        # Per-parameter pooling as R's pool(): Rubin's rules, Barnard-Rubin df
        # with dfcom the residual df of the first fitted model
        dfcom = _residual_df(self.model_results[0])
        table = pool_estimates(
            params_array,
            np.diagonal(cov_within_array, axis1=1, axis2=2),
            dfcom=dfcom,
            param_names=param_names,
        )

        # Rubin's rules for the full covariance matrices
        # 1. Pooled point estimates (qbar): average of individual estimates
        pooled_params = np.mean(params_array, axis=0)
        logger.debug(f"Computed pooled parameter estimates: {pooled_params}")

        # 2. Within-imputation variance (ubar): average of individual covariances
        cov_within = np.mean(cov_within_array, axis=0)

        # 3. Between-imputation variance (b): covariance of parameter estimates across imputations
        # 4. Total covariance matrix: within + (1 + 1/m) * between
        if m > 1:
            cov_between = np.cov(params_array, rowvar=False, ddof=1).reshape(cov_within.shape)
            cov_total = cov_within + (1.0 + 1.0 / m) * cov_between
        else:
            # Nothing to pool: b cannot be estimated, t is the single analysis
            cov_between = np.full_like(cov_within, np.nan)
            cov_total = cov_within.copy()

        # 5. Fraction of missing information, R's fmi
        fmi = table["fmi"].to_numpy()

        # 6. Pooled scale (average of individual scales)
        pooled_scale = float(np.mean(scale_array))

        logger.debug(f"Computed within-imputation variance diagonal: {np.diag(cov_within)}")
        logger.debug(f"Computed between-imputation variance diagonal: {np.diag(cov_between)}")
        logger.debug(f"Computed total variance diagonal: {np.diag(cov_total)}")
        logger.debug(f"Computed degrees of freedom: {table['df'].to_numpy()}")
        logger.debug(f"Computed fraction of missing information: {fmi}")
        logger.debug(f"Computed pooled scale: {pooled_scale}")

        # Store results for backward compatibility
        self.exog_names = param_names
        if hasattr(self.model_results[0], "model") and hasattr(self.model_results[0].model, "endog_names"):
            self.endog_names = self.model_results[0].model.endog_names

        # Create MICEresult object
        logger.debug("Creating MICEresult object")

        # The MICEresult expects normalized covariance params (divided by
        # scale); models without a usable residual scale use 1
        if np.isfinite(pooled_scale) and pooled_scale > 0:
            result_scale = pooled_scale
        else:
            result_scale = 1.0
        normalized_cov_params = cov_total / result_scale

        pooled_result = MICEresult(
            self,
            pooled_params,
            normalized_cov_params,
            scale=result_scale,
            df=table["df"].to_numpy(),
            dfcom=dfcom,
        )
        pooled_result.frac_miss_info = fmi

        # Store additional pooling diagnostics
        pooled_result.cov_within = cov_within
        pooled_result.cov_between = cov_between
        pooled_result.cov_total = cov_total
        pooled_result.m = m
        pooled_result.table = table

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
            "pooled_result": pooled_result,
            "pooled_params": pooled_params,
            "pooled_covariance": cov_total,
            "within_covariance": cov_within,
            "between_covariance": cov_between,
            "fraction_missing_info": fmi,
            "pooled_scale": pooled_scale,
            "n_imputations": m,
            "parameter_names": param_names,
            "formula": getattr(self, "formula", None),
            "table": table,
        }

        return comprehensive_result

    def _impute_once(self, state: _ChainState, chain_idx: int, iterations: range) -> pd.DataFrame:
        """
        Run iterations of one chain from its state, as R's ``sampler``.

        The working data are built from the data and ``state.imp``, also when
        a run is continued; each iteration updates both, and ``state.rng``
        advances.

        Parameters
        ----------
        state : _ChainState
            The chain after its start (`_initial_imputation`) or after its
            last iteration; updated in place.
        chain_idx : int
            Index of the chain.
        iterations : range
            The indices of the iterations to run (0 for R's iteration 1).
            Possibly empty: then the chain is completed as it is.

        Returns
        -------
        pd.DataFrame
            The completed dataset of the chain, a new frame: the data with
            every where-cell replaced by its imputation (`_complete`).
        """
        logger.debug(f"Starting imputation cycle for chain {chain_idx}")
        if len(iterations) > 0:
            working = self._working_frame(state.imp)
            for iter_idx in iterations:
                logger.debug(f"Starting iteration {iter_idx + 1}/{iterations.stop} for chain {chain_idx}")
                self._iterate(working, state, iter_idx, chain_idx)
                logger.debug(f"Completed iteration {iter_idx + 1}")

        logger.debug(f"Completed imputation cycle for chain {chain_idx}")
        return _complete(self._chain_data, state.imp, self.where)

    def _working_frame(self, imp: dict[Hashable, pd.Series]) -> pd.DataFrame:
        """The working data of a chain, R's ``data`` inside ``sampler``.

        A copy of the data in which the missing where-cells of each visited
        column hold their imputations from `imp`; observed cells keep their
        values.
        """
        working = self._chain_data.copy(deep=True)
        for col in dict.fromkeys(self.visit_sequence):
            _refresh_working_column(working, col, imp[col], self.where[col].to_numpy(), self.id_obs[col].to_numpy())
        return working

    def _iterate(self, working: pd.DataFrame, state: _ChainState, iter_idx: int, chain_idx: int) -> None:
        """
        One iteration of a chain: each column of the visit sequence in turn, as R's ``sampler.univ``.

        For a column, the design matrix of its predictors is built from all
        rows of the working data, and ``cx`` marks the rows where it is
        complete. The model is fitted on the rows `_fit_rows` gives and the
        imputer fills those of `_rows_to_impute`; it receives all rows, with
        NaN in the predictors outside ``cx``. Its values go to
        ``state.imp`` (`_store_imputations`: NaN for a where-cell outside
        ``cx``) and from there to the working data
        (`_refresh_working_column`); then the column's ``post`` function,
        if any, replaces them (`_apply_post`). A passive column is computed
        from the working data instead (`_visit_passive`). A column that the
        visit sequence names twice is visited twice. After the last column,
        the chain statistics are computed from ``state.imp``.

        Parameters
        ----------
        working : pd.DataFrame
            The working data of the chain; updated in place.
        state : _ChainState
            The state of the chain; ``imp`` is updated in place and ``rng``
            advances.
        iter_idx : int
            Index of the iteration (0 for R's iteration 1), counted over
            `impute` and every `continue_chains` since.
        chain_idx : int
            Index of the chain.
        """
        iteration_start_time = time.time()
        rng = state.rng

        for col in self.visit_sequence:
            logger.debug(f"Processing column '{col}' (iteration {iter_idx + 1}, chain {chain_idx})")
            method_name = self.method[col]
            if method_name == PASSIVE_METHOD:
                # Computed from the working data, as R's passive branch of sampler
                self._visit_passive(working, state.imp[col], col, iter_idx, chain_idx)
                continue
            predictor_cols = self._predictors[col]
            logger.debug(f"Using {len(predictor_cols)} predictors for column '{col}'")
            observed = self.id_obs[col].to_numpy()
            where = self.where[col].to_numpy()
            imp = state.imp[col]

            # Design matrix of all rows of the working data, as R's sampler.univ
            x = obtain_design(working, predictor_cols)
            x_values = x.to_numpy()
            cx = ~np.isnan(x_values).any(axis=1)
            ry = _fit_rows(cx, observed, self.ignore)
            wy = _rows_to_impute(cx, where)
            if not wy.any():
                # No where-cell has complete predictors: all are missing; as
                # in R, post is applied all the same
                _store_imputations(imp, wy[where], None)
                _refresh_working_column(working, col, imp, where, observed)
                self._apply_post(working, imp, col, iter_idx, chain_idx)
                continue

            # In the first iteration of the run only, also when it is continued
            if iter_idx == 0:
                self._check_df(x_values.shape[1], int(ry.sum()), col, method_name, chain_idx)

            # Drop constant and linearly dependent columns, as R's remove.lindep
            keep, event = _remove_lindep(x, working[col], ry)
            if event is not None:
                self._log_event(out=event, meth=method_name, it=iter_idx + 1, im=chain_idx + 1, dep=col)
            x_values = x_values[:, keep]

            y = _target_values(working[col])

            imputer_func = get_imputer_func(method_name)
            logger.debug(f"Using imputation method '{method_name}' for column '{col}'")
            # Pass rng to all imputation methods for reproducibility
            method_params = dict(self._method_params[method_name], rng=rng)
            # An imputer whose signature has `log_event` may log events: it
            # passes R's `out`, the rest of the row is filled in here
            if "log_event" in inspect.signature(imputer_func).parameters:
                method_params["log_event"] = functools.partial(
                    self._log_event, meth=method_name, it=iter_idx + 1, im=chain_idx + 1, dep=col
                )
            try:
                imputed_values = imputer_func(y=y, id_obs=ry, id_mis=wy, x=x_values, **method_params)
            except ValueError as err:
                # The imputer does not know which column it imputes
                raise ValueError(
                    f"Column {col!r}, method {method_name!r} (iteration {iter_idx + 1}, imputation {chain_idx + 1}): "
                    f"{err}"
                ) from err
            logger.debug(f"Successfully imputed {len(imputed_values)} values for column '{col}'")

            # New floats take the float dtype of the column
            if method_name in _FLOAT_METHODS:
                imputed_values = _in_float_dtype(imputed_values, working[col].dtype)
            _store_imputations(imp, wy[where], imputed_values)
            _refresh_working_column(working, col, imp, where, observed)
            # R's post, right after the univariate imputation
            self._apply_post(working, imp, col, iter_idx, chain_idx)

        self._record_chain_statistics(state.imp, iter_idx, chain_idx)
        iteration_time = time.time() - iteration_start_time
        logger.debug(f"Iteration {iter_idx + 1} completed in {iteration_time:.2f} seconds")

    def _apply_post(self, working: pd.DataFrame, imp: pd.Series, col: Hashable, iter_idx: int, chain_idx: int) -> None:
        """R's ``post``: the function of `col` in `post` replaces the imputations of its visit, in place.

        Called right after each univariate imputation of `col`, also when
        none of its where-cells could be imputed (as R evaluates ``post``
        then too); a column without a ``post`` function is left as it is.
        The function gets copies of `imp` (the imputations of all
        where-cells of the column) and of the working data in the rows of
        those cells (`_post_values`); its values replace `imp`, and the
        missing where-cells of the working data are refreshed from them
        (`_refresh_working_column`), as R refreshes ``data`` after
        evaluating ``post``.

        Parameters
        ----------
        working : pd.DataFrame
            The working data of the chain; updated in place.
        imp : pd.Series
            The imputations of the where-cells of `col` (``state.imp[col]``);
            updated in place.
        col : label
            The column just imputed.
        iter_idx, chain_idx : int
            The iteration and the chain, for the messages.
        """
        function = self.post.get(col)
        if function is None:
            return
        where = self.where[col].to_numpy()
        context = f"Column {col!r}, post (iteration {iter_idx + 1}, imputation {chain_idx + 1})"
        values = _post_values(function, imp, working.take(np.flatnonzero(where)), context)
        _replace_imputations(imp, values)
        _refresh_working_column(working, col, imp, where, self.id_obs[col].to_numpy())

    def _visit_passive(
        self, working: pd.DataFrame, imp: pd.Series, col: Hashable, iter_idx: int, chain_idx: int
    ) -> None:
        """A visit of the passive column `col`, as R's passive branch of ``sampler``; in place.

        Every where-cell of `imp` gets the value of its row from the
        column's function on the working data (`_passive_values`), and the
        missing where-cells of the working data are refreshed from `imp`
        (`_refresh_working_column`): an observed where-cell keeps its
        observed value there, as for an imputed column. Nothing is drawn and
        nothing is logged; ``post`` does not apply.

        Parameters
        ----------
        working : pd.DataFrame
            The working data of the chain; updated in place.
        imp : pd.Series
            The values of the where-cells of `col` (``state.imp[col]``);
            updated in place.
        col : label
            The passive column.
        iter_idx, chain_idx : int
            The iteration and the chain, for the messages.
        """
        where = self.where[col].to_numpy()
        values = self._passive_values(col, working, f"iteration {iter_idx + 1}, imputation {chain_idx + 1}")
        _replace_imputations(imp, values[where])
        _refresh_working_column(working, col, imp, where, self.id_obs[col].to_numpy())

    def _passive_values(self, col: Hashable, working: pd.DataFrame, when: str):
        """The value of the passive column `col` in every row: its function on a copy of the working data.

        The function gets all rows (R evaluates the formula on the rows of
        the where-cells only, which matters for a function that is not
        row-wise, such as one that subtracts the column mean) and must return
        one value per row of values the column can hold. Returns them in the
        dtype of the column in the chains, without index; raises ValueError,
        naming the column and `when` ("iteration 1, imputation 2"),
        otherwise and for an exception inside the function (the original is
        the cause).
        """
        context = f"Column {col!r}, passive ({when})"
        try:
            result = self.passive[col](working.copy(deep=True))
        except Exception as err:
            raise ValueError(f"{context}: the function raised {type(err).__name__}: {err}") from err
        values = _values_per_entry(result, working.index, context, _PASSIVE_RESULT, "rows of the data")
        return _in_column_dtype(values, working[col].dtype, context)

    def _record_chain_statistics(self, imp: dict[Hashable, pd.Series], iter_idx: int, chain_idx: int) -> None:
        """Store the mean and variance of the imputations of each numeric visited column, as R's ``sampler``.

        Computed from `imp` at the end of the iteration, over all where-cells
        of the column; missing imputations are skipped (R's ``na.rm =
        TRUE``). The mean is NaN without any imputation, the variance with
        fewer than two, as R's ``NA``. They go to row `iter_idx`, which
        `_run_chains` has appended.
        """
        for col in self.chain_mean:
            values = imp[col].to_numpy(dtype=float, na_value=np.nan)
            values = values[~np.isnan(values)]
            mean_val = np.nanmean(values) if values.size > 0 else np.nan
            var_val = np.nanvar(values, ddof=1) if values.size > 1 else np.nan
            self.chain_mean[col][iter_idx, chain_idx] = mean_val
            self.chain_var[col][iter_idx, chain_idx] = var_val
            logger.debug(f"Chain statistics for '{col}': mean={mean_val:.4f}, variance={var_val:.4f}")

    def _check_df(self, n_predictors, n_observed, col, method_name, chain_idx):
        """Log a model with fewer observed cases than parameters, as R's ``check.df``."""
        if n_observed > 0 and n_observed - n_predictors - 1 < 1:
            self._log_event(
                out=f"df set to 1. # observed cases: {n_observed}  # predictors: {n_predictors + 1}",
                meth=method_name,
                it=1,
                im=chain_idx + 1,
                dep=col,
            )

    def _initial_imputation(self, rng: np.random.Generator) -> dict[Hashable, pd.Series]:
        """
        The start of one chain, as R's ``initialize.imp``.

        Each column with a method gets a value in each of its where-cells,
        taken from its observed values in the rows that are not ignored.
        "sample" draws them, as R does, column by column in data order.
        "meanobs" (not in R) gives a numeric column the observed value
        closest to its mean and any other column its most frequent observed
        value. A passive column starts in the same way, not from its
        function, as in R; one without any observed value in those rows
        starts from standard normal draws, as R's ``rnorm``, whatever
        `initial` says (its function returns numbers, see
        `_check_passive_start`). A column without a method is not started:
        its where-cells are missing.

        Parameters
        ----------
        rng : np.random.Generator
            The generator of the chain; "sample" draws from it.

        Returns
        -------
        dict of column label to pd.Series
            The chain's ``imp`` (see `_ChainState`), one entry per column
            with where-cells.
        """
        data = self._chain_data
        imp = {}
        for col in data.columns:
            where = self.where[col].to_numpy()
            if not where.any():
                continue
            start = data[col][where].copy()
            observed = self.id_obs[col].to_numpy()
            # R's ry at the start: observed and not ignored, whatever the predictors
            donors = observed & ~self.ignore
            if self.method[col] == "":
                if observed[where].any():
                    start.iloc[np.flatnonzero(observed[where])] = np.nan
            elif not donors.any():
                # A passive column without observed value (any other column
                # with a method has one): R's rnorm
                start.iloc[:] = _in_float_dtype(rng.standard_normal(int(where.sum())), start.dtype)
            elif self.initial == InitialMethod.SAMPLE.value:
                observed_values = data.loc[donors, col].values
                start.iloc[:] = _sample_start(observed_values, int(where.sum()), rng)
            elif self.initial == InitialMethod.MEANOBS.value:
                observed_values = data.loc[donors, col]
                if _column_kind(observed_values) == "numeric":
                    # the observed value closest to the mean (the first if tied)
                    values = observed_values.to_numpy(dtype=float)
                    closest_value = observed_values.iloc[int(np.argmin(np.abs(values - values.mean())))]
                else:
                    # categorical, object, string or boolean: the most
                    # frequent observed value (ties: the first level)
                    closest_value = observed_values.mode().iloc[0]
                start.iloc[:] = closest_value
            imp[col] = start
        return imp

    def _set_visit_sequence(self, visit_sequence, rng=None):
        """
        Set the visit sequence for imputation based on the input parameter.

        Parameters
        ----------
        visit_sequence : str, list of str or None
            Visit sequence specification. Can be:

            - None: data order with the passive columns last (R's default)
            - str: "roman", "arabic", "monotone", "revmonotone" or "random"
            - list of str: column names specifying the order to visit variables
                        Must include all columns that are imputed. Columns that are not
                        imputed (without where-cells, or with method "") are dropped from it.
                        A column named more than once is visited each time.
        rng : np.random.Generator, optional
            Draws the "random" sequence. If None, a fresh generator is used.

        Only columns that are imputed are visited: those with where-cells
        (`where`) and a method other than "", passive columns included. The
        keywords follow R's ``mice``: "roman" is data order, "arabic" its
        reverse, "monotone" sorts by the number of where-cells with a stable
        sort (ties keep data order) and "revmonotone" is the reverse of
        "monotone"; they place the passive columns like the others. Only
        None moves the passive columns to the end (`_passive_last`), as R
        does without a visit sequence.
        """
        imputed_columns = [col for col in self.data.columns if self.where[col].any() and self.method[col] != ""]

        # Validate using centralized validator
        validated_sequence, cols_not_imputed = check_visit_sequence(
            visit_sequence, list(self.data.columns), imputed_columns
        )

        if isinstance(visit_sequence, list):
            # Report columns that are not imputed if any were filtered out
            if cols_not_imputed:
                logger.warning(
                    f"Visit sequence includes columns that are not imputed (without where-cells, or method ''): "
                    f"{cols_not_imputed}. They are skipped."
                )
            self.visit_sequence = validated_sequence
        elif visit_sequence is None:
            # R's visitSequence = NULL: the imputed columns, then the passive ones
            self.visit_sequence = _passive_last(imputed_columns, self.method)
        else:
            # For string sequences, generate based on type
            monotone = _monotone(self.where, imputed_columns)
            if visit_sequence == VisitSequence.RANDOM.value:
                if rng is None:
                    rng = np.random.default_rng()
                self.visit_sequence = [imputed_columns[i] for i in rng.permutation(len(imputed_columns))]
            elif visit_sequence == VisitSequence.ROMAN.value:
                self.visit_sequence = list(imputed_columns)
            elif visit_sequence == VisitSequence.ARABIC.value:
                self.visit_sequence = imputed_columns[::-1]
            elif visit_sequence == VisitSequence.MONOTONE.value:
                self.visit_sequence = monotone
            elif visit_sequence == VisitSequence.REVMONOTONE.value:
                self.visit_sequence = monotone[::-1]
