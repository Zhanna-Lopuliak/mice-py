"""The MICE class: multiple imputation by chained equations, as R's ``mice``."""

from __future__ import annotations

import functools
import inspect
import logging
import time
import warnings
from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .constants import (
    DEFAULT_INITIAL_METHOD,
    DEFAULT_METHOD,
    DEFAULT_VISIT_SEQUENCE,
    ImputationMethod,
    InitialMethod,
    VisitSequence,
)
from .design import _as_numeric, _column_kind, _remove_lindep, find_collinear, obtain_design, quickpred
from .mice_result import MICEresult
from .pooling import _residual_df, pool_estimates
from .utils import get_imputer_func, get_imputer_params, method_prefix, split_method_param
from .validators import (
    check_initial_method,
    check_maxit,
    check_method,
    check_n_imputations,
    check_visit_sequence,
    validate_columns,
    validate_dataframe,
    validate_formula,
    validate_predictor_matrix,
)

if TYPE_CHECKING:
    from statsmodels.iolib.summary2 import Summary

logger = logging.getLogger(__name__)

# Deprecated keyword arguments of `MICE.impute`, and the matching arguments
# of `imputation.quickpred`
_QUICKPRED_ARGUMENTS = {
    "min_cor": "mincor",
    "min_puc": "minpuc",
    "include": "include",
    "exclude": "exclude",
    "correlation_method": "method",
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
        For the first invalid name; the message lists the valid names.
    """
    valid = _method_param_names()
    for key in params:
        match = split_method_param(key, valid)
        if match is not None and match[1] in valid[match[0]]:
            continue
        method = match[0] if match is not None else None
        if method is not None and not valid[method]:
            hint = f"Method '{method}' takes no parameters."
        elif method is not None:
            prefix = method_prefix(method)
            hint = f"Valid parameters for method '{method}': {', '.join(f'{prefix}_{p}' for p in valid[method])}."
        else:
            hint = (
                "Method parameters take the form <method>_<parameter>; valid ones: "
                f"{', '.join(f'{method_prefix(m)}_{p}' for m, names in valid.items() for p in names)}. "
                f"Deprecated keyword arguments: {', '.join(_QUICKPRED_ARGUMENTS)}."
            )
        raise TypeError(f"impute() got an unexpected keyword argument '{key}'. {hint}")


class MICE:
    """
    Multiple imputation by chained equations, as R's ``mice``.

    Create the object with the incomplete data, then call `impute`. To
    analyse the completed datasets, call `fit` with a model formula and
    combine the fits with `pool` (Rubin's rules, as R's ``pool``).

    Parameters
    ----------
    data : pd.DataFrame
        The incomplete data. Anything else is converted with
        ``pd.DataFrame(data)``. Columns may be numeric, boolean, categorical,
        object or string.

    Attributes
    ----------
    data : pd.DataFrame
        The validated input data.
    id_obs : dict of str to pd.Series
        Per column, a boolean Series that is True where the value is observed.
    id_mis : dict of str to pd.Series
        Per column, a boolean Series that is True where the value is missing.
    imputed_datasets : list of pd.DataFrame
        The completed datasets of the last call of `impute`.
    method : dict of str to str
        The imputation method of each column used by the last `impute`
        (``""`` for a column that is not imputed).
    predictor_matrix : pd.DataFrame
        The predictor matrix used by the last `impute`, after the setup step
        (square, in data order; rows are targets, columns predictors).
    visit_sequence : list of str
        The order in which the last `impute` visited the columns.
    chain_mean, chain_var : dict of str to np.ndarray
        For each imputed numeric column, the mean and variance of its imputed
        values, shape ``(maxit, n_imputations)``.
    logged_events : pd.DataFrame
        What the last call of `impute` changed in the imputation model, with
        R's columns ``it`` (iteration, 0 for the setup step), ``im``
        (imputation), ``dep`` (the target), ``meth`` and ``out``. Empty when
        nothing was logged. See `impute`.
    model_results : list
        The fitted models of the last call of `fit`, one per dataset.
    pooled_result : MICEresult
        The pooled estimates of the last call of `pool`.

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
        visit_sequence: str | list[str] = DEFAULT_VISIT_SEQUENCE,
        seed: int | Sequence[int] | None = None,
        **kwargs: Any,
    ) -> list[pd.DataFrame]:
        """
        Perform multiple imputation by chained equations.

        Parameters
        ----------
        n_imputations : int, default=5
            Number of imputations (R's ``m``), a positive integer.
        maxit : int, default=5
            Number of iterations of each chain, a positive integer.
        predictor_matrix : pd.DataFrame, optional
            0/1 matrix with column labels as index (targets) and columns
            (predictors); a 1 means that the column is used as a predictor of
            the row. It may cover only some columns (missing entries count as
            0), but every incomplete column needs a row. If None, every other
            column predicts every incomplete column (all ones with a zero
            diagonal), as in R. The setup step described in the Notes then
            removes constant and collinear predictors.
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

            The argument is:

            - str: the same method for all columns;
            - dict: the method of each listed column; columns not in it get
              ``"pmm"``;
            - None: ``"pmm"`` for every incomplete column, whatever its type
              (R uses ``pmm`` for numeric columns and logreg, polyreg or polr
              for factors, which this package does not have).

            As in R, a complete column always gets ``""``. A column with
            method ``""`` keeps its missing values; where it is used as a
            predictor, the rows in which it is missing are left out of the
            imputation model and are not imputed.
        visit_sequence : str or list of str, default="roman"
            Order in which the columns are visited in each iteration. Only
            incomplete columns that are imputed are visited.

            - "roman": data order (R's default);
            - "arabic": reverse data order;
            - "monotone": increasing number of missing values, ties in data order;
            - "revmonotone": the reverse of "monotone";
            - "random": a random order, drawn once per call from `seed`;
            - a list of column labels: that order. It must contain every
              column that is imputed; other columns in it are skipped.
        seed : int, sequence of int, or None, default=None
            Seed for the random number generators. It seeds one
            ``np.random.SeedSequence``, from which independent streams are
            spawned: one for the random visit sequence and one per chain.
            Chain ``k`` always receives the same stream, whatever
            ``n_imputations`` is. The same seed gives identical results;
            ``None`` draws fresh entropy from the operating system.

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

            - `output_dir`: deprecated and ignored; passing it issues a
              ``DeprecationWarning``.

            Deprecated, to be removed in the next version: `min_cor`,
            `min_puc`, `include`, `exclude` and `correlation_method`. When
            `predictor_matrix` is None, they build the predictor matrix with
            ``imputation.quickpred(data, mincor=min_cor, minpuc=min_puc,
            include=include, exclude=exclude, method=correlation_method)``;
            otherwise they are ignored. Passing any of them issues a
            ``DeprecationWarning``. Call `imputation.quickpred` and pass its
            result as `predictor_matrix` instead.

        Returns
        -------
        list of pd.DataFrame
            The `n_imputations` completed datasets.

        Raises
        ------
        ValueError
            For an invalid argument (number of imputations or iterations,
            method, initial method, visit sequence, predictor matrix), or when
            a method cannot impute a column (for example ``"midas"`` or
            ``"norm"`` on a categorical column, ``"logreg"`` on a column with
            more than two categories, ``"polyreg"`` on a column with more
            than 50 categories or a model with more weights than
            ``polyreg_max_weights``, ``"norm.nob"`` with too few observed
            values for its predictors, or a column without observed value in
            the rows usable for its model).
        TypeError
            For an unknown keyword argument.

        Warns
        -----
        UserWarning
            When the imputation model was changed (see Notes), with the
            number of logged events, as R's "Number of logged events"; also
            for more than 100 imputations or 50 iterations.
        DeprecationWarning
            For the deprecated keyword arguments listed above.

        Notes
        -----
        As R's ``mice``, the imputation model is edited before the chains
        start and before every univariate imputation; each change is a row
        of `logged_events`:

        - Setup (``it`` 0): a variable that is constant (variance below
          ``1000 * eps``; also when fewer than two values are observed, so
          also an entirely missing variable) is no longer used as a
          predictor (``meth`` "constant"); if it is incomplete, it is not
          imputed and keeps its missing values. The same holds for a
          variable that (nearly) duplicates another one
          (`imputation.design.find_collinear`, ``meth`` "collinear").
          Object and string columns count as categorical variables here.
        - Each univariate imputation: the predictors are dummy-coded by
          `imputation.design.obtain_design`, and columns that are constant
          or linearly dependent on the observed rows are dropped
          (`imputation.design.remove_lindep`); ``out`` names the dropped
          columns. In the first iteration, a model with fewer observed
          cases than parameters logs "df set to 1".
        - An imputer may log events itself: with ``polr_log_fallback=True``
          each fallback of ``"polr"`` to the multinomial model logs
          "polr falls back to multinom" (``meth`` "polr"), as R's
          ``polr.to.loggedEvents = TRUE``.
        """
        logger.info("Starting imputation process")
        logger.debug(
            f"Parameters: n_imputations={n_imputations}, maxit={maxit}, "
            f"initial={initial}, method={method}, visit_sequence={visit_sequence}"
        )

        start_time = time.time()

        check_n_imputations(n_imputations)
        check_maxit(maxit)
        check_initial_method(initial)
        n_imputations = int(n_imputations)
        maxit = int(maxit)

        # One seed sequence per call, split into independent streams: the
        # first for the random visit sequence, then one per chain. Chain k
        # always gets child k + 1, so its stream does not depend on n_imputations.
        seed_sequence = np.random.SeedSequence(seed)
        visit_stream = seed_sequence.spawn(1)[0]
        chain_streams = seed_sequence.spawn(n_imputations)

        if "output_dir" in kwargs:
            kwargs.pop("output_dir")
            warnings.warn(
                "The 'output_dir' argument of impute() is deprecated and ignored.",
                DeprecationWarning,
                stacklevel=2,
            )
        deprecated = [name for name in _QUICKPRED_ARGUMENTS if name in kwargs]
        quickpred_args = {_QUICKPRED_ARGUMENTS[name]: kwargs.pop(name) for name in deprecated}
        _check_method_params(kwargs)
        if deprecated:
            warnings.warn(
                f"The arguments {', '.join(deprecated)} of impute() are deprecated and will be removed "
                "in the next version. Build the predictor matrix with imputation.quickpred(data, "
                "mincor=..., minpuc=..., include=..., exclude=..., method=...) and pass it as "
                "predictor_matrix.",
                DeprecationWarning,
                stacklevel=2,
            )

        if predictor_matrix is None and not quickpred_args:
            predictor_matrix = _full_predictor_matrix(list(self.data.columns))
        elif predictor_matrix is None:
            predictor_matrix = quickpred(self.data, **quickpred_args)
            logger.debug(f"Predictor matrix from quickpred:\n{predictor_matrix}")
        else:
            if quickpred_args:
                logger.warning("Arguments %s only apply when predictor_matrix is None; they are ignored.", deprecated)
            predictor_matrix = validate_predictor_matrix(predictor_matrix, list(self.data.columns), self.data)
            logger.debug("Predictor matrix validated successfully")
        # Square over all columns, in data order; absent entries are 0
        columns = list(self.data.columns)
        predictor_matrix = predictor_matrix.reindex(index=columns, columns=columns, fill_value=0).astype(int)

        if method is not None:
            self.method = check_method(method, list(self.data.columns))
        else:
            self.method = check_method(DEFAULT_METHOD, list(self.data.columns))
        # As in R, a complete column has no imputation method
        for col in columns:
            if not self.id_mis[col].any():
                self.method[col] = ""
        logger.debug(f"Using imputation methods: {self.method}")

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

        self._set_visit_sequence(visit_sequence, np.random.default_rng(visit_stream))
        logger.debug(f"Visit sequence set to: {self.visit_sequence}")

        # Prepare chain statistics containers
        # Only track statistics for numeric columns that will be imputed (i.e., have missing values)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        numeric_cols_to_impute = [col for col in self.visit_sequence if col in numeric_cols]

        self.chain_mean = {
            col: np.full((self.maxit, self.n_imputations), np.nan, dtype=float) for col in numeric_cols_to_impute
        }
        self.chain_var = {
            col: np.full((self.maxit, self.n_imputations), np.nan, dtype=float) for col in numeric_cols_to_impute
        }

        # Fixed for the whole run: the predictors of each target, in data
        # order, and the parameters of each method
        self._predictors = {
            col: [c for c in columns if c != col and self.predictor_matrix.at[col, c] != 0]
            for col in self.visit_sequence
        }
        self._method_params: dict[str, dict[str, Any]] = {
            method_name: {} for method_name in set(self.method[col] for col in self.visit_sequence)
        }
        for key, value in self.imputation_params.items():
            match = split_method_param(key, param_names)
            if match is not None and match[0] in self._method_params:
                self._method_params[match[0]][match[1]] = value

        self.imputed_datasets = []
        individual_times = []

        for chain_idx in range(self.n_imputations):
            chain_start_time = time.time()
            logger.info(f"Starting imputation chain {chain_idx + 1}/{self.n_imputations}")
            chain_rng = np.random.default_rng(chain_streams[chain_idx])
            self.imputed_datasets.append(self._impute_once(chain_idx, chain_rng))
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

        self._report_logged_events()

        return self.imputed_datasets

    def _log_event(self, out, meth, it=0, im=0, dep=""):
        """Record one change of the imputation model, as R's ``updateLog``."""
        self._log.append((it, im, dep, meth, out))

    def _report_logged_events(self):
        """Fill `logged_events` and warn once if anything was logged.

        R runs the iterations in the outer loop and the imputations in the
        inner one; the chains here run one after the other. Sorting by
        iteration and imputation (stable) gives R's order.
        """
        if not self._log:
            self.logged_events = _empty_log()
            return
        events = pd.DataFrame(self._log, columns=list(_LOG_COLUMNS))
        events = events.sort_values(["it", "im"], kind="stable").reset_index(drop=True)
        self.logged_events = events.astype({"it": "int64", "im": "int64"})
        message = f"Number of logged events: {len(events)}."
        if self._not_imputed:
            message += (
                f" Not imputed because constant or collinear, their missing values are kept: {self._not_imputed}."
            )
        message += " See MICE.logged_events."
        logger.info(message)
        warnings.warn(message, UserWarning, stacklevel=3)

    def _edit_setup(self, predictor_matrix: pd.DataFrame) -> pd.DataFrame:
        """Remove constant and collinear variables, as R's ``mice.edit.setup``.

        A variable that is constant, or (nearly) collinear with another
        variable that is used as a predictor, stops being a predictor. If it
        has a method, the method becomes ``""``: it is not imputed, keeps its
        missing values and leaves the visit sequence. Each variable
        affected is logged once (``meth`` "constant" or "collinear").
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

    def _impute_once(self, chain_idx: int, rng: np.random.Generator):
        """
        Perform one complete imputation cycle.

        Parameters
        ----------
        chain_idx : int
            Index of the chain
        rng : np.random.Generator
            Random number generator of this chain; its state advances through
            the initial imputation and all iterations

        Returns
        -------
        pd.DataFrame
            A copy of the data with one complete imputation cycle applied
        """
        logger.debug(f"Starting imputation cycle for chain {chain_idx}")
        current_data = self.data.copy(deep=True)
        # The norm methods and mean impute floats: a nullable integer column
        # they impute becomes Float64, as R turns an integer column into double
        for col in self.visit_sequence:
            if self.method[col] in _FLOAT_METHODS and _is_nullable_integer(current_data[col].dtype):
                current_data[col] = current_data[col].astype("Float64")

        logger.debug("Performing initial imputation")
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
            predictor_cols = self._predictors[col]
            logger.debug(f"Using {len(predictor_cols)} predictors for column '{col}'")

            # Design matrix of the current completed data, as R's sampler.univ
            x = obtain_design(updated_data, predictor_cols)
            x_values = x.to_numpy()
            complete = ~np.isnan(x_values).any(axis=1)
            id_obs = self.id_obs[col].to_numpy() & complete
            id_mis = self.id_mis[col].to_numpy() & complete
            # A missing cell whose predictors are incomplete (a predictor that
            # is not imputed) is not imputed either and stays missing
            unimputable = self.id_mis[col].to_numpy() & ~complete
            if unimputable.any():
                updated_data.loc[unimputable, col] = np.nan
            if not id_mis.any():
                continue

            if iter_idx == 0:
                self._check_df(x_values.shape[1], int(id_obs.sum()), col, method_name, chain_idx)

            # Drop constant and linearly dependent columns, as R's remove.lindep
            keep, event = _remove_lindep(x, updated_data[col], id_obs)
            if event is not None:
                self._log_event(out=event, meth=method_name, it=iter_idx + 1, im=chain_idx + 1, dep=col)
            x_values = x_values[:, keep]

            y = _target_values(updated_data[col])
            if not complete.all():
                y, x_values = y[complete], x_values[complete]
                id_obs, id_mis = id_obs[complete], id_mis[complete]

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
            imputed_values = imputer_func(y=y, id_obs=id_obs, id_mis=id_mis, x=x_values, **method_params)
            logger.debug(f"Successfully imputed {len(imputed_values)} values for column '{col}'")

            # Assign imputed values; new floats take the float dtype of the column
            if method_name in _FLOAT_METHODS:
                imputed_values = _in_float_dtype(imputed_values, updated_data[col].dtype)
            updated_data.loc[self.id_mis[col].to_numpy() & complete, col] = imputed_values

            # Record chain statistics (only for numeric columns)
            if col in self.chain_mean:
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

    def _initial_imputation(self, data, rng=None):
        """
        Initialize missing values based on the initial method.

        "sample" draws from the observed values of each column, as R does.
        "meanobs" (not in R) fills a numeric column with the observed value
        closest to its mean and any other column with its most frequent
        observed value.
        """
        if rng is None:
            rng = np.random.default_rng()  # fresh random generator

        # Columns without a method are not imputed, not even initially
        if self.initial == InitialMethod.SAMPLE.value:
            for col in data.columns:
                if self.method[col] != "" and data[col].isna().any():
                    observed_values = data.loc[self.id_obs[col], col].values
                    data.loc[self.id_mis[col], col] = rng.choice(observed_values, size=self.id_mis[col].sum())

        elif self.initial == InitialMethod.MEANOBS.value:
            for col in data.columns:
                if self.method[col] != "" and data[col].isna().any():
                    observed_values = data.loc[self.id_obs[col], col]
                    if _column_kind(observed_values) == "numeric":
                        # the observed value closest to the mean (the first if tied)
                        values = observed_values.to_numpy(dtype=float)
                        closest_value = observed_values.iloc[int(np.argmin(np.abs(values - values.mean())))]
                    else:
                        # categorical, object, string or boolean: the most
                        # frequent observed value (ties: the first level)
                        closest_value = observed_values.mode().iloc[0]
                    data.loc[self.id_mis[col], col] = closest_value

    def _set_visit_sequence(self, visit_sequence, rng=None):
        """
        Set the visit sequence for imputation based on the input parameter.

        Parameters
        ----------
        visit_sequence : str or list of str
            Visit sequence specification. Can be:

            - str: "roman", "arabic", "monotone", "revmonotone" or "random"
            - list of str: column names specifying the order to visit variables
                        Must include all columns that are imputed. Columns that are not
                        imputed (complete, or with method "") are dropped from it.
        rng : np.random.Generator, optional
            Draws the "random" sequence. If None, a fresh generator is used.

        Only incomplete columns with a method other than "" are visited. The
        keywords follow R's ``mice``: "roman" is data order, "arabic" its
        reverse, "monotone" sorts by the number of missing values with a
        stable sort (ties keep data order) and "revmonotone" is the reverse
        of "monotone".
        """
        columns_with_missing = [col for col in self.data.columns if self.id_mis[col].any() and self.method[col] != ""]

        # Validate using centralized validator
        validated_sequence, cols_without_missing = check_visit_sequence(
            visit_sequence, list(self.data.columns), columns_with_missing
        )

        if isinstance(visit_sequence, list):
            # Report columns that are not imputed if any were filtered out
            if cols_without_missing:
                logger.warning(
                    f"Visit sequence includes columns that are not imputed (complete, or method ''): "
                    f"{cols_without_missing}. They are skipped."
                )
            self.visit_sequence = validated_sequence
        else:
            # For string sequences, generate based on type
            nmis = np.array([self.id_mis[col].sum() for col in columns_with_missing], dtype=int)
            monotone = [columns_with_missing[i] for i in np.argsort(nmis, kind="stable")]
            if visit_sequence == VisitSequence.RANDOM.value:
                if rng is None:
                    rng = np.random.default_rng()
                self.visit_sequence = [columns_with_missing[i] for i in rng.permutation(len(columns_with_missing))]
            elif visit_sequence == VisitSequence.ROMAN.value:
                self.visit_sequence = list(columns_with_missing)
            elif visit_sequence == VisitSequence.ARABIC.value:
                self.visit_sequence = columns_with_missing[::-1]
            elif visit_sequence == VisitSequence.MONOTONE.value:
                self.visit_sequence = monotone
            elif visit_sequence == VisitSequence.REVMONOTONE.value:
                self.visit_sequence = monotone[::-1]
