"""Design matrix, predictor selection and pruning, as in R mice.

Pure functions behind the setup step of :class:`imputation.MICE` and
behind each univariate imputation call. Each reproduces the rule of the
matching function of R ``mice`` 3.19.0:

- :func:`obtain_design` builds the numeric design matrix of the predictors of
  one target, as R's ``model.matrix`` does inside ``mice``
  (``mice:::obtain.design``), without the intercept column.
- :func:`remove_lindep` decides which columns of that design matrix enter the
  imputation model of one target (``mice:::remove.lindep``).
- :func:`find_collinear` lists the variables that (nearly) duplicate another
  variable, so that the setup step can stop using them
  (``mice:::find.collinear``).
- :func:`quickpred` builds a predictor matrix from simple statistics
  (``mice::quickpred``); it is also available as ``imputation.quickpred``.

The functions only compute; they neither log nor warn.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import linalg

__all__ = ["obtain_design", "remove_lindep", "find_collinear", "quickpred"]

#: The event R's ``remove.lindep`` logs when no column survives the variance
#: and correlation screen.
ALL_PREDICTORS_DROPPED = "All predictors are constant or have too high correlation."

#: R's ``contr.poly`` refuses more levels than this.
_MAX_POLY_LEVELS = 95


# ---------------------------------------------------------------------------
# Column types
# ---------------------------------------------------------------------------


def _column_kind(series: pd.Series) -> str:
    """Classify a column the way R's ``model.matrix`` treats it.

    Returns ``"numeric"``, ``"bool"``, ``"factor"`` (unordered categorical)
    or ``"ordered"`` (ordered categorical). Object and string columns are
    factors, unless every non-missing value is a boolean.
    """
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return "ordered" if dtype.ordered else "factor"
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    if pd.api.types.is_complex_dtype(dtype):
        raise TypeError(f"column {series.name!r}: complex values are not supported")
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
        if pd.api.types.infer_dtype(series, skipna=True) == "boolean":
            return "bool"
        return "factor"
    raise TypeError(f"column {series.name!r}: unsupported dtype {dtype}")


def _levels(series: pd.Series, kind: str) -> list:
    """Levels of a non-numeric column, in the order R would use.

    A Categorical keeps its own category order (unused categories included,
    as R keeps unused factor levels). Booleans have the levels FALSE, TRUE.
    Other columns use their sorted distinct values, R's default factor order.
    """
    if isinstance(series.dtype, pd.CategoricalDtype):
        return list(series.cat.categories)
    if kind == "bool":
        return [False, True]
    try:
        return sorted(pd.unique(series.dropna()))
    except TypeError as err:
        raise TypeError(f"column {series.name!r}: values of mixed types cannot be ordered as levels") from err


def _codes(series: pd.Series, levels: list) -> np.ndarray:
    """Zero-based level codes of a column, -1 where it is missing."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return series.cat.codes.to_numpy()
    return pd.Categorical(series, categories=levels).codes


def _as_numeric(values: Any) -> np.ndarray:
    """A column as float, with categories replaced by their level codes.

    This is what R's ``as.numeric`` does to a factor and ``data.matrix`` to
    a data frame column: numbers stay numbers, booleans become 0 and 1, and
    categorical values become the position of their level. The codes here
    start at 0 instead of R's 1, which changes no variance or correlation.
    Missing values become NaN.
    """
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    kind = _column_kind(series)
    if kind == "numeric":
        return series.to_numpy(dtype=float, na_value=np.nan)
    codes: np.ndarray = _codes(series, _levels(series, kind)).astype(float)
    codes[codes < 0] = np.nan
    return codes


# ---------------------------------------------------------------------------
# obtain_design
# ---------------------------------------------------------------------------


def _poly_contrasts(n_levels: int) -> np.ndarray:
    """Orthonormal polynomial contrasts for ``n_levels`` equally spaced levels.

    Column ``d - 1`` holds the polynomial of degree ``d`` (``d = 1, ...,
    n_levels - 1``) evaluated at the scores ``1, ..., n_levels``: the
    component of ``score ** d`` orthogonal to all lower degrees, scaled to
    unit length, with the sign that makes its leading coefficient positive.
    These are the contrasts of R's ``contr.poly``.
    """
    if n_levels > _MAX_POLY_LEVELS:
        raise ValueError(f"polynomial contrasts are limited to {_MAX_POLY_LEVELS} levels, as in R; got {n_levels}")
    scores = np.arange(1, n_levels + 1, dtype=float)
    vandermonde = (scores - scores.mean())[:, None] ** np.arange(n_levels)
    q, r = np.linalg.qr(vandermonde)
    # q * sign(diag(r)) is the Gram-Schmidt basis of the Vandermonde columns,
    # whatever sign convention the QR routine uses.
    return (q * np.sign(np.diag(r)))[:, 1:]


def _poly_suffixes(n_levels: int) -> list[str]:
    """Column name suffixes of polynomial contrasts: .L, .Q, .C, ^4, ^5, ..."""
    suffixes = [".L", ".Q", ".C"] + [f"^{degree}" for degree in range(4, n_levels)]
    return suffixes[: max(n_levels - 1, 0)]


def _design_columns(series: pd.Series, name: Hashable) -> tuple[list, list[np.ndarray]]:
    """Design matrix columns of one predictor: their names and values."""
    kind = _column_kind(series)
    if kind == "numeric":
        return [name], [series.to_numpy(dtype=float, na_value=np.nan)]

    levels = _levels(series, kind)
    codes = _codes(series, levels)
    missing = codes < 0
    if len(levels) < 2:
        # A single level carries no information: no column. R stops with
        # "contrasts can be applied only to factors with 2 or more levels".
        return [], []

    if kind == "ordered":
        contrasts = _poly_contrasts(len(levels))
        names = [f"{name}{suffix}" for suffix in _poly_suffixes(len(levels))]
    else:
        # treatment contrasts: one indicator per level except the first
        contrasts = np.eye(len(levels))[:, 1:]
        if kind == "bool":
            names = [f"{name}TRUE"]
        else:
            names = [f"{name}{level}" for level in levels[1:]]

    block = contrasts[np.where(missing, 0, codes)]
    block[missing] = np.nan
    return names, list(block.T)


def obtain_design(data: pd.DataFrame, predictors: Sequence[Hashable]) -> pd.DataFrame:
    """Numeric design matrix of the listed predictors, without intercept.

    Builds what R's ``model.matrix`` builds for the formula
    ``target ~ p1 + p2 + ...`` inside ``mice`` (``mice:::obtain.design``),
    with the intercept column removed. Predictors are expanded in the order
    given:

    - numeric columns: one column of floats, named as the predictor;
    - unordered Categoricals: treatment dummies, one per level except the
      first, in the Categorical's own level order (unused levels included),
      named ``<predictor><level>``, for example ``hypyes``;
    - object and ``string`` columns: like an unordered Categorical with the
      sorted distinct values as levels, R's default factor order;
    - ordered Categoricals: orthonormal polynomial contrasts as R's
      ``contr.poly``, named ``<predictor>.L``, ``.Q``, ``.C``, ``^4``, ...;
    - booleans (``bool``, pandas ``boolean``, or an object column holding
      only booleans): one column ``<predictor>TRUE`` with values 0 and 1.

    Missing values stay missing: a row with a missing categorical value has
    NaN in every column of that predictor.

    Parameters
    ----------
    data : pd.DataFrame
        Data holding the predictor columns.
    predictors : sequence of column labels
        Predictor columns to encode. A label listed twice is used once, as
        a term listed twice in an R formula.

    Returns
    -------
    pd.DataFrame
        Float design matrix with the index of `data`. It has no columns
        when `predictors` is empty.

    Raises
    ------
    KeyError
        If a predictor is not a column of `data`.
    TypeError
        If a predictor has a dtype R has no counterpart for here (datetime,
        complex, ...).
    ValueError
        If an ordered Categorical has more than 95 levels (R's limit for
        polynomial contrasts).

    Notes
    -----
    A categorical predictor with a single level contributes no column. R
    stops with an error there instead; inside ``mice`` such a variable is
    constant and is removed as a predictor before a design matrix is built.

    Levels of object and string columns are sorted by code point, as R
    sorts them in the C locale. In another locale R may order mixed-case
    labels differently, which changes which level is dropped.

    Examples
    --------
    >>> data = pd.DataFrame({"age": [21.0, 35.0, 52.0],
    ...                      "hyp": ["no", "yes", "no"]})
    >>> obtain_design(data, ["age", "hyp"])
        age  hypyes
    0  21.0     0.0
    1  35.0     1.0
    2  52.0     0.0
    """
    if isinstance(predictors, str):
        predictors = [predictors]
    names: list = []
    values: list[np.ndarray] = []
    for predictor in dict.fromkeys(predictors):
        column_names, column_values = _design_columns(data[predictor], predictor)
        names.extend(column_names)
        values.extend(column_values)
    matrix = np.column_stack(values) if values else np.empty((len(data), 0))
    return pd.DataFrame(matrix, index=data.index, columns=pd.Index(names, dtype=object))


# ---------------------------------------------------------------------------
# remove_lindep
# ---------------------------------------------------------------------------


def _column_variance(x: np.ndarray) -> np.ndarray:
    """Sample variance (denominator n - 1) of each column; NaN if n < 2."""
    n = x.shape[0]
    if n < 2:
        return np.full(x.shape[1], np.nan)
    centred = x - x.mean(axis=0)
    return np.einsum("ij,ij->j", centred, centred) / (n - 1)


def _correlation_with(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation of each column of ``x`` with ``y``.

    Every column of ``x`` and ``y`` must have positive variance.
    """
    dx = x - x.mean(axis=0)
    dy = y - y.mean()
    sx = np.sqrt(np.einsum("ij,ij->j", dx, dx))
    sy = np.sqrt(dy @ dy)
    return np.clip((dx.T @ dy) / (sx * sy), -1.0, 1.0)


def _correlation_matrix(x: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix of columns with positive variance."""
    centred = x - x.mean(axis=0)
    cov = centred.T @ centred / (x.shape[0] - 1)
    sd = np.sqrt(np.diag(cov))
    corr = np.clip(cov / np.outer(sd, sd), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _remove_lindep(
    x: ArrayLike | pd.DataFrame, y: ArrayLike, ry: ArrayLike, eps: float = 1e-4, maxcor: float = 0.99
) -> tuple[np.ndarray, str | None]:
    """:func:`remove_lindep` plus the event R would add to its log.

    Returns ``(keep, event)``. ``event`` is the ``out`` text that
    ``mice:::remove.lindep`` passes to ``updateLog``, or None when it logs
    nothing:

    - :data:`ALL_PREDICTORS_DROPPED` when no column survives the variance
      and correlation screen;
    - the names of the dropped columns, joined by ``", "``, when at least
      two columns survive that screen and some column is dropped, by the
      screen or by the eigenvalue step (``""`` when ``x`` has no names).

    Columns dropped by the screen alone, with at most one column left, are
    not logged; nor is anything when the observed ``y`` is constant. That
    is R's behaviour.
    """
    names = list(x.columns) if isinstance(x, pd.DataFrame) else None
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"x must be two-dimensional, got shape {x.shape}")
    ry = np.asarray(ry, dtype=bool)
    n_cols = x.shape[1]

    if n_cols == 0:
        return np.zeros(0, dtype=bool), None
    if eps == 0:
        return np.ones(n_cols, dtype=bool), None
    if eps < 0:
        raise ValueError(f"eps must not be negative, got {eps}")
    if not ry.any():
        # nothing observed: keep every predictor (R's allow.na = TRUE)
        return np.ones(n_cols, dtype=bool), None

    x_obs = x[ry]
    y_obs = _as_numeric(y)[ry]
    y_variance = _column_variance(y_obs[:, None])[0]
    if not y_variance >= eps:
        # Constant observed y: no predictor is useful. A single observed
        # value (undefined variance) is treated the same way; R stops with
        # an error there.
        return np.zeros(n_cols, dtype=bool), None

    keep = _column_variance(x_obs) > eps
    cor_y = np.full(n_cols, np.nan)
    cor_y[keep] = _correlation_with(x_obs[:, keep], y_obs)
    # signed correlation: a predictor close to -y is kept, as in R
    keep &= cor_y < maxcor
    if not keep.any():
        return keep, ALL_PREDICTORS_DROPPED
    k = int(keep.sum())
    if k == 1:
        return keep, None

    # Eigen decomposition of the correlations of the screened columns,
    # eigenvalues in decreasing order (R's eigen()).
    values, vectors = linalg.eigh(_correlation_matrix(x_obs[:, keep]), driver="evr")
    values, vectors = values[::-1], vectors[:, ::-1]
    # R recomputes the decomposition after each drop, but on a submatrix
    # selected with an index that R recycles to the full matrix. Every pass
    # therefore reads the decomposition of all screened columns: pass k
    # tests the k-th largest eigenvalue, and the column picked from its
    # eigenvector is counted among the columns still kept, which may be
    # a different column, or none when the position exceeds k.
    while values[k - 1] / values[0] < eps:
        position = int(np.argmax(np.abs(vectors[:, k - 1])))
        if position < k:
            keep[np.flatnonzero(keep)[position]] = False
        k -= 1
        if k == 0:
            raise ValueError(f"eps = {eps} is too large: no eigenvalue ratio reaches it")

    event = None
    if not keep.all():
        event = ", ".join(str(names[j]) for j in np.flatnonzero(~keep)) if names else ""
    return keep, event


def remove_lindep(
    x: ArrayLike | pd.DataFrame, y: ArrayLike, ry: ArrayLike, eps: float = 1e-4, maxcor: float = 0.99
) -> np.ndarray:
    """Columns of a design matrix to keep in one imputation model.

    Reproduces ``mice:::remove.lindep`` of R mice 3.19.0, which ``mice``
    applies to the design matrix of each target before every univariate
    imputation. On the rows where ``y`` is observed:

    1. if ``y`` is constant (variance below `eps`), no column is kept;
    2. a column is dropped if its variance is not above `eps`, or if its
       correlation with ``y`` is not below `maxcor` (signed: a column
       nearly equal to ``-y`` stays);
    3. if two or more columns remain, columns are dropped while the ratio
       of an eigenvalue of their correlation matrix to the largest one is
       below `eps`. See the notes for how R carries this out.

    Parameters
    ----------
    x : array-like or pd.DataFrame, shape (n, p)
        Numeric design matrix without intercept, for example from
        :func:`obtain_design`.
    y : array-like, shape (n,)
        The target. Categorical values count by their level codes, as
        ``as.numeric`` counts a factor in R.
    ry : array-like of bool, shape (n,)
        True where ``y`` is observed (and ``x`` complete).
    eps : float, default 1e-4
        Threshold for variances and eigenvalue ratios. ``0`` keeps every
        column.
    maxcor : float, default 0.99
        Columns whose correlation with ``y`` reaches this value are dropped.

    Returns
    -------
    np.ndarray of bool, shape (p,)
        True for the columns to keep. All True when no ``y`` is observed;
        empty when ``x`` has no columns.

    Raises
    ------
    ValueError
        If `eps` is negative, or so large that the eigenvalue step runs out
        of columns (R stops with an error there too).

    Notes
    -----
    Step 3 follows R literally, including a quirk that matters only when
    more than one column has to go. R drops the column with the largest
    absolute loading on the eigenvector of the smallest eigenvalue, but
    afterwards keeps using the eigen decomposition of all columns that
    passed step 2: the next pass looks at the second-smallest eigenvalue of
    that original matrix, and the position of its largest loading is
    counted among the columns still kept. So R can keep a column that is
    still collinear, or drop an unrelated one; this function does the same.

    When ``y`` has a single observed value R stops with an error; this
    function keeps no column instead.

    Ties between equal loadings (exact duplicate columns) go to the first
    one, as in R, but which loading is larger can depend on rounding in the
    eigen solver, so R and this function may drop different copies.

    Examples
    --------
    >>> rng = np.random.default_rng(1)
    >>> x1, x2 = rng.normal(size=30), rng.normal(size=30)
    >>> near_copy = x1 + rng.normal(scale=1e-3, size=30)
    >>> x = np.column_stack([x1, near_copy, np.full(30, 2.0), x2])
    >>> y = x1 + x2 + rng.normal(size=30)
    >>> remove_lindep(x, y, np.ones(30, dtype=bool)).tolist()
    [True, False, False, True]
    """
    keep, _ = _remove_lindep(x, y, ry, eps=eps, maxcor=maxcor)
    return keep


# ---------------------------------------------------------------------------
# find_collinear
# ---------------------------------------------------------------------------


def _pairwise_correlation(x: np.ndarray) -> np.ndarray:
    """Upper triangle of the pairwise-complete Pearson correlation matrix.

    Entry ``(i, j)`` with ``i < j`` uses the rows where both columns are
    observed; it is NaN when fewer than two such rows exist or either
    column is constant on them. The diagonal and lower triangle are NaN.
    """
    n_cols = x.shape[1]
    out = np.full((n_cols, n_cols), np.nan)
    present = ~np.isnan(x)
    for i in range(n_cols - 1):
        both = present[:, [i]] & present[:, i + 1 :]
        count = both.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_i = np.where(both, x[:, [i]], 0.0).sum(axis=0) / count
            mean_j = np.where(both, x[:, i + 1 :], 0.0).sum(axis=0) / count
            dev_i = np.where(both, x[:, [i]] - mean_i, 0.0)
            dev_j = np.where(both, x[:, i + 1 :] - mean_j, 0.0)
            sxy = (dev_i * dev_j).sum(axis=0)
            sxx = (dev_i * dev_i).sum(axis=0)
            syy = (dev_j * dev_j).sum(axis=0)
            corr = sxy / (np.sqrt(sxx) * np.sqrt(syy))
        corr[(count < 2) | (sxx == 0) | (syy == 0)] = np.nan
        out[i, i + 1 :] = np.clip(corr, -1.0, 1.0)
    return out


def find_collinear(data: pd.DataFrame, threshold: float = 0.999) -> list[Hashable]:
    """Variables that (nearly) duplicate another variable.

    Reproduces ``mice:::find.collinear`` of R mice 3.19.0, which the setup
    step of ``mice`` uses to stop imputing with collinear variables:

    1. categorical and boolean columns are replaced by their level codes
       (R's ``data.matrix``);
    2. the columns are ordered by their number of observed values, most
       first; ties keep the column order of `data`;
    3. a column is reported if its absolute pairwise-complete correlation
       with any column before it in that order reaches `threshold`.

    So of two collinear variables the one with more missing values is
    reported, and of two equally complete ones the later one.

    Parameters
    ----------
    data : pd.DataFrame
        The data, possibly incomplete.
    threshold : float, default 0.999
        Absolute correlation at which a variable counts as collinear.

    Returns
    -------
    list
        Column labels of `data`, in the order of step 2.

    Notes
    -----
    Constant columns have no correlation and are never reported here; the
    setup step treats them separately. The level codes of an unordered
    categorical depend on its level order, so for three or more levels the
    correlation does too, as in R.

    Examples
    --------
    >>> data = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0],
    ...                      "b": [2.0, 4.0, 6.0, np.nan],
    ...                      "c": [1.0, 0.0, 1.0, 0.0]})
    >>> find_collinear(data)
    ['b']
    """
    columns = list(data.columns)
    if not columns:
        return []
    matrix = np.column_stack([_as_numeric(data.iloc[:, j]) for j in range(len(columns))])
    n_observed = (~np.isnan(matrix)).sum(axis=0)
    order = np.argsort(-n_observed, kind="stable")
    corr = _pairwise_correlation(matrix[:, order])
    with np.errstate(invalid="ignore"):
        hit = np.abs(corr) >= threshold  # NaN compares False
    collinear = hit.any(axis=0)
    return [columns[order[j]] for j in np.flatnonzero(collinear)]


# ---------------------------------------------------------------------------
# quickpred
# ---------------------------------------------------------------------------

_CORRELATION_METHODS = ("pearson", "kendall", "spearman")


def _correlation_columns_with(a: np.ndarray, b: np.ndarray, method: str) -> np.ndarray:
    """Correlation of each column of ``a`` with ``b``, neither with missing values.

    NaN where a column or ``b`` is constant (or fewer than two rows exist),
    as R's ``cor`` returns NA there. Spearman ranks the values with average
    ranks for ties; Kendall is tau-b, as in R.
    """
    if a.shape[0] < 2:
        return np.full(a.shape[1], np.nan)
    if method == "kendall":
        from scipy.stats import kendalltau

        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.array([kendalltau(column, b)[0] for column in a.T], dtype=float)
        constant = (np.ptp(a, axis=0) == 0) | (np.ptp(b) == 0)
        out[constant] = np.nan
        return out
    if method == "spearman":
        from scipy.stats import rankdata

        a = rankdata(a, axis=0)
        b = rankdata(b)
    da = a - a.mean(axis=0)
    db = b - b.mean()
    scale = np.sqrt(np.einsum("ij,ij->j", da, da) * (db @ db))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (db @ da) / scale
    out[scale == 0] = np.nan
    return np.clip(out, -1.0, 1.0)


def _column_labels(names: Any, columns: list, argument: str) -> list:
    """``include`` / ``exclude`` as a list of column labels; unknown labels raise."""
    if names is None:
        return []
    if isinstance(names, str) or not isinstance(names, (list, tuple, set, pd.Index, np.ndarray)):
        names = [names]
    names = list(names)
    unknown = [name for name in names if name not in columns]
    if unknown:
        raise ValueError(f"quickpred: {argument} names columns that are not in the data: {unknown}")
    return names


def quickpred(
    data: pd.DataFrame,
    mincor: float | ArrayLike = 0.1,
    minpuc: float | ArrayLike = 0.0,
    include: Hashable | Sequence[Hashable] | None = None,
    exclude: Hashable | Sequence[Hashable] | None = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """Select predictors by simple statistics, as R's ``quickpred``.

    Reproduces ``mice::quickpred`` of R mice 3.19.0. For every pair of a
    target (row) and a predictor (column) it computes, on the pairwise
    complete cases,

    - the absolute correlation between the two columns, and
    - the absolute correlation between the target's response indicator
      (1 where the target is observed, 0 where it is missing) and the
      predictor,

    and uses the predictor if the larger of the two exceeds `mincor`
    (strictly). Then:

    1. if `minpuc` is not 0, a predictor is dropped when its proportion of
       usable cases is below `minpuc`: of the rows where the target is
       missing, the share where the predictor is observed;
    2. the columns in `exclude` are dropped as predictors of every target;
    3. the columns in `include` are added as predictors of every target
       (so a column in both lists is included);
    4. the diagonal is set to 0, and so are the rows of complete columns.

    Parameters
    ----------
    data : pd.DataFrame
        The incomplete data. Categorical, object, string and boolean columns
        enter by their level codes, as R's ``data.matrix`` does (see Notes).
    mincor : float or array-like, default 0.1
        Minimum absolute correlation. A vector of length ``n_columns`` holds
        one threshold per target, a square matrix one per pair, as in R.
    minpuc : float or array-like, default 0.0
        Minimum proportion of usable cases, shaped like `mincor`. With 0 no
        predictor is dropped for this reason.
    include : label or list of labels, optional
        Columns used as predictors of every target.
    exclude : label or list of labels, optional
        Columns never used as predictors.
    method : {"pearson", "kendall", "spearman"}, default "pearson"
        Correlation coefficient.

    Returns
    -------
    pd.DataFrame
        Integer 0/1 matrix with the columns of `data` as index (targets) and
        as columns (predictors); pass it to ``MICE.impute`` as
        ``predictor_matrix``.

    Raises
    ------
    ValueError
        For an unknown `method`, or `include` / `exclude` naming a column
        that is not in `data`.

    Notes
    -----
    Differences from R: `include` and `exclude` take exact column labels
    (R matches partial names with ``pmatch``), and an unknown label raises
    instead of being ignored. The level codes of an unordered categorical
    with three or more levels depend on its level order, so its
    correlations do too, as in R; object and string columns use their
    sorted values as levels.

    Examples
    --------
    >>> import imputation
    >>> pred = imputation.quickpred(data, mincor=0.3, include=["age"])  # doctest: +SKIP
    >>> imputation.MICE(data).impute(predictor_matrix=pred)  # doctest: +SKIP
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    if method not in _CORRELATION_METHODS:
        raise ValueError(f"quickpred: method must be one of {_CORRELATION_METHODS}, got {method!r}")
    columns = list(data.columns)
    include = _column_labels(include, columns, "include")
    exclude = _column_labels(exclude, columns, "exclude")
    n_cols = len(columns)
    predictors = np.zeros((n_cols, n_cols), dtype=int)
    if n_cols == 0:
        return pd.DataFrame(predictors, index=data.columns, columns=data.columns)

    x = np.column_stack([_as_numeric(data.iloc[:, j]) for j in range(n_cols)])
    observed = ~np.isnan(x)

    # 1) correlation of the columns, pairwise complete
    v = pd.DataFrame(x).corr(method=method).to_numpy()
    # 2) correlation of the target's response indicator (row) with the
    #    predictor's values (column), on the rows where the predictor is observed
    u = np.empty((n_cols, n_cols))
    for j in range(n_cols):
        rows = observed[:, j]
        u[:, j] = _correlation_columns_with(observed[rows].astype(float), x[rows, j], method)
    largest = np.fmax(np.nan_to_num(np.abs(v)), np.nan_to_num(np.abs(u)))

    mincor = np.asarray(mincor, dtype=float)
    minpuc = np.asarray(minpuc, dtype=float)
    # A vector holds one value per target (row), as R recycles it by column
    if mincor.ndim == 1:
        mincor = mincor[:, None]
    if minpuc.ndim == 1:
        minpuc = minpuc[:, None]
    predictors[largest > mincor] = 1

    if np.any(minpuc != 0):
        missing = ~observed
        usable = missing.T.astype(int) @ observed.astype(int)  # target missing, predictor observed
        with np.errstate(invalid="ignore", divide="ignore"):
            puc = usable / missing.sum(axis=0)[:, None]
        predictors[puc < minpuc] = 0  # NaN (complete target) compares False

    predictors[:, [columns.index(c) for c in exclude]] = 0
    predictors[:, [columns.index(c) for c in include]] = 1
    np.fill_diagonal(predictors, 0)
    predictors[observed.all(axis=0), :] = 0
    return pd.DataFrame(predictors, index=data.columns, columns=data.columns)
