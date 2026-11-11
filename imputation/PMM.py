import pandas as pd
import numpy as np
from .sampler import norm_draw

def pmm(y, id_obs, x, id_mis = None, donors = 5, matchtype = 1,
                    quantify = True, ridge = 1e-5, ls_meth = "qr", rng = None):
    """
       Predictive Mean Matching (PMM) imputation.

       This function imputes missing values in a variable `y` using predictive mean matching.
       The method is based on Rubin's (1987) Bayesian linear regression and mimics the behavior
       of the R `mice` package's PMM imputation method.

       Parameters
       ----------
       y : array-like (1D), shape (n_samples,)
           Target variable to be imputed. Numeric, or categorical of any
           non-numeric dtype (object, category, bool, string).

       id_obs : array-like of bool, shape (n_samples,)
           Logical array indicating which elements of `y` are observed (True) or missing (False).

       x : array-like (2D), shape (n_samples, n_features)
           Numeric design matrix of predictors. Must have no missing values.

       id_mis : array-like of bool, shape (n_samples,), optional
           Logical array indicating which values should be imputed.
           If None, `id_mis` is set to the complement of `id_obs`.

       donors : int, default=5
           Number of donors to draw from the observed cases when imputing missing values.
           Clamped to the range from 1 to the number of observed values.

       matchtype : int, default=1
           Type of matching:
           - 0: Predicted value of y_obs vs predicted value of y_mis
           - 1: Predicted value of y_obs vs drawn value of y_mis (default)
           - 2: Drawn value of y_obs vs drawn value of y_mis

       quantify : bool, default=True
           If True and `y` is categorical, its categories are replaced by the
           optimal scaling of `quantify()` before the regression. If False,
           categories are replaced by integer codes (less accurate).

       ridge : float, default=1e-5
           Ridge regularization parameter used in `norm_draw()` to stabilize estimation.
           Increase for multicollinear data, decrease to reduce bias.

       ls_meth : str, default="qr"
           Least squares method passed to `norm_draw()`: "qr", "ridge" or "svd".

       rng : np.random.Generator, optional
           Random number generator for reproducibility. If None, a fresh generator is used.

       Returns
       -------
       y_imp : np.ndarray
           Imputed values for the `id_mis` positions only (as in R):
           ``y[id_obs][idx]``, so observed values or category labels of `y`.

       Notes
       -----
       Based on:
       - Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*.
       - Van Buuren, S. & Groothuis-Oudshoorn, K. (2011). `mice` R package.

       Examples
       --------
       >>> y = np.array([7, np.nan, 9, 10, 11])
       >>> id_obs = ~np.isnan(y)
       >>> x = np.array([[1, 2], [3, 4], [5, 7], [7, 8], [9, 10]])
       >>> pmm(y=y, id_obs=id_obs, x=x, donors=3)
    """
    # Validate predictors (x): must be numeric and contain no missing values
    if isinstance(x, pd.DataFrame):
        non_numeric_cols = x.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            raise ValueError(
                f"Predictors must be numeric for pmm. Non-numeric predictors found: {non_numeric_cols}"
            )
        missing_cols = x.columns[x.isna().any()].tolist()
        if missing_cols:
            raise ValueError(
                f"Predictors must not contain missing values for pmm. Columns with missing values: {missing_cols}"
            )
        x = x.to_numpy(dtype=float)
    else:
        x = np.asarray(x)
        # Try to coerce to float to detect non-numeric entries early
        try:
            x = x.astype(float)
        except Exception:
            raise ValueError("Predictors must be numeric for pmm. Could not convert 'x' to numeric array.")
        if np.isnan(x).any():
            raise ValueError("Predictors must not contain missing values for pmm.")
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    id_obs = np.asarray(id_obs, dtype=bool)
    id_mis = ~id_obs if id_mis is None else np.asarray(id_mis, dtype=bool)

    # Create default RNG if not provided
    if rng is None:
        rng = np.random.default_rng()

    # Add a column of ones to the matrix x
    x = np.c_[np.ones(x.shape[0]), x]

    # Numeric version of y for the regression; categories are quantified
    values = np.asarray(y)
    if _is_categorical(y):
        if quantify:
            ynum = _quantify(y, id_obs, x)
        else:
            ynum = np.full(len(values), np.nan)
            ynum[id_obs] = _category_codes(y, id_obs)
    else:
        ynum = _as_float(y)

    # Parameter estimation
    p = norm_draw(ynum, id_obs, x, ridge=ridge, rng=rng, ls_meth=ls_meth)

    #dotproduct x @ parameter = predicted values
    if matchtype == 0:
        yhatobs = np.dot(x[id_obs, :], p["coef"])
        yhatmis = np.dot(x[id_mis, :], p["coef"])
    elif matchtype == 1:
        yhatobs = np.dot(x[id_obs, :], p["coef"])
        yhatmis = np.dot(x[id_mis, :], p["beta"])
    elif matchtype == 2:
        yhatobs = np.dot(x[id_obs, :], p["beta"])
        yhatmis = np.dot(x[id_mis, :], p["beta"])
    else:
        raise ValueError(f"matchtype must be 0, 1 or 2, got {matchtype!r}")

    idx = matchindex(yhatobs, yhatmis, k=donors, rng=rng)

    # The observed values (or labels) of the selected donors
    return values[id_obs][idx]
def quantify(y, id_obs, x):
    """
    Optimal scaling of a categorical variable, as R mice's ``quantify()``.

    The categories of the observed `y` are replaced by the scores that
    correlate best with the predictors: a canonical correlation analysis,
    without centering, of the category indicators against `x`. The first
    canonical pair is the constant (the indicators sum to one and `x` holds
    the intercept), so the second canonical variate of the indicator side is
    used, standardised to mean 0 and standard deviation 1 (divisor n - 1).

    Parameters
    ----------
    y : array-like, shape (n_samples,)
        Categorical variable; only the entries where `id_obs` is True are used.
    id_obs : array-like of bool, shape (n_samples,)
        True where `y` is observed.
    x : array-like, shape (n_samples, n_features)
        Numeric predictors without missing values, including the intercept
        column (as pmm passes them).

    Returns
    -------
    np.ndarray of float, shape (n_samples,)
        The scores of the observed entries; NaN where `y` is missing. The
        scores are defined up to their sign. If fewer than two categories are
        observed, every observed entry scores 0.
    """
    id_obs = np.asarray(id_obs, dtype=bool)
    codes = _category_codes(y, id_obs)
    ynum = np.full(len(id_obs), np.nan)
    n_levels = int(codes.max()) + 1 if codes.size else 0
    if n_levels < 2:
        ynum[id_obs] = 0.0
        return ynum
    # one indicator column per observed category
    indicators = (codes[:, None] == np.arange(n_levels)).astype(float)
    predictors = np.asarray(x, dtype=float)[id_obs]
    if predictors.ndim == 1:
        predictors = predictors.reshape(-1, 1)
    basis_y = _column_space(indicators)
    basis_x = _column_space(predictors)
    # canonical directions: singular vectors of the product of the two bases;
    # full_matrices keeps a second direction when x spans a single dimension
    u, _, _ = np.linalg.svd(basis_y.T @ basis_x, full_matrices=True)
    score = basis_y @ u[:, 1]
    ynum[id_obs] = (score - score.mean()) / score.std(ddof=1)
    return ynum


# pmm's argument `quantify` hides the function inside pmm
_quantify = quantify


def matchindex(d, t, k=5, rng=None):
    """
    Index of a matched donor for every target, as R mice 3.19.0's ``matchindex()``.

    For each value in `t`, one of the `k` donors in `d` nearest to it is
    drawn at random: a number ``h`` is drawn uniformly from 1 to `k` and the
    ``h``-th nearest donor is returned. Ties between donors are broken by one
    random permutation of the donors per call, shared by all targets. Among
    donors at the same distance on both sides of a target, the larger one
    comes first.

    Parameters
    ----------
    d : array-like of float, shape (n_donors,)
        Values of the donors (e.g. predicted values of the observed cases).
    t : array-like of float, shape (n_targets,)
        Values of the targets (e.g. predicted values of the missing cases).
    k : int, default=5
        Number of nearest donors to draw from; clamped to ``[1, n_donors]``.
    rng : np.random.Generator, optional
        Random number generator. If None, a fresh generator is used. Draws
        one permutation of the donors, then one integer per target.

    Returns
    -------
    np.ndarray of int, shape (n_targets,)
        Positions in `d` of the matched donors.

    Examples
    --------
    >>> d = np.array([-5, 5, 0, 10, 12])
    >>> t = np.array([-6, -4, 0, 2, 4, -2, 6])
    >>> matchindex(d, t, k=1)
    array([0, 0, 2, 2, 1, 2, 1])
    """
    d = np.asarray(d, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    if rng is None:
        rng = np.random.default_rng()
    n_donors = d.size
    if n_donors == 0:
        raise ValueError("matchindex needs at least one donor")
    k = int(min(max(k, 1), n_donors))

    # Sort the donors after one random shuffle, so that tied donors end up in
    # random order; `order` maps sorted positions back to positions in d.
    shuffle = rng.permutation(n_donors)
    order = shuffle[np.argsort(d[shuffle], kind="stable")]
    ds = d[order]
    h = rng.integers(1, k + 1, size=t.size)

    # Donors ds[:split] lie left of each target, ds[split:] right of it (or
    # on it). Walking outwards, the next donor is the left one if it is
    # strictly closer, else the right one. The h nearest donors consist of
    # some number `a` from the left and h - a from the right. `a` is found by
    # bisection, for all targets at once: the a-th nearest left donor is among
    # the h nearest if it is closer than the (h - a)-th nearest right donor.
    split = np.searchsorted(ds, t, side="left")
    n_right = n_donors - split
    lo = np.maximum(h - n_right, 0)
    hi = np.minimum(h, split)
    while np.any(lo < hi):
        active = lo < hi
        mid = (lo + hi + 1) // 2
        left = ds[np.clip(split - mid, 0, n_donors - 1)]  # mid-th nearest on the left
        m = h - mid  # right donors that may come before it
        right = ds[np.clip(split + m, 0, n_donors - 1)]  # (m + 1)-th nearest on the right
        inside = (m >= n_right) | (t - left < right - t)
        lo = np.where(active & inside, mid, lo)
        hi = np.where(active & ~inside, mid - 1, hi)
    a = lo
    b = h - a
    # The h-th nearest is the later of the a-th left and the b-th right donor
    left_pos = np.clip(split - a, 0, n_donors - 1)
    right_pos = np.clip(split + b - 1, 0, n_donors - 1)
    left_last = (b == 0) | ((a > 0) & ~(t - ds[left_pos] < ds[right_pos] - t))
    return order[np.where(left_last, left_pos, right_pos)]


def _is_categorical(y):
    """True unless `y` has a numeric, non-boolean dtype."""
    dtype = getattr(y, "dtype", None)
    if dtype is None:
        dtype = np.asarray(y).dtype
    return not (pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype))


def _as_float(y):
    """`y` as a float array with NaN for missing entries (also for pandas NA)."""
    if isinstance(y, (pd.Series, pd.Index, pd.api.extensions.ExtensionArray)):
        return pd.Series(y).to_numpy(dtype=float, na_value=np.nan)
    return np.asarray(y, dtype=float)


def _category_codes(y, id_obs):
    """Integer codes 0, 1, ... of the observed categories of `y`.

    The order is the order of the categories for a pandas categorical, and
    the sorted order of the observed values otherwise (R's factor order).
    """
    dtype = getattr(y, "dtype", None)
    if isinstance(dtype, pd.CategoricalDtype):
        codes = np.asarray(pd.Categorical(y).codes)[id_obs]
        _, codes = np.unique(codes, return_inverse=True)  # drop unused categories
        return codes.ravel()
    observed = np.asarray(y, dtype=object)[id_obs]
    try:
        codes, _ = pd.factorize(observed, sort=True)
    except TypeError:  # values that cannot be ordered: order of appearance
        codes, _ = pd.factorize(observed, sort=False)
    return np.asarray(codes)


def _column_space(a, tol=1e-7):
    """Orthonormal basis of the column space of `a` (columns of an SVD)."""
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    rank = int(np.sum(s > tol * s.max(initial=0.0)))
    return u[:, :max(rank, 1)]
