import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .design import obtain_design

logger = logging.getLogger(__name__)


def _design_matrix(x) -> np.ndarray:
    """Predictors as a 2-D float array.

    A DataFrame with categorical, object, string or boolean columns is
    dummy-coded by :func:`imputation.design.obtain_design`, as MICE does;
    numeric input is only converted to float.
    """
    if isinstance(x, pd.DataFrame) and not x.select_dtypes(exclude=[np.number]).columns.empty:
        x = obtain_design(x, list(x.columns))
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def _is_numeric_target(y) -> bool:
    """True if `y` has a numeric dtype that is neither boolean nor categorical.

    Such a target gets a regression tree; any other target (categorical,
    object, string, boolean) a classification tree.
    """
    dtype = getattr(y, "dtype", None)
    if dtype is None:
        dtype = np.asarray(y).dtype
    return pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)


def _label_codes(y_obs: np.ndarray) -> np.ndarray:
    """Integer codes of the observed labels, in sorted label order if they sort.

    scikit-learn's classifiers accept numbers and strings as labels but not,
    for example, an object array of booleans. The trees are fitted on these
    codes; the imputations are drawn from the labels themselves. Sorted codes
    give the same tree as fitting on sortable labels directly.
    """
    try:
        codes, _ = pd.factorize(y_obs, sort=True)
    except TypeError:  # labels that cannot be ordered: order of appearance
        codes, _ = pd.factorize(y_obs, sort=False)
    return np.asarray(codes)


def _draw_pooled_donors(
    nodes_obs: np.ndarray,
    nodes_mis: np.ndarray,
    y_obs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one donor for each missing row from the observed rows in its leaves.

    For each missing row the pool holds, for every tree, the observed values
    that fall into the same leaf of that tree, whether they were used to
    grow the tree or not. An observed row enters the pool once for every
    tree in which it shares the leaf, so the pool has multiplicities. One
    value is drawn uniformly from the pool. This is R mice's rule for
    ``rf`` (the donors of all trees, unlisted, then ``sample(, 1)``); with a
    single tree it is the rule of ``cart``.

    Parameters
    ----------
    nodes_obs : np.ndarray
        Leaf ids of the observed rows, shape ``(n_obs, n_trees)`` (or
        ``(n_obs,)`` for one tree).
    nodes_mis : np.ndarray
        Leaf ids of the missing rows, shape ``(n_mis, n_trees)`` (or
        ``(n_mis,)``).
    y_obs : np.ndarray
        The observed values, length ``n_obs``.
    rng : np.random.Generator
        Source of the draws: one integer per missing row.

    Returns
    -------
    np.ndarray
        ``n_mis`` values of `y_obs`.

    Raises
    ------
    ValueError
        If a missing row shares no leaf with any observed row (impossible
        for leaves of a tree grown on observed rows).
    """
    nodes_obs = np.asarray(nodes_obs)
    nodes_mis = np.asarray(nodes_mis)
    if nodes_obs.ndim == 1:
        nodes_obs = nodes_obs[:, None]
    if nodes_mis.ndim == 1:
        nodes_mis = nodes_mis[:, None]
    y_obs = np.asarray(y_obs)
    n_mis, n_trees = nodes_mis.shape
    rows = np.arange(n_mis)

    # Per tree, the observed rows sorted by leaf: the members of one leaf are
    # contiguous, starting at `start` and `count` long
    order = np.argsort(nodes_obs, axis=0, kind="stable")
    sorted_leaves = np.take_along_axis(nodes_obs, order, axis=0)
    start = np.empty((n_mis, n_trees), dtype=np.intp)
    count = np.empty((n_mis, n_trees), dtype=np.intp)
    for t in range(n_trees):
        first = np.searchsorted(sorted_leaves[:, t], nodes_mis[:, t], side="left")
        start[:, t] = first
        count[:, t] = np.searchsorted(sorted_leaves[:, t], nodes_mis[:, t], side="right") - first
    pool_size = count.sum(axis=1)
    if np.any(pool_size == 0):
        raise ValueError("a missing row shares no leaf with any observed row")

    # Position in the pool of each missing row, then the tree and the leaf member it falls on
    position = rng.integers(pool_size)
    cumulative = np.cumsum(count, axis=1)
    tree = (cumulative <= position[:, None]).sum(axis=1)
    offset = position - (cumulative[rows, tree] - count[rows, tree])
    return y_obs[order[start[rows, tree] + offset, tree]]


def cart(
    y: Union[pd.Series, np.ndarray],
    id_obs: np.ndarray,
    x: Union[pd.DataFrame, np.ndarray],
    id_mis: Optional[np.ndarray] = None,
    min_samples_leaf: int = 5,
    cp: float = 1e-4,
    min_samples_split: Optional[int] = None,
    max_depth: int = 30,
    bootstrap: bool = False,
    ccp_alpha: Optional[float] = None,
    tree_params: Optional[dict] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Impute missing values using Classification and Regression Trees (CART).

    This function is designed to be compatible with the MICE framework and
    follows R's ``mice.impute.cart``, which grows the tree with ``rpart``.

    Parameters
    ----------
    y : Union[pd.Series, np.ndarray]
        Target variable with missing values
    id_obs : np.ndarray
        Boolean mask of observed values in y (True for observed, False for missing)
    x : Union[pd.DataFrame, np.ndarray]
        Predictor variables (must be fully observed). A DataFrame with
        categorical, object, string or boolean columns is dummy-coded first by
        :func:`imputation.design.obtain_design`, as MICE does. Without any
        predictor, the imputations are draws from the observed values.
    id_mis : np.ndarray, optional
        Boolean mask of missing values to impute. If None, uses ~id_obs
    min_samples_leaf : int, default=5
        Minimum number of observations in a leaf (rpart's ``minbucket``).
    cp : float, default=1e-4
        Complexity parameter, relative as rpart's ``cp``: a numeric target's
        tree is pruned with ``ccp_alpha = cp * var(y)``, the population
        variance of the values the tree is fitted on. Not used for a
        categorical target.
    min_samples_split : int, optional
        Minimum number of observations in a node for a split to be tried
        (rpart's ``minsplit``). None means ``3 * min_samples_leaf``, rpart's
        default when only ``minbucket`` is given, as in R mice.
    max_depth : int, default=30
        Maximum depth of the tree (rpart's ``maxdepth``).
    bootstrap : bool, default=False
        If True, the tree is fitted on a bootstrap sample of the observed
        rows; the donors are still the original observed rows in each leaf.
        R does not bootstrap; this was the behaviour of mice-py 0.1.
    ccp_alpha : float, optional
        Cost-complexity pruning parameter in scikit-learn's units. If given,
        it replaces the pruning derived from `cp`, also for a categorical
        target (which is otherwise not pruned).
    tree_params : dict, optional
        Further arguments for scikit-learn's ``DecisionTreeRegressor`` or
        ``DecisionTreeClassifier``, e.g. ``{"max_leaf_nodes": 20}``. It must
        not set ``min_samples_leaf``, ``min_samples_split``, ``max_depth``,
        ``ccp_alpha`` or ``random_state``.
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None, a fresh generator is used.

    Returns
    -------
    np.ndarray
        Imputed values for missing positions only (matching R implementation).
        A numeric target gets a regression tree. Any other target (categorical,
        object, string, boolean) gets a classification tree, and its
        imputations are observed labels of `y`, so booleans stay booleans.

    Raises
    ------
    ValueError
        If `y` has no observed value and some value is to be imputed.

    Notes
    -----
    The procedure follows R's ``mice.impute.cart``:

    1. Fit a regression tree (numeric target) or a classification tree
       (any other target) on the observed rows.
    2. For each missing value, find the leaf it falls into.
    3. Draw one of the observed values in that leaf at random.

    For a categorical target R draws a level with the class proportions of
    the leaf, which is the same as drawing one of its observed values.
    A categorical target whose observed values are all equal is imputed
    with that value. With the default settings a numeric target's tree has
    the same leaves as rpart's in our comparisons; scikit-learn grows
    classification trees differently from rpart, and breaks ties between
    equally good splits at random where rpart takes the first.
    """
    logger.debug("Starting CART imputation.")

    tree_params = dict(tree_params or {})
    reserved = sorted(
        {"min_samples_leaf", "min_samples_split", "max_depth", "ccp_alpha", "random_state"} & set(tree_params)
    )
    if reserved:
        raise TypeError(
            f"tree_params must not set {reserved}: pass min_samples_leaf, min_samples_split, "
            "max_depth and ccp_alpha as arguments of cart; the random state is drawn from rng."
        )

    x = _design_matrix(x)
    id_obs = np.asarray(id_obs, dtype=bool)

    if id_mis is None:
        id_mis = ~id_obs
    id_mis = np.asarray(id_mis, dtype=bool)

    if rng is None:
        rng = np.random.default_rng()

    min_samples_leaf = max(1, int(min_samples_leaf))
    if min_samples_split is None:
        min_samples_split = 3 * min_samples_leaf

    if x.shape[1] == 0:
        x = np.ones((len(x), 1))

    x_obs = x[id_obs]
    x_mis = x[id_mis]
    is_numeric = _is_numeric_target(y)
    y_obs = np.asarray(y)[id_obs]

    if len(x_mis) == 0:
        return np.array([])
    if len(y_obs) == 0:
        raise ValueError(
            "cart: the target has no observed value in the rows used for the model, so there is no donor to draw from."
        )

    if is_numeric:
        target = y_obs.astype(float)
    else:
        target = _label_codes(y_obs)
        if target.max() == 0:  # a single observed label
            return np.repeat(y_obs[:1], len(x_mis))

    if bootstrap:
        fit_rows = rng.choice(len(y_obs), size=len(y_obs), replace=True)
        x_fit, target_fit = x_obs[fit_rows], target[fit_rows]
    else:
        x_fit, target_fit = x_obs, target

    if is_numeric:
        logger.debug("Performing regression tree imputation.")
        alpha = cp * float(np.var(target_fit)) if ccp_alpha is None else ccp_alpha
        tree_class = DecisionTreeRegressor
    else:
        logger.debug("Performing classification tree imputation.")
        alpha = 0.0 if ccp_alpha is None else ccp_alpha
        tree_class = DecisionTreeClassifier

    tree = tree_class(
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_depth=max_depth,
        ccp_alpha=alpha,
        random_state=int(rng.integers(2**31 - 1)),
        **tree_params,
    )
    tree.fit(x_fit, target_fit)

    imputed_values = _draw_pooled_donors(tree.apply(x_obs), tree.apply(x_mis), y_obs, rng)
    logger.debug(f"CART imputation complete. Imputed {len(imputed_values)} values.")
    return imputed_values
