import pandas as pd
import numpy as np
from .sampler import *
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cross_decomposition import CCA
from sklearn.neighbors import KDTree

def pmm(y, id_obs, x, id_mis = None, donors = 5, matchtype = 1,
                    quantify = True, ridge = 1e-5, matcher = "NN", rng = None, **kwargs):
    """
       Predictive Mean Matching (PMM) imputation.

       This function imputes missing values in a variable `y` using predictive mean matching.
       The method is based on Rubin's (1987) Bayesian linear regression and mimics the behavior
       of the R `mice` package's PMM imputation method.

       Parameters
       ----------
       y : array-like (1D), shape (n_samples,)
           Target variable to be imputed. Can be numeric or categorical.

       id_obs : array-like of bool, shape (n_samples,)
           Logical array indicating which elements of `y` are observed (True) or missing (False).

       x : array-like (2D), shape (n_samples, n_features)
           Numeric design matrix of predictors. Must have no missing values.

       id_mis : array-like of bool, shape (n_samples,), optional
           Logical array indicating which values should be imputed.
           If None, `id_mis` is set to the complement of `id_obs`.

       donors : int, default=5
           Number of donors to draw from the observed cases when imputing missing values.

       matchtype : int, default=1
           Type of matching:
           - 0: Predicted value of y_obs vs predicted value of y_mis
           - 1: Predicted value of y_obs vs drawn value of y_mis (default)
           - 2: Drawn value of y_obs vs drawn value of y_mis

       quantify : bool, default=True
           If True and `y` is categorical, factor levels are replaced by the first canonical variate (via CCA).
           If False, categorical values are replaced by integer codes (less accurate).

       ridge : float, default=1e-5
           Ridge regularization parameter used in `norm_draw()` to stabilize estimation.
           Increase for multicollinear data, decrease to reduce bias.

       matcher : str, default="NN"
           Matching method. Currently only "NN" (nearest neighbor) is supported.

       **kwargs : dict
           Additional arguments passed to `norm_draw()`, such as `ls_meth`.

       Returns
       -------
       y_imp : np.ndarray
           Imputed values for missing positions only (matching R implementation).
           Returns object array if `y` was categorical, else float array.

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
        x = x.to_numpy()
    else:
        x = np.asarray(x)
        # Try to coerce to float to detect non-numeric entries early
        try:
            _ = x.astype(float, copy=False)
        except Exception:
            raise ValueError("Predictors must be numeric for pmm. Could not convert 'x' to numeric array.")
        if np.isnan(x).any():
            raise ValueError("Predictors must not contain missing values for pmm.")

    if id_mis is None:
        id_mis = ~id_obs

    # Add a column of ones to the matrix x
    x = np.c_[np.ones(x.shape[0]), x]
    ynum = y
    # Quantify categories for categorical data y
    if y.dtype == "object":
        if quantify:
            # quantify function returns the numeric transformation of the factor
            #Experimental has different output than R
            #id to retransform cca to categories back
            ynum, id = quantify_cca(y, id_obs, x)
        else:
            ynum, id = pd.factorize(y)

    # Create default RNG if not provided
    if rng is None:
        rng = np.random.default_rng()
    
    # Parameter estimation
    p = norm_draw(ynum, id_obs, x, ridge=ridge, rng=rng, **kwargs)

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

    idx = matcherid(d = yhatobs, t = yhatmis, matcher = "NN", k = donors, rng = rng)
    
    # Get the observed values that were selected as donors
    donor_values = ynum[id_obs][idx]
    
    # Handle categorical data retransformation if needed
    if y.dtype == "object":
        if quantify:
            #retransform cca numericals to categories
            donor_values_obj = donor_values.astype(object)
            for col in id.columns:
                val = id.at[0, col]
                mask = np.isclose(donor_values, val)  # Use donor values here
                donor_values_obj[mask] = col
            donor_values = donor_values_obj
    
    return donor_values
def quantify_cca(y, id_obs, x):
    """
    Factorize a categorical variable y into numeric values via optimal scaling
    using Canonical Correlation Analysis (CCA) with predictors x.

    Parameters
    ----------
    y : array-like, categorical variable with missing values
    id_obs : boolean array-like, mask indicating observed (True) and missing (False) in y
    x : array-like or DataFrame, predictors without missing values corresponding to y

    Returns
    -------
    ynum : numpy.ndarray
        Numeric transformation of y with missing positions as np.nan.
    id : pandas.DataFrame
        DataFrame representing the canonical components for the observed y.

    Notes
    -----
    This method encodes y as one-hot vectors, then applies CCA to find
    numeric representations that maximize correlation with predictors x.
    """
    # Subset y and x based on id_obs
    xd = np.array(x)[id_obs]
    yf = np.array(y)[id_obs]
    encoder = OneHotEncoder(sparse_output=False, drop=None)
    yf = encoder.fit_transform(yf.reshape(-1, 1))

    #canonical correlation analysis to find "correlation"
    cca = CCA(scale=False, n_components=min(xd.shape[1], yf.shape[1]))
    # yf design matrix, xd data
    cca.fit(X=yf, y=xd)
    # yf design matrix, xd data
    xd_c, yf_c = cca.transform(X=yf, y=xd)
    scaler = StandardScaler()
    y_t = scaler.fit_transform(yf_c[:, 1].reshape(-1, 1)).flatten()
    ynum = np.array([np.nan] * len(y), dtype=np.float64)
    ynum[id_obs] = y_t
    id = pd.DataFrame([y_t], columns=y[id_obs].values)
    return ynum, id
def matcherid(d, t, matcher = "NN", k = 10, radius = 3, rng = None):
    """
    Find donor indices matching missing values based on specified matching method.

    Parameters
    ----------
    d : np.array
        Numeric vector of observed values (donor pool).
    t : np.array
        Numeric vector of missing values to be matched.
    matcher : str, optional
        Matching method to use:
        - "NN": Randomly selects one from the k nearest neighbors (default).
        - "fixedNN": Randomly selects one donor within a fixed radius.
    k : int, optional
        Number of nearest neighbors to consider (only for "NN" matcher).
    radius : float, optional
        Radius threshold for fixedNN matcher (only for "fixedNN" matcher).
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None, a fresh generator is used.

    Returns
    -------
    list of int
        List of indices corresponding to chosen donors in d for each element in t.

    Raises
    ------
    ValueError
        If an unknown matcher method is specified.

    Examples
    --------
    >>> d = np.array([-5, 6, 0, 10, 12])
    >>> t = np.array([-6])
    >>> matcherid(d, t, matcher="NN", k=3)
    [0]
    >>> matcherid(d, t, matcher="fixedNN", radius=5)
    [0]
    """
    if rng is None:
        rng = np.random.default_rng()
    if matcher == "NN": #random from n closest Donors
        idx = []
        tree = KDTree(d.reshape(-1, 1), leaf_size = 40)
        #returns index k NN indices choose 1 on random
        dist, ind = tree.query(t.reshape(-1, 1), k = k)
        for list in ind:
            idx.append(rng.choice(list))
        #returns indices of one random nearest neighbor for each t
        return idx
    elif matcher == "fixedNN": #fixed radius nearest neighbour
        idx = []
        tree = KDTree(d.reshape(-1, 1), leaf_size = 40)
        #returns index k NN indices choose 1 on random
        ind = tree.query_radius(t.reshape(-1, 1), radius)
        for list in ind:
            idx.append(rng.choice(list))
        #returns indices of one random nearest neighbor
        return idx
    else:
        raise ValueError("unknown matcher")