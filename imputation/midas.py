import numpy as np
def bootfunc_plain(n):
    """
        Generates bootstrap weights for n observations using simple random sampling with replacement.

        This function simulates a nonparametric bootstrap by randomly drawing `n` integers
        from the range 1 to n (inclusive), with replacement. It returns the count of how many times
        each index (1-based) is selected, producing a frequency table that can be used
        as weights in e.g. MIDAS imputation.

        Parameters
        ----------
        n : int
            The number of observations to sample and also the length of the resulting weight vector.

        Returns
        -------
        weights : ndarray of shape (n,)
            An array of integers indicating how often each index (1-based) was selected in the bootstrap sample.
        """
    #random n int with size n from 1:n
    random = np.random.choice(n, size=n, replace=True)+1
    #returns histogram of drawn ints
    table, _ = np.histogram(random, bins=np.arange(1, n + 2))
    return table
def minmax(x, domin=True, domax=True):
    maxx = np.sqrt(np.finfo(float).max)
    minx = np.sqrt(np.finfo(float).eps)
    if domin:
        x = np.minimum(x, maxx)
    if domax:
        x = np.maximum(x, minx)
    return x
def compute_beta(x, m):
    A = x[:m**2].reshape((m, m))
    b = x[m**2:]
    return np.linalg.solve(A, b)
def midas(y, id_obs, x, id_mis=None, ridge=1e-5, midas_kappa=None, outout=True, **kwargs):
    """
        MIDAS Imputation: Multiple Imputation with Distant Average Substitution.

        This function implements the MIDAS imputation algorithm for continuous variables,
        as introduced by Gaffert et al. (2018).

        It operates by weighting observed donors based on the similarity between predicted values,
        with optional leave-one-out model estimation for increased fidelity.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            The target variable with missing values to be imputed. Must be numeric.

        id_obs : array-like of bool of shape (n_samples,)
            Logical array indicating observed values in `y`. True where `y` is observed, False where missing.

        x : array-like of shape (n_samples, n_features)
            Design matrix of predictor variables. Must be fully observed.

        id_mis : np.ndarray, optional
            Boolean mask of missing values to impute. If None, uses ~id_obs.

        ridge : float, default=1e-5
            Ridge penalty used in regularized regression to stabilize the solution in the presence of multicollinearity.
            - Set lower (e.g. 1e-6) to reduce bias in noisy data.
            - Set higher (e.g. 1e-4) if collinearity is suspected.

        midas_kappa : float or None, default=None
            Controls the sharpness of donor weighting. If None, the optimal value is estimated
            based on R² as described by Siddique and Belin (2008). A common fallback is 3.

        outout : bool, default=True
            If True, uses leave-one-out regression for each donor (slow but MI-proper).
            If False, a single model is estimated for all donors and recipients.
            WARNING: Setting `outout=False` may produce biased estimates and is not fully supported.

        **kwargs : dict
            Additional arguments (not used in this method).

        Returns
        -------
        y_imp : np.ndarray
            Imputed values for missing positions only (matching R implementation).

        Notes
        -----
        - Based on: Gaffert, P., Meinfelder, F., & van den Bosch, V. (2018).
          "Towards an MI-proper Predictive Mean Matching."
        - Related: Siddique, J. & Belin, T. R. (2008). "Multiple Imputation Using an Iterative Hot-Deck with Distance-Based Donor Selection."

        Examples
        --------
        >>> y = np.array([7, np.nan, 9, 10, 11])
        >>> id_obs = ~np.isnan(y)
        >>> x = np.array([[1, 2], [3, 4], [5, 6], [7, 13], [11, 10]])
        >>> midas(y, id_obs, x)
        array([9.0])
        """
    # Validate predictors (x): must be numeric and contain no missing values
    import pandas as pd
    if isinstance(x, pd.DataFrame):
        non_numeric_cols = x.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            raise ValueError(
                f"Predictors must be numeric for midas. Non-numeric predictors found: {non_numeric_cols}"
            )
        missing_cols = x.columns[x.isna().any()].tolist()
        if missing_cols:
            raise ValueError(
                f"Predictors must not contain missing values for midas. Columns with missing values: {missing_cols}"
            )
        x = x.to_numpy()
    else:
        x = np.asarray(x)
        try:
            _ = x.astype(float, copy=False)
        except Exception:
            raise ValueError("Predictors must be numeric for midas. Could not convert 'x' to numeric array.")
        if np.isnan(x).any():
            raise ValueError("Predictors must not contain missing values for midas.")

    # Validate target (y): must be numeric for midas
    y_array = np.asarray(y)
    try:
        y_numeric = y_array.astype(float)
    except Exception:
        raise ValueError("Target y must be numeric for midas. Found non-numeric values.")

    if id_mis is None:
        id_mis = ~id_obs
    #machine epsilon
    sminx = np.finfo(float).eps ** (1 / 4)

    x = np.asarray(x, dtype=float)
    x = np.c_[np.ones(x.shape[0]), x]
    y = y_numeric.astype(float, copy=False)
    nobs = np.sum(id_obs)
    n = len(id_obs)
    m = x.shape[1]
    yobs = y[id_obs]
    xobs = x[id_obs, :]
    xmis = x[id_mis, :]
    #P Step
    omega = bootfunc_plain(nobs)

    CX = omega.reshape(-1, 1) * xobs
    XCX = xobs.T @ CX
##
    if ridge > 0:
        dia = np.diag(XCX)
        dia = dia * np.concatenate(([1], np.repeat(1 + ridge, m - 1)))
        np.fill_diagonal(XCX, dia)

    diag0 = np.where(np.diag(XCX) == 0)[0]
    if len(diag0) > 0:
        XCX[diag0, diag0] = max(sminx, ridge)

    Xy = CX.T @ yobs

    #CX = observed data * bootstrap frequencies
    #XCX = observed data * CX
    beta = np.linalg.solve(XCX, Xy)
    yhat_obs = xobs @ beta

    if midas_kappa is None:
        mean_y = np.dot(yobs, omega) / nobs
        eps = yobs - yhat_obs
        r2 = 1 - (np.dot(omega, eps ** 2) / np.dot(omega, (yobs - mean_y) ** 2))
        #section 5.3.1
        #min function is used correction gets active for r2>.999 only because division by 0
        #if r2 cannot be determined (eg zero variance in yhat), use 3 as suggested by Siddique / Belin
        #if taking delta as in the paper there are numerical errors needing to be fixed
        if r2 < 1:
            midas_kappa = min((50 * r2 / (1 - r2))** (3 / 8),100)
        if np.isnan(midas_kappa):
            midas_kappa = 3

    if outout:
        XXarray_pre = np.array([np.outer(xobs[i], xobs[i]).flatten() * omega[i] for i in range(nobs)]).T
        ridgeind = np.arange(1, m) * (m + 1)
        if ridge > 0:
            XXarray_pre[ridgeind, :] *= (1 + ridge)

        XXarray = XCX.ravel()[:, None] - XXarray_pre
        diag0 = np.where(np.diag(XXarray) == 0)[0]
        if len(diag0) > 0:
            XXarray[diag0, diag0] = max(sminx, ridge)
        Xyarray = Xy.ravel()[:, None] - (xobs * yobs[:, None] * omega[:, None]).T

        ##solve(a = matrix(head(x, m^2), m), b = tail(x, m)) for each column
        stacked_array = np.vstack((XXarray, Xyarray))
        BETAarray = np.apply_along_axis(compute_beta, axis=0, arr=stacked_array, m=m)

        # y
        YHATdon = np.sum(xobs * BETAarray.T, axis=1)
        YHATrec = xmis @ BETAarray

        # distance matrix
        dist_mat = YHATdon - YHATrec
    else:
        yhat_mis = xmis @ beta
        dist_mat = (yhat_obs[:, np.newaxis] - np.tile(yhat_mis, (nobs, 1))).T

    delta_mat = 1 / (np.abs(dist_mat) ** midas_kappa)
    delta_mat = minmax(delta_mat)

    probs = delta_mat * omega

    csums = minmax(np.nansum(probs, axis=1))
    probs /= csums[:, np.newaxis]
    probs = probs.T

    index = np.array([
        np.random.choice(nobs, size=1, replace=False, p=probs[:, j])[0]
        for j in range(probs.shape[1])
    ])

    imputed_values = yobs[index]
    #PLF correction implemented needs to be saved globally over iterations
    #mean(1 / rowSums((t(delta.mat) / csums)^2))
    #consists
    row_sums = np.sum((delta_mat / csums[:, np.newaxis])**2, axis=1)

    #mean(1 / rowSums((t(delta.mat) / csums)^2))
    neff = np.mean(1 / row_sums)
    return imputed_values
