import numpy as np
from scipy.stats import chi2
def sym(x):
    """
        Ensures the input square matrix is symmetric by averaging it with its transpose.

        Parameters
        ----------
        x : np.ndarray
            A square numpy matrix.

        Returns
        -------
        np.ndarray
            A symmetric matrix computed as (x + x.T) / 2.

        """
    return (x + x.T) / 2
def norm_draw(y, ry, x, rank_adjust=True, **kwargs):
    """
        Bayesian linear regression draw of regression coefficients and residual variance,
        based on the least squares parameters from `estimice()`.

        This function replicates the `mice.impute.norm.draw()` algorithm from the R mice package,
        as described in Rubin (1987, p. 167).

        Parameters
        ----------
        y : np.ndarray
            Numeric vector of length n, containing the variable to be imputed.
        ry : np.ndarray of bool
            Boolean mask vector of length n, where True indicates observed values of `y` and False indicates missing values.
        x : np.ndarray
            Numeric design matrix of shape (n, p) with predictors for `y`. Must have no missing values.
        rank_adjust : bool, optional
            If True, replaces any NaN coefficients with zeros. This is relevant only when
            the least squares method is "qr" and the predictor matrix is rank-deficient.
            Default is True.
        **kwargs : dict
            Additional keyword arguments passed to `estimice()`, e.g., `ls_meth` to specify the least squares method.

        Returns
        -------
        dict
            Dictionary containing:
            - 'coef': Least squares coefficient estimates (numpy array).
            - 'beta': Bayesian drawn regression coefficients (numpy array).
            - 'sigma': Drawn residual standard deviation (float).
            - 'estimation': Least squares method used (string).

        Notes
        -----
        The residual variance sigma is drawn from a scaled chi-square distribution, and
        the regression coefficients beta are drawn from a multivariate normal centered
        at the least squares estimates with variance scaled by sigma.

        References
        ----------
        Rubin, D. B. (1987). Multiple Imputation for Nonresponse in Surveys.
        Wiley, p. 167.

        Examples
        --------
        >>> import numpy as np
        >>> y = np.array([1.0, 2.0, np.nan, 4.0])
        >>> ry = ~np.isnan(y)
        >>> x = np.array([[1, 0], [1, 1], [1, 2], [1, 3]])
        >>> result = norm_draw(y, ry, x, ls_meth='qr')
        >>> print(result['beta'])
        """
    #Draw from estimice
    p = estimice(x[ry, :], y[ry], **kwargs)
    #sqrt(sum((p$r)^2) / rchisq(n = 1,df = p$df)) #one random variate with p$df normal noise
    #sqrt because we need sigma not sigma^2 for beta later
    sigma_star = np.sqrt(np.sum(p["r"] ** 2) / chi2.rvs(df = p["df"], size = 1))
    # #cholesky needs matrix to be symmetrical, must be positive definite -> use sym() (A+A.T)/2
    # #np.linalg.cholesky returns lower triangular matrix
    chol = np.linalg.cholesky(sym(p["v"]))
    # #coef + lower triangular matrix from Cholesky Decomposition * random n draws from standard normal * sigma
    beta_star = p["c"] + (chol.T @ np.random.normal(size=x.shape[1])) * sigma_star
    #return list
    parm = {
        "coef": p["c"],
        "beta": beta_star,
        "sigma": sigma_star,
        "estimation": p["ls_meth"]
    }
    #Replaces NaN with 0 if rank_adjust = True
    if np.any(np.isnan(parm["coef"])) and rank_adjust:
        parm["coef"] = np.nan_to_num(parm["coef"], nan=0.0)
        parm["beta"] = np.nan_to_num(parm["beta"], nan=0.0)
    return parm
def estimice(x, y, ls_meth="qr", ridge=1e-5):
    """
        Computes least squares estimates, residuals, variance-covariance matrix,
        and degrees of freedom using different methods: ridge regression, QR decomposition, or Singular Value Decomposition.

        Parameters
        ----------
        x : np.ndarray
            Numeric design matrix with shape (n_samples, n_predictors). Must not contain missing values.
        y : np.ndarray
            Numeric vector of responses to be imputed, with possible missing values.
        ls_meth : str, optional
            Least squares method to use. Options are:
            - "qr": QR decomposition (default)
            - "ridge": Ridge regression
            - "svd": Singular Value Decomposition
        ridge : float, optional
            Ridge penalty size for ridge regression. Default is 1e-5, balancing stability and bias.

        Returns
        -------
        dict
            Dictionary containing:
            - 'c': Least squares coefficient estimates (numpy array).
            - 'r': Residuals (numpy array).
            - 'v': Variance-covariance matrix of coefficients (numpy array).
            - 'df': Degrees of freedom (int).
            - 'ls_meth': Method used (str).

        Examples
        --------
        >>> import numpy as np
        >>> x = np.array([[1, 2], [3, 4], [5, 6]])
        >>> y = np.array([7, np.nan, 9])
        >>> # Assuming you handle missing y externally, e.g. ry = ~np.isnan(y)
        >>> result = estimice(x[~np.isnan(y)], y[~np.isnan(y)], ls_meth="qr")
        >>> print(result['c'])
        [-6.   6.5]
        """

    #degrees of freedom = length of y - number of columns of x, mininmum df = 1
    #c = coefficients, f = fitted values, r = residuals
    df = max(len(y) - x.shape[1], 1)
    #QR Decomposition
    if ls_meth == "qr":
        #QR decomposition
        qr = np.linalg.qr(x)
        c = np.linalg.solve(qr.R, (qr.Q).T @ y)
        f = x @ c
        r = y - f
        rr = (qr.R).T @ qr.R
        v = np.linalg.solve(rr, np.eye(rr.shape[1]))

        return {
            "c": c.flatten(),  # transpose to match shape
            "r": r.flatten(),
            "v": v,
            "df": df,
            "ls_meth": ls_meth
        }
    #Ridge Regression
    elif ls_meth == "ridge":
        xx = x.T @ x
        pen = ridge * np.eye(xx.shape[0]) * xx
        v = np.linalg.solve(xx + pen, np.eye(xx.shape[1]))
        c = y.T @ x @ v
        r = y - x @ c.T
        return {
            "c": c.flatten(),
            "r": r.flatten(),
            "v": v,
            "df": df,
            "ls_meth": ls_meth
        }
    #Singular Value Decomposition
    elif ls_meth == "svd":
        svd = np.linalg.svd(x, full_matrices=False)
        c = svd.Vh @ (((svd.U).T @ y) / svd.S)
        f = x @ c
        r = f - y
        v = np.linalg.solve((svd.Vh).T * svd.S ** 2 @ svd.Vh, (np.eye((svd.S).shape[0])))
        return {
            "c": c.flatten(),
            "r": r.flatten(),
            "v": v,
            "df": df,
            "ls_meth": ls_meth
        }

