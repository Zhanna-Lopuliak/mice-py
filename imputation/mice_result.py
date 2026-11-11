"""Pooled results container for `MICE` following Rubin's rules.

Separated into its own module so it can be reused and keeps `MICE.py` lighter.
"""

import numpy as np
from scipy import stats
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.iolib import summary2


class MICEresult(LikelihoodModelResults):
    """Holds pooled parameter estimates after multiple imputations.

    Inference follows R mice: every parameter has its own Barnard-Rubin
    degrees of freedom (``df``), and standard errors, t statistics, p-values
    and confidence intervals use the t distribution with those degrees of
    freedom.

    Parameters
    ----------
    model : object
        The object that produced the pooled estimates (a `MICE` instance).
    params : array_like
        Pooled estimates.
    normalized_cov_params : array_like
        Pooled covariance divided by ``scale``.
    scale : float, default=1.0
        Scale of the covariance.
    df : array_like, optional
        Barnard-Rubin degrees of freedom of each parameter. Defaults to
        infinity, which gives normal-theory inference.
    dfcom : float, default=np.inf
        Residual degrees of freedom of the complete-data analysis. Also
        stored as ``df_resid``; inference does not use it directly.
    """

    def __init__(self, model, params, normalized_cov_params, scale=1.0, df=None, dfcom=np.inf):
        super().__init__(model, params, normalized_cov_params, scale=scale)
        # Fixed here so that cov_params() does not change with `scale`
        self.cov_params_default = np.asarray(normalized_cov_params, dtype=float) * scale
        self.use_t = True
        if df is None:
            df = np.full(len(np.atleast_1d(params)), np.inf)
        self.df = np.asarray(df, dtype=float)
        self.dfcom = float(dfcom)
        self.df_resid = self.dfcom

    @property
    def pvalues(self):
        """Two-sided p-values from the t distribution with each parameter's df."""
        # R floors the df at 0.001 for the p-value.
        return 2.0 * stats.t.sf(np.abs(self.tvalues), np.maximum(self.df, 0.001))

    def conf_int(self, alpha=0.05, cols=None):
        """Confidence intervals from the t distribution with each parameter's df.

        Returns an array with one row per parameter: lower and upper limit.
        """
        bse = np.asarray(self.bse)
        params = np.asarray(self.params)
        lower = params + stats.t.ppf(alpha / 2.0, self.df) * bse
        upper = params + stats.t.ppf(1.0 - alpha / 2.0, self.df) * bse
        ci = np.column_stack([lower, upper])
        return ci if cols is None else ci[np.asarray(cols)]

    def _no_joint_tests(self, *args, **kwargs):
        raise NotImplementedError(
            "Tests of linear combinations of pooled estimates need their own "
            "pooling rule (for example D1 in R mice), which is not implemented. "
            "Per-parameter tests are in `pvalues`, `conf_int()` and the 'table' "
            "returned by MICE.pool()."
        )

    t_test = f_test = wald_test = _no_joint_tests

    def summary(self, title: str = None, alpha: float = 0.05):  # type: ignore[override]
        """Return a statsmodels summary object with df and FMI columns."""
        smry = summary2.Summary()
        float_fmt = "%8.3f"

        scale = getattr(self, "scale", np.nan)
        info = {
            "Dependent variable:": getattr(self.model, "endog_names", "-"),
            "Sample size:": f"{self.model.data.shape[0]:d}" if hasattr(self.model, "data") else "-",
            "Scale": f"{scale:.2f}" if np.isfinite(scale) else "-",
            "M": f"{getattr(self, 'm', len(getattr(self.model, 'imputed_datasets', []))):d}",
        }
        smry.add_dict(info, align="l", float_format=float_fmt)

        params_df = summary2.summary_params(self, alpha=alpha)
        params_df["df"] = self.df
        params_df["FMI"] = getattr(self, "frac_miss_info", np.nan)
        smry.add_df(params_df, float_format=float_fmt)
        smry.add_title(title=title, results=self)

        return smry
