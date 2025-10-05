"""Pooled results container for `MICE` following Rubin's rules.

Separated into its own module so it can be reused and keeps `MICE.py` lighter.
"""

import numpy as np
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.iolib import summary2


class MICEresult(LikelihoodModelResults):
    """Holds pooled parameter estimates after multiple imputations."""

    def __init__(self, model, params, normalized_cov_params):
        super().__init__(model, params, normalized_cov_params)
        # For simple descriptive pooling we set residual DOF to infinity so
        # t-values effectively become z-values.
        self.df_resid = np.inf

    # noinspection PyMethodMayBeStatic
    def summary(self, title: str = None, alpha: float = 0.05):  # type: ignore[override]
        """Return a statsmodels summary object with an FMI column."""
        smry = summary2.Summary()
        float_fmt = "%8.3f"

        info = {
            "Dependent variable:": getattr(self.model, "endog_names", "-"),
            "Sample size:": "%d" % self.model.data.shape[0] if hasattr(self.model, "data") else "-",
            "Scale": "%.2f" % self.scale,
            "M": "%d" % len(getattr(self.model, "imputed_datasets", [])),
        }
        smry.add_dict(info, align="l", float_format=float_fmt)

        params_df = summary2.summary_params(self, alpha=alpha)
        params_df["FMI"] = getattr(self, "frac_miss_info", np.nan)
        smry.add_df(params_df, float_format=float_fmt)
        smry.add_title(title=title, results=self)

        return smry 