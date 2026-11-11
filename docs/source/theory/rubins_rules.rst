Rubin's Rules
=============

How the results of *m* analyses of completed datasets are combined. These are
the formulas of R's ``pool()``, and those of ``MICE.pool`` and
``imputation.pooling.pool_estimates``.

Estimate and variances
----------------------

For a scalar parameter with estimates :math:`\hat\theta_1, \dots, \hat\theta_m`
and squared standard errors :math:`U_1, \dots, U_m`:

**Pooled estimate**

.. math::

   \bar\theta = \frac{1}{m}\sum_{i=1}^{m} \hat\theta_i

**Within-imputation variance**

.. math::

   \bar U = \frac{1}{m}\sum_{i=1}^{m} U_i

**Between-imputation variance**

.. math::

   B = \frac{1}{m-1}\sum_{i=1}^{m} (\hat\theta_i - \bar\theta)^2

**Total variance**

.. math::

   T = \bar U + \left(1 + \frac{1}{m}\right) B

The standard error of :math:`\bar\theta` is :math:`\sqrt{T}`.

Degrees of freedom
------------------

With

.. math::

   \lambda = \frac{(1 + 1/m) B}{T}, \qquad
   \nu_\text{old} = \frac{m - 1}{\lambda^2},

Rubin's large-sample degrees of freedom are :math:`\nu_\text{old}`. Barnard
and Rubin (1999) adjust them for the complete-data degrees of freedom
:math:`\nu_\text{com}` (``dfcom``, for example :math:`n - p` in a linear
regression):

.. math::

   \nu_\text{obs} = \frac{\nu_\text{com} + 1}{\nu_\text{com} + 3}\,
                    \nu_\text{com}\,(1 - \lambda), \qquad
   \nu = \frac{\nu_\text{old}\,\nu_\text{obs}}{\nu_\text{old} + \nu_\text{obs}}.

When :math:`\nu_\text{com}` is infinite, :math:`\nu = \nu_\text{old}`. The
interval is :math:`\bar\theta \pm t_{\nu,\,0.975} \sqrt{T}` and the p-value
comes from the t distribution with :math:`\nu` degrees of freedom.

Fraction of missing information
-------------------------------

Two related quantities describe how much of the uncertainty is due to the
missing data:

.. math::

   r = \frac{(1 + 1/m) B}{\bar U}, \qquad
   \lambda = \frac{(1 + 1/m) B}{T}, \qquad
   \gamma = \frac{r + 2/(\nu + 3)}{r + 1}.

:math:`r` is the relative increase in variance (``riv``), :math:`\lambda`
the proportion of the total variance due to missingness (``lambda``) and
:math:`\gamma` the fraction of missing information (``fmi``). mice-py 0.1
reported :math:`\lambda` under the name FMI; since 0.2 ``fmi`` is
:math:`\gamma`, as in R.

With *m* = 1 nothing can be pooled: *B* cannot be estimated, and the single
analysis is reported with :math:`\nu_\text{com}` degrees of freedom.

In mice-py
----------

``MICE.pool()["table"]`` has one row per term with the columns ``estimate``,
``ubar``, ``b``, ``t``, ``dfcom``, ``df``, ``riv``, ``lambda``, ``fmi``,
``std_error``, ``statistic``, ``p_value``, ``conf_low`` and ``conf_high``;
see :doc:`../user_guide/pooling_analysis`.
