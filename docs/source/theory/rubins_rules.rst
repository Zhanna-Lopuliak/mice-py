Rubin's Rules
=============

This page explains the mathematical foundations of pooling results from multiple 
imputed datasets.

The Pooling Problem
-------------------

After multiple imputation, you have *m* complete datasets and *m* sets of results:

.. math::

   \hat{\theta}_1, \hat{\theta}_2, ..., \hat{\theta}_m

where :math:`\hat{\theta}_i` is the estimate from the i-th imputed dataset.

How do you combine these into a single estimate with appropriate uncertainty?

Simply averaging would ignore the uncertainty from imputation. Rubin's rules 
provide the correct solution.

Rubin's Rules for Scalar Estimates
-----------------------------------

For a scalar parameter :math:`\theta` (e.g., a regression coefficient):

Pooled Estimate
~~~~~~~~~~~~~~~

The pooled estimate is simply the average:

.. math::

   \bar{\theta} = \frac{1}{m}\sum_{i=1}^{m} \hat{\theta}_i

Within-Imputation Variance
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Average of the squared standard errors:

.. math::

   \bar{U} = \frac{1}{m}\sum_{i=1}^{m} SE_i^2

This represents the average sampling variance within each imputed dataset.

Between-Imputation Variance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Variance of the estimates across imputations:

.. math::

   B = \frac{1}{m-1}\sum_{i=1}^{m} (\hat{\theta}_i - \bar{\theta})^2

This represents the additional variance due to missing data.

Total Variance
~~~~~~~~~~~~~~

The total variance combines both sources plus a correction:

.. math::

   T = \bar{U} + B + \frac{B}{m}

The term :math:`B/m` accounts for the finite number of imputations.

Standard Error
~~~~~~~~~~~~~~

The pooled standard error is:

.. math::

   SE_{pooled} = \sqrt{T}

Degrees of Freedom
------------------

The degrees of freedom for inference are adjusted to account for the finite number 
of imputations:

Old Degrees of Freedom
~~~~~~~~~~~~~~~~~~~~~~

Barnard & Rubin (1999) formula:

.. math::

   df_{old} = (m-1)\left[1 + \frac{1}{m+1}\frac{\bar{U}}{B}\right]^2

This was used historically.

Complete-Data Degrees of Freedom
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you know the complete-data degrees of freedom :math:`df_{com}` (e.g., n - p - 1 
for regression):

.. math::

   df_{obs} = \frac{df_{com} + 1}{df_{com} + 3} \cdot df_{com} \cdot (1 - \gamma)

where

.. math::

   \gamma = \frac{(1 + 1/m)B}{T}

Combined Degrees of Freedom
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   df = \left(\frac{1}{df_{old}} + \frac{1}{df_{obs}}\right)^{-1}

In practice, mice-py uses these adjusted degrees of freedom for inference.

Statistical Inference
---------------------

Confidence Intervals
~~~~~~~~~~~~~~~~~~~~

:math:`(1-\alpha)` confidence interval:

.. math::

   \bar{\theta} \pm t_{df, \alpha/2} \cdot SE_{pooled}

where :math:`t_{df, \alpha/2}` is the critical value from a t-distribution with 
:math:`df` degrees of freedom.

Hypothesis Tests
~~~~~~~~~~~~~~~~

Test statistic:

.. math::

   t = \frac{\bar{\theta}}{SE_{pooled}}

Compare to :math:`t_{df}` distribution for p-value.

Fraction of Missing Information
--------------------------------

FMI quantifies how much uncertainty is due to missing data:

Formula
~~~~~~~

.. math::

   \lambda = \frac{B + B/m}{T} = \frac{(1 + 1/m)B}{\bar{U} + (1 + 1/m)B}

Interpretation
~~~~~~~~~~~~~~

- **FMI = 0**: No missing information (equivalent to complete data)
- **FMI = 0.5**: Half the variance is due to missingness
- **FMI = 1**: All uncertainty from missing data (very rare)

**Typical values**: 0.05 to 0.30

**Rule of thumb**:
   - FMI < 0.1: Low impact of missingness
   - 0.1 ≤ FMI < 0.3: Moderate impact
   - FMI ≥ 0.3: High impact, consider more imputations

Relative Increase in Variance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Related metric:

.. math::

   r = \frac{(1 + 1/m)B}{\bar{U}}

This is the proportional increase in variance due to missing data.

Relationship to number of imputations:

.. math::

   \lambda = \frac{r}{1 + r}

Example Calculation
-------------------

Suppose you have m=5 imputations with results:

.. code-block:: text

   Imputation   Estimate   SE
   1            2.34       0.45
   2            2.51       0.43
   3            2.42       0.46
   4            2.48       0.44
   5            2.39       0.45

Step-by-Step
~~~~~~~~~~~~

1. **Pooled estimate**:

   .. math::

      \bar{\theta} = \frac{2.34 + 2.51 + 2.42 + 2.48 + 2.39}{5} = 2.428

2. **Within-imputation variance**:

   .. math::

      \bar{U} = \frac{0.45^2 + 0.43^2 + 0.46^2 + 0.44^2 + 0.45^2}{5} = 0.1992

3. **Between-imputation variance**:

   .. math::

      B &= \frac{1}{4}[(2.34-2.428)^2 + ... + (2.39-2.428)^2] \\
        &= 0.00397

4. **Total variance**:

   .. math::

      T = 0.1992 + 0.00397 + \frac{0.00397}{5} = 0.2040

5. **Standard error**:

   .. math::

      SE = \sqrt{0.2040} = 0.4517

6. **FMI**:

   .. math::

      \lambda = \frac{(1 + 1/5) \times 0.00397}{0.2040} = 0.0234

**Interpretation**: Only 2.3% of uncertainty is due to missing data.

How Many Imputations?
----------------------

Relationship with FMI
~~~~~~~~~~~~~~~~~~~~~

The efficiency of using *m* imputations (relative to infinite imputations) is:

.. math::

   \text{Efficiency} = \left(1 + \frac{\lambda}{m}\right)^{-1}

For 95% efficiency:

.. math::

   m \approx 20\lambda

**Examples**:
   - If FMI = 0.1, need m ≈ 2 (so m=5 is plenty)
   - If FMI = 0.3, need m ≈ 6 (so m=10 is good)
   - If FMI = 0.5, need m ≈ 10 (so m=20 is better)

Modern Recommendations
~~~~~~~~~~~~~~~~~~~~~~

Historical recommendation was m=5, but modern advice:

- **Minimum**: m = 20
- **High missingness**: m = 50-100
- **Rule of thumb**: m ≈ percentage of incomplete cases

Vectorized Rubin's Rules
-------------------------

For multivariate estimates (e.g., all regression coefficients), Rubin's rules 
apply element-wise:

.. math::

   \bar{\boldsymbol{\theta}} &= \frac{1}{m}\sum_{i=1}^{m} \hat{\boldsymbol{\theta}}_i \\
   \bar{\mathbf{U}} &= \frac{1}{m}\sum_{i=1}^{m} \mathbf{U}_i \\
   \mathbf{B} &= \frac{1}{m-1}\sum_{i=1}^{m} (\hat{\boldsymbol{\theta}}_i - \bar{\boldsymbol{\theta}})(\hat{\boldsymbol{\theta}}_i - \bar{\boldsymbol{\theta}})^T \\
   \mathbf{T} &= \bar{\mathbf{U}} + \mathbf{B} + \frac{\mathbf{B}}{m}

where quantities are now matrices.

Why Rubin's Rules Work
----------------------

Theoretical Justification
~~~~~~~~~~~~~~~~~~~~~~~~~~

Under MAR and correct imputation models, Rubin's rules produce:

1. **Unbiased estimates**: :math:`E[\bar{\theta}] = \theta`
2. **Valid standard errors**: Account for both sampling and imputation uncertainty
3. **Correct coverage**: 95% CIs contain true value approximately 95% of the time
4. **Proper p-values**: Type I error controlled at nominal level

The key insight is that :math:`B` captures the uncertainty about the missing data, 
while :math:`\bar{U}` captures sampling uncertainty.

Assumptions
~~~~~~~~~~~

Rubin's rules are valid when:

✓ Imputation model is correct (or approximately so)
✓ Data is MAR
✓ Analysis model is compatible with imputation model
✓ Sufficient imputations (m) are used

Common Mistakes
---------------

Mistake 1: Using Single Imputation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Don't just pick one imputed dataset:

❌ :math:`SE = SE_1` (too small, ignores between-imputation variance)

✓ :math:`SE = \sqrt{\bar{U} + B + B/m}` (correct)

Mistake 2: Averaging Standard Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Don't average the SEs:

❌ :math:`SE = \frac{1}{m}\sum SE_i` (wrong)

✓ Average the *variances*: :math:`\bar{U} = \frac{1}{m}\sum SE_i^2`

Mistake 3: Ignoring Between-Imputation Variance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Don't forget :math:`B`:

❌ :math:`SE = \sqrt{\bar{U}}` (ignores imputation uncertainty)

✓ :math:`SE = \sqrt{\bar{U} + B + B/m}` (correct)

Implementation in mice-py
--------------------------

The ``pool()`` method implements Rubin's rules:

.. code-block:: python

   mice.fit('outcome ~ predictor')
   pooled = mice.pool(summ=True)

Output includes:

- **Estimate**: :math:`\bar{\theta}`
- **Std.Error**: :math:`SE_{pooled}`
- **t-statistic**: :math:`\bar{\theta}/SE_{pooled}`
- **df**: Adjusted degrees of freedom
- **p-value**: From t-distribution
- **CI**: Confidence interval
- **FMI**: Fraction of missing information

See :doc:`../user_guide/pooling_analysis` for practical usage.

Advanced Topics
---------------

Combining Other Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~

Rubin's rules work for any scalar or vector quantity. For other statistics (e.g., 
R², χ² tests), special methods may be needed.

D1 and D2 Statistics
~~~~~~~~~~~~~~~~~~~~

For testing multiple parameters simultaneously (e.g., overall model fit):

- **D1**: Test statistic accounting for within-imputation uncertainty
- **D2**: Test statistic accounting for between-imputation uncertainty
- **D3**: Combined test (Li, Meng, Raghunathan & Rubin, 1991)

These are more complex and beyond basic usage.

Summary
-------

**Key formulas**:

.. math::

   \bar{\theta} &= \frac{1}{m}\sum \hat{\theta}_i \\
   T &= \bar{U} + B + \frac{B}{m} \\
   SE &= \sqrt{T} \\
   FMI &= \frac{(1+1/m)B}{T}

**Usage**: 
   - Always pool using Rubin's rules
   - Check FMI to assess impact of missingness
   - Use adequate number of imputations (m ≥ 20)

References
----------

- Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*
- Barnard, J., & Rubin, D. B. (1999). Small-sample degrees of freedom with 
  multiple imputation
- See :doc:`../references` for complete bibliography

See Also
--------

- :doc:`multiple_imputation_theory` for why pooling is necessary
- :doc:`../user_guide/pooling_analysis` for practical implementation
- :doc:`../user_guide/best_practices` for general guidance

