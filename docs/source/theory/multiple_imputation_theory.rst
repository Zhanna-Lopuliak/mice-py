Multiple Imputation Theory
==========================

This page explains why multiple imputation works and how MICE implements it.

The Missing Data Problem
-------------------------

When you have missing data, you face two challenges:

1. **Loss of information**: Fewer observations means less statistical power
2. **Potential bias**: If data aren't MCAR, complete-case analysis can be biased

Traditional approaches have serious limitations:

**Complete Case Analysis (Listwise Deletion)**
   - Throws away incomplete observations
   - Wastes information
   - Can introduce bias under MAR or MNAR

**Single Imputation (Mean, Median, etc.)**
   - Preserves sample size
   - But underestimates uncertainty
   - Distorts relationships between variables
   - SEs are too small, p-values too significant

Why Multiple Imputation?
-------------------------

Multiple imputation solves both problems:

1. **Recovers information** by using observed data to predict missing values
2. **Accounts for uncertainty** by creating multiple different imputations
3. **Produces valid statistical inference** under MAR

The Key Insight
~~~~~~~~~~~~~~~

Since we don't know the true missing values, we acknowledge uncertainty by:

- Creating multiple plausible versions of the complete data
- Analyzing each version separately
- Combining results in a way that properly reflects uncertainty

This is better than pretending we know the missing values (single imputation) or 
throwing away data (complete case analysis).

The Multiple Imputation Process
--------------------------------

Multiple imputation involves three steps:

1. Imputation
~~~~~~~~~~~~~

Create *m* complete datasets, each with different imputed values:

.. code-block:: text

   Original data        →    Dataset 1, Dataset 2, ..., Dataset m
   (with missing)            (all complete, different imputations)

2. Analysis
~~~~~~~~~~~

Analyze each complete dataset separately using standard methods:

.. code-block:: text

   Dataset 1  →  Analysis  →  Estimates₁, SE₁
   Dataset 2  →  Analysis  →  Estimates₂, SE₂
   ...
   Dataset m  →  Analysis  →  Estimatesₘ, SEₘ

3. Pooling
~~~~~~~~~~

Combine results using Rubin's rules to get final estimates:

.. code-block:: text

   Estimates₁...ₘ, SE₁...ₘ  →  Pooling  →  Final estimate, SE, CI

Theoretical Foundations
-----------------------

Bayesian Framework
~~~~~~~~~~~~~~~~~~

Multiple imputation has a Bayesian justification. For each imputation:

1. Draw parameters from their posterior distribution given observed data
2. Draw missing values from their posterior predictive distribution given parameters

This generates imputations that reflect uncertainty about both parameters and 
missing values.

Repeated Imputation
~~~~~~~~~~~~~~~~~~~

By repeating this process *m* times, we get *m* samples from the posterior 
distribution of the missing data.

The variation *between* imputations reflects uncertainty about the missing values.
The variation *within* each imputation reflects sampling variability.

Rubin's rules combine both sources of uncertainty properly.

Why It Works Under MAR
~~~~~~~~~~~~~~~~~~~~~~~

Under MAR, the distribution of missing values can be inferred from:

- The observed values of the same variable
- The observed relationships with other variables

MICE leverages these relationships to create plausible imputations.

What is MICE?
-------------

**MICE** (Multiple Imputation by Chained Equations) is a flexible implementation 
of multiple imputation that:

1. Imputes one variable at a time using the others as predictors
2. Iterates through all variables multiple times (chained equations)
3. Repeats the entire process to create multiple imputed datasets

Why "Chained Equations"?
~~~~~~~~~~~~~~~~~~~~~~~~~

Each incomplete variable gets its own imputation model:

.. math::

   Y_1 &\sim f_1(Y_2, Y_3, ..., Y_p) \\
   Y_2 &\sim f_2(Y_1, Y_3, ..., Y_p) \\
   &\vdots \\
   Y_p &\sim f_p(Y_1, Y_2, ..., Y_{p-1})

These equations are "chained" together—imputing one variable affects the imputation 
of the next.

The MICE Algorithm
------------------

**Initialization** (iteration 0):
   Fill in missing values using simple method (mean, mode, or random sample)

**Iteration** t (repeat for t = 1, 2, ..., maxit):
   For each variable with missing data:
   
   1. Set its imputed values back to missing
   2. Fit a model using complete observations
   3. Predict missing values from the fitted model
   4. Add appropriate random variation
   5. Fill in missing values with predictions

**Multiple Imputations**:
   Repeat entire process m times to get m complete datasets

Convergence
~~~~~~~~~~~

After enough iterations, the imputed values stabilize (converge). The algorithm 
creates a Markov chain that eventually samples from the correct posterior distribution.

Theoretical Properties
----------------------

Proper vs Improper Imputation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Proper imputation**: Incorporates appropriate randomness so that:

- Multiple imputations differ from each other
- Pooled estimates are valid for inference
- Standard errors reflect uncertainty

**Improper imputation**: Deterministic (like mean imputation):

- All imputations would be identical
- Underestimates standard errors
- Invalid inference

MICE uses proper imputation methods that include random variation.

Congeniality
~~~~~~~~~~~~

**Congenial models**: Imputation and analysis models are compatible.

**Uncongenial models**: Imputation and analysis models are incompatible.

In practice:
   - Make imputation model at least as complex as analysis model
   - Include all variables that will be in your analysis
   - Include auxiliary variables that help prediction

MICE doesn't require a joint model for all variables (unlike some methods), making 
it flexible but requiring care about congeniality.

Limitations
-----------

Assumptions
~~~~~~~~~~~

MICE assumes:

1. **MAR**: Missingness depends only on observed data
2. **Model correctness**: Imputation models are correctly specified
3. **Congeniality**: Imputation and analysis models are compatible

If these don't hold:
   - Results may be biased
   - Consider sensitivity analyses
   - Be transparent about assumptions

Theoretical Justification
~~~~~~~~~~~~~~~~~~~~~~~~~~

MICE doesn't have a single joint distribution for all variables (unlike some 
multivariate normal imputation methods). Instead, it specifies conditional 
distributions.

**Concern**: These conditionals may not be compatible with any joint distribution.

**Practical reality**: This rarely causes problems. Simulations show MICE works 
well even when theoretical justification is questionable (Azur et al., 2011).

When MICE Works Well
--------------------

MICE is effective when:

✓ Data is MAR (or MCAR)
✓ Missingness is moderate (<30-40%)
✓ Relationships between variables are clear
✓ Sample size is adequate
✓ Imputation models are well-specified

MICE may struggle when:

✗ Data is MNAR
✗ Missingness is very high (>50%)
✗ Sample size is very small
✗ Variables are weakly related
✗ Many complex interactions exist

Comparison to Other Methods
----------------------------

Joint Modeling (Multivariate Normal)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Approach**: Assume all variables follow multivariate normal distribution

**Pros**: 
   - Strong theoretical foundation
   - Well-understood properties

**Cons**:
   - Restrictive assumption (normality)
   - Doesn't handle mixed data types naturally
   - Can't easily incorporate complex relationships

**MICE advantage**: More flexible, handles mixed data types, allows variable-specific 
models

Maximum Likelihood
~~~~~~~~~~~~~~~~~~

**Approach**: Estimate parameters directly using all available data

**Pros**:
   - Efficient
   - Single analysis (no imputation)
   - Well-understood theory

**Cons**:
   - Limited to specific models
   - Not available for all analyses
   - Harder to implement for complex models

**MICE advantage**: Works with any analysis method, more flexible

Practical Implications
----------------------

How Many Imputations?
~~~~~~~~~~~~~~~~~~~~~

**Theory**: More imputations = more accurate inference

**Old recommendation**: m = 5 sufficient

**Modern recommendation**: m = 20+ (or higher with high missingness)

**Rule of thumb**: m ≈ percentage of incomplete cases

Number of Iterations
~~~~~~~~~~~~~~~~~~~~

**Minimum**: 10 iterations

**Typical**: 20 iterations

**Check**: Convergence diagnostics (trace plots should be flat)

Validity of Inference
~~~~~~~~~~~~~~~~~~~~~

Under MAR with correct models, multiple imputation produces:

- **Unbiased estimates**: On average, same as if data were complete
- **Valid standard errors**: Properly account for uncertainty
- **Correct coverage**: 95% CIs contain true value 95% of the time
- **Appropriate p-values**: Type I error rate controlled

Key Takeaways
-------------

1. **Multiple imputation** > single imputation > complete case analysis
2. **MICE** is a flexible, practical implementation of multiple imputation
3. **Assumes MAR**: Make this plausible by including good predictors
4. **Requires pooling**: Must use Rubin's rules to combine results
5. **Check convergence**: Ensure algorithm has stabilized
6. **Use enough imputations**: m = 20+ recommended

References
----------

Key papers:

- Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*
- Van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate Imputation 
  by Chained Equations in R
- Azur, M. J., et al. (2011). Multiple imputation by chained equations: what is it 
  and how does it work?

See :doc:`../references` for complete bibliography.

See Also
--------

- :doc:`missing_data_mechanisms` for why MAR matters
- :doc:`rubins_rules` for how pooling works
- :doc:`method_details` for specific imputation methods
- :doc:`../user_guide/mice_overview` for practical usage

