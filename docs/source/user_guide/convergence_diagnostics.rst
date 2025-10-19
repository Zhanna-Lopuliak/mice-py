Convergence Diagnostics
=======================

After running MICE, it's crucial to check whether the algorithm has converged. This 
guide explains what convergence means and how to assess it.

What is Convergence?
--------------------

**Convergence** means the MICE algorithm has stabilized—the imputed values are no 
longer systematically changing from one iteration to the next.

Why It Matters
~~~~~~~~~~~~~~

If MICE hasn't converged:
   - Imputed values may be unreliable
   - Statistical inferences may be biased
   - More iterations are needed

How MICE Converges
~~~~~~~~~~~~~~~~~~

During each iteration, MICE updates the imputed values for each variable. Initially, 
these updates cause substantial changes, but as the algorithm proceeds, changes should 
become smaller and stabilize.

Chain Statistics
----------------

MICE tracks two key statistics across iterations for each variable:

**Chain Mean**
   The mean of the imputed values at each iteration

**Chain Variance**
   The variance of the imputed values at each iteration

These "chains" should:
   1. Start from initial values
   2. Potentially drift in early iterations
   3. Stabilize after some iterations (convergence!)

Visualizing Convergence
-----------------------

The primary tool for checking convergence is plotting the chain statistics:

.. code-block:: python

   from plotting.diagnostics import plot_chain_stats
   
   # After running MICE
   mice = MICE(df)
   mice.impute(n_imputations=5, maxit=20)
   
   # Plot convergence
   plot_chain_stats(
       chain_mean=mice.chain_mean,
       chain_var=mice.chain_var,
       save_path='convergence.png'
   )

Interpreting the Plots
~~~~~~~~~~~~~~~~~~~~~~~

**What to look for**:

✓ **Stable horizontal lines**: Means and variances have stabilized
✓ **No systematic trends**: Values aren't consistently increasing or decreasing
✓ **Mixing of chains**: If multiple imputations are shown, they should overlap

✗ **Trending lines**: Values still changing systematically
✗ **Unstable oscillations**: Large swings even in later iterations
✗ **Separated chains**: Different imputations have very different patterns

Example: Good Convergence
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Mean Chain Plot:
   ────────────────────
   │ ∼∼∼∼∼∼∼∼∼∼∼∼∼∼∼  ← Stable, flat line
   │
   │
   ├───────────────────
   0   5   10  15  20  iterations

Example: Poor Convergence
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Mean Chain Plot:
   ────────────────────
   │         ╱
   │       ╱
   │     ╱              ← Still trending upward
   │   ╱
   │ ╱
   ├───────────────────
   0   5   10  15  20  iterations

Numerical Assessment
--------------------

While visual inspection is primary, you can also check numerically:

.. code-block:: python

   import numpy as np
   
   # Get chain means for a specific variable
   var_name = 'income'
   chain = mice.chain_mean[var_name]
   
   # Check if last few iterations are stable
   last_5 = chain[-5:]
   variation = np.std(last_5) / np.mean(last_5)  # Coefficient of variation
   
   if variation < 0.01:  # Less than 1% variation
       print(f"{var_name}: Converged")
   else:
       print(f"{var_name}: May need more iterations")

What to Do If Not Converged
----------------------------

Increase Iterations
~~~~~~~~~~~~~~~~~~~

The simplest solution:

.. code-block:: python

   # Try more iterations
   mice.impute(n_imputations=5, maxit=50)  # Increased from 10 to 50
   
   # Check again
   plot_chain_stats(mice.chain_mean, mice.chain_var)

Most convergence issues are resolved by running more iterations.

Adjust Initial Values
~~~~~~~~~~~~~~~~~~~~~

Try different initial imputation:

.. code-block:: python

   # Use mean instead of sample for initialization
   mice.impute(n_imputations=5, maxit=20, initial='mean')

Simplify Predictor Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~

Too many predictors or multicollinearity can slow convergence:

.. code-block:: python

   from imputation.utils import quickpred
   
   # Use automatic selection with higher threshold
   predictor_matrix = quickpred(df, mincor=0.3)
   mice.impute(predictor_matrix=predictor_matrix, maxit=20)

Change Method
~~~~~~~~~~~~~

Some methods converge faster than others:

.. code-block:: python

   # Try a different method
   mice.impute(method='cart', maxit=20)  # Instead of PMM

How Many Iterations?
--------------------

**Default**: 10 iterations
   Sufficient for many datasets

**Recommendation**: 15-20 iterations
   Safer choice, check convergence diagnostics

**Complex data**: 30-50+ iterations
   - High missingness (>30%)
   - Many variables
   - Complex relationships

**Rule of thumb**: Run until chains are flat for at least 5 iterations

Convergence by Variable
------------------------

Different variables may converge at different rates:

.. code-block:: python

   # Check each variable separately
   for var in mice.chain_mean.keys():
       chain = mice.chain_mean[var]
       plt.figure()
       plt.plot(chain)
       plt.title(f'Convergence: {var}')
       plt.xlabel('Iteration')
       plt.ylabel('Mean')
       plt.savefig(f'convergence_{var}.png')
       plt.close()

Variables with more missingness or weaker predictive relationships typically need 
more iterations.

Other Diagnostic Checks
-----------------------

Compare Observed vs Imputed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Even if chains converge, check that imputed values are reasonable:

.. code-block:: python

   from plotting.diagnostics import stripplot, densityplot
   
   missing_pattern = df.notna().astype(int)
   
   # Stripplot: visual check
   stripplot(mice.imputed_datasets, missing_pattern)
   
   # Density plot: distributional check
   densityplot(mice.imputed_datasets, missing_pattern)

Look for:
   - Imputed values (red) within range of observed values (blue)
   - Similar distributions between observed and imputed
   - No impossible values

Check Variability Between Imputations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multiple imputations should differ from each other:

.. code-block:: python

   # For a specific variable
   var = 'income'
   imputed_values = [dataset[var] for dataset in mice.imputed_datasets]
   
   # Check standard deviation across imputations
   sd_across = np.std(imputed_values, axis=0)
   
   print(f"Mean SD across imputations: {sd_across.mean()}")

If imputations are nearly identical, you may need more iterations or a less 
deterministic method.

Common Convergence Issues
--------------------------

Slow Convergence
~~~~~~~~~~~~~~~~

**Symptoms**: Chains still changing after many iterations

**Causes**:
   - High dimensionality
   - Weak predictor relationships
   - High missingness
   - Multicollinearity

**Solutions**:
   - Use quickpred to select predictors
   - Increase ridge parameter in PMM
   - Try different method (CART/RF)
   - More iterations

Non-Convergence
~~~~~~~~~~~~~~~

**Symptoms**: Chains never stabilize, even after 50+ iterations

**Causes**:
   - Perfect multicollinearity
   - Circular dependencies
   - Insufficient data
   - Model misspecification

**Solutions**:
   - Check for perfectly correlated variables
   - Remove redundant predictors
   - Simplify predictor matrix
   - Consider different imputation strategy

Oscillating Chains
~~~~~~~~~~~~~~~~~~

**Symptoms**: Chains oscillate rather than stabilize

**Causes**:
   - Conflicting information from different predictors
   - Overfitting with complex methods

**Solutions**:
   - Use simpler method (PMM instead of RF)
   - Regularize more strongly
   - Reduce predictor complexity

Separated Chains
~~~~~~~~~~~~~~~~

**Symptoms**: Different imputation chains don't mix

**Causes**:
   - Insufficient iterations
   - Bimodal or complex distributions
   - Categorical variables with many levels

**Solutions**:
   - More iterations
   - Check if true multimodality exists
   - Use method appropriate for data type

Documenting Convergence
------------------------

When reporting results, document:

1. **Number of iterations used**
2. **Assessment method** (visual inspection of chain plots)
3. **Conclusion** (e.g., "All variables converged by iteration 15")
4. **Any issues** encountered and how addressed

Example Documentation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   We ran MICE for 20 iterations. Convergence was assessed by visual 
   inspection of trace plots for mean and variance. All variables 
   showed stable chains by iteration 12. Imputed values were within 
   the range of observed values for all variables.

Best Practices
--------------

1. **Always check convergence**: Never skip this step
2. **Visual inspection first**: Plots are more informative than statistics
3. **Be conservative**: If unsure, run more iterations
4. **Check all variables**: Don't just look at your outcome variable
5. **Document your assessment**: Include in your methods section
6. **Look at early iterations**: They can reveal problems with initialization
7. **Compare multiple runs**: Rerun with different seeds to check stability

Quick Convergence Checklist
----------------------------

Before finalizing your imputation:

☐ Chain plots show stable horizontal lines for all variables
☐ No systematic trends in the last 5-10 iterations
☐ Imputed values are in reasonable range
☐ Distributions of observed and imputed values are similar
☐ Multiple imputations show appropriate variability
☐ Convergence achieved with acceptable number of iterations (<50)

If all checked, your imputation is ready for analysis!

Next Steps
----------

- Learn about :doc:`pooling_analysis` to analyze your imputed data
- Review :doc:`best_practices` for overall guidance
- See examples of complete workflows in :doc:`../examples/index`

