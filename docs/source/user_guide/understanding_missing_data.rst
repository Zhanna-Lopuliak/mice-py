Understanding Missing Data
==========================

Before using MICE, it's important to understand the nature of missing data in your dataset 
and how it affects statistical analysis.

What is Missing Data?
---------------------

Missing data occurs when no value is stored for a variable in an observation. In pandas, 
missing values are typically represented as ``NaN`` (Not a Number) or ``None``.

.. code-block:: python

   import pandas as pd
   import numpy as np
   
   # Example with missing data
   df = pd.DataFrame({
       'age': [25, 30, np.nan, 45],
       'income': [50000, np.nan, 60000, 75000],
       'city': ['NYC', 'LA', np.nan, 'Chicago']
   })
   
   # Check for missing values
   print(df.isnull().sum())

Why Missing Data Matters
-------------------------

Missing data can:

1. **Reduce statistical power** by decreasing the effective sample size
2. **Introduce bias** if data is not missing randomly
3. **Complicate analysis** by making many statistical methods inapplicable
4. **Reduce representativeness** of your sample

Types of Missing Data Mechanisms
---------------------------------

Understanding *why* data is missing is crucial for choosing the appropriate handling method.

MCAR: Missing Completely at Random
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data is MCAR when the probability of missingness is the same for all observations.

**Example**: Survey responses are lost due to a random computer glitch.

**Characteristics**:
   - Missingness is unrelated to any observed or unobserved variables
   - Complete case analysis is unbiased (but inefficient)
   - Easiest case to handle

**Test**: Little's MCAR test can help assess this assumption.

MAR: Missing at Random
~~~~~~~~~~~~~~~~~~~~~~

Data is MAR when the probability of missingness depends on observed variables but not 
on the missing values themselves.

**Example**: Younger people are less likely to report their income, but among people 
of the same age, income values are missing randomly.

**Characteristics**:
   - Missingness can be predicted from other observed variables
   - Multiple imputation is appropriate
   - Most common assumption in practice

MNAR: Missing Not at Random
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data is MNAR when the probability of missingness depends on the unobserved (missing) 
values themselves.

**Example**: People with higher incomes are less likely to report their income.

**Characteristics**:
   - Missingness depends on the value that would have been observed
   - Most difficult case
   - Requires specialized methods or sensitivity analysis

.. note::
   MICE assumes data is MAR. If you suspect MNAR, consider sensitivity analyses or 
   specialized methods beyond standard multiple imputation.

Visualizing Missing Data Patterns
----------------------------------

Use the plotting utilities to understand your missing data:

.. code-block:: python

   from plotting.utils import md_pattern_like, plot_missing_data_pattern
   
   # Create missing data pattern summary
   pattern = md_pattern_like(df)
   print(pattern)
   
   # Visualize the pattern
   plot_missing_data_pattern(pattern, save_path='missing_pattern.png')

The pattern shows:
   - Which combinations of variables have missing values
   - How many cases have each pattern
   - Total missingness per variable

Missing Data Patterns
---------------------

Univariate Pattern
~~~~~~~~~~~~~~~~~~

Only one variable has missing values. This is the simplest pattern.

.. code-block:: python

   # Only 'income' has missing values
   df = pd.DataFrame({
       'age': [25, 30, 35, 45],
       'income': [50000, np.nan, 60000, np.nan]
   })

Monotone Pattern
~~~~~~~~~~~~~~~~

Variables can be ordered such that if variable X is missing, all variables after X 
are also missing.

**Common in**:
   - Longitudinal studies with dropout
   - Surveys with skip patterns

Non-monotone (General) Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Missing values appear in an arbitrary pattern across variables. This is the most 
common and complex case, where MICE is particularly useful.

Strategies for Handling Missing Data
-------------------------------------

Complete Case Analysis
~~~~~~~~~~~~~~~~~~~~~~

Delete all rows with any missing values.

**Pros**: Simple, standard methods apply
**Cons**: Reduces sample size, can introduce bias, wastes information

.. code-block:: python

   df_complete = df.dropna()

Single Imputation
~~~~~~~~~~~~~~~~~

Replace missing values once (e.g., with mean, median, or mode).

**Pros**: Preserves sample size
**Cons**: Underestimates standard errors, ignores imputation uncertainty

.. code-block:: python

   # Mean imputation (NOT recommended)
   df_filled = df.fillna(df.mean())

Multiple Imputation (MICE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Replace missing values multiple times to account for uncertainty.

**Pros**: 
   - Preserves sample size
   - Accounts for imputation uncertainty
   - Produces valid statistical inferences under MAR
   
**Cons**: 
   - More complex
   - Requires more computation
   - Assumes MAR

.. code-block:: python

   from imputation import MICE
   
   mice = MICE(df)
   mice.impute(n_imputations=5, maxit=10)

When to Use MICE
----------------

MICE is appropriate when:

✓ You have missing data in multiple variables
✓ The missing data mechanism is likely MAR
✓ You want valid statistical inferences
✓ You have sufficient observed data to predict missing values
✓ The relationships between variables can be modeled

MICE may not be suitable when:

✗ Data is MNAR (consider sensitivity analysis)
✗ Missingness is >50% in key variables
✗ Sample size is very small
✗ Missingness is systematic and predictable by design

Examining Your Data
-------------------

Before imputation, always:

1. **Check missingness percentages**:

   .. code-block:: python
   
      missing_pct = df.isnull().mean() * 100
      print(missing_pct)

2. **Visualize patterns**:

   .. code-block:: python
   
      from plotting.utils import md_pattern_like
      pattern = md_pattern_like(df)
      print(pattern)

3. **Consider the mechanism**: Think about *why* the data might be missing

4. **Look for patterns**: Are certain groups more likely to have missing data?

Next Steps
----------

Now that you understand missing data:

- Learn how :doc:`mice_overview` works
- Explore different :doc:`imputation_methods`
- Read about :doc:`best_practices` for missing data analysis

