Missing Data Mechanisms
=======================

Understanding why data is missing is crucial for choosing appropriate imputation 
methods and interpreting results.

Three Types of Missingness
---------------------------

Rubin (1976) defined three missing data mechanisms that describe the relationship 
between missingness and data values:

MCAR: Missing Completely at Random
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Definition**: The probability that a value is missing is the same for all observations.

Mathematically: :math:`P(R | Y_{obs}, Y_{mis}) = P(R)`

where :math:`R` is the missingness indicator, :math:`Y_{obs}` are observed values, 
and :math:`Y_{mis}` are missing values.

**In plain language**: Missingness doesn't depend on any data values, observed or unobserved.

**Examples**:
   - Lab samples randomly lost due to equipment failure
   - Survey pages randomly torn from questionnaires
   - Data accidentally deleted by computer error

**Practical implications**:
   - Complete case analysis is unbiased (but inefficient)
   - Any imputation method will work
   - Rarest mechanism in practice

**Test**: Little's MCAR test can statistically test this assumption

MAR: Missing at Random
~~~~~~~~~~~~~~~~~~~~~~

**Definition**: The probability that a value is missing depends on observed data 
but not on the missing values themselves.

Mathematically: :math:`P(R | Y_{obs}, Y_{mis}) = P(R | Y_{obs})`

**In plain language**: Missingness can be predicted from other observed variables, 
but not from the values that are actually missing.

**Examples**:
   - Younger people less likely to report income (age is observed, income is missing)
   - Sicker patients more likely to miss follow-up visits (baseline health observed)
   - Men less likely to answer questions about emotions (gender observed)

**Practical implications**:
   - Complete case analysis may be biased
   - Multiple imputation is appropriate
   - Most common assumption for MICE

**Key assumption**: MICE assumes MAR

MNAR: Missing Not at Random
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Definition**: The probability that a value is missing depends on the unobserved 
values themselves.

Mathematically: :math:`P(R | Y_{obs}, Y_{mis})` depends on :math:`Y_{mis}`

**In plain language**: The missingness depends on what the value would have been.

**Examples**:
   - People with higher incomes less likely to report income
   - More depressed individuals less likely to complete depression surveys
   - Severely ill patients more likely to drop out of study

**Practical implications**:
   - Most difficult case
   - Standard MICE may produce biased results
   - Requires specialized methods or sensitivity analysis
   - Consider carefully whether data is truly MNAR

Distinguishing Between Mechanisms
----------------------------------

The challenge is that MAR and MNAR often look the same in your data—you can't 
definitively test which one you have.

Clues for MAR vs MNAR
~~~~~~~~~~~~~~~~~~~~~

**Evidence for MAR**:
   - Missingness strongly predicted by observed variables
   - Patterns make sense given what you know about data collection
   - Missingness related to measured characteristics

**Suspicion of MNAR**:
   - Sensitive questions (income, drug use, mental health)
   - Self-reporting of stigmatized conditions
   - Attrition related to treatment effects
   - Dropout related to unobserved severity

**Best practice**: Assume MAR but conduct sensitivity analyses to assess impact if 
data were MNAR.

Implications for Analysis
--------------------------

Complete Case Analysis
~~~~~~~~~~~~~~~~~~~~~~

Only uses observations with complete data.

.. list-table::
   :header-rows: 1

   * - Mechanism
     - Bias?
     - Efficiency
   * - MCAR
     - No
     - Poor (loses data)
   * - MAR
     - Often yes
     - Poor
   * - MNAR
     - Usually yes
     - Poor

Single Imputation
~~~~~~~~~~~~~~~~~

Fill in missing values once (e.g., with means).

.. list-table::
   :header-rows: 1

   * - Mechanism
     - Bias?
     - SE estimation
   * - MCAR
     - Sometimes
     - Too small
   * - MAR
     - Often yes
     - Too small
   * - MNAR
     - Usually yes
     - Too small

**Problem**: Treats imputed values as if they were observed, underestimating uncertainty.

Multiple Imputation (MICE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creates multiple imputations accounting for uncertainty.

.. list-table::
   :header-rows: 1

   * - Mechanism
     - Bias?
     - SE estimation
   * - MCAR
     - No
     - Correct
   * - MAR
     - No (if model correct)
     - Correct
   * - MNAR
     - Possibly yes
     - Uncertain

**Assumption**: MICE assumes MAR

Making MAR More Plausible
--------------------------

You can strengthen the MAR assumption by:

**1. Include predictors of missingness**

.. code-block:: python

   # If older people less likely to report income
   # Include age in imputation model
   mice = MICE(df[['age', 'income', 'other_vars']])

**2. Include variables correlated with incomplete variables**

.. code-block:: python

   # Include education, occupation (correlated with income)
   mice = MICE(df[['age', 'income', 'education', 'occupation']])

**3. Use auxiliary variables**

Variables not in your analysis model but helpful for prediction:

.. code-block:: python

   # Include zip code, even if not in analysis
   # It helps predict income
   mice = MICE(df[['income', 'analysis_vars', 'zip_code']])

**4. Model the missingness mechanism explicitly**

If you know what predicts missingness, include those variables.

Practical Guidelines
--------------------

Assuming MAR
~~~~~~~~~~~~

MICE assumes MAR. This is reasonable when:

✓ You have good predictors of missingness in your data
✓ Missingness patterns make sense given data collection
✓ You include auxiliary variables
✓ The amount of missingness is moderate (<30%)

Be cautious with MAR when:

✗ Missingness is on sensitive variables
✗ No observed variables predict missingness well
✗ Missingness is very high (>50%)
✗ Strong theoretical reasons suggest MNAR

Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~

When you suspect MNAR, conduct sensitivity analyses:

1. **Pattern mixture models**: Analyze complete and incomplete cases separately
2. **Tipping point analysis**: How wrong can MAR be before conclusions change?
3. **Expert elicitation**: Get expert opinions on likely values
4. **Compare methods**: Does conclusion hold with different imputation methods?

.. code-block:: python

   # Compare results with different methods
   methods = ['pmm', 'cart', 'rf', 'midas']
   results = {}
   
   for method in methods:
       mice = MICE(df)
       mice.impute(method=method)
       mice.fit('outcome ~ predictors')
       results[method] = mice.pool(summ=True)
   
   # Check if conclusions consistent across methods

Testing Assumptions
-------------------

Little's MCAR Test
~~~~~~~~~~~~~~~~~~

Tests the null hypothesis that data is MCAR:

- **p < 0.05**: Reject MCAR (data is MAR or MNAR)
- **p > 0.05**: Consistent with MCAR (but doesn't prove it)

.. note::
   Most real data fails this test, suggesting MAR or MNAR. This is why multiple 
   imputation is generally preferred over complete case analysis.

Compare Complete vs Incomplete
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compare characteristics
   complete_cases = df.dropna()
   incomplete_cases = df[df.isnull().any(axis=1)]
   
   print("Complete cases:")
   print(complete_cases.describe())
   
   print("\nIncomplete cases:")
   print(incomplete_cases.describe())

Large systematic differences suggest MAR or MNAR.

Examining Patterns
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from plotting.utils import md_pattern_like
   
   pattern = md_pattern_like(df)
   print(pattern)

Look for:
   - Which variables are missing together
   - Whether patterns suggest a mechanism

Reporting Missing Data Mechanisms
----------------------------------

In your paper, discuss:

1. **Amount and pattern of missingness**
2. **Suspected mechanism** (MAR assumed, reasons why)
3. **Variables included** in imputation to support MAR
4. **Sensitivity analyses** if MNAR suspected
5. **Limitations** regarding assumptions

Example Text
~~~~~~~~~~~~

.. code-block:: text

   Income data were missing for 23% of participants. Missingness was more 
   common among younger participants (p<0.01) and those with less education 
   (p<0.001), suggesting a MAR mechanism. We included age, education, 
   occupation, and zip code in the imputation model to make the MAR assumption 
   more plausible. While we cannot rule out MNAR (e.g., higher earners less 
   willing to report), sensitivity analyses (see Appendix) suggest conclusions 
   are robust to moderate violations of MAR.

Summary
-------

- **MCAR**: Missingness unrelated to any variables (rare)
- **MAR**: Missingness predictable from observed variables (MICE assumption)
- **MNAR**: Missingness related to unobserved values (problematic)

**For MICE**:
   - Assumes MAR
   - Include good predictors of missingness
   - Conduct sensitivity analyses if MNAR suspected
   - Be transparent about assumptions

See Also
--------

- :doc:`multiple_imputation_theory` for how MICE handles MAR data
- :doc:`../user_guide/understanding_missing_data` for practical guidance
- :doc:`../user_guide/best_practices` for implementation recommendations

