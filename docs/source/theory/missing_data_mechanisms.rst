Missing Data Mechanisms
=======================

Three types of missingness (Rubin, 1976):

MCAR: Missing Completely at Random
-----------------------------------

**Definition**: Probability of missingness is the same for all observations.

Mathematically: :math:`P(R | Y_{obs}, Y_{mis}) = P(R)`

**Example**: Data lost due to random computer error.

**For MICE**: Complete case analysis is unbiased under MCAR, but MICE improves efficiency.

MAR: Missing at Random
----------------------

**Definition**: Probability of missingness depends on observed data but not on the missing values themselves.

Mathematically: :math:`P(R | Y_{obs}, Y_{mis}) = P(R | Y_{obs})`

**Example**: Younger people less likely to report income (age observed, income missing).

**For MICE**: This is the key assumption. MICE produces valid results under MAR.

MNAR: Missing Not at Random
----------------------------

**Definition**: Probability of missingness depends on the unobserved (missing) values.

**Example**: People with higher incomes less likely to report income.

**For MICE**: MICE may produce biased results under MNAR. Consider sensitivity analyses.

Practical Implications
----------------------

**Making MAR plausible**: Include variables that:

- Predict missingness
- Correlate with incomplete variables
- Help explain why data is missing

See :doc:`../user_guide/understanding_missing_data` for practical guidance.
