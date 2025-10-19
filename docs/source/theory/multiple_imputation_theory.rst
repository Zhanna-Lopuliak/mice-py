Multiple Imputation Theory
==========================

Why Multiple Imputation?
-------------------------

**Problem with complete case analysis**:
   - Wastes data
   - Can introduce bias under MAR/MNAR

**Problem with single imputation**:
   - Treats imputed values as if they were observed
   - Underestimates uncertainty
   - Standard errors too small, p-values too optimistic

**Multiple imputation solution**:
   - Creates multiple plausible versions of complete data
   - Properly accounts for imputation uncertainty
   - Produces valid inference under MAR

The Three-Step Process
-----------------------

1. **Imputation**: Create *m* complete datasets with different imputed values

2. **Analysis**: Analyze each dataset separately

3. **Pooling**: Combine results using Rubin's rules

What is MICE?
-------------

**MICE** (Multiple Imputation by Chained Equations) imputes one variable at a time:

1. **Initialize**: Fill missing values with simple method (mean/sample)

2. **Iterate**: For each incomplete variable:
   
   - Set its imputed values to missing
   - Fit model using other variables as predictors
   - Predict and fill missing values
   - Repeat for all variables

3. **Converge**: Continue iterations until chains stabilize

4. **Repeat**: Create *m* complete datasets

Key Points
----------

**Assumptions**:
   - Data is MAR
   - Imputation models are correctly specified
   - Sufficient iterations for convergence

**Advantages**:
   - Flexible (different methods for different variables)
   - Handles mixed data types
   - No joint distribution required

**Limitations**:
   - Assumes MAR (problems if MNAR)
   - Requires convergence checking
   - Conditional models may not have joint distribution

When MICE Works
---------------

✓ Data is MAR
✓ Clear relationships between variables
✓ Moderate missingness (<30-40%)
✓ Adequate sample size

See :doc:`../user_guide/mice_overview` for usage details.
