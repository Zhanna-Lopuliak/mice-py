Understanding Missing Data
==========================

Checking for Missing Data
--------------------------

.. code-block:: python

   import pandas as pd
   import numpy as np
   
   # Check for missing values
   print(df.isnull().sum())
   
   # Missing percentages
   missing_pct = df.isnull().mean() * 100
   print(missing_pct)

Visualizing Missing Patterns
-----------------------------

.. code-block:: python

   from plotting.utils import md_pattern_like, plot_missing_data_pattern
   
   # Create pattern summary
   pattern = md_pattern_like(df)
   print(pattern)
   
   # Visualize
   plot_missing_data_pattern(pattern, save_path='missing_pattern.png')

The pattern shows:
   - Which variables have missing values
   - Which combinations occur together
   - How many cases have each pattern

Types of Missing Data
----------------------

**MCAR (Missing Completely at Random)**
   Missingness unrelated to any variables. Complete case analysis is unbiased.

**MAR (Missing at Random)**
   Missingness depends on observed variables. MICE assumes MAR.

**MNAR (Missing Not at Random)**
   Missingness depends on unobserved values. Requires specialized methods.

See :doc:`../theory/missing_data_mechanisms` for technical definitions.

When to Use MICE
----------------

Use MICE when:
   - Multiple variables have missing data
   - Data is likely MAR
   - You want valid statistical inference

MICE assumes MAR. Include predictors of missingness to make this more plausible.

Next Steps
----------

- :doc:`mice_overview` - Learn how MICE works
- :doc:`imputation_methods` - Choose an imputation method
