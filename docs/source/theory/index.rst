Theory & Background
===================

This section provides theoretical background on multiple imputation, MICE, and the 
methods implemented in mice-py.

.. toctree::
   :maxdepth: 2

   missing_data_mechanisms
   multiple_imputation_theory
   method_details
   rubins_rules

Overview
--------

Understanding the theory behind MICE helps you:

- Choose appropriate methods for your data
- Interpret results correctly
- Recognize when assumptions may be violated
- Make informed decisions about imputation parameters

These pages provide concise theoretical background focusing on practical implications 
rather than extensive mathematical derivations.

**Missing Data Mechanisms**
   Understand MCAR, MAR, and MNAR and why they matter.

**Multiple Imputation Theory**
   Learn why multiple imputation is superior to single imputation.

**Method Details**
   Brief overview of how each imputation method works.

**Rubin's Rules**
   Understand how results are pooled and what FMI means.

For Practitioners
-----------------

If you're primarily interested in using MICE effectively, focus on:

1. :doc:`missing_data_mechanisms` - To understand when MICE is appropriate
2. :doc:`multiple_imputation_theory` - To understand why multiple imputation works
3. :doc:`rubins_rules` - To interpret your pooled results

For Researchers
---------------

If you need deeper understanding for methodological research:

- Start with :doc:`multiple_imputation_theory`
- Review :doc:`method_details` for algorithmic details
- Consult the references for mathematical proofs and detailed derivations

Key References
--------------

The theory behind MICE is based on foundational work by:

- **Rubin (1987)**: *Multiple Imputation for Nonresponse in Surveys* - 
  Established theoretical framework
- **Van Buuren & Groothuis-Oudshoorn (2011)**: *mice: Multivariate Imputation 
  by Chained Equations in R* - Original MICE implementation
- **Little & Rubin (2019)**: *Statistical Analysis with Missing Data* - 
  Comprehensive theoretical treatment

See :doc:`../references` for complete bibliography.

