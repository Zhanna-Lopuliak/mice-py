API Reference
=============

The reference is generated from the docstrings. mice-py installs two
packages: ``imputation`` (imputation and pooling) and ``plotting``
(diagnostic plots).

.. toctree::
   :maxdepth: 2

   mice
   methods
   design
   pooling
   plotting
   utilities

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Name
     - Purpose
   * - :class:`imputation.MICE`
     - impute, fit and pool (:doc:`mice`)
   * - :func:`imputation.quickpred`
     - predictor matrix by R's ``quickpred`` rule (:doc:`design`)
   * - ``imputation.PMM.pmm``, ``imputation.midas.midas``,
       ``imputation.cart.cart``, ``imputation.rf.rf``,
       ``imputation.sample.sample``
     - the univariate imputation methods (:doc:`methods`)
   * - ``imputation.design``
     - design matrix, removal of dependent and collinear predictors
       (:doc:`design`)
   * - ``imputation.pooling``, ``imputation.mice_result.MICEresult``
     - Rubin's rules, pooled results (:doc:`pooling`)
   * - ``plotting``
     - diagnostic plots (:doc:`plotting`)
   * - ``imputation.configure_logging`` and ``imputation.logging_config``
     - logging helpers (:doc:`utilities`)
   * - ``imputation.sampler``, ``imputation.constants``,
       ``imputation.utils``, ``imputation.validators``
     - building blocks used by the above (:doc:`utilities`)
