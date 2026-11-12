Examples
========

Jupyter notebooks in the `examples/ folder
<https://github.com/Zhanna-Lopuliak/mice-py/tree/main/examples>`_ of the
repository. They use the ``nhanes`` data of R mice (25 rows, 4 columns), in
``examples/data/nhanes.csv``, and run from top to bottom with mice-py 0.3.
The data are numeric, so the default method is pmm for every column.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Notebook
     - Content
   * - `01_basic_imputation.ipynb
       <https://github.com/Zhanna-Lopuliak/mice-py/blob/main/examples/01_basic_imputation.ipynb>`_
     - Loading data, the missing-data pattern, a first imputation, the
       completed datasets, the trace plot.
   * - `02_imputation_methods.ipynb
       <https://github.com/Zhanna-Lopuliak/mice-py/blob/main/examples/02_imputation_methods.ipynb>`_
     - pmm, cart and rf for all columns, a method per column, comparing the
       results in plots.
   * - `03_advanced_parameters.ipynb
       <https://github.com/Zhanna-Lopuliak/mice-py/blob/main/examples/03_advanced_parameters.ipynb>`_
     - Method parameters, predictor matrices and ``quickpred``, number of
       imputations and iterations, visit sequences, initial values.
   * - `04_analysis_workflow.ipynb
       <https://github.com/Zhanna-Lopuliak/mice-py/blob/main/examples/04_analysis_workflow.ipynb>`_
     - Fitting a model to every completed dataset and pooling with Rubin's
       rules.
   * - `05_diagnostic_plots.ipynb
       <https://github.com/Zhanna-Lopuliak/mice-py/blob/main/examples/05_diagnostic_plots.ipynb>`_
     - Every plot function and its options.
   * - `06_missing_data_patterns.ipynb
       <https://github.com/Zhanna-Lopuliak/mice-py/blob/main/examples/06_missing_data_patterns.ipynb>`_
     - Missing-data patterns of simulated MCAR, MAR and MNAR data, and of
       ``nhanes``.

Running the notebooks
---------------------

From a clone of the repository:

.. code-block:: bash

   uv run --extra notebooks jupyter lab examples/

or, without uv, ``pip install -e ".[notebooks]"`` and ``jupyter lab
examples/``. The notebooks read ``data/nhanes.csv`` relative to their own
folder. To run all of them as tests, as the CI does:

.. code-block:: bash

   uv run --extra notebooks pytest --nbmake examples/
