Examples
========

Comprehensive Jupyter notebook tutorials are available in the `examples/ folder 
<https://github.com/Zhanna-Lopuliak/mice-py/tree/main/examples>`_ of the repository.

Available Notebooks
-------------------

The notebooks are designed to be explored sequentially, building from basic to 
advanced usage:

01. Basic Imputation
~~~~~~~~~~~~~~~~~~~~

`01_basic_imputation.ipynb <https://github.com/Zhanna-Lopuliak/mice-py/blob/main/examples/01_basic_imputation.ipynb>`_

**Introduction to MICE workflow**

Learn the fundamentals:
   - Loading data and inspecting missing values
   - Visualizing missing data patterns
   - Running basic imputation with default settings
   - Accessing and exporting imputed datasets
   - Checking convergence with chain statistics

**Best for**: Complete beginners to MICE

02. Imputation Methods
~~~~~~~~~~~~~~~~~~~~~~

`02_imputation_methods.ipynb <https://github.com/Zhanna-Lopuliak/mice-py/blob/main/examples/02_imputation_methods.ipynb>`_

**Comparing imputation methods**

Explore different imputation approaches:
   - PMM (Predictive Mean Matching)
   - CART (Classification and Regression Trees)
   - Random Forest
   - Using the same method for all columns vs different methods per column
   - Visual comparison of results
   - Guidelines for method selection

**Best for**: Understanding which method to use for your data

03. Advanced Parameters
~~~~~~~~~~~~~~~~~~~~~~~

`03_advanced_parameters.ipynb <https://github.com/Zhanna-Lopuliak/mice-py/blob/main/examples/03_advanced_parameters.ipynb>`_

**Advanced parameter tuning**

Master parameter customization:
   - Method-specific parameters (e.g., ``pmm_donors``, ``cart_max_depth``)
   - Predictor matrix control (auto-generated vs custom)
   - Imputation control (``n_imputations``, ``maxit``, ``visit_sequence``)
   - Initial imputation methods
   - Comprehensive example combining multiple parameters

**Best for**: Fine-tuning imputation for specific needs

04. Analysis Workflow
~~~~~~~~~~~~~~~~~~~~~

`04_analysis_workflow.ipynb <https://github.com/Zhanna-Lopuliak/mice-py/blob/main/examples/04_analysis_workflow.ipynb>`_

**Complete statistical analysis workflow**

Learn the full imputation-to-inference pipeline:
   - Imputing missing data
   - Fitting statistical models with ``.fit(formula)``
   - Pooling results using Rubin's rules with ``.pool()``
   - Interpreting pooled estimates, standard errors, and FMI

**Best for**: Complete real-world analysis workflows

05. Diagnostic Plots
~~~~~~~~~~~~~~~~~~~~

`05_diagnostic_plots.ipynb <https://github.com/Zhanna-Lopuliak/mice-py/blob/main/examples/05_diagnostic_plots.ipynb>`_

**Visual diagnostics and plotting**

Comprehensive guide to all plotting capabilities:
   - Missing data pattern visualization
   - Convergence diagnostics (trace plots)
   - Distribution comparison plots (stripplot, boxplot, density)
   - Relationship plots (scatter plots)
   - Customization options and saving plots

**Best for**: Assessing imputation quality visually

Dataset
-------

All examples use the **NHANES (National Health and Nutrition Examination Survey)** 
dataset, which is included in the ``examples/data/`` folder. This is a small, 
well-documented dataset commonly used in missing data literature, making it ideal 
for learning.

Learning Paths
--------------

New to MICE?
~~~~~~~~~~~~

**Recommended sequence**:

1. :doc:`../quickstart` - Get the basics in 5 minutes
2. `01_basic_imputation.ipynb`_ - Hands-on introduction
3. `05_diagnostic_plots.ipynb`_ - Learn to assess quality
4. `02_imputation_methods.ipynb`_ - Choose the right method
5. `04_analysis_workflow.ipynb`_ - Complete workflow from start to finish

Already Familiar with MICE?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jump directly to:
   - `04_analysis_workflow.ipynb`_ for the complete workflow
   - `03_advanced_parameters.ipynb`_ for fine-tuning
   - :doc:`../api/index` for detailed API documentation

Want to Dive Deeper?
~~~~~~~~~~~~~~~~~~~~

After working through the notebooks:
   - Read :doc:`../theory/index` for theoretical background
   - Explore :doc:`../user_guide/index` for detailed guidance
   - Check :doc:`../user_guide/best_practices` for expert tips

Quick Code Snippets
--------------------

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

   from imputation import MICE
   import pandas as pd
   
   # Load data
   df = pd.read_csv('data.csv')
   
   # Impute
   mice = MICE(df)
   mice.impute(n_imputations=5, maxit=10, method='pmm')
   
   # Analyze
   mice.fit('outcome ~ predictor1 + predictor2')
   results = mice.pool(summ=True)
   print(results)

With Diagnostics
~~~~~~~~~~~~~~~~

.. code-block:: python

   from imputation import MICE, configure_logging
   from plotting.diagnostics import plot_chain_stats, stripplot
   from plotting.utils import md_pattern_like
   
   # Enable logging
   configure_logging(level='INFO')
   
   # Check missing patterns
   pattern = md_pattern_like(df)
   print(pattern)
   
   # Impute
   mice = MICE(df)
   mice.impute(n_imputations=20, maxit=20, method='pmm')
   
   # Check convergence
   plot_chain_stats(mice.chain_mean, mice.chain_var, 
                    save_path='convergence.png')
   
   # Compare distributions
   missing_pattern = df.notna().astype(int)
   stripplot(mice.imputed_datasets, missing_pattern,
             save_path='stripplot.png')
   
   # Analyze and pool
   mice.fit('outcome ~ age + gender + treatment')
   results = mice.pool(summ=True)
   print(results)

Different Methods for Different Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   method_dict = {
       'age': 'pmm',           # Numeric, continuous
       'income': 'midas',      # Numeric, skewed
       'education': 'cart',    # Categorical, ordered
       'city': 'sample',       # Categorical, many levels
       'health_score': 'rf'    # Complex relationships
   }
   
   mice.impute(n_imputations=20, method=method_dict)

Running the Notebooks
---------------------

Local Installation
~~~~~~~~~~~~~~~~~~

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/Zhanna-Lopuliak/mice-py.git
      cd mice-py

2. Install with Jupyter:

   .. code-block:: bash

      pip install -e .
      pip install jupyter

3. Launch Jupyter:

   .. code-block:: bash

      jupyter notebook examples/

4. Open any notebook and run the cells

Online Viewing
~~~~~~~~~~~~~~

You can view the notebooks directly on GitHub without running them:

`View on GitHub <https://github.com/Zhanna-Lopuliak/mice-py/tree/main/examples>`_

Additional Resources
--------------------

After working through these examples:

- **User Guide**: :doc:`../user_guide/index` - Detailed documentation
- **Theory**: :doc:`../theory/index` - Theoretical foundations
- **API Reference**: :doc:`../api/index` - Complete API documentation
- **Best Practices**: :doc:`../user_guide/best_practices` - Expert tips

Getting Help
------------

If you have questions:

1. Check the relevant :doc:`../user_guide/index` section
2. Review the :doc:`../api/index` for detailed function documentation
3. Search existing `GitHub Issues <https://github.com/Zhanna-Lopuliak/mice-py/issues>`_
4. Open a new issue if needed

Contributing Examples
---------------------

Have a useful example or tutorial? We welcome contributions!

See :doc:`../contributing` for guidelines on contributing to the documentation.

