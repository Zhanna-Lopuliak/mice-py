Changelog
=========

For a detailed changelog, see `CHANGELOG.md <https://github.com/Zhanna-Lopuliak/mice-py/blob/main/CHANGELOG.md>`_ in the repository.

Version 0.1.0
-------------

**Initial Release**

Core Features
~~~~~~~~~~~~~

- Complete MICE implementation with convergence tracking
- Five imputation methods: PMM, CART, Random Forest, MIDAS, Sample
- Rubin's rules pooling with fraction of missing information (FMI)
- Formula-based model fitting and analysis
- Comprehensive input validation
- Professional logging system

Imputation Methods
~~~~~~~~~~~~~~~~~~

- **PMM**: Predictive Mean Matching with Bayesian bootstrap
- **CART**: Classification and Regression Trees
- **Random Forest**: Ensemble method with configurable parameters
- **MIDAS**: Distance-aided substitution for small samples
- **Sample**: Simple random sampling

Configuration Options
~~~~~~~~~~~~~~~~~~~~~

- Customizable predictor matrices
- Multiple visit sequence strategies
- Method-specific parameter tuning
- Initial imputation methods
- Flexible method assignment per variable

Diagnostic Tools
~~~~~~~~~~~~~~~~

- Convergence diagnostics (chain statistics)
- Stripplots for observed vs imputed comparison
- Density plots for distribution comparison
- Box plots for distribution visualization
- Missing data pattern visualization
- XY plots for bivariate relationships

Statistical Analysis
~~~~~~~~~~~~~~~~~~~~

- Formula-based model specification (Patsy syntax)
- Automatic pooling using Rubin's rules
- Fraction of Missing Information (FMI) calculation
- Confidence intervals and p-values
- Degrees of freedom adjustment

Documentation
~~~~~~~~~~~~~

- Comprehensive Sphinx documentation
- User guide with detailed explanations
- Theory section with mathematical background
- API reference for all modules
- Jupyter notebook examples
- Best practices guide

Testing
~~~~~~~

- Extensive test suite with pytest
- Unit tests for all core functions
- Integration tests for workflows
- Coverage tracking

Development
~~~~~~~~~~~

- MIT License
- GitHub repository with CI/CD
- ReadTheDocs integration
- Development, testing, and documentation dependencies

Contributors
~~~~~~~~~~~~

- Anna-Carolina Haensch
- The Anh Vu  
- Zhanna Lopuliak

Future Plans
------------

Potential future enhancements (not yet implemented):

- Additional imputation methods (e.g., lasso, ridge)
- Parallel processing for large datasets
- GPU acceleration for random forest
- More sophisticated predictor matrix algorithms
- Additional diagnostic plots
- Integration with scikit-learn pipelines
- Categorical variable handling improvements
- Time series imputation methods

Stay tuned for updates!

Reporting Issues
----------------

Found a bug or have a feature request? 

Open an issue on `GitHub <https://github.com/Zhanna-Lopuliak/mice-py/issues>`_.

