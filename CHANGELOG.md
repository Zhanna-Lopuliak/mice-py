# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.1.0] - 2025-05-10

### Added
- Core MICE implementation with five imputation methods:
  - PMM (Predictive Mean Matching)
  - CART (Classification and Regression Trees)
  - Random Forest
  - MIDAS (Multiple Imputation with Distant Average Substitution)
  - Sample (Random sampling from observed values)
- Automatic predictor matrix estimation using quickpred algorithm
- Rubin's rules for pooling multiple imputation results
- Formula-based model fitting with statsmodels integration
- Comprehensive diagnostic plotting tools:
  - Stripplots for observed vs imputed values
  - Box plots for distribution comparison
  - Density plots (KDE)
  - XY scatter plots for bivariate relationships
  - Convergence diagnostics for MICE chains
- Missing data pattern visualization
- Configurable logging system following Python best practices
- Comprehensive input validation for all parameters
- Extensive test suite with unit and integration tests
- Sphinx documentation with ReadTheDocs support
- Jupyter notebook examples for plotting and diagnostics

### Features
- Support for mixed data types (numeric and categorical)
- Flexible visit sequences (monotone, random, custom)
- Method-specific parameter passing
- Chain statistics tracking (mean and variance)
- Fraction of missing information (FMI) calculation
- Multiple initial imputation strategies
- Customizable predictor matrices
- Output directory management for plots
- Progress tracking with tqdm integration

### Documentation
- Detailed API reference
- Method-specific documentation
- Logging configuration guide
- Testing guide with coverage goals
- Pooling module documentation
- Multiple usage examples

### Testing
- Unit tests for all imputation methods
- Validation tests for input parameters
- Integration tests for complete workflows
- Plotting functionality tests
- Test coverage tracking with pytest-cov

---

## Version History Notes

### Pre-release Development
This package was developed as part of a master's thesis at Ludwig Maximilian University of Munich, Statistics Department.

### Key Research Findings
- PMM performs reliably under MCAR and mild MAR conditions
- MIDAS provides robust performance under skewed distributions and small samples
- CART and Random Forest effectively handle non-linear relationships

---

## Future Roadmap

### Planned Features
- [ ] Additional imputation methods (Bayesian approaches)
- [ ] Parallel processing for multiple imputations
- [ ] Advanced diagnostics (convergence tests)
- [ ] Export/import of imputation results
- [ ] GPU acceleration for large datasets

## Links

- **Documentation**: [Read the Docs](https://mice-py.readthedocs.io/)
- **Repository**: [GitHub](https://github.com/Zhanna-Lopuliak/mice-py)

---

[Unreleased]: https://github.com/[username]/mice-py/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/[username]/mice-py/releases/tag/v0.1.0
