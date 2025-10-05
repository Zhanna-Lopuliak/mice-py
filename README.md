# mice-py

A comprehensive Python implementation of **Multiple Imputation by Chained Equations (MICE)** for handling missing data in statistical analysis and machine learning workflows.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/mice-py/badge/?version=latest)](https://mice-py.readthedocs.io/en/latest/?badge=latest)

## Features

- **Multiple Imputation Methods**: Choose from five robust imputation strategies
  - **PMM** (Predictive Mean Matching) - Maintains distributional properties
  - **CART** (Classification and Regression Trees) - Handles non-linear relationships
  - **Random Forest** - Captures complex interactions
  - **MIDAS** (Multiple Imputation with Distant Average Substitution) - Efficient for small samples
  - **Sample** - Simple random sampling from observed values

- **Flexible Configuration**:
  - Automatic predictor matrix estimation
  - Custom visit sequences for imputation order
  - Method-specific parameter control
  - Mixed data types (numeric and categorical)

- **Statistical Pooling**:
  - Rubin's rules for combining estimates
  - Fraction of missing information (FMI)
  - Confidence intervals and standard errors
  - Formula-based model fitting with statsmodels integration

- **Diagnostic Tools**:
  - Convergence diagnostics (chain statistics)
  - Stripplots, box plots, and density plots
  - Missing data pattern visualization
  - XY plots for bivariate relationships

- **Production-Ready**:
  - Comprehensive input validation
  - Configurable logging system
  - Extensive test suite
  - Well-documented API

## Installation

### From GitHub

```bash
pip install git+https://github.com/Zhanna-Lopuliak/mice-py.git
```

### Development Installation

```bash
git clone https://github.com/Zhanna-Lopuliak/mice-py.git
cd mice-py
pip install -e .
```

### With Optional Dependencies

```bash
# For testing
pip install -e ".[test]"

# For development
pip install -e ".[dev]"

# For documentation
pip install -e ".[docs]"
```

## Quick Start

```python
import pandas as pd
import numpy as np
from imputation import MICE

# Load your data with missing values
df = pd.read_csv("your_data.csv")

# Initialize MICE
mice = MICE(df)

# Perform multiple imputation
mice.impute(
    n_imputations=5,      # Number of imputed datasets to create
    maxit=10,             # Number of MICE iterations
    method='pmm'          # Imputation method (pmm, cart, rf, midas, sample)
)

# Access imputed datasets
imputed_datasets = mice.imputed_datasets

# Fit a model on imputed data
mice.fit('outcome ~ predictor1 + predictor2 + predictor3')

# Pool results using Rubin's rules
pooled_results = mice.pool(summ=True)
print(pooled_results)
```

## Complete Workflow Example

```python
import pandas as pd
import numpy as np
from imputation import MICE, configure_logging
from plotting.utils import md_pattern_like, plot_missing_data_pattern
from plotting.diagnostics import stripplot, densityplot, plot_chain_stats

# Optional: Enable logging to track progress
configure_logging(level='INFO')

# 1. Load data
df = pd.DataFrame({
    'age': [25, 30, np.nan, 45, 50, np.nan, 35, 40],
    'income': [50000, np.nan, 60000, np.nan, 80000, 70000, np.nan, 75000],
    'education': ['Bachelor', 'Master', 'Bachelor', np.nan, 'PhD', 'Master', 'Bachelor', np.nan],
    'employed': [1, 1, 0, 1, 1, np.nan, 1, 0]
})

# 2. Examine missing data patterns
pattern = md_pattern_like(df)
print("Missing Data Pattern:\n", pattern)
plot_missing_data_pattern(pattern, save_path='missing_pattern.png')

# 3. Set up predictor matrix (optional - auto-estimated if not provided)
predictor_matrix = pd.DataFrame(1, index=df.columns, columns=df.columns)
np.fill_diagonal(predictor_matrix.values, 0)

# 4. Perform imputation
mice = MICE(df)
mice.impute(
    n_imputations=5,
    maxit=15,
    predictor_matrix=predictor_matrix,
    method='cart',  # Can also use dict for column-specific methods
    initial='sample'
)

# 5. Check convergence
plot_chain_stats(
    chain_mean=mice.chain_mean,
    chain_var=mice.chain_var,
    save_path='convergence.png'
)

# 6. Visualize imputations
missing_pattern = df.notna().astype(int)
stripplot(mice.imputed_datasets, missing_pattern, save_path='stripplot.png')
densityplot(mice.imputed_datasets, missing_pattern, save_path='density.png')

# 7. Analyze imputed data
mice.fit('income ~ age + education + employed')
results = mice.pool(summ=True)
print("\nPooled Analysis Results:")
print(results)
```

## Using Different Methods

```python
# Use the same method for all variables
mice.impute(n_imputations=5, method='pmm')

# Use different methods for different variables
method_dict = {
    'age': 'pmm',
    'income': 'cart',
    'education': 'sample',
    'employed': 'rf'
}
mice.impute(n_imputations=5, method=method_dict)

# Pass method-specific parameters
mice.impute(
    n_imputations=5,
    method='pmm',
    pmm_donors=5  # PMM-specific parameter
)
```

## Logging Configuration

The package is silent by default. Enable logging when needed:

```python
from imputation import configure_logging

# Basic configuration
configure_logging()

# Detailed debugging
configure_logging(level='DEBUG')

# Custom log directory
configure_logging(log_dir='my_logs', level='INFO')

# Quiet mode (errors only)
configure_logging(console_level='ERROR')
```

See `LOGGING.md` for detailed logging documentation.

## Requirements

- Python ≥ 3.9
- numpy ≥ 1.24
- pandas ≥ 2.0
- scipy ≥ 1.10
- scikit-learn ≥ 1.3
- statsmodels ≥ 0.14
- matplotlib ≥ 3.7
- seaborn ≥ 0.12
- patsy ≥ 1.0
- tqdm ≥ 4.60

## Documentation

Full documentation is available at [Read the Docs](https://mice-py.readthedocs.io/) (powered by Sphinx).

### Documentation includes:
- Complete API reference
- Method descriptions and mathematical details
- Advanced usage examples
- Troubleshooting guide

## Testing

```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests
pytest

# Run with coverage
pytest --cov=imputation --cov=plotting --cov-report=term-missing

# Run specific test categories
pytest tests/test_imputation/     # Core functionality
pytest tests/test_plotting/       # Plotting functions
pytest tests/test_integration.py  # End-to-end workflows
```

See `tests/README.md` for detailed testing documentation.

## Project Structure

```
mice-py/
├── imputation/          # Core MICE implementation
│   ├── MICE.py         # Main MICE class
│   ├── PMM.py          # Predictive Mean Matching
│   ├── cart.py         # CART imputation
│   ├── rf.py           # Random Forest imputation
│   ├── midas.py        # MIDAS imputation
│   ├── sample.py       # Simple sampling
│   ├── pooling.py      # Rubin's rules pooling
│   ├── validators.py   # Input validation
│   └── utils.py        # Helper functions
├── plotting/            # Diagnostic visualization
│   ├── diagnostics.py  # Diagnostic plots
│   └── utils.py        # Missing data pattern plots
├── tests/               # Test suite
├── docs/                # Sphinx documentation source
└── examples/            # Usage examples
```

## Background

This package was developed as part of a master's thesis at Ludwig Maximilian University of Munich, Statistics Department. It provides a modular Python framework for multiple imputation inspired by the R package `mice`, with full flexibility in defining distance metrics, donor selection rules, and imputation parameters.

### Key Findings from Research

Based on extensive simulation studies (675 configurations across continuous, semi-continuous, and discrete target variables):

- **PMM** performs reliably under MCAR and mild MAR, particularly with symmetric distributions and large samples
- **PMM** may struggle under skewed distributions or structured missingness
- **MIDAS** consistently matches or outperforms PMM in coverage and standard error estimation, especially under skewness or small sample sizes
- **CART** and **Random Forest** handle non-linear relationships and interactions effectively
- Method choice should consider data characteristics, missingness patterns, and sample size

## Contributing

Contributions are welcome! Please feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Anna-Carolina Haensch** - Anna-Carolina.Haensch@stat.uni-muenchen.de
- **The Anh Vu** - Theanh_v99@yahoo.de
- **Zhanna Lopuliak** - zhanna.lopuliak@gmail.com

## Citation

If you use this package in your research, please cite:

```bibtex
@software{mice-py,
  title = {mice-py: Multiple Imputation by Chained Equations in Python},
  author = {Haensch, Anna-Carolina and Vu, The Anh and Lopuliak, Zhanna},
  year = {2025},
  url = {https://github.com/Zhanna-Lopuliak/mice-py}
}
```

## References

- Van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate imputation by chained equations in R. *Journal of statistical software*, 45, 1-67.
- Rubin, D. B. (1987). *Multiple imputation for nonresponse in surveys*. New York: John Wiley & Sons.
