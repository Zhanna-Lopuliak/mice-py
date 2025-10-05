# Testing Guide

This document describes the testing strategy and how to run tests for the MICE package.

## Overview

The testing suite covers two main modules:
- **`imputation/`**: Core MICE functionality, validation, and imputation methods
- **`plotting/`**: Diagnostic plots and visualization utilities

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures and configuration
├── test_integration.py            # End-to-end workflow tests
├── test_imputation/
│   ├── __init__.py
│   ├── test_mice.py              # Core MICE class tests
│   ├── test_validators.py        # Input validation tests
│   └── test_methods.py           # Individual imputation method tests
└── test_plotting/
    ├── __init__.py
    ├── test_diagnostics.py       # Diagnostic plotting tests
    └── test_utils.py             # Plotting utility tests
```

## Running Tests

### Quick Start

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-timeout

# Run all tests
pytest

# Run with coverage
pytest --cov=imputation --cov=plotting --cov-report=term-missing
```

### Using the Test Runner Script

```bash
# Run all tests
python run_tests.py

# Run specific test suites
python run_tests.py --suite imputation
python run_tests.py --suite plotting
python run_tests.py --suite integration

# Run with coverage
python run_tests.py --coverage

# Skip slow tests
python run_tests.py --fast
```

### Individual Test Categories

```bash
# Core MICE functionality
pytest tests/test_imputation/test_mice.py

# Validation functions
pytest tests/test_imputation/test_validators.py

# Imputation methods
pytest tests/test_imputation/test_methods.py

# Plotting functions
pytest tests/test_plotting/

# Integration tests
pytest tests/test_integration.py
```

## Test Categories

### Unit Tests
- **Validation Tests**: Input validation, parameter checking, error handling
- **Method Tests**: Individual imputation algorithms (PMM, CART, RF, Sample, MIDAS)
- **Plotting Tests**: Individual plotting functions and utilities

### Integration Tests
- **End-to-End Workflows**: Complete MICE workflows from data to results
- **Plotting Integration**: MICE results with diagnostic plotting
- **Error Recovery**: Graceful handling of edge cases

## Test Data

Tests use several types of synthetic data:

- **`sample_data_complete`**: Complete dataset without missing values
- **`sample_data_missing`**: Dataset with realistic missing patterns (20% BMI, 15% cholesterol, 10% smoker)
- **`small_data_missing`**: Minimal dataset for quick tests
- **`imputed_datasets`**: Mock imputed datasets for plotting tests
- **`missing_pattern`**: Binary pattern matrices for plotting

## What's Tested

### Imputation Module (`imputation/`)

#### Core MICE Class (`test_mice.py`)
- ✅ Initialization with valid/invalid data
- ✅ Missing value tracking and statistics
- ✅ Basic imputation workflow
- ✅ Different imputation methods
- ✅ Custom predictor matrices
- ✅ Logging integration

#### Validation Functions (`test_validators.py`)
- ✅ DataFrame validation and cleaning
- ✅ Parameter validation (n_imputations, maxit, methods)
- ✅ Predictor matrix validation
- ✅ Formula validation
- ✅ Error handling and edge cases

#### Imputation Methods (`test_methods.py`)
- ✅ Sample imputation (random sampling)
- ✅ PMM (Predictive Mean Matching)
- ✅ CART (Classification and Regression Trees)
- ✅ Random Forest imputation
- ✅ MIDAS (Multiple Imputation with Distant Average Substitution)
- ✅ Method interface consistency
- ✅ Parameter handling

### Plotting Module (`plotting/`)

#### Diagnostic Plots (`test_diagnostics.py`)
- ✅ Stripplots (observed vs imputed values)
- ✅ Box-and-whisker plots
- ✅ Density plots (KDE)
- ✅ Scatter plots (XY plots)
- ✅ Chain statistics plots
- ✅ Plot saving and customization
- ✅ Error handling

#### Utility Functions (`test_utils.py`)
- ✅ Missing data pattern analysis (`md_pattern_like`)
- ✅ Missing data pattern visualization
- ✅ Different data types handling
- ✅ Edge cases and error handling

### Integration Tests (`test_integration.py`)
- ✅ Complete MICE workflows
- ✅ MICE + plotting workflows
- ✅ Multiple method comparisons
- ✅ Logging integration
- ✅ Mixed data types
- ✅ Error recovery
- ✅ Basic performance checks

## Test Configuration

### pytest.ini
- Test discovery patterns
- Output formatting
- Timeout settings (5 minutes)
- Warning filters

### Fixtures (conftest.py)
- Reproducible test data (seed=42)
- Temporary directories for file outputs
- Matplotlib non-interactive backend
- Shared test utilities


## Coverage Goals

Target coverage levels:
- **Core functionality**: >90%
- **Validation functions**: >95%
- **Plotting functions**: >80%
- **Overall**: >85%

## Adding New Tests

When adding new functionality:

1. **Unit tests**: Test individual functions/methods
2. **Integration tests**: Test how components work together
3. **Edge cases**: Test error conditions and boundary cases
4. **Documentation**: Update this guide if needed

### Test Naming Convention

```python
class TestClassName:
    def test_function_basic(self):
        """Test basic functionality."""
        
    def test_function_with_parameters(self):
        """Test with different parameters."""
        
    def test_function_error_handling(self):
        """Test error conditions."""
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure package is installed in development mode
   ```bash
   pip install -e .
   ```

2. **Matplotlib backend errors**: Tests use 'Agg' backend automatically

3. **Random seed issues**: Tests use fixed seeds for reproducibility

4. **Memory issues**: Large datasets are avoided in tests

### Debugging Tests

```bash
# Run single test with verbose output
pytest tests/test_imputation/test_mice.py::TestMICEInitialization::test_init_with_valid_data -v

# Run with Python debugger
pytest --pdb tests/test_imputation/test_mice.py

# Show print statements
pytest -s tests/test_imputation/test_mice.py
```

## Performance Considerations

Tests are designed to be:
- **Fast**: Most tests complete in <1 second
- **Lightweight**: Small datasets to avoid memory issues
- **Reliable**: Fixed random seeds for reproducible results
- **Isolated**: Each test is independent

For performance testing of larger datasets, use the examples in the main package rather than the test suite.
