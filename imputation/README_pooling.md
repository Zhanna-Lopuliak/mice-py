# Standalone Pooling Module

The `pooling.py` module provides standalone functions for pooling descriptive statistics from multiple imputed datasets using Rubin's rules. This module is completely independent of the MICE implementation and can be used with any complete datasets.

## Features

- **Standalone functionality**: Works with any list of complete DataFrames
- **Rubin's rules implementation**: Proper statistical pooling with variance calculations
- **Flexible input sources**: Pool from datasets in memory, files, or other sources
- **Mixed data types**: Supports both numeric and categorical variables
- **Backward compatibility**: MICE.pool() now uses this module internally
- **Rich output**: Detailed results with diagnostic statistics

## Quick Start

### Basic Usage

```python
from imputation.pooling import pool_descriptive_statistics
import pandas as pd

# Assume you have multiple complete imputed datasets
imputed_datasets = [df1, df2, df3, df4, df5]  # List of pd.DataFrame

# Pool descriptive statistics
result = pool_descriptive_statistics(imputed_datasets)

# View summary
print(result.summary())

# Access individual components
print(f"Pooled estimates: {result.estimates}")
print(f"Total variances: {result.variances}")
print(f"Fraction of missing info: {result.frac_miss_info}")
```

### Pool from Files

```python
from imputation.pooling import pool_from_files

# Pool datasets stored in CSV files
file_paths = ['imp1.csv', 'imp2.csv', 'imp3.csv']
result = pool_from_files(file_paths)
```

### Pool Subset of Columns

```python
from imputation.zh.pooling import pool_subset

# Pool only specific columns
result = pool_subset(datasets, columns=['age', 'income', 'education'])
```

### MICE Integration (Backward Compatible)

```python
from imputation.zh.MICE import MICE

mice = MICE(data_with_missing)
mice.impute(n_imputations=5)

# This now uses the standalone pooling module internally
mice.pool()

# Access both interfaces
traditional_result = mice.result          # MICEresult (backward compatible)
standalone_result = mice.pooling_result   # PoolingResult (new interface)
```

## API Reference

### Core Functions

#### `pool_descriptive_statistics(datasets, include_numeric=True, include_categorical=True)`

Pool descriptive statistics across multiple imputed datasets.

**Parameters:**
- `datasets` (List[pd.DataFrame]): List of complete imputed datasets
- `include_numeric` (bool): Include numeric columns in pooling
- `include_categorical` (bool): Include categorical columns in pooling

**Returns:**
- `PoolingResult`: Object containing pooled estimates and diagnostics

#### `pool_from_files(file_paths, read_kwargs=None, **pooling_kwargs)`

Pool datasets stored in files.

**Parameters:**
- `file_paths` (List[str]): List of file paths to imputed datasets
- `read_kwargs` (dict): Keyword arguments for pd.read_csv()
- `**pooling_kwargs`: Additional arguments for pool_descriptive_statistics()

#### `pool_subset(datasets, columns=None, **pooling_kwargs)`

Pool descriptive statistics for a subset of columns.

**Parameters:**
- `datasets` (List[pd.DataFrame]): List of complete imputed datasets
- `columns` (List[str]): Column names to include in pooling
- `**pooling_kwargs`: Additional arguments for pool_descriptive_statistics()

#### `apply_rubins_rules(estimates, variances)`

Apply Rubin's rules to combine estimates and variances.

**Parameters:**
- `estimates` (np.ndarray): Array of parameter estimates (n_imputations × n_parameters)
- `variances` (np.ndarray): Array of within-imputation variances

**Returns:**
- `tuple`: (pooled_estimates, total_variances, within_variance, between_variance, frac_miss_info)

### PoolingResult Class

Container for pooled multiple imputation results.

**Attributes:**
- `estimates`: Pooled parameter estimates
- `variances`: Total variances for each parameter
- `within_variance`: Average within-imputation variance
- `between_variance`: Between-imputation variance
- `frac_miss_info`: Fraction of missing information
- `param_names`: Names of pooled parameters
- `n_imputations`: Number of imputations used
- `sample_size`: Sample size of each dataset

**Methods:**
- `summary()`: Return detailed summary DataFrame
- `__str__()`, `__repr__()`: String representations

## What Gets Pooled

### Numeric Columns
- **Parameter**: Sample mean per column
- **Within-imputation variance**: `Var(mean) = s² / n` (with ddof=1)

### Categorical Columns
- **Parameter**: Per-level proportions for each column
- **Within-imputation variance**: `Var(p) = p(1-p) / n`
- **Parameter names**: Formatted as `column[level]`

## Statistical Details

The module implements Rubin's rules for multiple imputation:

1. **Pooled estimate**: `q̄ = (1/m) Σ qᵢ`
2. **Within-imputation variance**: `ū = (1/m) Σ uᵢ`
3. **Between-imputation variance**: `b = (1/(m-1)) Σ (qᵢ - q̄)²`
4. **Total variance**: `t = ū + (1 + 1/m) × b`
5. **Fraction of missing information**: `λ = ((1 + 1/m) × b) / t`

Where:
- `m` = number of imputations
- `qᵢ` = parameter estimate from imputation i
- `uᵢ` = within-imputation variance from imputation i

## Input Requirements

### Valid Datasets
- All datasets must be pandas DataFrames
- All datasets must have the same shape and column names
- Datasets should be complete (no missing values)
- At least one dataset is required

### Supported Data Types
- **Numeric**: int, float, and other numeric types
- **Categorical**: object, category, bool types

## Error Handling

The module includes comprehensive validation:
- Dataset format and consistency checks
- Missing value warnings
- Column existence validation
- Robust handling of edge cases (single imputation, empty datasets)

## Integration Benefits

### For MICE Users
- Seamless backward compatibility
- Access to both traditional and modern interfaces
- No code changes required for existing workflows

### For Advanced Users
- Direct access to pooling functionality
- Work with datasets from any source
- More flexible parameter control
- Better performance for large-scale pooling

### For Researchers
- Standard implementation of Rubin's rules
- Detailed diagnostic statistics
- Export/import friendly results
- Integration with other analysis pipelines

## Examples

See `example_standalone_pooling.py` for comprehensive usage examples including:
- Basic pooling with mixed data types
- File-based pooling workflow
- Subset pooling for specific analyses
- MICE backward compatibility demonstration
