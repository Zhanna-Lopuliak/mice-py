# Plotting

This folder contains all plotting and visualization functions for the MICE (Multiple Imputation by Chained Equations) implementation. It combines diagnostic plots for analyzing imputed datasets and utilities for visualizing missing data patterns.

## Contents

### `diagnostics.py`
Contains diagnostic visualization functions for analyzing imputed datasets:

- `stripplot()`: Creates stripplots showing observed and imputed values
- `bwplot()`: Creates box-and-whisker plots for observed and imputed values  
- `densityplot()`: Creates density plots (KDE) for observed and imputed values
- `densityplot_split()`: Creates separate density plots for one column for observed data and each imputation
- `xyplot()`: Creates scatter plots of two columns, showing observed and imputed values
- `plot_chain_stats()`: Plots convergence diagnostics for MICE chains (mean and variance)

All diagnostic functions support:
- Custom colors for observed and imputed values
- Option to merge imputations into a single plot or show separate plots
- Automatic handling of missing values
- Optional saving to file

### `utils.py`
Contains utility functions for missing data pattern analysis and visualization:

- `md_pattern_like()`: Replicates R's mice::md.pattern() behavior, showing missing data patterns as 1 (observed) and 0 (missing), with counts per pattern and per column
- `plot_missing_data_pattern()`: Creates a visual representation of missing data patterns

### Example Files

- `diagnostics_example.ipynb`: Jupyter notebook demonstrating diagnostic plotting tools
- `visualization_example.ipynb`: Jupyter notebook demonstrating missing data pattern visualization

## Usage Examples

### Diagnostic Plots

```python
from plotting.diagnostics import stripplot, bwplot, densityplot, xyplot

# Create stripplot for imputed data
stripplot(
    imputed_datasets=[df1, df2, df3],  # List of imputed datasets as pandas DataFrames
    missing_pattern=missing_df,        # DataFrame indicating missing values (0=missing, 1=observed)
    columns=['col1', 'col2'],          # Optional: specific columns to plot
    merge_imputations=False,           # Show separate plots for each imputation
    observed_color='blue',             # Color for observed values
    imputed_color='red'                # Color for imputed values
)

# Create box-and-whisker plots
bwplot(
    imputed_datasets=imputed_datasets,
    missing_pattern=missing_pattern,
    merge_imputations=False
)

# Create density plots
densityplot(
    imputed_datasets=imputed_datasets,
    missing_pattern=missing_pattern
)

# Create scatter plots (XY plots)
xyplot(
    imputed_datasets=imputed_datasets,
    missing_pattern=missing_pattern,
    x="age",      # X-axis variable
    y="bmi"       # Y-axis variable (should contain missing values)
)
```

### Missing Data Pattern Analysis

```python
from plotting.utils import md_pattern_like, plot_missing_data_pattern

# Analyze missing data patterns
pattern_df = md_pattern_like(df)
print("Missing-data pattern:\n", pattern_df)

# Visualize missing data patterns
plot_missing_data_pattern(
    pattern_df, 
    title="Missing-Data Pattern",
    save_path='missing_data_pattern.png'
)
```

### Convergence Diagnostics

```python
from plotting.diagnostics import plot_chain_stats

# Plot convergence of chain statistics
plot_chain_stats(
    chain_mean=mice_imputer.chain_mean,
    chain_var=mice_imputer.chain_var,
    columns=['col1', 'col2'],  # Optional: specific columns
    save_path='convergence_plots.png'
)
```

## Dependencies

- `pandas`: Data manipulation and analysis
- `matplotlib`: Core plotting library
- `seaborn`: Statistical data visualization
- `numpy`: Numerical computations

## Notes

- All plotting functions handle missing values automatically
- Plots can be saved to files by providing a `save_path` parameter
- The diagnostic plots are designed to help assess the quality of MICE imputations
- Missing data pattern functions help understand the structure of missingness in your data 