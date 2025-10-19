# MICE Examples

This folder contains comprehensive Jupyter notebook examples demonstrating all aspects of the MICE implementation.

## 📚 Notebooks

The notebooks are designed to be explored sequentially, building from basic to advanced usage:

### [01_basic_imputation.ipynb](01_basic_imputation.ipynb)
**Introduction to MICE workflow**

Learn the fundamentals:
- Loading data and inspecting missing values
- Visualizing missing data patterns
- Running basic imputation with default settings
- Accessing and exporting imputed datasets
- Checking convergence with chain statistics

---

### [02_imputation_methods.ipynb](02_imputation_methods.ipynb)
**Comparing imputation methods**

Explore different imputation approaches:
- PMM (Predictive Mean Matching)
- CART (Classification and Regression Trees)
- Random Forest
- Using the same method for all columns vs different methods per column
- Visual comparison of results
- Guidelines for method selection
---

### [03_advanced_parameters.ipynb](03_advanced_parameters.ipynb)
**Advanced parameter tuning**

Master parameter customization:
- Method-specific parameters (e.g., `pmm_donors`, `cart_max_depth`)
- Predictor matrix control (auto-generated vs custom)
- Imputation control (`n_imputations`, `maxit`, `visit_sequence`)
- Initial imputation methods
- Comprehensive example combining multiple parameters


---

### [04_analysis_workflow.ipynb](04_analysis_workflow.ipynb)
**Complete statistical analysis workflow**

Learn the full imputation-to-inference pipeline:
- Imputing missing data
- Fitting statistical models with `.fit(formula)`
- Pooling results using Rubin's rules with `.pool()`
- Interpreting pooled summary tables
---

### [05_diagnostic_plots.ipynb](05_diagnostic_plots.ipynb)
**Visual diagnostics and plotting**

Comprehensive guide to all plotting capabilities:
- Missing data pattern visualization
- Convergence diagnostics
- Distribution comparison plots (stripplot, boxplot, density)
- Relationship plots (scatter plots)
- Customization options and saving plots

---

### Dataset

All examples use the NHANES (National Health and Nutrition Examination Survey) dataset, which is included in the `data/` folder. This is a small, well-documented dataset commonly used in missing data literature.

## 📖 Learning Path

**New to MICE?** Start here:
1. `01_basic_imputation.ipynb` - Learn the basics
2. `05_diagnostic_plots.ipynb` - Understand how to assess imputation quality
3. `02_imputation_methods.ipynb` - Learn about different methods
4. `04_analysis_workflow.ipynb` - Complete analysis from start to finish

**Want to dive deeper?**
- `03_advanced_parameters.ipynb` - Fine-tune your imputations

**Already familiar with MICE?**
- Jump directly to `04_analysis_workflow.ipynb` for the complete workflow