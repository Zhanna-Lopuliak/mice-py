# Examples

Jupyter notebooks for mice-py 0.2. They use R mice's `nhanes` data (25 rows:
age group, body mass index, hypertension, cholesterol), in `data/nhanes.csv`,
and read it relative to this folder.

| Notebook | Content |
|---|---|
| [01_basic_imputation.ipynb](01_basic_imputation.ipynb) | Loading data, the missing-data pattern, a first imputation with the default settings, the completed datasets, the trace plot |
| [02_imputation_methods.ipynb](02_imputation_methods.ipynb) | pmm, cart, rf and midas for all columns, a method per column, comparing the results in plots |
| [03_advanced_parameters.ipynb](03_advanced_parameters.ipynb) | Method parameters (`pmm_donors`, `cart_max_depth`, ...), predictor matrices and `imputation.quickpred`, number of imputations and iterations, visit sequences, initial values, seeds |
| [04_analysis_workflow.ipynb](04_analysis_workflow.ipynb) | Fitting a model to every completed dataset with `fit` and pooling with Rubin's rules with `pool` |
| [05_diagnostic_plots.ipynb](05_diagnostic_plots.ipynb) | Every plot function: pattern, trace, strip, box, density and scatter plots, colors, saving |
| [06_missing_data_patterns.ipynb](06_missing_data_patterns.ipynb) | Missing-data patterns of simulated MCAR, MAR and MNAR data, and of `nhanes` |

New to MICE: 01, 05, 02, 04, then 03.

## Running

From the repository root:

```bash
uv run --extra notebooks jupyter lab examples/
```

Without uv: `pip install -e ".[notebooks]"`, then `jupyter lab examples/`.

The stored outputs were produced with mice-py 0.2.0. To check that every
notebook runs from top to bottom, as the CI does:

```bash
uv run --extra notebooks pytest --nbmake examples/
```

The documentation explains the same topics in more detail:
[mice-py.readthedocs.io](https://mice-py.readthedocs.io/).
