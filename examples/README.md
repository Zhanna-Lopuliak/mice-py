# Examples

Jupyter notebooks for mice-py 0.3. They read their data relative to this folder:

- `data/nhanes.csv`: R mice's `nhanes` data (25 rows: age group, body mass
  index, hypertension, cholesterol), used by every notebook;
- `data/boys.csv`: R mice's `boys` data (748 rows: age, height, weight, body
  mass index, head circumference, genital development, pubic hair, testicular
  volume and region of Dutch boys), written from R mice 3.19.0 with
  `write.csv(mice::boys, row.names = FALSE)`, used by notebook 07.

| Notebook | Content |
|---|---|
| [01_basic_imputation.ipynb](01_basic_imputation.ipynb) | Loading data, the missing-data pattern, a first imputation with the default settings, the completed datasets, the trace plot |
| [02_imputation_methods.ipynb](02_imputation_methods.ipynb) | pmm, cart, rf and midas for all columns, a method per column, comparing the results in plots |
| [03_advanced_parameters.ipynb](03_advanced_parameters.ipynb) | Method parameters (`pmm_donors`, `cart_max_depth`, ...), predictor matrices and `imputation.quickpred`, number of imputations and iterations, visit sequences, initial values, seeds |
| [04_analysis_workflow.ipynb](04_analysis_workflow.ipynb) | Fitting a model to every completed dataset with `fit` and pooling with Rubin's rules with `pool` |
| [05_diagnostic_plots.ipynb](05_diagnostic_plots.ipynb) | Every plot function: pattern, trace, strip, box, density and scatter plots, colors, saving |
| [06_missing_data_patterns.ipynb](06_missing_data_patterns.ipynb) | Missing-data patterns of simulated MCAR, MAR and MNAR data, and of `nhanes` |
| [07_categorical_data.ipynb](07_categorical_data.ipynb) | The default method by column type (`imputation.make_method`), categorical columns imputed with `polr` and `polyreg`, `method` dicts and `default_method`, the model-based methods for numeric columns (`norm`, ..., `mean`), method parameters, the checks before the chains start, fit and pool; with `boys` |

New to MICE: 01, 05, 02, 04, 03, then 07.

## Running

From the repository root:

```bash
uv run --extra notebooks jupyter lab examples/
```

Without uv: `pip install -e ".[notebooks]"`, then `jupyter lab examples/`.

The stored outputs of 01 to 06 were produced with mice-py 0.2.0 (0.3.0 gives
the same numbers), those of 07 with mice-py 0.3.0. To check that every
notebook runs from top to bottom, as the CI does:

```bash
uv run --extra notebooks pytest --nbmake examples/
```

The documentation explains the same topics in more detail:
[mice-py.readthedocs.io](https://mice-py.readthedocs.io/).
