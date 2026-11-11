# mice-py

Multiple imputation by chained equations (MICE) in Python: a port of the R
package [mice](https://cran.r-project.org/package=mice) for pandas data
frames, checked against R mice 3.19.0.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/mice-py/badge/?version=latest)](https://mice-py.readthedocs.io/en/latest/?badge=latest)

## What it does

- **Imputation** with five of R mice's methods: predictive mean matching
  (`pmm`, the default), `midas` (R's `midastouch`), classification and
  regression trees (`cart`), random forests (`rf`) and random draws
  (`sample`). Numeric, boolean and categorical columns.
- **R's model set-up**: full predictor matrix by default, `quickpred`,
  removal of constant and collinear variables (with R's logged events), R's
  dummy coding of categorical predictors.
- **Pooling** with Rubin's rules and Barnard-Rubin degrees of freedom, for
  any statsmodels formula model or for estimates computed elsewhere.
- **Diagnostic plots** after R mice: missing-data pattern, strip, box,
  density and scatter plots of observed and imputed values, trace plots.

## Relation to R mice

mice-py 0.2 was compared with R mice 3.19.0 (R 4.6.2) in four layers: exact
comparisons of the deterministic parts, the distribution of the imputed
values of each method, a simulation study in which both engines impute the
same datasets, and tests of invariants. Every comparison with R agrees (the
random ones within Monte Carlo error) except random forests with categorical
predictors, where scikit-learn and ranger grow different trees. Details:
[Validation against R mice](https://mice-py.readthedocs.io/en/latest/r_mice/validation.html)
and [Differences from R mice](https://mice-py.readthedocs.io/en/latest/r_mice/differences.html).

**Upgrading from 0.1:** version 0.2 uses R's defaults and corrects several
methods, so results change. See
[Upgrading from 0.1 to 0.2](https://mice-py.readthedocs.io/en/latest/upgrading.html)
and the [changelog](CHANGELOG.md).

## Installation

```bash
pip install git+https://github.com/Zhanna-Lopuliak/mice-py.git
```

Python 3.9 to 3.13. This installs two packages, `imputation` and `plotting`.

## Quick start

```python
import numpy as np
import pandas as pd
from imputation import MICE

# Example data: y depends on x1 and x2; 30% of x1 is missing
rng = np.random.default_rng(0)
n = 200
df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
df["y"] = 0.5 + df["x1"] + 0.5 * df["x2"] + rng.normal(size=n)
df.loc[rng.random(n) < 0.3, "x1"] = np.nan

mice = MICE(df)
datasets = mice.impute(n_imputations=5, seed=1)  # pmm, 5 iterations: R's defaults
mice.fit("y ~ x1 + x2")                          # OLS on each completed dataset
print(mice.pool(summ=True))                      # Rubin's rules, Barnard-Rubin df
```

In R this is `imp <- mice(df, m = 5, seed = 1)`,
`fit <- with(imp, lm(y ~ x1 + x2))`, `summary(pool(fit))`; the numbers differ
because the random number generators do. `mice.pool()["table"]` gives R's
full table (`estimate`, `ubar`, `b`, `t`, `df`, `riv`, `lambda`, `fmi`,
intervals) as a data frame.

## Documentation

[mice-py.readthedocs.io](https://mice-py.readthedocs.io/): the
[quick start](https://mice-py.readthedocs.io/en/latest/quickstart.html) with
diagnostics, the
[user guide](https://mice-py.readthedocs.io/en/latest/user_guide/index.html)
(methods and their parameters, predictor matrices, convergence, plots,
pooling, logging) and the
[API reference](https://mice-py.readthedocs.io/en/latest/api/index.html). The
sources are in `docs/source/`; example notebooks are in `examples/`.

## Development

```bash
git clone https://github.com/Zhanna-Lopuliak/mice-py.git
cd mice-py
uv sync          # package, test tools, ruff, mypy, pre-commit
uv run pytest    # all tests, including the comparisons with R
```

See the [contributing guide](https://mice-py.readthedocs.io/en/latest/contributing.html),
[`tests/README.md`](tests/README.md) and
[`simulations/README.md`](simulations/README.md).

## Project structure

```
mice-py/
├── imputation/        # MICE class, imputation methods, design, pooling
├── plotting/          # diagnostic plots and the missing-data pattern
├── tests/             # unit, parity (against R fixtures) and property tests
│   └── reference/     # R scripts and the fixtures they produce
├── simulations/       # simulation study, mice-py against R mice
├── examples/          # Jupyter notebooks and the nhanes data
└── docs/              # Sphinx documentation
```

## Background

mice-py was developed as part of a master's thesis at the Department of
Statistics, Ludwig Maximilian University of Munich. `simulations/legacy/`
holds result files of the thesis's simulation study; the code that produced
them is not in this repository.

## License

MIT; see [LICENSE](LICENSE).

## Authors

- **Zhanna Lopuliak** - zhanna.lopuliak@gmail.com
- **The Anh Vu** - Theanh_v99@yahoo.de
- **Anna-Carolina Haensch** - Anna-Carolina.Haensch@stat.uni-muenchen.de

## Citation

```bibtex
@software{mice-py,
  title = {mice-py: Multiple Imputation by Chained Equations in Python},
  author = {Lopuliak, Zhanna and Vu, The Anh and Haensch, Anna-Carolina},
  year = {2025},
  url = {https://github.com/Zhanna-Lopuliak/mice-py}
}
```

## References

- Van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate
  imputation by chained equations in R. *Journal of Statistical Software*,
  45(3), 1-67.
- Van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.).
  Chapman & Hall/CRC.
- Rubin, D. B. (1987). *Multiple imputation for nonresponse in surveys*. New
  York: John Wiley & Sons.
