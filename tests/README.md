# Tests

The test suite covers the `imputation` and `plotting` packages and compares
them with R mice 3.19.0. It runs without R: the R results are committed as
fixtures in `reference/fixtures`.

## Running

From the repository root, after `uv sync`:

```bash
uv run pytest                                        # everything (about 2,500 tests)
uv run pytest -m parity                              # only the comparisons with R
uv run pytest -m "not slow"                          # without the slow tree tests
uv run pytest tests/test_imputation/test_mice.py     # one file
uv run pytest --cov --cov-report=term-missing        # with coverage (CI requires 89%)
uv run pytest --doctest-modules imputation plotting  # the examples in the docstrings
```

Without uv: `pip install -e ".[test]"`, then `pytest`. Plots use the `Agg`
backend (set in `conftest.py`), so no window opens.

## Layout

| Path | What it tests |
|---|---|
| `test_imputation/` | Unit tests: `MICE` (`test_mice.py`), the imputation methods and the model fits (`test_methods.py`), `imputation.design` (`test_design.py`), the validators, logging |
| `test_plotting/` | Unit tests of the plot functions and the missing-data pattern |
| `test_integration.py` | End-to-end workflows: impute, fit, pool, plot |
| `test_parity/` | Comparisons with R mice (marker `parity`), in three layers, see below |
| `test_properties/` | Layer 4: invariants every completed dataset must satisfy, seeds, argument handling, the default method of each column type, edge cases, pooling properties, the plotting contract; partly with `hypothesis` |
| `reference/` | The R scripts (`reference/r/`) and the fixtures they produce; see [`reference/README.md`](reference/README.md) |
| `conftest.py`, `_logged_events.py` | Shared fixtures and helpers |

The parity layers:

| Files | Layer | What |
|---|---|---|
| `test_l1_*.py` | L1, exact parity | deterministic pieces (least squares, coefficient draw, quantify, pmm with one donor, donor matching, setup step, `make.method`, design, quickpred, md.pattern, pooling, midastouch probabilities) against R to a relative tolerance of 1e-8; the model-based methods (norm family, mean, logreg, logreg.boot, polyreg, polr) with R's random numbers replayed by a strict `ScriptedGenerator`, and the fits behind them (`augment`, `glm.fit`, `optim`'s BFGS, `nnet::multinom`, `MASS::polr`; probabilities of polyreg and polr to 1e-6); `test_l1_mutants.py` checks that the tests catch defects put in on purpose |
| `test_scripted_generator.py` | helper of L1 | the scripted random number generator that serves R's draws |
| `test_l2_*.py` | L2, distribution parity | the imputed values of many calls per method, cell by cell, against R's (chi-square and z tests), and full chains with pmm and with the default methods; `test_l2_mutants.py` checks that the tests catch old and new defects; `test_l2_stats.py` tests the statistics themselves |
| `test_l3_mini_simulation.py` | L3, in-suite | a small simulation (scenario S4, pmm, midas and norm, 200 replications) against stored R results |

Counts at version 0.3.0 (`uv run pytest --collect-only -q <path>`): 763 L1
tests (`test_l1_*.py`, 31 of them mutants), 26 tests of the scripted
generator, 112 L2 tests, 6 in-suite L3 tests, 1304 L4 tests
(`test_properties/`) and 276 unit and workflow tests (`test_imputation/`,
`test_plotting/`, `test_integration.py`): 2487 in all.

The full simulation study (L3) is in `simulations/`. The method and the
results of all layers are described in the documentation page
"Validation against R mice" (`docs/source/r_mice/validation.rst`).

## Markers

Declared in `pyproject.toml`:

- `parity`: comparisons with R (all of `test_parity/`).
- `slow`: the tree tests of L2 that take several seconds.
- `plotting`: the plotting contract tests (`unit` and `integration` are
  declared but not used).

`--strict-markers` is on, so a misspelled marker is an error.

## Warnings

Warnings are not suppressed globally. A test that expects one of the
package's warnings checks it with `pytest.warns`. The few third-party warnings
that cannot be avoided are filtered in `pyproject.toml`, each with its
reason.

## Adding tests

- Put unit tests next to the existing ones of the same module.
- A behavior that R mice defines belongs in `test_parity/`, with a fixture
  from a script in `reference/r/`; see `reference/README.md` for the naming
  and the format.
- An invariant that should hold for every method belongs in
  `test_properties/`; `test_properties/_helpers.py` has the method list and
  the assertions.
- Seed every random generator (`np.random.default_rng(seed)`, or `seed=` in
  `MICE.impute`).
