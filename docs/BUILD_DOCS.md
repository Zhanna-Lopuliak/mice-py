# Building the documentation

The documentation is a Sphinx project in `docs/source` (reStructuredText,
theme `sphinx_rtd_theme`). The API reference is generated from the
docstrings of the installed package with autodoc; the version shown comes
from the installed package metadata (`pyproject.toml`).

## Build

From the repository root:

```bash
uv run --extra docs sphinx-build -W --keep-going -b html docs/source docs/build/html
open docs/build/html/index.html       # macOS; xdg-open on Linux
```

`-W --keep-going` turns every warning into an error but reports all of them
first. The CI job "Docs" (`.github/workflows/ci.yml`) and Read the Docs
(`.readthedocs.yaml`, `fail_on_warning: true`) build with the same
strictness, so a build that warns locally fails there.

Without uv: `pip install -e ".[docs]"`, then the same `sphinx-build`
command. For a clean build remove `docs/build/` first. The intersphinx links
to Python, NumPy, pandas and scikit-learn need network access.

## Layout

```
docs/source/
├── conf.py
├── index.rst
├── installation.rst, quickstart.rst, upgrading.rst
├── user_guide/        # how to use the package; examples start from the quick start's data
├── r_mice/            # validation against R mice, differences from R mice
├── theory/            # missing-data mechanisms, multiple imputation, Rubin's rules
├── examples/          # links to the notebooks in examples/
├── api/               # autodoc pages, one object documented on one page only
└── contributing.rst, changelog.rst, references.rst
```

`changelog.rst` links to `CHANGELOG.md` instead of copying it.

## Conventions

- Every code block in the quick start and the user guide must run as
  written: the user-guide pages assume the data frame `df` of the quick
  start's "Example data" block and run their blocks in order. Use
  `save_path=` in plot examples so that they also run without a display.
- Document each object once. A second `automodule` of the same module
  produces "duplicate object description" warnings, which fail the build.
- Numbers about R parity come from `simulations/results/` and
  `python -m tests.test_parity._l2_report`; update
  `r_mice/validation.rst` when they change.
