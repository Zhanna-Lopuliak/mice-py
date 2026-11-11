Contributing
============

Bug reports, fixes, documentation and new methods are welcome. Open an issue
on `GitHub <https://github.com/Zhanna-Lopuliak/mice-py/issues>`_ with a
minimal example and your versions (Python, numpy, pandas, scikit-learn), or
a pull request against ``main``.

Setup
-----

The project uses `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   git clone https://github.com/Zhanna-Lopuliak/mice-py.git
   cd mice-py
   uv sync                       # package (editable), test extra, ruff, mypy, pre-commit
   uv run pre-commit install     # ruff check and ruff format on every commit

``uv sync`` installs the ``dev`` dependency group by default. The docs and
notebook tools are extras: ``uv run --extra docs ...`` and
``uv run --extra notebooks ...`` install them on demand.

Checks
------

The same commands run in CI (``.github/workflows/ci.yml``):

.. code-block:: bash

   uv run ruff check .                     # lint
   uv run ruff format --check .            # formatting (line length 120)
   uv run mypy                             # type check of imputation and plotting
   uv run pytest                           # all tests
   uv run pytest --cov --cov-report=term-missing   # with coverage; CI requires 89%
   uv run pytest --doctest-modules imputation plotting
   uv run --extra docs sphinx-build -W --keep-going -b html docs/source docs/build/html
   uv run --extra notebooks pytest --nbmake examples/

CI runs the tests on Python 3.9 to 3.13 on Linux and on 3.13 on macOS.

Tests
-----

``tests/README.md`` describes the layout. In short:

- ``tests/test_imputation``, ``tests/test_plotting``,
  ``tests/test_integration.py``: unit and workflow tests;
- ``tests/test_parity``: comparisons with R mice 3.19.0 (marker ``parity``),
  against the fixtures in ``tests/reference/fixtures``;
- ``tests/test_properties``: invariants and edge cases, partly with
  hypothesis.

A change that affects imputations should keep the parity tests passing. If R
behaves differently from what you expect, check R first: the fixtures are
produced by the scripts in ``tests/reference/r`` (see
``tests/reference/README.md``). To regenerate them you need R with mice
3.19.0, ranger, rpart and jsonlite:

.. code-block:: bash

   Rscript tests/reference/r/generate_all.R
   git diff --exit-code tests/reference/fixtures

The workflow ``.github/workflows/r-fixtures.yml`` does the same on GitHub
(started by hand). The simulation study against R is described in
``simulations/README.md`` and on :doc:`r_mice/validation`.

Style
-----

- ruff for lint and formatting, configured in ``pyproject.toml``.
- Type hints on the public API; mypy must pass.
- NumPy-style docstrings, with a blank line before every list so that Sphinx
  renders it. Examples in docstrings are run as doctests.
- The library does not print: use ``warnings.warn`` for what the user must
  see and ``logging.getLogger(__name__)`` for the rest.
- When porting from R mice, read the source of the ``v3.19.0`` tag and say
  in the docstring which R function is followed. mice-py is derived from
  mice and, like mice, is licensed under the GPL, version 2 or later; see
  ``COPYRIGHT``.

Adding an imputation method
---------------------------

1. Write the function in a new module ``imputation/<name>.py`` with the
   signature of the others:
   ``f(y, id_obs, x, id_mis=None, <parameters>, rng=None) -> np.ndarray``.
   It returns the imputations of the cells in ``id_mis`` (all missing cells
   if ``None``) and draws all random numbers from ``rng``. It may also take
   a parameter ``log_event``: ``MICE.impute`` then passes a callable that
   takes one string and records it as a row of ``MICE.logged_events``, with
   the iteration, imputation, column and method filled in.
2. Add the name to ``ImputationMethod`` in ``imputation/constants.py`` and
   the function to ``_IMPUTER_MAP`` in ``imputation/utils.py``. Its
   parameters then become ``<name>_<parameter>`` arguments of
   ``MICE.impute``.
3. Add unit tests, and the method to the property tests in
   ``tests/test_properties``. If R mice has the method, add parity tests.
4. Document it in :doc:`user_guide/imputation_methods` and
   ``docs/source/api/methods.rst``, and add a line to ``CHANGELOG.md``.

Documentation
-------------

The docs are in ``docs/source`` (Sphinx, reStructuredText); see
``docs/BUILD_DOCS.md``. Build them with warnings as errors before a pull
request:

.. code-block:: bash

   uv run --extra docs sphinx-build -W --keep-going -b html docs/source docs/build/html

Every code block of the README, the quick start and the user guide should
run as written.

License
-------

Contributions are licensed under the GNU General Public License, version 2
or later, as the rest of the project; see ``LICENSE`` and ``COPYRIGHT``.
