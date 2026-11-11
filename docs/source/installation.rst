Installation
============

Requirements
------------

Python 3.9 to 3.13, and

- numpy >= 1.24
- pandas >= 2.0
- scipy >= 1.10
- scikit-learn >= 1.3
- statsmodels >= 0.14
- matplotlib >= 3.7
- seaborn >= 0.12

pip installs them with the package. The package is not on PyPI; install it
from GitHub.

With pip
--------

.. code-block:: bash

   pip install git+https://github.com/Zhanna-Lopuliak/mice-py.git

This installs two packages, ``imputation`` and ``plotting``:

.. code-block:: python

   import imputation
   import plotting

   print(imputation.__version__)   # 0.2.0

Optional extras:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Extra
     - Installs
   * - ``test``
     - pytest, pytest-cov, pytest-timeout, hypothesis
   * - ``docs``
     - sphinx, sphinx-rtd-theme
   * - ``notebooks``
     - jupyter, nbmake (to run the example notebooks)

For example ``pip install "mice-py[notebooks] @
git+https://github.com/Zhanna-Lopuliak/mice-py.git"``.

From a clone
------------

The repository uses `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   git clone https://github.com/Zhanna-Lopuliak/mice-py.git
   cd mice-py
   uv sync                     # the package, the test extra and the dev tools
   uv run python -c "import imputation; print(imputation.__version__)"

Without uv, ``pip install -e ".[test]"`` in a virtual environment gives the
package and the test tools. See :doc:`contributing` for the development
setup.

Troubleshooting
---------------

- ``ModuleNotFoundError: No module named 'imputation'``: the package is not
  installed in the Python you are running; check with
  ``python -m pip show mice-py``.
- Dependency conflicts: install into a fresh virtual environment
  (``python -m venv .venv``).
- Report problems on the `issue tracker
  <https://github.com/Zhanna-Lopuliak/mice-py/issues>`_ with the output of
  ``python -c "import sys, numpy, pandas, sklearn; print(sys.version, numpy.__version__, pandas.__version__, sklearn.__version__)"``.
