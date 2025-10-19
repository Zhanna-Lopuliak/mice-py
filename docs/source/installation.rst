Installation
============

Requirements
------------

**Python Version**
   Python 3.9 or higher is required.

**Core Dependencies**
   - numpy >= 1.24
   - pandas >= 2.0
   - scipy >= 1.10
   - scikit-learn >= 1.3
   - statsmodels >= 0.14
   - matplotlib >= 3.7
   - seaborn >= 0.12
   - patsy >= 1.0
   - tqdm >= 4.60

Installation Methods
--------------------

From GitHub
~~~~~~~~~~~

Install directly from the GitHub repository:

.. code-block:: bash

   pip install git+https://github.com/Zhanna-Lopuliak/mice-py.git

This is the recommended method for most users.

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For contributors or those who want to modify the code:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/Zhanna-Lopuliak/mice-py.git
   cd mice-py
   
   # Install in development mode
   pip install -e .

Development mode allows you to make changes to the code that take effect immediately without reinstalling.

Optional Dependencies
---------------------

Install with Testing Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -e ".[test]"

This installs:
   - pytest >= 6.0
   - pytest-cov >= 2.10
   - pytest-timeout >= 1.4

Install with Documentation Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -e ".[docs]"

This installs:
   - sphinx >= 7.0
   - sphinx-rtd-theme >= 1.2

Install with Development Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -e ".[dev]"

This installs all testing tools plus:
   - black (code formatting)
   - flake8 (linting)
   - mypy (type checking)

Verifying Installation
----------------------

After installation, verify that the package is correctly installed:

.. code-block:: python

   import imputation
   from imputation import MICE
   
   print(f"mice-py version: {imputation.__version__}")

If this runs without errors, the installation was successful.

Troubleshooting
---------------

Import Errors
~~~~~~~~~~~~~

If you encounter import errors after installation:

1. Check that you're using Python 3.9 or higher:

   .. code-block:: bash

      python --version

2. Verify the package is installed:

   .. code-block:: bash

      pip list | grep mice-py

3. Try reinstalling:

   .. code-block:: bash

      pip uninstall mice-py
      pip install git+https://github.com/Zhanna-Lopuliak/mice-py.git

Dependency Conflicts
~~~~~~~~~~~~~~~~~~~~

If you have dependency conflicts:

1. Create a fresh virtual environment:

   .. code-block:: bash

      python -m venv mice_env
      source mice_env/bin/activate  # On Windows: mice_env\Scripts\activate
      pip install git+https://github.com/Zhanna-Lopuliak/mice-py.git

2. Or use conda:

   .. code-block:: bash

      conda create -n mice_env python=3.9
      conda activate mice_env
      pip install git+https://github.com/Zhanna-Lopuliak/mice-py.git

Version Compatibility
~~~~~~~~~~~~~~~~~~~~~

The package has been tested with:
   - Python 3.9, 3.10, 3.11, 3.12
   - numpy 1.24+
   - pandas 2.0+
   - scikit-learn 1.3+

If you experience issues with specific versions, please report them on the `GitHub Issues page <https://github.com/Zhanna-Lopuliak/mice-py/issues>`_.

Next Steps
----------

After successful installation:

- Read the :doc:`quickstart` guide for a quick introduction
- Explore the :doc:`user_guide/index` for detailed usage information
- Check out the :doc:`examples/index` for practical examples

