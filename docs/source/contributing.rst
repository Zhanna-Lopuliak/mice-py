Contributing
============

We welcome contributions to mice-py! This guide will help you get started.

Ways to Contribute
------------------

There are many ways to contribute:

**Report Bugs**
   Found a bug? Open an issue on GitHub with:
   
   - Clear description of the problem
   - Minimal reproducible example
   - Your environment (Python version, OS, package versions)

**Suggest Features**
   Have an idea for improvement? Open an issue describing:
   
   - The proposed feature
   - Use cases and benefits
   - Possible implementation approach

**Improve Documentation**
   Documentation can always be better:
   
   - Fix typos or unclear explanations
   - Add examples
   - Improve docstrings
   - Write tutorials

**Submit Code**
   Code contributions are welcome:
   
   - Bug fixes
   - New features
   - Performance improvements
   - Additional imputation methods

Development Setup
-----------------

Fork and Clone
~~~~~~~~~~~~~~

1. Fork the repository on GitHub

2. Clone your fork:

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/mice-py.git
      cd mice-py

3. Add upstream remote:

   .. code-block:: bash

      git remote add upstream https://github.com/Zhanna-Lopuliak/mice-py.git

Install Development Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install in development mode with all dependencies
   pip install -e ".[dev]"

This installs:
   - The package in editable mode
   - Testing tools (pytest, pytest-cov)
   - Development tools (black, flake8, mypy)
   - Documentation tools (sphinx, sphinx-rtd-theme)

Create a Branch
~~~~~~~~~~~~~~~

.. code-block:: bash

   git checkout -b feature/your-feature-name

Or for bug fixes:

.. code-block:: bash

   git checkout -b fix/issue-description

Development Workflow
--------------------

Making Changes
~~~~~~~~~~~~~~

1. Make your changes to the code
2. Add or update tests
3. Update documentation if needed
4. Run tests locally

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   pytest
   
   # Run with coverage
   pytest --cov=imputation --cov=plotting --cov-report=term-missing
   
   # Run specific test file
   pytest tests/test_imputation/test_mice.py
   
   # Run specific test
   pytest tests/test_imputation/test_mice.py::test_basic_imputation

Code Style
~~~~~~~~~~

We follow PEP 8 style guidelines:

.. code-block:: bash

   # Format code with black
   black imputation/ plotting/ tests/
   
   # Check with flake8
   flake8 imputation/ plotting/ tests/
   
   # Type checking (optional but encouraged)
   mypy imputation/

Writing Tests
~~~~~~~~~~~~~

- Add tests for new features
- Ensure bug fixes include regression tests
- Maintain or improve code coverage
- Tests should be clear and well-documented

Example test:

.. code-block:: python

   def test_basic_imputation():
       """Test that basic imputation runs without errors."""
       df = pd.DataFrame({
           'a': [1, 2, np.nan, 4],
           'b': [5, np.nan, 7, 8]
       })
       
       mice = MICE(df)
       mice.impute(n_imputations=2, maxit=2)
       
       assert len(mice.imputed_datasets) == 2
       assert mice.imputed_datasets[0].isnull().sum().sum() == 0

Documentation
~~~~~~~~~~~~~

- Update docstrings for new/modified functions
- Follow NumPy docstring format
- Add examples to docstrings when helpful
- Update relevant user guide sections

Example docstring:

.. code-block:: python

   def my_function(x, y, method='default'):
       """
       Brief one-line description.
       
       More detailed description of what the function does,
       including any important details about behavior.
       
       Parameters
       ----------
       x : array-like
           Description of x parameter
       y : int
           Description of y parameter
       method : str, default='default'
           Description of method parameter
           
       Returns
       -------
       result : float
           Description of return value
           
       Examples
       --------
       >>> my_function([1, 2, 3], 5)
       42.0
       """
       pass

Submitting Changes
------------------

Commit Your Changes
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git add .
   git commit -m "Brief description of changes"

Write clear commit messages:
   - Use present tense ("Add feature" not "Added feature")
   - First line should be brief (50 chars or less)
   - Add detailed description after blank line if needed

Push to Your Fork
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git push origin feature/your-feature-name

Open a Pull Request
~~~~~~~~~~~~~~~~~~~

1. Go to the original repository on GitHub
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill in the PR template:
   
   - Clear title
   - Description of changes
   - Related issue number (if applicable)
   - Checklist of completed items

5. Submit the PR

Pull Request Guidelines
-----------------------

Before submitting:

☐ All tests pass
☐ Code follows style guidelines
☐ New tests added for new features
☐ Documentation updated
☐ Commit messages are clear
☐ No merge conflicts with main branch

Your PR should:

- Have a clear purpose (one feature or fix per PR)
- Include tests for new functionality
- Update relevant documentation
- Pass all CI checks

After Submission
~~~~~~~~~~~~~~~~

- Respond to review comments
- Make requested changes
- Push updates to your branch (PR will update automatically)
- Be patient and respectful

Code Review Process
-------------------

What to Expect
~~~~~~~~~~~~~~

- Maintainers will review your PR
- They may request changes or ask questions
- Discussion helps improve the code
- Multiple rounds of review are normal

Tips for Success
~~~~~~~~~~~~~~~~

- Keep PRs focused and reasonably sized
- Respond to feedback promptly
- Be open to suggestions
- Ask questions if anything is unclear

Adding New Imputation Methods
------------------------------

To add a new imputation method:

1. **Create method file**: ``imputation/new_method.py``

2. **Implement function**:

   .. code-block:: python

      def new_method(y, id_obs, x, id_mis=None, **kwargs):
          """
          Docstring describing the method.
          
          Parameters should follow PMM/CART conventions.
          """
          # Implementation
          pass

3. **Add to utils.py**: Register in ``get_imputer_func()``

4. **Add tests**: ``tests/test_imputation/test_methods.py``

5. **Update documentation**:
   
   - API reference
   - User guide
   - Examples

6. **Add to constants**: If needed, update ``constants.py``

Building Documentation
----------------------

Local Build
~~~~~~~~~~~

.. code-block:: bash

   cd docs
   sphinx-build -b html source build

Then open ``docs/build/index.html`` in your browser.

Auto-rebuild
~~~~~~~~~~~~

.. code-block:: bash

   pip install sphinx-autobuild
   sphinx-autobuild docs/source docs/build

Then visit http://127.0.0.1:8000

Getting Help
------------

If you need help:

- Check existing issues and PRs
- Read the :doc:`user_guide/index`
- Ask in your PR or issue
- Contact maintainers

Code of Conduct
---------------

Be Respectful
~~~~~~~~~~~~~

- Treat everyone with respect
- Be welcoming to newcomers
- Value diverse perspectives
- Focus on constructive feedback

Be Collaborative
~~~~~~~~~~~~~~~~

- Help others
- Share knowledge
- Give credit where due
- Work together toward shared goals

License
-------

By contributing, you agree that your contributions will be licensed under the 
MIT License.

Questions?
----------

Open an issue on GitHub if you have questions about contributing.

Thank You!
----------

Thank you for contributing to mice-py! Your help makes this project better for everyone.

