# Building Documentation Locally

This guide explains how to build and preview the documentation on your local machine before publishing to ReadTheDocs.

## Documentation Structure

The documentation is organized into several sections:

- **Getting Started**: Installation and quickstart guides
- **User Guide**: Comprehensive usage documentation (7 guides)
- **Theory & Background**: Theoretical foundations (4 pages)
- **Examples**: Links to Jupyter notebook tutorials
- **API Reference**: Complete API documentation (5 modules)
- **Development**: Contributing guidelines, changelog, references

## Prerequisites

Install documentation dependencies:

```bash
pip install -e ".[docs]"
```

Or install manually:

```bash
pip install sphinx sphinx-rtd-theme
```

## Build Commands

### Build HTML Documentation

From the repository root:

```bash
cd docs
sphinx-build -b html source build
```

Or use the Sphinx Makefile (if you have one):

```bash
cd docs
make html
```

### View the Documentation

After building, open in your browser:

```bash
# On macOS
open build/index.html

# On Linux
xdg-open build/index.html

# On Windows
start build/index.html
```

Or manually navigate to: `docs/build/index.html`

## Clean Build

To remove old builds and start fresh:

```bash
cd docs
rm -rf build/
sphinx-build -b html source build
```

## Common Build Commands

```bash
# Build HTML (default)
sphinx-build -b html source build

# Build with verbose output (helpful for debugging)
sphinx-build -b html source build -v

# Build and show warnings as errors (strict mode)
sphinx-build -b html source build -W

# Auto-rebuild on file changes (requires sphinx-autobuild)
pip install sphinx-autobuild
sphinx-autobuild source build
# Then open http://127.0.0.1:8000 in your browser
```

## Troubleshooting

### Import Errors

If you get import errors when building:

```bash
# Make sure package is installed in development mode
pip install -e .

# Verify Python can find your package
python -c "import imputation; print(imputation.__file__)"
```

### Missing Dependencies

If Sphinx can't find extensions:

```bash
pip install sphinx sphinx-rtd-theme
```

### API Documentation Not Updating

If changes to docstrings aren't showing up:

```bash
# Clean build directory
rm -rf build/
# Rebuild
sphinx-build -b html source build
```

## Directory Structure

The documentation source files are organized as:

```
docs/source/
├── index.rst                  # Main landing page
├── installation.rst           # Installation guide
├── quickstart.rst            # Quick start tutorial
├── user_guide/               # User guides (7 pages)
│   ├── index.rst
│   ├── understanding_missing_data.rst
│   ├── mice_overview.rst
│   ├── imputation_methods.rst
│   ├── predictor_matrices.rst
│   ├── convergence_diagnostics.rst
│   ├── pooling_analysis.rst
│   └── best_practices.rst
├── theory/                   # Theoretical background (4 pages)
│   ├── index.rst
│   ├── missing_data_mechanisms.rst
│   ├── multiple_imputation_theory.rst
│   ├── method_details.rst
│   └── rubins_rules.rst
├── examples/                 # Examples section
│   └── index.rst
├── api/                      # API reference (5 modules)
│   ├── index.rst
│   ├── mice.rst
│   ├── methods.rst
│   ├── pooling.rst
│   ├── plotting.rst
│   └── utilities.rst
├── contributing.rst          # Contribution guidelines
├── changelog.rst            # Version history
└── references.rst           # Bibliography
```

## Automatic Documentation Generation

The API documentation is manually organized but uses autodoc for docstrings.
If you need to regenerate:

```bash
# The API docs use automodule directives
# Just rebuild to pull latest docstrings
sphinx-build -b html source build
```

## Preview Before Publishing

**Always** build and preview locally before pushing to GitHub:

1. Build docs: `sphinx-build -b html source build`
2. Check for warnings or errors in the output
3. Open `build/index.html` in browser
4. Verify all pages render correctly
5. Check that navigation works
6. Verify API documentation is complete
7. Push to GitHub when satisfied

## Continuous Integration

ReadTheDocs will automatically build your documentation when you push to GitHub. You can check build status at:

- https://readthedocs.org/projects/mice-py/builds/

## Tips

- Use `sphinx-autobuild` for live preview during development
- Check for broken links: `sphinx-build -b linkcheck source build`
- Build PDF (requires LaTeX): `sphinx-build -b latex source build && cd build && make`
- Use `-W` flag to treat warnings as errors in CI
