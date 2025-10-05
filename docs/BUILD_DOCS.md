# Building Documentation Locally

This guide explains how to build and preview the documentation on your local machine before publishing to ReadTheDocs.

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

## Automatic Documentation Generation

If you want to auto-generate API documentation from docstrings:

```bash
# Install sphinx-apidoc
pip install sphinx

# Generate RST files from code
sphinx-apidoc -f -o docs/source/api imputation/
sphinx-apidoc -f -o docs/source/api plotting/

# Then build
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
