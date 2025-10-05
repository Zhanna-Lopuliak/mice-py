# ReadTheDocs Setup Guide

This guide walks you through setting up automatic documentation hosting on ReadTheDocs for the mice-py package.

## What You'll Get

- 📚 Beautiful, searchable documentation at `https://mice-py.readthedocs.io/`
- 🔄 Automatic updates whenever you push to GitHub
- 📱 Mobile-friendly responsive design
- 🔍 Built-in search functionality
- 📖 Version control for documentation
- 🌐 Free hosting forever (for open source projects)

## Quick Setup (5 Minutes)

### Step 1: Push to GitHub

Make sure your repository is on GitHub with all the latest changes:

```bash
git add .
git commit -m "Prepare documentation for ReadTheDocs"
git push origin main
```

### Step 2: Sign Up for ReadTheDocs

1. Go to **https://readthedocs.org/**
2. Click **"Sign Up"**
3. Choose **"Sign up with GitHub"** (easiest option)
4. Authorize ReadTheDocs to access your repositories

### Step 3: Import Your Project

1. After signing in, click your username → **"My Projects"**
2. Click **"Import a Project"**
3. Find **`mice-py`** in your repository list
4. Click the **"+"** button next to it
5. Verify the settings:
   - **Name**: `mice-py`
   - **Repository URL**: (auto-filled from GitHub)
   - **Default branch**: `main`
6. Click **"Next"**

### Step 4: First Build

ReadTheDocs will automatically:
- Detect your `readthedocs.yaml` configuration
- Install your package with dependencies
- Build the documentation
- Publish it online

This takes 2-3 minutes. You can watch the progress in the **"Builds"** tab.

### Step 5: Access Your Docs

Once the build completes (green checkmark ✅), your documentation is live at:

```
https://mice-py.readthedocs.io/en/latest/
```

## Configuration Files Explained

You already have all the necessary files:

### 1. `readthedocs.yaml` (Root Directory)

```yaml
version: 2
build:
  os: ubuntu-22.04
  tools:
    python: "3.9"
python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - docs
sphinx:
  configuration: docs/source/conf.py
```

**What it does:**
- Tells ReadTheDocs to use Python 3.9
- Installs your package with docs dependencies
- Points to your Sphinx configuration

### 2. `docs/source/conf.py` (Sphinx Configuration)

```python
project = 'mice-py'
copyright = '2025, Anna-Carolina Haensch, The Anh Vu, Zhanna Lopuliak'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',     # Auto-generate API docs from docstrings
    'sphinx.ext.napoleon',    # Support NumPy/Google docstring style
    'sphinx.ext.viewcode',    # Add "View Source" links
    'sphinx.ext.intersphinx', # Link to other projects' docs
]

html_theme = "sphinx_rtd_theme"  # ReadTheDocs theme
```

**What it does:**
- Sets project metadata
- Configures Sphinx extensions for API documentation
- Sets the visual theme

### 3. `pyproject.toml` (Docs Dependencies)

```toml
[project.optional-dependencies]
docs = [
    "sphinx>=7.0",
    "sphinx-rtd-theme>=1.2",
]
```

**What it does:**
- Lists packages needed to build documentation
- Installed automatically by ReadTheDocs

## Automatic Updates

After setup, ReadTheDocs automatically rebuilds your documentation:

✅ **Every time you push to GitHub** (any branch)
✅ **For pull requests** (if you enable it in settings)
✅ **On a schedule** (optional, for external data sources)

## Monitoring Builds

View build status and logs:

1. Go to https://readthedocs.org/projects/mice-py/
2. Click **"Builds"** tab
3. See build history with status:
   - ✅ Green = Success
   - ❌ Red = Failed
   - 🟡 Yellow = In Progress

Click any build to see detailed logs.

## Advanced Configuration

### Multiple Versions

ReadTheDocs automatically creates documentation for:
- **latest**: Your default branch (main)
- **stable**: Your latest tagged release
- Each Git tag (e.g., v0.1.0, v0.2.0)

Users can switch versions using a dropdown in the docs.

### Custom Domain

You can use your own domain:

1. In ReadTheDocs project settings
2. Go to **Admin → Domains**
3. Add your custom domain (e.g., docs.your-project.com)
4. Update your DNS settings

### Build Notifications

Get notified of build failures:

1. **Admin → Notifications**
2. Add email address
3. Choose notification triggers

### PR Preview Builds

Preview docs for pull requests before merging:

1. **Admin → Advanced Settings**
2. Enable **"Build pull requests for this project"**

Now each PR gets a preview link!

## Troubleshooting

### Build Fails with Import Error

**Problem:** Can't import your package

**Solution:** Make sure `readthedocs.yaml` installs your package:
```yaml
python:
  install:
    - method: pip
      path: .  # Install package from root
```

### Missing Dependencies

**Problem:** Module not found during build

**Solution:** Add dependencies to `pyproject.toml`:
```toml
[project.optional-dependencies]
docs = [
    "sphinx>=7.0",
    "sphinx-rtd-theme>=1.2",
    "any-other-needed-package>=version",
]
```

### Documentation Not Updating

**Problem:** Changes don't appear after pushing

**Solutions:**
1. Check build status - did it succeed?
2. Clear browser cache
3. Trigger manual rebuild: **Builds → Build Version**
4. Check you're viewing `/latest/` not an old version

### Theme Issues

**Problem:** Documentation looks broken

**Solution:** Verify theme is installed:
```toml
docs = [
    "sphinx-rtd-theme>=1.2",
]
```

## Best Practices

### 1. Write Good Docstrings

Use NumPy or Google style docstrings:

```python
def impute(self, n_imputations=5):
    """
    Perform multiple imputation.
    
    Parameters
    ----------
    n_imputations : int, default=5
        Number of imputations to create
        
    Returns
    -------
    list of pd.DataFrame
        List of imputed datasets
        
    Examples
    --------
    >>> mice = MICE(data)
    >>> mice.impute(n_imputations=5)
    """
```

### 2. Test Locally First

Before pushing, build docs locally:

```bash
cd docs
sphinx-build -b html source build
open build/index.html  # macOS
```

See `docs/BUILD_DOCS.md` for detailed local build instructions.

### 3. Version Your Docs

Use Git tags for releases:

```bash
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0
```

ReadTheDocs will automatically build docs for this version.

### 4. Add Links to README

Add a documentation badge to your README:

```markdown
[![Documentation Status](https://readthedocs.org/projects/mice-py/badge/?version=latest)](https://mice-py.readthedocs.io/en/latest/?badge=latest)
```

### 5. Keep Docs Updated

When you add new features:
1. Write docstrings
2. Update relevant RST files if needed
3. Push to GitHub
4. ReadTheDocs builds automatically!

## Additional Resources

- **ReadTheDocs Documentation**: https://docs.readthedocs.io/
- **Sphinx Documentation**: https://www.sphinx-doc.org/
- **Sphinx RTD Theme**: https://sphinx-rtd-theme.readthedocs.io/
- **Your Build Logs**: https://readthedocs.org/projects/mice-py/builds/

## Need Help?

If you encounter issues:

1. Check build logs on ReadTheDocs
2. Test build locally: `cd docs && sphinx-build -b html source build`
3. Verify all configuration files are committed to GitHub
4. Check ReadTheDocs documentation for common issues

## Summary Checklist

- [x] Configuration files ready (`readthedocs.yaml`, `conf.py`)
- [x] Dependencies specified in `pyproject.toml`
- [ ] Repository pushed to GitHub
- [ ] ReadTheDocs account created and linked to GitHub
- [ ] Project imported on ReadTheDocs
- [ ] First build successful
- [ ] Documentation accessible online
- [ ] Badge added to README

That's it! Your documentation is now live and will update automatically. 🎉
