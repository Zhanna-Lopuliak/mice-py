# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#
# Build with warnings as errors, as CI and Read the Docs do:
#   uv run --extra docs sphinx-build -W --keep-going -b html docs/source docs/build/html

from importlib.metadata import version as _installed_version

# -- Project information -----------------------------------------------------

project = "mice-py"
copyright = "2025, Anna-Carolina Haensch, The Anh Vu, Zhanna Lopuliak"
author = "Anna-Carolina Haensch, The Anh Vu, Zhanna Lopuliak"

# The version of the installed package (pyproject.toml is the single source).
# The docs are built against the installed package, so autodoc documents the
# same code.
release = _installed_version("mice-py")
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",  # API reference from the docstrings
    "sphinx.ext.napoleon",  # NumPy-style docstrings
    "sphinx.ext.viewcode",  # links to the highlighted source
    "sphinx.ext.intersphinx",  # links to Python, NumPy, pandas, scikit-learn
    "sphinx.ext.mathjax",  # formulas
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# Each object is documented once, on one page of api/. Undocumented members
# are left out: the dataclass fields of PoolingResult, for example, are
# described in its Attributes section.
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
    "prev_next_buttons_location": "both",
    "style_external_links": True,
}

html_title = f"{project} {version}"
html_short_title = "mice-py"
html_show_sourcelink = True
