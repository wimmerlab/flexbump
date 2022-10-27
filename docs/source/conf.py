# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from urllib.request import urlretrieve
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(1, os.path.abspath('../../src'))
sys.path.insert(2, os.path.abspath('../../src/lib'))

# -- Project information -----------------------------------------------------

project = 'Flexible integration of continuous sensory evidence in perceptual estimation tasks'
copyright = '2022, Jose M. Esnaola-Acebes, Alex Roxin, Klaus Wimmer'
author = 'Jose M. Esnaola-Acebes, Alex Roxin, Klaus Wimmer'

# The full version, including alpha/beta/rc tags
release = 'GPL-3.0'


# -- General configuration ---------------------------------------------------

autodoc_member_order = "bysource"
autodoc_inherit_docstrings = True  # This option includes references (links) to the methods of the inherited class
add_function_parentheses = True
add_module_names = True

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",  # This extension enables rendering of type hints
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
    "nbsphinx_link",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinx_design",
    "sphinxarg.ext",
    "sphinxcontrib.youtube",
]

master_doc = "index"
# Napoleon settings
napoleon_google_docstring = False

# To do extension options
todo_include_todos = True

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    "papersize": "letterpaper",
    # The font size ('10pt', '11pt' or '12pt').
    "pointsize": "11pt",
    # Additional stuff for the LaTeX preamble.
    "preamble": r"""
        \usepackage{charter}
        % \usepackage[defaultsans]{lato}
        % \usepackage{inconsolata}
    """,
}

autodoc_default_options = {"members": True, "inherited-members": True}
# generate autosummary even if no references
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
panels_add_bootstrap_css = False  # Panel extension loads it by default, and it is already loaded
html_theme = "sphinx_book_theme"
html_title = "Flexible integration of continuous sensory evidence in perceptual estimation tasks"
html_logo = sys.path[0] + "/docs/source/_static/flex_bump.svg"
html_favicon = sys.path[0] + "/docs/source/_static/flexbump_icon.svg"

htmlhelp_basename = "flex_bump"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ["crm.css"]
html_js_files = [
    'read-more.js',
]
html_theme_options = {
    "repository_url": "https://github.com/wimmerlab/flexbump",
    "collapse_navigation": True,
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "use_edit_page_button": False,
    "repository_branch": "main",
    "path_to_docs": "docs",
    "launch_buttons": {
        "colab_url": "https://colab.research.google.com",
        "notebook_interface": "jupyterlab",
    },
    "extra_navbar": "",  # Custom sidebar footer ("" means empty)
    "home_page_in_toc": True,  # landing page in TOC
    "show_navbar_depth": 2,  # Depth of left sidebar lists to expand (default is 1)
    "toc_title": "Go to",  # Title of the secondary (right) sidebar
    "announcement": "⚠ Warning! Under development! ⚠",  # Announcement banner (disappears when scrolling)
}

# Notebook related options
ipython_warning_is_error = False
# We need to execute import the modules if we want to use them as .. ipython:: python
ipython_execlines = [
    "import numpy as np",
    "import pandas as pd",
    # This ensures correct rendering on system with console encoding != utf8
    # (windows). It forces pandas to encode its output reprs using utf8
    # wherever the docs are built. The docs' target is the browser, not
    # the console, so this is fine.
    'pd.options.display.encoding="utf8"',
]

# -----------------------------------------------------------------------------
# Intersphinx configuration
# -----------------------------------------------------------------------------
intersphinx_mapping = {
    'neps': ('https://numpy.org/neps', None),
    'python': ('https://docs.python.org/3.9', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'imageio': ('https://imageio.readthedocs.io/en/stable', None),
    'skimage': ('https://scikit-image.org/docs/stable', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
    'pd': ('https://pandas.pydata.org/pandas-docs/stable', None),
    'scipy-lecture-notes': ('https://scipy-lectures.org', None),
    'pytest': ('https://docs.pytest.org/en/stable', None),
    'numpy-tutorials': ('https://numpy.org/numpy-tutorials', None),
    'numpydoc': ('https://numpydoc.readthedocs.io/en/latest', None),
    'dlpack': ('https://dmlc.github.io/dlpack/latest', None),
    'seaborn': ("https://seaborn.pydata.org/", None),
}

# -----------------------------------------------------------------------------
# External files (e.g. unicode characters)
# -----------------------------------------------------------------------------
symbol_pages = ["https://docutils.sourceforge.io/docutils/parsers/rst/include/isoamsa.txt",
                "https://docutils.sourceforge.io/docutils/parsers/rst/include/xhtml1-symbol.txt"
                ]

for page in symbol_pages:
    try:
        urlretrieve(page, f"_templates/{page.split('/')[-1]}")
    except FileNotFoundError:
        os.mkdir("_templates")
        urlretrieve(page, f"_templates/{page.split('/')[-1]}")
