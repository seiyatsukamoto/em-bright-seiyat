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
import builtins
from pathlib import Path
import sys
import toml

builtins.__sphinx_build__ = True
sys.path.insert(0, str(Path(__file__).parents[1]))
# -- Project information -----------------------------------------------------

pyproject_toml = toml.load(Path(__file__).parents[1] / 'pyproject.toml')
project = pyproject_toml['tool']['poetry']['name']
author = ', '.join(pyproject_toml['tool']['poetry']['authors'])
copyright = '2021, Deep Chatterjee'

# The full version, including alpha/beta/rc tags
release = pyproject_toml['tool']['poetry']['version']


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'nbsphinx'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = ['.rst', '.md', '.ipynb']
source_parsers = {
    '.md': 'recommonmark.parser.CommonMarkParser'
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

autodoc_mock_imports = [
    'numpy',
    'numpy.core',
    'scipy',
    'h5py',
    'astropy',
    'lal',
    'lalsimulation',
    'argparse',
    'configparser'
]

autodoc_default_options = {
    'members': None,
    'undoc-members': None,
    'show-inheritance': None
}
autodoc_member_order = 'bysource'
autosummary_generate = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
