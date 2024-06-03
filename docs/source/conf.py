# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Add documentation stored in other directories
import os
import sys

# couldn't get any other way to work, so 
# this should guarantee that sphinx is able
# to find the dir
p1 = os.path.dirname(os.path.realpath(__file__))
p2 = "..\.."
sys.path.insert(0, os.path.join(p1, p2))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'riserfit'
copyright = '2024, Lennart Grimm'
author = 'Lennart Grimm'
release = '0.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.napoleon']

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "nbsphinx_link"
]
napoleon_google_docstring = False
napoleon_numpy_docstring = True
# napoleon_use_param = False
# napoleon_use_ivar = True

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

templates_path = ['_templates']

html_theme = 'sphinx_rtd_theme'

#html_theme = 'alabaster'
# html_static_path = ['_static']
