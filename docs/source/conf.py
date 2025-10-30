# Configuration file for the Sphinx documentation builder.
# this sphinx file was inspired by https://sphinx-doc.org and
# https://github.com/NIFTy-PPL/JAXbind/blob/main/docs/source/conf.py
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
needs_sphinx = "3.2.0"

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Motif Reactor Simulation, Analysis and Inference Kit'
copyright = '2025, Johannes Harth-Kitzerow'
author = 'Johannes Harth-Kitzerow'
release = "0.1.0" # morsaik.__version__
version = release[:-2]

language = "en"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        'sphinx.ext.napoleon', # support for NumPy and Google style docstrings
        'sphinx.ext.imgmath', # render math as images
        'sphinx.ext.viewcode', # add links to highlighted source code
        'sphinx.ext.intersphinx', # links to other sphinx docs
        'myst_parser', # parse markdown
        ]
master_doc = 'index'

myst_enable_extenstions = [
        'amsmath',
        'dollarmath',
        'strikethrough',
        'tasklist'
]

intersphinx_mapping = {"numpy": ("https://numpy.org/doc/stable/", None),
                       #"matplotlib": ('https://matplotlib.org/stable/', None),
                       "nifty8": ('https://ift.pages.mpcdf.de/nifty', None),
                       "scipy": ('https://docs.scipy.org/doc/scipy/reference/', None),
                       }


autodoc_default_options = {
        'member-order' : 'bysource',
        'special-members' : '__init__, domain.__init__, get.__init__, infer.__init__, plot.__init__, read.__init__, util.__init__'
        }
add_module_names = False

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_references = True

imgmath_embed = True

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_last_updated_fmt = "%b %d, %Y"
html_theme = "pydata_sphinx_theme"
html_context = {"default_mode": "dark"}
# html_static_path = ['_static']
