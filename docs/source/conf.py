# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PrjLECM'
copyright = '2025, Bhalotra, Fernandez, Wang, Yang'
author = 'Bhalotra, Fernandez, Wang, Yang'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']


import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))
import inspect
from pathlib import Path

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.intersphinx',     # Add this
    'sphinx.ext.linkcode',         # Add this for "view source" links
    'sphinx.ext.viewcode',         # Alternative to linkcode (simpler)
    'sphinx.ext.mathjax', # Required to render math in HTML
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

sphinx_gallery_conf = {
    'examples_dirs': [
        '../examples/equi'
    ],
    'gallery_dirs': [
        'auto_examples/equi'
    ],
    'filename_pattern': r'.*\.py$',
    'ignore_pattern': r'__init__\.py',
    'backreferences_dir': 'gen_modules/backreferences',
    'doc_module': ('prjlecm',),
    'reference_url': {
        'prjlecm': None,
    },
    'image_srcset': ["1x"],
    'subsection_order': None,
    # Use MyST for markdown parsing
    'pypandoc': False,
    'use_pypandoc': False,
}

sphinx_gallery_conf = {
    'examples_dirs': [
        '../examples/equi'
    ],   # paths to your example scripts
    'gallery_dirs': [
        'auto_examples/equi'
    ],   # where to save gallery generated output
    'filename_pattern': r'.*\.py$',   # include all .py files
    'ignore_pattern': r'__init__\.py',
    'backreferences_dir': 'gen_modules/backreferences',  # backreferences for modules
    'doc_module': ('prjlabbcmww',),  # your package name
    'reference_url': {
        'prjlabbcmww': None,  # use local docs
    },
    'image_srcset': ["1x"],  # clickable images at original resolutions
    'subsection_order': None,  # optional: control order of examples within each gallery
}

# to generate links to GitHub source code, works with sphinx.ext.linkcode
def linkcode_resolve(domain, info):
    """Link to GitHub source code"""
    if domain != 'py':
        return None
    if not info['module']:
        return None
    
    # Get the object
    obj = None
    try:
        obj = __import__(info['module'])
        for part in info['module'].split('.')[1:]:
            obj = getattr(obj, part)
        if info['fullname']:
            for part in info['fullname'].split('.'):
                obj = getattr(obj, part)
    except Exception:
        return None
    
    # Get source file
    try:
        fn = inspect.getsourcefile(obj)
        if fn is None:
            return None
        
        # Get line numbers
        source, lineno = inspect.getsourcelines(obj)
        
        # Convert to relative path from repo root
        fn = os.path.relpath(fn, start=os.path.dirname(os.path.dirname(__file__)))
        
        # Construct GitHub URL
        # Replace with your actual GitHub repo
        github_user = "FanWangEcon"
        github_repo = "PrjLECM"
        github_branch = "main"
        
        return f"https://github.com/{github_user}/{github_repo}/blob/{github_branch}/{fn}#L{lineno}-L{lineno + len(source) - 1}"
    except Exception:
        return None