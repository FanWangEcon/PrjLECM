# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PrjLECM'
copyright = '2025, Bhalotra, Fernandez, Wang, Xun'
author = 'Bhalotra, Fernandez, Wang, Xun'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
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

templates_path = ['_templates']
exclude_patterns = []

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

sphinx_gallery_conf = {
    'examples_dirs': [
        '../examples/demand',
        '../examples/supply',
        '../examples/equi'
    ],   # paths to your example scripts
    'gallery_dirs': [
        'auto_examples/demand',
        'auto_examples/supply',
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

    
# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "github_url": "https://github.com/FanWangEcon/PrjLECM",
    "logo": {
        "text": "PrjLECM",
    },
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "navbar_align": "left",
    "show_toc_level": 3,
    "navigation_depth": 4,
    "show_nav_level": 2,
    "navigation_with_keys": True,
    "collapse_navigation": False,
    "header_links_before_dropdown": 4,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/FanWangEcon/PrjLECM",
            "icon": "fa-brands fa-github",
        },
    ],
    "use_edit_page_button": True,
    "show_prev_next": True,
    "search_bar_text": "Search this site...",
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    
    # Default to light mode
    "theme_switcher_button": True,
    "default_mode": "light",
}

# Enable section numbering (we'll override the style with CSS)
html_secnumber_suffix = ". "
numfig = True
numfig_secnum_depth = 3

html_context = {
    "github_user": "FanWangEcon",
    "github_repo": "PrjLECM",
    "github_version": "main",
    "doc_path": "docs/source",
}

html_static_path = ['_static']
html_css_files = ['custom.css']

html_sidebars = {
    "**": ["search-field", "sidebar-nav-bs"]
}

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

# Linkcode configuration
def linkcode_resolve(domain, info):
    """Link to GitHub source code"""
    if domain != 'py':
        return None
    if not info['module']:
        return None
    
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
    
    try:
        fn = inspect.getsourcefile(obj)
        if fn is None:
            return None
        
        source, lineno = inspect.getsourcelines(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(os.path.dirname(__file__)))
        
        github_user = "FanWangEcon"
        github_repo = "PrjLECM"
        github_branch = "main"
        
        return f"https://github.com/{github_user}/{github_repo}/blob/{github_branch}/{fn}#L{lineno}-L{lineno + len(source) - 1}"
    except Exception:
        return None