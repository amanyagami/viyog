"""Sphinx configuration for the Viyog documentation."""

import sys
from pathlib import Path

# make the installable package (src/ layout) importable for autodoc
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

project = "Viyog"
author = "Aman Singh"
copyright = "2026, Aman Singh"

# read the version without importing the package (torch isn't installed on RTD)
_init = (Path(__file__).parents[2] / "src" / "viyog" / "__init__.py").read_text()
release = next((ln.split('"')[1] for ln in _init.splitlines()
                if ln.startswith("__version__")), "0.0.0")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",      # Google-style docstrings
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "show-inheritance": True}

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

# heavy/optional runtime deps are mocked so the docs build without installing them
autodoc_mock_imports = ["torch", "numpy", "sklearn", "scipy"]

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

html_theme = "sphinx_rtd_theme"
templates_path = []
html_static_path = []
