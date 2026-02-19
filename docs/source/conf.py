# docs/source/conf.py  (replace or merge into your existing conf.py)

import sys
from pathlib import Path

# repo root so "src" is importable
sys.path.insert(0, str(Path(__file__).parents[2].resolve()))  # repo root so "src" is importable
# If your package root is different, adjust path accordingly (e.g., "../..", "../").

project = "Viyog"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Google / NumPy style docstrings
    "sphinx.ext.autosummary",  # generate summaries
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "recommonmark",  # if you want to use Markdown (or use myst-parser)
    "sphinx_rtd_theme",
    "nbsphinx",  # optional: for Jupyter notebooks in docs
    "sphinx_gallery.gen_gallery",  # optional: for script -> gallery (sphinx-gallery)
]

# autosummary will create .rst files for each listed object under _autosummary/
autosummary_generate = True

# show type hints in descriptions (less noisy signatures)
autodoc_typehints = "description"

# prefer NumPy-style docstrings (you can use google style if you prefer)
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = True

# If torch is not installed in the doc-build environment, mock it so docs can build.
# If you DO have torch installed in the environment, you can remove this line.
autodoc_mock_imports = ["torch"]

# ordering and member options
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# HTML theme (optional)
html_theme = "sphinx_rtd_theme"

# paths for templates/static if you need them remain unchanged
templates_path = ["_templates"]
html_static_path = ["_static"]
