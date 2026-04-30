# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Neural Reconstruction Engine (NRE)"
copyright = "2024, NVIDIA"
author = "NVIDIA - Toronto AI Lab"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinxcontrib.autodoc_pydantic",
]

# autosummary_generate = True

todo_include_todos = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "PyTorch": ("https://pytorch.org/docs/2.4/", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable/", None),
}

autodoc_default_options = {
    "show-inheritance": False,
    "undoc-members": False,
}

autodoc_mock_imports = [
    "tinycudann",  # ignore for docs as some CI machines might require a special CC70 version of the wheel
    "libs.nrend",
    "libs.nrend.renderer",
    "libs.nrend.renderer_test_case",
]

master_doc = "index"

templates_path = ["_templates"]
exclude_patterns = ["_build"]
html_static_path = ["_static"]

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
import sphinx_rtd_theme


html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": -1,
}
