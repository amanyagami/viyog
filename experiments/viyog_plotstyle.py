"""Shared matplotlib style for all Viyog experiment plots.

Import this module at the top of every plot script.  Calling ``import
viyog_plotstyle`` (or ``from viyog_plotstyle import *``) applies the shared
rcParams immediately as a side-effect so all subsequent figures inherit the
style automatically.

Typical usage::

    import viyog_plotstyle as vs

    fig, ax = plt.subplots()
    ax.bar(x, y, color=vs.C_OURS)
    vs.add_panel_label(ax, "(a)")
    vs.savefig_pdf(fig, "output/fig1.pdf")

Color palette
-------------
All constants follow the Okabe-Ito colorblind-safe palette so that figures
remain legible in greyscale and for the most common forms of color-vision
deficiency.

Notes:
-----
``pdf.fonttype=42`` and ``ps.fonttype=42`` embed TrueType fonts instead of
Type-3 bitmaps.  This is **required** to pass IEEE PDF-eXpress validation.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Okabe-Ito palette constants
# ---------------------------------------------------------------------------

C_OURS: str = "#009E73"
"""Green — always used for the Viyog method."""

C_BASE: str = "#D55E00"
"""Vermillion — always used for baseline methods."""

C_COMP: str = "#0072B2"
"""Blue — complement statistic / secondary comparisons."""

C_ACCENT: str = "#E69F00"
"""Orange — EOT / stochastic variants."""

C_REF: str = "#555555"
"""Gray — reference lines and annotations."""

C_WARN: str = "#CC79A7"
"""Pink — ViT-B outlier values, warnings."""


# ---------------------------------------------------------------------------
# Style application
# ---------------------------------------------------------------------------


def apply_style() -> None:
    """Apply the shared Viyog rcParams to the current matplotlib session.

    Called automatically at module import time.  Safe to call again if
    rcParams have been modified by other code (e.g. seaborn imports).
    """
    mpl.rcParams.update(
        {
            # Resolution
            "figure.dpi": 150,
            "savefig.dpi": 300,
            # Typography
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            # Spines
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Grid
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.25,
            "grid.linestyle": ":",
            # Font embedding — CRITICAL for IEEE PDF-eXpress
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------


def despine(ax: plt.Axes) -> None:
    """Remove the top and right spines from *ax*.

    Parameters
    ----------
    ax:
        The :class:`matplotlib.axes.Axes` instance to modify in place.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_label(
    ax: plt.Axes,
    label: str,
    x: float = -0.12,
    y: float = 1.05,
) -> None:
    """Add a bold panel letter (e.g. ``'(a)'``) to *ax* in axes coordinates.

    Parameters
    ----------
    ax:
        Target axes.
    label:
        Panel label string, e.g. ``'(a)'`` or ``'A'``.
    x:
        Horizontal position in axes coordinates (default ``-0.12``).
    y:
        Vertical position in axes coordinates (default ``1.05``).
    """
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


# ---------------------------------------------------------------------------
# Figure saving
# ---------------------------------------------------------------------------


def savefig_pdf(fig: plt.Figure, path: str) -> None:
    """Save *fig* to *path* as a PDF with tight bounding box.

    Asserts that ``pdf.fonttype`` is 42 before saving to catch any accidental
    rcParams reset that would produce a PDF-eXpress-rejected file.

    Parameters
    ----------
    fig:
        The :class:`matplotlib.figure.Figure` to save.
    path:
        Destination file path (should end in ``.pdf``).

    Raises:
    ------
    AssertionError
        If ``matplotlib.rcParams['pdf.fonttype']`` is not 42 at call time.
    """
    assert mpl.rcParams["pdf.fonttype"] == 42, (
        f"pdf.fonttype={mpl.rcParams['pdf.fonttype']}; must be 42 for IEEE PDF-eXpress. "
        "Call viyog_plotstyle.apply_style() before saving."
    )
    fig.savefig(path, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Apply at import time
# ---------------------------------------------------------------------------

apply_style()
