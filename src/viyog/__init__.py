"""Viyog package — thin imports and re-exports.

Keep implementation in a separate module to satisfy packaging/tooling rules.
"""

from .main import Viyog, viyog_metrics

__all__ = ["Viyog", "viyog_metrics"]
