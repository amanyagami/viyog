"""Viyog — separate adversarial (ADV) from out-of-distribution (OOD) inputs.

A training-free, gradient-free, post-hoc detector that reads the dormant-band
roughness of a model's first convolutional layer. See :class:`Viyog`.
"""

from .main import Viyog, viyog_metrics

__version__ = "0.1.3"
__all__ = ["Viyog", "__version__", "viyog_metrics"]
