"""Compatibility exports for DeepSlice neural network construction.

Historically this module was empty. Keeping these re-exports avoids dead code
while providing a stable import location for architecture helpers.
"""

from .neural_network import initialise_network, load_xception_weights

__all__ = ["initialise_network", "load_xception_weights"]
