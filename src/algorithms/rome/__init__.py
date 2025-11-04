"""
ROME: Rank-One Model Editing

This module implements the ROME algorithm for precise model editing
using rank-one updates to transformer weights.
"""

from .rome_main import apply_rome_to_model
from .rome_hparams import ROMEHyperParams
from .compute_u import compute_u
from .compute_v import compute_v
from .layer_stats import layer_stats

__all__ = [
    "apply_rome_to_model",
    "ROMEHyperParams",
    "compute_u", 
    "compute_v",
    "layer_stats"
]
