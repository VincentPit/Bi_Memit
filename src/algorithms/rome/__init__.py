"""
ROME: Rank-One Model Editing

This module implements the ROME algorithm for precise model editing
using rank-one updates to transformer weights, with enhanced bidirectional capabilities.
"""

from .rome_main import apply_rome_to_model
from .rome_hparams import ROMEHyperParams
from .compute_u import compute_u
from .compute_v import compute_v
from .layer_stats import layer_stats

# Bidirectional ROME implementation
try:
    from .bidirectional_rome import (
        apply_bidirectional_rome_to_model,
        BidirectionalROME,
        compute_rome_bidirectional_metrics
    )
    BIDIRECTIONAL_AVAILABLE = True
except ImportError:
    BIDIRECTIONAL_AVAILABLE = False

__all__ = [
    "apply_rome_to_model",
    "ROMEHyperParams",
    "compute_u", 
    "compute_v",
    "layer_stats"
]

# Add bidirectional exports if available
if BIDIRECTIONAL_AVAILABLE:
    __all__.extend([
        "apply_bidirectional_rome_to_model",
        "BidirectionalROME", 
        "compute_rome_bidirectional_metrics"
    ])
