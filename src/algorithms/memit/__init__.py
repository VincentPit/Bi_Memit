"""
MEMIT: Mass-Editing Memory in a Transformer

This module implements the MEMIT algorithm for editing thousands of facts
in transformer models simultaneously, with enhanced bidirectional capabilities.
"""

from .memit_main import apply_memit_to_model
from .memit_hparams import MEMITHyperParams
from .compute_ks import compute_ks
from .compute_z import compute_z

# Bidirectional MEMIT implementation
try:
    from .bidirectional_memit import (
        apply_bidirectional_memit_to_model,
        BidirectionalMEMIT
    )
    BIDIRECTIONAL_AVAILABLE = True
except ImportError:
    BIDIRECTIONAL_AVAILABLE = False

__all__ = [
    "apply_memit_to_model",
    "MEMITHyperParams", 
    "compute_ks",
    "compute_z"
]

# Add bidirectional exports if available
if BIDIRECTIONAL_AVAILABLE:
    __all__.extend([
        "apply_bidirectional_memit_to_model",
        "BidirectionalMEMIT"
    ])