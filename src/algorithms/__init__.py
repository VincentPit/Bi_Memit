"""
Algorithm implementations for model editing.

This module contains implementations of:
- MEMIT: Mass-Editing Memory in a Transformer (with bidirectional enhancements)
- ROME: Rank-One Model Editing (with bidirectional enhancements)
- MEND: Model Editor Networks using Gradient Decomposition
- Bidirectional Core: Advanced bidirectional consistency framework
"""

from .memit import apply_memit_to_model, MEMITHyperParams
from .rome import apply_rome_to_model, ROMEHyperParams
from .mend import MENDHyperParams, MendRewriteExecutor

# Core bidirectional functionality
try:
    from .bidirectional_core import (
        BidirectionalEditProcessor,
        BidirectionalConsistencyTracker,
        RelationshipPreservationModule,
        validate_bidirectional_consistency
    )
    BIDIRECTIONAL_CORE_AVAILABLE = True
except ImportError:
    BIDIRECTIONAL_CORE_AVAILABLE = False

# Enhanced bidirectional algorithms
try:
    from .memit import apply_bidirectional_memit_to_model, BidirectionalMEMIT
    BIDIRECTIONAL_MEMIT_AVAILABLE = True
except ImportError:
    BIDIRECTIONAL_MEMIT_AVAILABLE = False

try:
    from .rome import apply_bidirectional_rome_to_model, BidirectionalROME
    BIDIRECTIONAL_ROME_AVAILABLE = True
except ImportError:
    BIDIRECTIONAL_ROME_AVAILABLE = False

__all__ = [
    "apply_memit_to_model", "MEMITHyperParams",
    "apply_rome_to_model", "ROMEHyperParams", 
    "MENDHyperParams", "MendRewriteExecutor"
]

# Add bidirectional functionality if available
if BIDIRECTIONAL_CORE_AVAILABLE:
    __all__.extend([
        "BidirectionalEditProcessor",
        "BidirectionalConsistencyTracker", 
        "RelationshipPreservationModule",
        "validate_bidirectional_consistency"
    ])

if BIDIRECTIONAL_MEMIT_AVAILABLE:
    __all__.extend([
        "apply_bidirectional_memit_to_model", 
        "BidirectionalMEMIT"
    ])

if BIDIRECTIONAL_ROME_AVAILABLE:
    __all__.extend([
        "apply_bidirectional_rome_to_model",
        "BidirectionalROME"
    ])