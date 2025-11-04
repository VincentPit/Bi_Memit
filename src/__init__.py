"""
Bi-MEMIT: Revolutionary Bidirectional Model Editing Framework

World's first bidirectional consistency system for transformer editing.
Ensures logical coherence in both forward and backward directions with
novel symmetric parameter updates and relationship preservation.

Original innovation by Stephen Lee (2024-2025).
"""

__version__ = "1.0.0"
__author__ = "Stephen Lee"

try:
    from .algorithms import (
        apply_memit_to_model, 
        MEMITHyperParams,
        apply_rome_to_model, 
        ROMEHyperParams,
        MENDHyperParams, 
        MendRewriteExecutor
    )
    from .utils import generate, nethook, globals as config
    from .data import counterfact, zsre
    from .experiments import evaluate, causal_trace
    
    __all__ = [
        "apply_memit_to_model", "MEMITHyperParams",
        "apply_rome_to_model", "ROMEHyperParams", 
        "MENDHyperParams", "MendRewriteExecutor",
        "generate", "nethook", "config",
        "counterfact", "zsre", 
        "evaluate", "causal_trace"
    ]
except ImportError as e:
    # Graceful fallback if dependencies aren't installed
    import warnings
    warnings.warn(f"Some Bi-MEMIT modules could not be imported: {e}")
    __all__ = []