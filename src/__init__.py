"""
Bi-MEMIT: Bi-directional Mass Editing Memory in a Transformer

A library for efficiently editing thousands of facts in transformer models
using MEMIT, ROME, and MEND algorithms.
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