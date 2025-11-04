"""
MEMIT: Mass-Editing Memory in a Transformer

This module implements the MEMIT algorithm for editing thousands of facts
in transformer models simultaneously.
"""

from .memit_main import apply_memit_to_model
from .memit_hparams import MEMITHyperParams
from .compute_ks import compute_ks
from .compute_z import compute_z

__all__ = [
    "apply_memit_to_model",
    "MEMITHyperParams", 
    "compute_ks",
    "compute_z"
]