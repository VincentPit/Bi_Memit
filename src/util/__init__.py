"""
Utility module compatibility layer for Bi-MEMIT

This module provides compatibility with the original util imports
while maintaining the new project structure.
"""

# Import from the legacy utils for compatibility
import sys
from pathlib import Path

# Add legacy utils to path
legacy_utils_path = Path(__file__).parent / "utils_legacy"
if str(legacy_utils_path) not in sys.path:
    sys.path.insert(0, str(legacy_utils_path))

# Re-export everything from legacy utils for compatibility
try:
    from utils_legacy.globals import *
    from utils_legacy.generate import *
    from utils_legacy.hparams import *
    from utils_legacy.logit_lens import *
    from utils_legacy.nethook import *
    from utils_legacy.perplexity import *
    from utils_legacy.runningstats import *
except ImportError as e:
    print(f"Warning: Could not import legacy utils: {e}")
    # Provide minimal fallbacks if needed