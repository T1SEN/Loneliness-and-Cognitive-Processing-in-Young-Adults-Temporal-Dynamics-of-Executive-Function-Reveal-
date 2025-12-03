"""
Backward compatibility shim.
The actual implementation lives in analysis/utils/data_loader_utils.py.
"""

import sys
from pathlib import Path

# Ensure parent directory is in path for imports
_this_dir = Path(__file__).parent
if str(_this_dir.parent) not in sys.path:
    sys.path.insert(0, str(_this_dir.parent))

from analysis.utils.data_loader_utils import *  # noqa: F401,F403
