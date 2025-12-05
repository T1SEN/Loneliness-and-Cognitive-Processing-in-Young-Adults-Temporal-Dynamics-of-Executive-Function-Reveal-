"""
Legacy Data Loader Compatibility Layer
======================================

Archive scripts import ``analysis.utils.data_loader_utils`` (and even
``utils.data_loader_utils``) to access master datasets, summary tables,
and shared constants.  The active codebase has migrated these helpers
into ``analysis.preprocessing``; this module simply re-exports the new
implementations so the legacy scripts keep working without modification.
"""

from analysis import preprocessing as _pre
from analysis.preprocessing import *  # noqa: F401,F403

# Mirror the public API from analysis.preprocessing for compatibility
__all__ = getattr(_pre, "__all__", [])
