"""Stroop LMM package shim (points to publication/3_stroop_lmm)."""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
_IMPL_DIR = Path(__file__).resolve().parent.parent / "3_stroop_lmm"
if _IMPL_DIR.exists():
    __path__.append(str(_IMPL_DIR))
