"""
Gender Interaction Analyses
===========================

UCLA x Gender interaction tests and synthesis.
"""

from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "results" / "publication" / "gender_analysis" / "interactions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

__all__ = ['OUTPUT_DIR']
