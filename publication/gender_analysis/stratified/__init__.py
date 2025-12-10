"""
Gender-Stratified Analyses
==========================

Task-specific analyses stratified by gender.
"""

from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "results" / "publication" / "gender_analysis" / "stratified"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

__all__ = ['OUTPUT_DIR']
